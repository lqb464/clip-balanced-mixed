import hashlib
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from vendor.clip_model import build_CLIP_from_openai_pretrained


class Encoder(nn.Module):
    def __init__(self, pretrained='ViT-B/16'):
        super().__init__()
        self.clip, _ = build_CLIP_from_openai_pretrained(pretrained, (384, 128), 16)
        self.clip.float()

    def forward(self, images, tokens):
        image_features, text_features = self.clip(images, tokens)
        return (image_features[:, 0].float(),
                text_features[torch.arange(len(tokens), device=tokens.device),
                              tokens.argmax(-1)].float())


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def retrieval_loss(images, texts, pids, scale=50.0):
    logits = scale * (F.normalize(images.float(), dim=-1) @
                      F.normalize(texts.float(), dim=-1).T)
    labels = pids[:, None].eq(pids[None, :]).float()
    labels = labels / labels.sum(1, keepdim=True)
    target_log = (labels + 1e-8).log()
    return sum((F.softmax(scores, 1) * (F.log_softmax(scores, 1) - target_log))
               .sum(1).mean() for scores in (logits, logits.T))


def gradient_cached_step(model, batch, optimizer, scaler, microbatch=16):
    """Recompute encoder chunks; compute one loss over the entire sampler batch.

    This backbone has no dropout or batch normalization. The no-grad pass and
    replay therefore produce the same features without storing activations.
    """
    device = next(model.parameters()).device
    images, tokens, pids, image_ids = batch
    optimizer.zero_grad(set_to_none=True)
    cached_images, cached_texts = [], []
    with torch.no_grad():
        for start in range(0, len(pids), microbatch):
            with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                i, t = model(images[start:start + microbatch].to(device),
                             tokens[start:start + microbatch].to(device))
            cached_images.append(i)
            cached_texts.append(t)
    i = torch.cat(cached_images).detach().requires_grad_()
    t = torch.cat(cached_texts).detach().requires_grad_()
    pid = pids.to(device)
    loss = retrieval_loss(i, t, pid)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite loss: {loss.item()}')
    scaler.scale(loss).backward()
    for start in range(0, len(pids), microbatch):
        with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.float16):
            outputs = model(images[start:start + microbatch].to(device),
                            tokens[start:start + microbatch].to(device))
        torch.autograd.backward(outputs, (i.grad[start:start + microbatch],
                                         t.grad[start:start + microbatch]))
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        same = pids[:, None].eq(pids[None, :])
        same.fill_diagonal_(False)
        pairs = torch.triu(same, 1)
        duplicate = image_ids[:, None].eq(image_ids[None, :]) & pairs
    return {'loss': loss.item(), 'unique_pids': pids.unique().numel(),
            'positive_anchor_fraction': same.any(1).float().mean().item(),
            'positive_pairs': pairs.sum().item(), 'same_image_pairs': duplicate.sum().item(),
            'amp_skipped_step': int(scaler.get_scale() < old_scale)}


@torch.no_grad()
def extract_features(model, dataset, batch_size=64, workers=2):
    model.eval()
    device = next(model.parameters()).device
    images, texts = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    for batch in loader:
        with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.float16):
            i, t = model(batch[0].to(device), batch[1].to(device))
        images.append(F.normalize(i.float(), dim=1).cpu())
        texts.append(F.normalize(t.float(), dim=1).cpu())
    return torch.cat(images), torch.cat(texts)


def retrieval_metrics(images, texts, rows):
    first = {}
    for index, row in enumerate(rows):
        first.setdefault(row[1], index)
    gallery_indices = list(first.values())
    gallery = images[gallery_indices]
    pids = torch.tensor([r[0] for r in rows])
    gallery_pids = pids[gallery_indices]
    totals = dict(R1=0.0, R5=0.0, R10=0.0, mAP=0.0, mINP=0.0)
    for start in range(0, len(texts), 256):
        ranks = (texts[start:start + 256] @ gallery.T).argsort(dim=1, descending=True, stable=True)
        matches = gallery_pids[ranks].eq(pids[start:start + 256, None])
        counts = matches.sum(1)
        if (counts == 0).any():
            raise ValueError('Query has no same-PID gallery image')
        position = torch.arange(1, len(gallery) + 1)
        for k in (1, 5, 10):
            totals[f'R{k}'] += matches[:, :k].any(1).sum().item()
        totals['mAP'] += ((matches.cumsum(1) / position) * matches).sum(1).div(counts).sum().item()
        last = (matches * position).max(1).values
        totals['mINP'] += (counts / last).sum().item()
    return {k: 100 * v / len(texts) for k, v in totals.items()}


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def learning_rate(base, epoch, epochs):
    if epoch < 5:
        return base * (0.1 + 0.9 * epoch / 5)
    return base * (1 + math.cos(math.pi * (epoch - 5) / max(1, epochs - 5))) / 2
