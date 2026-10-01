import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sampler import BalancedMixedIndexSampler, NegativeNeighborIndex


def _global_neighbors(image_by_row, texts, rows, representatives, unique_images,
                      device, topk=64, chunk_size=128):
    pids = torch.as_tensor([int(row[0]) for row in rows], device=device)
    unique_pids = pids[torch.as_tensor(representatives, device=device)]
    directions = (
        (image_by_row, texts, pids),
        (texts, unique_images, unique_pids),
        (texts, texts, pids),
    )
    result = np.full((len(rows), 3, topk), -1, dtype=np.int32)
    row_map = torch.as_tensor(representatives, device=device)
    for direction, (queries, candidates, candidate_pids) in enumerate(directions):
        width = min(topk, len(candidates))
        for start in range(0, len(rows), chunk_size):
            scores = queries[start:start + chunk_size] @ candidates.T
            scores.masked_fill_(pids[start:start + chunk_size, None].eq(candidate_pids), -torch.inf)
            values, indices = scores.topk(width, dim=1)
            if direction == 1:
                indices = row_map[indices]
            indices[~values.isfinite()] = -1
            result[start:start + len(indices), direction, :width] = indices.cpu().numpy()
    return result


@torch.inference_mode()
def refresh_balanced_mixed_epoch(train_loader, model, args, epoch, logger=None):
    sampler = getattr(train_loader, "sampler", None)
    if not isinstance(sampler, BalancedMixedIndexSampler):
        return

    sampler.set_epoch(epoch)
    interval = max(1, int(args.sampler_refresh_every))
    previous = getattr(sampler, "last_refresh_epoch", None)
    if sampler.negative_index is not None and previous is not None and epoch - previous < interval:
        return

    started = time.time()
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms

    rows = sampler.rows
    first_by_image = {}
    for row_index, row in enumerate(rows):
        first_by_image.setdefault(int(row[1]), row_index)
    representatives = list(first_by_image.values())
    eval_transform = build_transforms(img_size=args.img_size, is_train=False)
    image_set = ImageDataset(
        [rows[index][0] for index in representatives],
        [rows[index][2] for index in representatives],
        eval_transform,
    )
    text_set = TextDataset([row[0] for row in rows], [row[3] for row in rows],
                           text_length=args.text_length)
    batch_size = min(64, int(args.batch_size))
    workers = int(args.num_workers)
    image_loader = DataLoader(image_set, batch_size=batch_size, num_workers=workers,
                              shuffle=False, pin_memory=True)
    text_loader = DataLoader(text_set, batch_size=batch_size, num_workers=workers,
                             shuffle=False, pin_memory=True)

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    image_features, text_features = [], []
    is_dm_adapter = hasattr(args, "num_experts")
    try:
        for _, images in image_loader:
            images = images.to(device, non_blocking=True)
            features = (model.encode_image(images, 0) if is_dm_adapter
                        else model.encode_image(images))
            if isinstance(features, (tuple, list)):
                features = features[0]
            image_features.append(F.normalize(features.float(), dim=-1).cpu())
        for _, tokens in text_loader:
            tokens = tokens.to(device, non_blocking=True)
            features = (model.encode_text(tokens, 0) if is_dm_adapter
                        else model.encode_text(tokens))
            if isinstance(features, (tuple, list)):
                features = features[0]
            text_features.append(F.normalize(features.float(), dim=-1).cpu())
    finally:
        model.train(was_training)

    unique_images = torch.cat(image_features).to(device)
    image_position = {image_id: pos for pos, image_id in enumerate(first_by_image)}
    image_by_row = unique_images[[image_position[int(row[1])] for row in rows]]
    text_matrix = torch.cat(text_features).to(device)
    ranked = _global_neighbors(image_by_row, text_matrix, rows, representatives,
                               unique_images, device)
    index = NegativeNeighborIndex(
        rows, ranked, checkpoint=f"{type(model).__name__}:epoch-{epoch}"
    )
    sampler.set_negative_index(index)
    sampler.last_refresh_epoch = int(epoch)
    message = (f"Balanced Mixed mining refreshed at epoch {epoch}: "
               f"{len(rows)} train records, {len(representatives)} unique images, "
               f"elapsed {time.time() - started:.1f}s")
    (logger or logging.getLogger("sampler")).info(message)
