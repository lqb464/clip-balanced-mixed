import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_cache import validate_cache
from dataio import PairDataset, read_rows
from engine import (Encoder, atomic_save, extract_features, gradient_cached_step,
                    learning_rate, restore_rng, retrieval_metrics, rng_state, state_digest)
from sampler import BalancedMixedIndexSampler, row_fingerprint


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def epoch_indices(rows, name, seed, epoch, cache=None):
    if name == 'random':
        return np.random.default_rng(seed + epoch).permutation(len(rows)).tolist(), {}
    if cache is None:
        raise ValueError('Balanced Mixed requires its verified negative cache')
    sampler = BalancedMixedIndexSampler(rows, 128, 4, negative_index=cache, seed=seed)
    sampler.set_epoch(epoch)
    return list(sampler), sampler.diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--sampler', choices=['random', 'balanced_mixed'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default='cache/pretrained_neighbors.npz')
    parser.add_argument('--pretrained', default='ViT-B/16')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, choices=[128], default=128)
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-hours', type=float, default=9)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.microbatch, args.eval_batch_size, args.save_every) < 1:
        parser.error('Epochs, microbatch, evaluation batch and save interval must be positive')
    if args.max_hours <= 0 or args.lr <= 0:
        parser.error('max-hours and lr must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; use the Kaggle notebook')
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / 'last.pt'
    if last_path.exists() and not args.resume:
        raise FileExistsError(f'{last_path} exists; use --resume')
    if args.resume and not last_path.exists():
        raise FileNotFoundError(last_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    train_rows = read_rows(args.root, 'train')
    val_rows = read_rows(args.root, 'val')
    test_rows = read_rows(args.root, 'test')
    splits = [{r[0] for r in rows} for rows in (train_rows, val_rows, test_rows)]
    if any(splits[a] & splits[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Train, val and test PID sets must be disjoint')
    model = Encoder(args.pretrained)
    model_hash = state_digest(model)
    cache, cache_meta = None, {}
    if args.sampler == 'balanced_mixed':
        cache, cache_meta = validate_cache(args.cache, train_rows, model_hash)
    signature = dict(sampler=args.sampler, loss='sdm', epochs=args.epochs,
                     torch_version=str(torch.__version__), numpy_version=np.__version__,
                     batch_size=128, microbatch=args.microbatch, seed=args.seed, lr=args.lr,
                     weight_decay=4e-5, initial_model=model_hash,
                     train_rows=row_fingerprint(train_rows),
                     val_rows=row_fingerprint(val_rows), test_rows=row_fingerprint(test_rows),
                     cache=cache_meta.get('sha256'), image_size=[384, 128],
                     optimizer='AdamW', schedule='5_epoch_warmup_cosine', temperature=0.02)
    model.cuda()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=4e-5, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda')
    epoch, next_batch, best_r1, best_epoch = 0, 0, -1.0, None
    history, accumulated = [], []
    if args.resume:
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            changed = [k for k in signature if saved['signature'].get(k) != signature[k]]
            raise ValueError(f'Resume configuration/data/cache changed: {changed}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        epoch, next_batch = saved['epoch'], saved['next_batch']
        best_r1, best_epoch = saved['best_r1'], saved['best_epoch']
        history, accumulated = saved['history'], saved['accumulated']
        restore_rng(saved['rng'])
        if best_epoch is not None and not (output / 'best.pt').exists():
            raise FileNotFoundError('Restore best.pt together with last.pt')
        del saved
    write_json(output / 'config.json', {**vars(args), 'signature': signature,
                                      'torch': torch.__version__})
    train_data = PairDataset(train_rows, training=True, seed=args.seed)
    val_data = PairDataset(val_rows)

    def save():
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, next_batch=next_batch,
                         best_r1=best_r1, best_epoch=best_epoch, history=history,
                         accumulated=accumulated, signature=signature, rng=rng_state()), last_path)
        write_json(output / 'metrics.json', history)
        write_json(output / 'progress.json', dict(completed_epochs=epoch,
                   next_batch=next_batch, target_epochs=args.epochs, best_val_R1=best_r1))

    def expired():
        return time.monotonic() - started >= args.max_hours * 3600

    while epoch < args.epochs:
        indices, diagnostics = epoch_indices(train_rows, args.sampler, args.seed, epoch, cache)
        train_data.epoch = epoch
        loader = DataLoader(train_data, batch_size=128, sampler=indices[next_batch * 128:],
                            drop_last=False, num_workers=args.workers, pin_memory=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        lr = learning_rate(args.lr, epoch, args.epochs)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        epoch_started = time.monotonic()
        for batch in loader:
            stats = gradient_cached_step(model, batch, optimizer, scaler, args.microbatch)
            stats['samples'] = len(batch[2])
            accumulated.append(stats)
            next_batch += 1
            if next_batch == 1 or next_batch % 10 == 0:
                print(f'{args.sampler} epoch {epoch + 1}/{args.epochs} '
                      f'batch {next_batch}/{math.ceil(len(indices)/128)} '
                      f'loss={stats["loss"]:.4f} '
                      f'peak_GPU_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}', flush=True)
            if next_batch % args.save_every == 0 or expired():
                save()
            if expired():
                print('Paused with last.pt saved. Resume to reach 60 total epochs.', flush=True)
                return
        i, t = extract_features(model, val_data, args.eval_batch_size, args.workers)
        val = retrieval_metrics(i, t, val_rows)
        if val['R1'] > best_r1:
            best_r1, best_epoch = val['R1'], epoch + 1
            atomic_save(dict(model=model.state_dict(), epoch=best_epoch,
                             val=val, signature=signature), output / 'best.pt')
        total = sum(s['samples'] for s in accumulated)
        means = {key: sum(s[key] * s['samples'] for s in accumulated) / total
                 for key in ('loss', 'positive_anchor_fraction')}
        means.update({key: sum(s[key] for s in accumulated) / len(accumulated)
                      for key in ('unique_pids', 'positive_pairs')})
        means['same_image_pairs'] = sum(s['same_image_pairs'] for s in accumulated)
        means['amp_skipped_steps'] = sum(s['amp_skipped_step'] for s in accumulated)
        means['samples_seen'] = total
        history.append(dict(epoch=epoch + 1, lr=lr, train=means, val=val,
                            sampler=diagnostics, session_epoch_seconds=time.monotonic()-epoch_started))
        print(json.dumps(history[-1]), flush=True)
        epoch += 1
        next_batch, accumulated = 0, []
        save()
        if expired() and epoch < args.epochs:
            return
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    del best
    i, t = extract_features(model, PairDataset(test_rows), args.eval_batch_size, args.workers)
    result = dict(status='complete', completed_epochs=epoch, best_epoch=best_epoch,
                  best_val_R1=best_r1, test=retrieval_metrics(i, t, test_rows), signature=signature)
    write_json(output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
