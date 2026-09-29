import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dataio import PairDataset, read_rows
from engine import Encoder, extract_features, state_digest
from sampler import NegativeNeighborIndex, row_fingerprint


def file_digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_cache(path, rows, model_hash):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['model_sha256'] != model_hash or meta['rows'] != row_fingerprint(rows):
        raise ValueError('Cache must use the same initial CLIP weights and training rows')
    if meta['sha256'] != file_digest(path):
        raise ValueError('Cache checksum mismatch')
    if meta['topk'] != 64:
        raise ValueError('This experiment requires 64 cached negative ranks')
    return NegativeNeighborIndex.load(path, rows), meta


@torch.no_grad()
def neighbors(images, texts, rows, device, topk=64, chunk=128):
    first = {}
    for index, row in enumerate(rows):
        first.setdefault(row[1], index)
    representative = torch.tensor(list(first.values()), dtype=torch.long, device=device)
    pid = torch.tensor([r[0] for r in rows], device=device)
    images, texts = images.to(device), texts.to(device)
    gallery = images[representative]
    result = np.full((len(rows), 3, topk), -1, dtype=np.int32)
    directions = [(images, texts, pid, None),
                  (texts, gallery, pid[representative], representative),
                  (texts, texts, pid, None)]
    for direction, (queries, candidates, candidate_pid, mapping) in enumerate(directions):
        k = min(topk, len(candidates))
        for start in range(0, len(rows), chunk):
            scores = queries[start:start + chunk] @ candidates.T
            scores.masked_fill_(pid[start:start + chunk, None].eq(candidate_pid), -torch.inf)
            values, indices = scores.topk(k, dim=1)
            if mapping is not None:
                indices = mapping[indices]
            indices[~values.isfinite()] = -1
            result[start:start + len(indices), direction, :k] = indices.cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default='cache/pretrained_neighbors.npz')
    parser.add_argument('--pretrained', default='ViT-B/16')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Build the full training cache on Kaggle GPU')
    rows = read_rows(args.root, 'train')
    model = Encoder(args.pretrained)
    model_hash = state_digest(model)
    output = Path(args.output)
    if output.exists():
        validate_cache(output, rows, model_hash)
        print(f'Existing cache verified: {output}', flush=True)
        return
    model.cuda()
    images, texts = extract_features(model, PairDataset(rows), args.batch_size, args.workers)
    ranked = neighbors(images, texts, rows, 'cuda')
    cache = NegativeNeighborIndex(rows, ranked, checkpoint=f'initial-clip:{model_hash}')
    cache.save(output)
    meta = dict(model_sha256=model_hash, rows=row_fingerprint(rows), topk=64,
                sha256=file_digest(output), split='train', pretrained=args.pretrained,
                image_size=[384, 128], frozen=True)
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f'Created frozen training-only cache: {output}', flush=True)


if __name__ == '__main__':
    main()
