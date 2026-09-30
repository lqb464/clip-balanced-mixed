import json
from pathlib import Path

import numpy as np
import torch

from rde_runtime import PROTOCOL, file_digest
from sampler import NegativeNeighborIndex, row_fingerprint


def validate_cache(path, rows, model_hash=None):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta.get('protocol') != PROTOCOL:
        raise ValueError('This RDE experiment requires a newly built RDE cache')
    if meta['rows'] != row_fingerprint(rows) or meta['sha256'] != file_digest(path):
        raise ValueError('Cache dataset/checksum mismatch')
    if model_hash is not None and meta['model_sha256'] != model_hash:
        raise ValueError('Cache must use the identical initial RDE CLIP backbone')
    if meta['topk'] != 64 or meta['split'] != 'train':
        raise ValueError('Expected 64 negative ranks from the training split only')
    return NegativeNeighborIndex.load(path, rows), meta


@torch.no_grad()
def neighbors(images, texts, rows, device, topk=64, chunk=128):
    first = {}
    for index, row in enumerate(rows):
        first.setdefault(row[1], index)
    representative = torch.tensor(list(first.values()), dtype=torch.long, device=device)
    pid = torch.tensor([r[0] for r in rows], device=device)
    images, texts = images.to(device), texts.to(device)
    result = np.full((len(rows), 3, topk), -1, dtype=np.int32)
    directions = [(images, texts, pid, None),
                  (texts, images[representative], pid[representative], representative),
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
