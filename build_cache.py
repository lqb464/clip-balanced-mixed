import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cache_utils import neighbors, validate_cache
from rde_runtime import (PROTOCOL, activate_rde, file_digest, seed_everything,
                         state_digest, write_json)
from sampler import NegativeNeighborIndex, row_fingerprint

activate_rde()
from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.options import get_args


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default='cache/rde_pretrained_neighbors.npz')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Build this cache on a CUDA GPU')
    seed_everything(1)
    dataset = RSTPReid(args.root)
    rows = dataset.train
    settings = get_args([])
    model = build_model(settings, len(dataset.train_id_container))
    model_hash = state_digest(model.base_model)
    output = Path(args.output)
    if output.exists():
        validate_cache(output, rows, model_hash)
        print(f'Cache verified: {output}', flush=True)
        return
    model.cuda().eval()
    first = {}
    for i, row in enumerate(rows):
        first.setdefault(row[1], i)
    unique_rows = list(first.values())
    images = ImageDataset([rows[i][0] for i in unique_rows],
                          [rows[i][2] for i in unique_rows], build_transforms(is_train=False))
    texts = TextDataset([r[0] for r in rows], [r[3] for r in rows])
    image_features, text_features = [], []
    for _, batch in DataLoader(images, args.batch_size, num_workers=args.workers):
        image_features.append(F.normalize(model.encode_image(batch.cuda()), dim=1).cpu())
    for _, batch in DataLoader(texts, args.batch_size, num_workers=args.workers):
        text_features.append(F.normalize(model.encode_text(batch.cuda()), dim=1).cpu())
    images_by_id = torch.cat(image_features)
    image_position = {rows[i][1]: pos for pos, i in enumerate(unique_rows)}
    images_by_row = images_by_id[[image_position[r[1]] for r in rows]]
    ranked = neighbors(images_by_row, torch.cat(text_features), rows, 'cuda')
    index = NegativeNeighborIndex(rows, ranked, checkpoint=f'initial-rde-clip:{model_hash}')
    index.save(output)
    write_json(output.with_suffix('.json'), dict(protocol=PROTOCOL, model_sha256=model_hash,
               rows=row_fingerprint(rows), topk=64, sha256=file_digest(output),
               split='train', pretrained='ViT-B/16', image_size=[384, 128],
               frozen=True, features='initial BGE CLS/EOT', precision='original RDE mixed fp16'))
    print(f'Created training-only cache: {output}', flush=True)


if __name__ == '__main__':
    main()
