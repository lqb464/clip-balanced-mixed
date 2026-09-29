import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from vendor.simple_tokenizer import SimpleTokenizer


def read_rows(root, split):
    root = Path(root)
    if not (root / 'data_captions.json').is_file():
        root = root / 'RSTPReid'
    with (root / 'data_captions.json').open(encoding='utf-8') as f:
        annotations = json.load(f)
    rows = []
    image_id = 0
    for item in annotations:
        if item['split'] != split:
            continue
        path = root / 'imgs' / item['img_path']
        if not path.is_file():
            raise FileNotFoundError(path)
        for caption in item['captions']:
            rows.append((int(item['id']), image_id, str(path), caption))
        image_id += 1
    if not rows:
        raise ValueError(f'Empty {split} split at {root}')
    return rows


def image_transform(training):
    operations = [T.Resize((384, 128), interpolation=InterpolationMode.BICUBIC)]
    if training:
        operations += [T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((384, 128))]
    operations += [T.ToTensor(), T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                           (0.26862954, 0.26130258, 0.27577711))]
    if training:
        operations += [T.RandomErasing(p=0.5, scale=(0.02, 0.4), value=0)]
    return T.Compose(operations)


class PairDataset(Dataset):
    def __init__(self, rows, training=False, seed=1, epoch=0):
        self.rows, self.seed, self.epoch = rows, seed, epoch
        self.training = training
        self.transform = image_transform(training)
        tokenizer = SimpleTokenizer()
        start, end = tokenizer.encoder['<|startoftext|>'], tokenizer.encoder['<|endoftext|>']
        self.tokens = torch.zeros(len(rows), 77, dtype=torch.long)
        for i, row in enumerate(rows):
            ids = [start] + tokenizer.encode(row[3])[:75] + [end]
            self.tokens[i, :len(ids)] = torch.tensor(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        pid, image_id, path, _ = self.rows[index]
        with Image.open(path) as image:
            # Row/epoch-based augmentation is stable after skipping resumed batches.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + self.epoch * len(self.rows) + index)
                image = self.transform(image.convert('RGB'))
        return image, self.tokens[index], pid, image_id
