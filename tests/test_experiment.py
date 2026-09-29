import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from dataio import PairDataset, read_rows

from build_cache import neighbors
from engine import atomic_save, gradient_cached_step, retrieval_loss, retrieval_metrics
from sampler import BalancedMixedIndexSampler, NegativeNeighborIndex
from train import epoch_indices
from vendor.clip_model import CLIP
from engine import Encoder


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(8, 5)
        self.text = nn.Linear(8, 5)

    def forward(self, images, texts):
        return self.image(images), self.text(texts)


class ExperimentTests(unittest.TestCase):
    def test_annotation_pairs_and_resumed_augmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'RSTPReid'
            (root / 'imgs').mkdir(parents=True)
            pixels = np.arange(40 * 30 * 3, dtype=np.uint8).reshape(40, 30, 3)
            Image.fromarray(pixels).save(root / 'imgs' / 'one.png')
            (root / 'data_captions.json').write_text(json.dumps([
                dict(split='train', id=1, img_path='one.png', captions=['first', 'second'])]))
            rows = read_rows(folder, 'train')
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1:3], rows[1][1:3])
            dataset = PairDataset(rows, training=True, seed=4, epoch=7)
            image = dataset[1][0]
            torch.manual_seed(918)
            restored = PairDataset(rows, training=True, seed=4, epoch=7)
            torch.testing.assert_close(image, restored[1][0], atol=0, rtol=0)

    def test_actual_clip_backbone_gradient_replay(self):
        torch.manual_seed(1)
        full = Encoder.__new__(Encoder)
        nn.Module.__init__(full)
        full.clip = CLIP(embed_dim=32, image_resolution=(32, 16), vision_layers=1,
                         vision_width=64, vision_patch_size=16, stride_size=16,
                         context_length=5, vocab_size=8, transformer_width=64,
                         transformer_heads=1, transformer_layers=1)
        cached = copy.deepcopy(full)
        batch = (torch.randn(4, 3, 32, 16), torch.tensor([[6, 1, 7, 0, 0]] * 4),
                 torch.tensor([0, 0, 1, 2]), torch.arange(4))
        opt = torch.optim.SGD(full.parameters(), lr=0.01)
        i, t = full(*batch[:2])
        loss = retrieval_loss(i, t, batch[2])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
        opt.step()
        gradient_cached_step(cached, batch, torch.optim.SGD(cached.parameters(), lr=0.01),
                             torch.amp.GradScaler('cuda', enabled=False), 2)
        for a, b in zip(full.parameters(), cached.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)

    def test_optimizer_checkpoint_resumes_same_updates(self):
        torch.manual_seed(17)
        model = ToyEncoder()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        batch = (torch.randn(8, 8), torch.randn(8, 8), torch.arange(8) // 2, torch.arange(8))
        gradient_cached_step(model, batch, optimizer, scaler, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict()), path)
            gradient_cached_step(model, batch, optimizer, scaler, 3)
            restored = ToyEncoder()
            opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            saved = torch.load(path, weights_only=True)
            restored.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            gradient_cached_step(restored, batch, opt, scaler, 3)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_cached_gradient_matches_full_batch_sdm(self):
        torch.manual_seed(8)
        full = ToyEncoder()
        cached = copy.deepcopy(full)
        batch = (torch.randn(17, 8), torch.randn(17, 8),
                 torch.arange(17) // 2, torch.arange(17))
        optimizer = torch.optim.SGD(full.parameters(), lr=0.01)
        images, texts = full(*batch[:2])
        loss = retrieval_loss(images, texts, batch[2])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
        optimizer.step()
        cached_optimizer = torch.optim.SGD(cached.parameters(), lr=0.01)
        result = gradient_cached_step(cached, batch, cached_optimizer,
                                      torch.amp.GradScaler('cuda', enabled=False), 4)
        self.assertAlmostEqual(result['loss'], loss.item(), places=5)
        for a, b in zip(full.parameters(), cached.parameters()):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_unique_image_gallery_and_multiple_pid_positives(self):
        rows = [(1, 0, 'a', 'a'), (1, 0, 'a', 'b'), (2, 1, 'b', 'c')]
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        metrics = retrieval_metrics(features, features, rows)
        for value in metrics.values():
            self.assertAlmostEqual(value, 100.0)
        wrong_text = features[[2, 2, 0]]
        metrics = retrieval_metrics(features, wrong_text, rows)
        self.assertEqual(metrics['R1'], 0.0)
        self.assertEqual(metrics['mAP'], 50.0)

    def test_cache_excludes_same_pid_and_maps_images(self):
        rows = [(pid, pid * 2 + image, f'{pid}-{image}', str(caption))
                for pid in range(4) for image in range(2) for caption in range(2)]
        torch.manual_seed(3)
        features = torch.nn.functional.normalize(torch.randn(len(rows), 6), dim=1)
        ranked = neighbors(features, features, rows, 'cpu', topk=3, chunk=3)
        index = NegativeNeighborIndex(rows, ranked)
        for row in range(len(rows)):
            for candidate in index.neighbors[row].flatten():
                if candidate >= 0:
                    self.assertNotEqual(rows[row][0], rows[candidate][0])
        self.assertTrue((ranked[:, 1] % 2 == 0).all())

    def test_sampler_coverage_and_resume_order(self):
        rows = [(pid, pid * 5 + image, f'{pid}-{image}', str(caption))
                for pid in range(150) for image in range(5) for caption in range(2)]
        candidates = np.array([[(i + j * 10) % len(rows) for j in range(1, 65)]
                               for i in range(len(rows))], dtype=np.int32)
        index = NegativeNeighborIndex(rows, np.repeat(candidates[:, None, :], 3, axis=1))
        for name in ('random', 'balanced_mixed'):
            indices, _ = epoch_indices(rows, name, 1, 2, index)
            self.assertEqual(sorted(indices), list(range(len(rows))))
            repeated, _ = epoch_indices(rows, name, 1, 2, index)
            self.assertEqual(indices[128 * 3:], repeated[128 * 3:])
            other, _ = epoch_indices(rows, name, 1, 3, index)
            self.assertNotEqual(indices, other)
        sampler = BalancedMixedIndexSampler(rows, 128, 4, negative_index=index)
        order = list(sampler)
        self.assertEqual(sampler.diagnostics['mining_enabled'], True)
        first = [rows[i] for i in order[:128]]
        self.assertEqual(len({r[0] for r in first}), 124)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'last.pt'
            atomic_save({'epoch': 2, 'next_batch': 3}, checkpoint)
            self.assertEqual(torch.load(checkpoint, weights_only=True)['next_batch'], 3)
            self.assertFalse(checkpoint.with_suffix('.pt.tmp').exists())


if __name__ == '__main__':
    unittest.main()
