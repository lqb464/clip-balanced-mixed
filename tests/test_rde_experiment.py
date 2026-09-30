import copy
import ast
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from cache_utils import neighbors, validate_cache
from compare import assert_reference_config, compare
from launch import command_for
from rde_runtime import (PROTOCOL, RDE_ROOT, activate_rde, file_digest, restore_rng,
                         rng_state, training_signature, write_json)
from sampler import BalancedMixedIndexSampler, NegativeNeighborIndex, row_fingerprint

activate_rde()
from datasets.bases import ImageTextDataset
from datasets.build import build_dataloader
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.metrics import rank
from utils.options import get_args
from processor import processor as training

ROOT = Path(__file__).resolve().parents[1]


class RDEExperimentTests(unittest.TestCase):
    def test_original_gmm_pass_produces_per_row_labels(self):
        class FakeModel:
            args = SimpleNamespace(noisy_rate=0.0, dataset_name='RSTPReid')
            def eval(self):
                return self
            def compute_per_loss(self, batch):
                index = batch['index']
                losses = (index >= 20).float() + index.float() / 1000
                return losses, losses.flip(0), losses, losses
        class FakeLoader:
            dataset = SimpleNamespace(real_correspondences=np.ones(40),
                                      __len__=lambda: 40)
            def __iter__(self):
                yield dict(index=torch.arange(40))
        original_to = torch.Tensor.to
        def cpu_transfer(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                return tensor
            return original_to(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'to', cpu_transfer):
            a, b = training.get_loss(FakeModel(), FakeLoader())
        self.assertEqual(tuple(a.shape), (40,))
        self.assertEqual(tuple(b.shape), (40,))
        self.assertEqual(a[:20].sum().item(), 20)
        self.assertEqual(a[20:].sum().item(), 0)
        self.assertEqual(b[:20].sum().item(), 0)
        self.assertEqual(b[20:].sum().item(), 20)

    def test_original_rde_forward_backward_with_tiny_clip(self):
        from model.build import build_model
        from model.clip_model import CLIP
        backbone = CLIP(embed_dim=512, image_resolution=(64, 32), vision_layers=1,
                        vision_width=64, vision_patch_size=16, stride_size=16,
                        context_length=8, vocab_size=8, transformer_width=64,
                        transformer_heads=1, transformer_layers=1)
        with patch('model.build.build_CLIP_from_openai_pretrained',
                   return_value=(backbone, {'embed_dim': 512})):
            model = build_model(get_args([]), 4)
        batch = dict(images=torch.randn(4, 3, 64, 32),
                     caption_ids=torch.tensor([[6, 1, 2, 3, 7, 0, 0, 0]] * 4),
                     pids=torch.tensor([0, 0, 1, 2]), label_hat=torch.ones(4))
        # The original TAL implementation explicitly calls .cuda(); only route
        # that device transfer to CPU here, leaving all loss math unchanged.
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
            result = model(batch)
            loss = result['bge_loss'] + result['tse_loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        self.assertIsNotNone(model.base_model.visual.conv1.weight.grad)
        self.assertIsNotNone(model.texual_emb_layer.linear.weight.grad)
        self.assertIsNotNone(model.visul_emb_layer.fc.weight.grad)

    def test_epoch_boundary_pause_and_resume_do_not_repeat_epoch(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
            def forward(self, batch):
                return dict(bge_loss=self.weight.square(), tse_loss=self.weight.square(),
                            temperature=torch.tensor(0.02))
        class Loader:
            batch_size = 64
            def __init__(self):
                self.sampler = SimpleNamespace(diagnostics={}, set_epoch=self.set_epoch)
                self.phases = []
            def set_epoch(self, value):
                self.phases.append(value)
            def __len__(self):
                return 1
            def __iter__(self):
                yield dict(images=torch.ones(64, 1), index=torch.arange(64))
        class Evaluation:
            def eval(self, model):
                self.last_metrics = {'BGE+TSE': {'R1': 50.0}}
                return 50.0
        original_to = torch.Tensor.to
        def cpu_transfer(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                return tensor
            return original_to(tensor, *args, **kwargs)
        args = get_args(['--num_epoch', '2', '--max-hours', '0.000000000001'])
        with tempfile.TemporaryDirectory() as folder:
            args.output_dir = folder
            args.distributed = False
            model, loader = Toy(), Loader()
            optimizer = build_optimizer(args, model)
            scheduler = build_lr_scheduler(args, optimizer)
            checkpointer = Checkpointer(model, optimizer, scheduler, folder, True)
            with patch.object(torch.Tensor, 'to', cpu_transfer), \
                 patch.object(training, 'get_loss', return_value=(torch.ones(64), torch.ones(64))), \
                 patch.object(training, 'SummaryWriter'):
                done = training.do_train(1, args, model, loader, Evaluation(), optimizer,
                                         scheduler, checkpointer, signature={'fixture': True})
                self.assertFalse(done)
                saved = torch.load(Path(folder) / 'resume.pth', weights_only=False)
                self.assertEqual(saved['epoch'], 1)
                self.assertEqual(saved['best_epoch'], 1)
                self.assertEqual(loader.phases, [2, 3])
                done = training.do_train(saved['epoch'] + 1, args, model, loader,
                                         Evaluation(), optimizer, scheduler, checkpointer,
                                         resume_state=saved, signature={'fixture': True})
                self.assertTrue(done)
                self.assertEqual(loader.phases, [2, 3, 4, 5])
                self.assertEqual([r['epoch'] for r in json.loads(
                    (Path(folder) / 'metrics.json').read_text())], [1, 2])
                self.assertTrue((Path(folder) / 'best.pth').exists())
                self.assertTrue((Path(folder) / 'last.pth').exists())

    def test_original_model_losses_heads_optimizer_unchanged(self):
        manifest = json.loads((RDE_ROOT / 'SOURCE_MANIFEST.json').read_text())
        for path, expected in manifest.items():
            source = (RDE_ROOT / path).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, path)
        functions = json.loads((RDE_ROOT / 'SOURCE_FUNCTIONS.json').read_text())
        for key, expected in functions.items():
            path, name = key.split(':')
            tree = ast.parse((RDE_ROOT / path).read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, expected, key)

    def test_defaults_match_actual_old_config(self):
        args = get_args([])
        assert_reference_config(vars(args))
        self.assertEqual(args.sampler, 'balanced_mixed')
        self.assertEqual(args.positive_pairs, 4)
        bad = vars(args).copy()
        bad['batch_size'] = 128
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            assert_reference_config(bad)

    def test_actual_dataloader_with_balanced_sampler(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'RSTPReid'
            (root / 'imgs').mkdir(parents=True)
            annotation = []
            for pid in range(80):
                for image in range(2):
                    name = f'{pid}_{image}.png'
                    Image.new('RGB', (16, 32), color=(pid, image, 30)).save(root / 'imgs' / name)
                    annotation.append(dict(split='train', id=pid, img_path=name,
                                           captions=['person in a shirt', 'a walking person']))
            for split, pid in [('val', 80), ('test', 81)]:
                annotation.append(dict(split=split, id=pid, img_path='0_0.png',
                                       captions=['a person', 'walking']))
            (root / 'data_captions.json').write_text(json.dumps(annotation))
            args = get_args(['--root_dir', folder, '--num_workers', '0'])
            args.distributed = False
            rows = [(item['id'], i, str(root / 'imgs' / item['img_path']), caption)
                    for i, item in enumerate(annotation[:160]) for caption in item['captions']]
            candidates = np.array([[(i + j * 4) % len(rows) for j in range(1, 65)]
                                   for i in range(len(rows))], dtype=np.int32)
            index = NegativeNeighborIndex(rows, np.repeat(candidates[:, None, :], 3, axis=1))
            cache = Path(folder) / 'cache.npz'
            index.save(cache)
            write_json(cache.with_suffix('.json'), dict(protocol=PROTOCOL,
                       rows=row_fingerprint(rows), sha256=file_digest(cache), topk=64,
                       split='train', model_sha256='fixture'))
            args.sampler_mining_cache = str(cache)
            loader, _, _, classes = build_dataloader(args)
            self.assertEqual(classes, 80)
            order = list(loader.sampler)
            self.assertEqual(sorted(order), list(range(320)))
            first = next(iter(loader))
            self.assertEqual(first['images'].shape, (64, 3, 384, 128))
            self.assertEqual(first['caption_ids'].shape, (64, 77))
            self.assertEqual(first['pids'].unique().numel(), 60)
            self.assertTrue(loader.sampler.diagnostics['mining_enabled'])
            validate_cache(cache, rows, 'fixture')
            with self.assertRaises(ValueError):
                validate_cache(cache, rows, 'wrong weights')

    def test_optimizer_parameter_rates_and_resume_scheduler_rng(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = nn.Linear(2, 2)
                self.visul_emb_layer = nn.Linear(2, 2)
                self.texual_emb_layer = nn.Linear(2, 2)
        args = get_args([])
        model = Toy()
        optimizer = build_optimizer(args, model)
        groups = dict(zip((n for n, _ in model.named_parameters()), optimizer.param_groups))
        self.assertEqual(groups['base_model.weight']['lr'], 1e-5)
        self.assertEqual(groups['base_model.bias']['lr'], 2e-5)
        self.assertEqual(groups['visul_emb_layer.weight']['lr'], 0.001)
        self.assertEqual(groups['texual_emb_layer.bias']['lr'], 0.001)
        self.assertEqual(groups['base_model.bias']['weight_decay'], 0)
        self.assertEqual(optimizer.defaults['eps'], 1e-3)
        scheduler = build_lr_scheduler(args, optimizer)
        for _ in range(6):
            optimizer.zero_grad()
            model.base_model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
        with tempfile.TemporaryDirectory() as folder:
            checkpointer = Checkpointer(model, optimizer, scheduler, folder, True)
            state = rng_state()
            expected = (random.random(), np.random.random(), torch.rand(1))
            checkpointer.save('resume', epoch=6, rng=state, signature=training_signature(args))
            saved = torch.load(Path(folder) / 'resume.pth', weights_only=False)
            fresh = Toy()
            opt = build_optimizer(args, fresh)
            sched = build_lr_scheduler(args, opt)
            fresh.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
            sched.load_state_dict(saved['scheduler'])
            self.assertEqual(scheduler.get_last_lr(), sched.get_last_lr())
            restore_rng(saved['rng'])
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], np.random.random())
            torch.testing.assert_close(expected[2], torch.rand(1))
            self.assertFalse((Path(folder) / 'resume.pth.tmp').exists())

    def test_neighbors_exclude_same_pid_and_deduplicate_gallery(self):
        rows = [(pid, pid * 2 + image, str(image), str(caption))
                for pid in range(5) for image in range(2) for caption in range(2)]
        features = torch.nn.functional.normalize(torch.randn(len(rows), 8), dim=1)
        cache = neighbors(features, features, rows, 'cpu', topk=3, chunk=3)
        index = NegativeNeighborIndex(rows, cache)
        self.assertTrue((cache[:, 1] % 2 == 0).all())
        for row, candidates in enumerate(index.neighbors):
            for candidate in candidates.flatten():
                if candidate >= 0:
                    self.assertNotEqual(rows[row][0], rows[candidate][0])

    def test_rde_retrieval_metric(self):
        similarity = torch.eye(10)
        cmc, mAP, mINP, _ = rank(similarity, torch.arange(10), torch.arange(10))
        self.assertEqual(cmc[0].item(), 100)
        self.assertEqual(mAP.item(), 100)
        self.assertEqual(mINP.item(), 100)

    def test_comparison_uses_best_and_last_not_training_peak(self):
        old = json.loads((ROOT / 'reference_old.json').read_text())
        args = get_args([])
        signature = training_signature(args)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'balanced_mixed_seed1'
            output.mkdir()
            result = dict(status='complete', completed_epochs=60, signature=signature,
                          test=copy.deepcopy(old['test']))
            result['test']['best']['BGE+TSE']['R1'] += 1
            write_json(output / 'result.json', result)
            with patch('builtins.print'):
                report = compare(folder)
            self.assertAlmostEqual(report['difference_percentage_points']['best']['BGE+TSE']['R1'], 1)
            self.assertEqual(report['difference_percentage_points']['last']['BGE+TSE']['R1'], 0)

    def test_launch_keeps_batch64_and_identity_k4(self):
        args = SimpleNamespace(root='data', cache='cache.npz', seed=1, max_hours=9)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            cmd = command_for(args, 'balanced_mixed', output)
            self.assertEqual(cmd[cmd.index('--batch_size') + 1], '64')
            self.assertEqual(cmd[cmd.index('--num_instance') + 1], '4')
            self.assertEqual(cmd[cmd.index('--num_epoch') + 1], '60')
            (output / 'resume.pth').touch()
            self.assertIn('--resume', command_for(args, 'balanced_mixed', output))


if __name__ == '__main__':
    unittest.main()
