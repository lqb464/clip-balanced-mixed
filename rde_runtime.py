import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROTOCOL = 'rde_old_batch64_tal_rbs_v1'
RDE_ROOT = Path(__file__).resolve().parent / 'rde'


def activate_rde():
    path = str(RDE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def seed_everything(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def training_signature(args):
    excluded = {'root_dir', 'output_dir', 'resume', 'resume_ckpt_file',
                'max_hours', 'sampler_mining_cache', 'local_rank'}
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'numpy', 'scikit-learn', 'scipy')}
    return dict(protocol=PROTOCOL, libraries=versions,
                config={k: v for k, v in vars(args).items() if k not in excluded})
