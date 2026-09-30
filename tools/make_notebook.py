import json
from pathlib import Path
import textwrap


def cell(source):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=textwrap.dedent(source).strip().splitlines(keepends=True))


cells = [
    cell("""
    import os
    import sys
    import subprocess
    from pathlib import Path

    REPO = Path('/kaggle/working/clip-balanced-mixed')
    if not REPO.exists():
        subprocess.run(['git', 'clone', 'https://github.com/lqb464/clip-balanced-mixed.git',
                        str(REPO)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
    os.chdir(REPO)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
    subprocess.run(['nvidia-smi'], check=True)
    """),
    cell("""
    import shutil

    DATA_ROOT = '/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark'
    WORK = Path('/kaggle/working/rde_experiment')
    RESUME_FROM = ''  # Previous saved output's rde_experiment folder; leave empty for first run.
    SAMPLERS = ['balanced_mixed']  # Optional: ['balanced_mixed', 'identity'] on T4 x2.
    GPUS = ['0', '1']
    BATCH_SIZE = 64
    EPOCHS = 60
    SEED = 1
    MAX_TRAIN_HOURS = 9
    assert BATCH_SIZE == 64 and EPOCHS == 60, 'Keep the actual old RDE batch64/60-epoch setup.'
    assert (Path(DATA_ROOT) / 'RSTPReid' / 'data_captions.json').is_file(), DATA_ROOT
    if RESUME_FROM:
        source = Path(RESUME_FROM)
        assert (source / 'cache').is_dir() and (source / 'runs').is_dir(), source
        if not WORK.exists():
            shutil.copytree(source, WORK)
        else:
            print('WORK already exists; keeping this session data instead of overwriting.')
    WORK.mkdir(parents=True, exist_ok=True)
    CACHE = WORK / 'cache' / 'rde_pretrained_neighbors.npz'
    print('RDE + TAL/RBS, BGE+TSE, batch 64, 60 epochs; only the sampler changes.')
    print('Old reference: Identity K=4; best checkpoint test R@1 = 63.80%.')
    """),
    cell("""
    subprocess.run([sys.executable, '-u', 'build_cache.py', '--root', DATA_ROOT,
                    '--output', str(CACHE)],
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}, check=True)
    """),
    cell("""
    subprocess.run([sys.executable, '-u', 'launch.py', '--root', DATA_ROOT,
                    '--runs', str(WORK / 'runs'), '--cache', str(CACHE),
                    '--samplers', *SAMPLERS, '--gpus', *GPUS,
                    '--seed', str(SEED), '--max-hours', str(MAX_TRAIN_HOURS)], check=True)
    """),
    cell("""
    subprocess.run([sys.executable, 'compare.py', '--runs', str(WORK / 'runs'),
                    '--seed', str(SEED)], check=True)
    print('Save Version to preserve rde_experiment/cache and rde_experiment/runs.')
    print('If training paused before 60 epochs, add that output next session and set RESUME_FROM.')
    print('Preserve resume.pth and best.pth together. Do not reuse CLIP+SDM checkpoints.')
    """)
]

notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python',
                                              name='python3'),
                              language_info=dict(name='python', version='3.12')))
path = Path(__file__).resolve().parents[1] / 'kaggle_train.ipynb'
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding='utf-8')
for index, item in enumerate(cells):
    compile(''.join(item['source']), f'cell {index}', 'exec')
print(f'Generated and syntax-checked {len(cells)} cells: {path}')
