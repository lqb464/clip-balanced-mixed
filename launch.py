import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--cache', default='cache/pretrained_neighbors.npz')
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--samplers', nargs='+', choices=['random', 'balanced_mixed'],
                        default=['random', 'balanced_mixed'])
    parser.add_argument('--max-hours', type=float, default=9)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--microbatch', type=int, default=16)
    args = parser.parse_args()
    if len(args.gpus) < len(args.samplers) or len(set(args.gpus)) != len(args.gpus):
        parser.error('Assign one distinct GPU per sampler (or run one sampler at a time)')
    if len(set(args.samplers)) != len(args.samplers):
        parser.error('Duplicate samplers are not allowed')
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        for gpu, sampler in zip(args.gpus, args.samplers):
            output = runs / f'{sampler}_seed{args.seed}'
            output.mkdir(parents=True, exist_ok=True)
            result_path = output / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['completed_epochs'] != args.epochs:
                    raise ValueError('Completed output has a different epoch target; use a new runs directory')
                print(f'{sampler}: already complete', flush=True)
                continue
            command = [sys.executable, '-u', 'train.py', '--root', args.root,
                       '--sampler', sampler, '--output', str(output), '--cache', args.cache,
                       '--epochs', str(args.epochs), '--seed', str(args.seed),
                       '--microbatch', str(args.microbatch), '--max-hours', str(args.max_hours)]
            if (output / 'last.pt').exists():
                command.append('--resume')
            log = (output / 'console.log').open('a', encoding='utf-8')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1'}
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((sampler, process, log, output))
            print(f'Started {sampler} on GPU {gpu}; log: {output / "console.log"}', flush=True)
        while any(p.poll() is None for _, p, _, _ in processes):
            for sampler, process, _, output in processes:
                progress = output / 'progress.json'
                if progress.exists():
                    print(f'{sampler}: {progress.read_text().strip()}', flush=True)
                if process.poll() not in (None, 0):
                    print(f'{sampler} failed; inspect {output / "console.log"}', flush=True)
            time.sleep(30)
        failed = [sampler for sampler, process, _, _ in processes if process.returncode != 0]
        if failed:
            raise RuntimeError(f'Failed runs: {failed}')
        subprocess.run([sys.executable, 'compare.py', '--runs', str(runs),
                        '--seed', str(args.seed)], check=True)
    finally:
        for _, process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == '__main__':
    main()
