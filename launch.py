import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rde_runtime import PROTOCOL


def command_for(args, sampler, output):
    command = [sys.executable, '-u', 'train.py', '--root_dir', args.root,
               '--sampler', sampler, '--output_dir', str(output),
               '--sampler-mining-cache', args.cache, '--seed', str(args.seed),
               '--num_epoch', '60', '--lr-total-epochs', '60', '--batch_size', '64',
               '--num_instance', '4', '--positive-pairs', '4',
               '--max-hours', str(args.max_hours)]
    if (output / 'resume.pth').exists():
        command.append('--resume')
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--cache', default='cache/rde_pretrained_neighbors.npz')
    parser.add_argument('--samplers', nargs='+', choices=['balanced_mixed', 'identity', 'random'],
                        default=['balanced_mixed'])
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--max-hours', type=float, default=9)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if len(args.samplers) > len(args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error('One independent sampler per distinct GPU; select at most two on T4 x2')
    if len(set(args.samplers)) != len(args.samplers) or args.max_hours <= 0:
        parser.error('Duplicate samplers or invalid time budget')
    processes = []
    try:
        for gpu, sampler in zip(args.gpus, args.samplers):
            output = Path(args.runs) / f'{sampler}_seed{args.seed}'
            output.mkdir(parents=True, exist_ok=True)
            result_path = output / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['signature']['protocol'] != PROTOCOL:
                    raise ValueError('Choose a new output folder; this result uses another pipeline')
                print(f'{sampler}: already complete', flush=True)
                continue
            log = (output / 'console.log').open('a', encoding='utf-8')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1'}
            process = subprocess.Popen(command_for(args, sampler, output), env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((sampler, process, log, output))
            print(f'Started {sampler} on GPU {gpu}. Log: {output / "console.log"}', flush=True)
        while any(p.poll() is None for _, p, _, _ in processes):
            for sampler, process, _, output in processes:
                path = output / 'progress.json'
                if path.exists():
                    print(f'{sampler}: {path.read_text().strip()}', flush=True)
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'{sampler} failed; inspect {output / "console.log"}')
            time.sleep(30)
        failed = [name for name, process, _, _ in processes if process.returncode != 0]
        if failed:
            raise RuntimeError(f'Failed runs: {failed}')
        subprocess.run([sys.executable, 'compare.py', '--runs', args.runs,
                        '--seed', str(args.seed)], check=True)
    finally:
        for _, process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == '__main__':
    main()
