import argparse
import json
from pathlib import Path

from rde_runtime import PROTOCOL, write_json


REFERENCE = Path(__file__).with_name('reference_old.json')


def assert_reference_config(config):
    expected = json.loads(REFERENCE.read_text())['training_config']
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            raise ValueError(f'Not comparable to old RDE: {key}={actual!r}, expected {value!r}')


def compare(runs, seed=1):
    path = Path(runs) / f'balanced_mixed_seed{seed}' / 'result.json'
    if not path.exists():
        print('Balanced Mixed has not finished 60 epochs. Save output and resume.')
        return None
    current = json.loads(path.read_text())
    if current['signature']['protocol'] != PROTOCOL or current['status'] != 'complete':
        raise ValueError('Expected a completed RDE Balanced Mixed run')
    if current['completed_epochs'] != 60:
        raise ValueError('The historical comparison requires 60 completed epochs')
    assert_reference_config(current['signature']['config'])
    old = json.loads(REFERENCE.read_text())
    deltas = {}
    for checkpoint in ('best', 'last'):
        print(f'\n{checkpoint}.pth | Old Identity K=4 vs Balanced Mixed | percentage points')
        print('Branch       Metric       Old       New     New-Old')
        deltas[checkpoint] = {}
        for branch in ('BGE', 'TSE', 'BGE+TSE'):
            deltas[checkpoint][branch] = {}
            for metric in ('R1', 'R5', 'R10', 'mAP', 'mINP'):
                a = old['test'][checkpoint][branch][metric]
                b = current['test'][checkpoint][branch][metric]
                deltas[checkpoint][branch][metric] = b - a
                print(f'{branch:12s} {metric:6s} {a:9.3f} {b:9.3f} {b-a:+10.3f}')
    report = dict(reference=old, current=current, difference_percentage_points=deltas,
                  caveat='Historical one-seed comparison; test-selected checkpoints, not held-out model selection.')
    write_json(Path(runs) / f'comparison_seed{seed}.json', report)
    print(report['caveat'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    compare(args.runs, args.seed)
