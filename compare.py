import argparse
import json
from pathlib import Path


def compare(runs, seed):
    results = {}
    for sampler in ('random', 'balanced_mixed'):
        path = Path(runs) / f'{sampler}_seed{seed}' / 'result.json'
        if not path.exists():
            print(f'{sampler}: incomplete; resume from last.pt. No final comparison yet.')
            return None
        results[sampler] = json.loads(path.read_text())
    keys = set(results['random']['signature']) - {'sampler', 'cache'}
    if any(results['random']['signature'][k] != results['balanced_mixed']['signature'][k]
           for k in keys):
        raise ValueError('Cannot compare runs with different training configurations or data')
    for result in results.values():
        if result['status'] != 'complete' or result['completed_epochs'] != result['signature']['epochs']:
            raise ValueError('Both runs must have finished the configured training horizon')
    delta = {k: results['balanced_mixed']['test'][k] - results['random']['test'][k]
             for k in ('R1', 'R5', 'R10', 'mAP', 'mINP')}
    report = dict(seed=seed, results=results, balanced_minus_random_percentage_points=delta,
                  selection='best validation R1; test evaluated after training')
    path = Path(runs) / f'comparison_seed{seed}.json'
    path.write_text(json.dumps(report, indent=2))
    print('Metric         Random  Balanced Mixed  Difference (pp)')
    for key, value in delta.items():
        print(f'{key:8s} {results["random"]["test"][key]:12.3f} '
              f'{results["balanced_mixed"]["test"][key]:15.3f} {value:+16.3f}')
    print('One seed is an initial result; repeat training seeds before a general claim.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    compare(args.runs, args.seed)
