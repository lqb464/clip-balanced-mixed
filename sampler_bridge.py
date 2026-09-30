from cache_utils import validate_cache
from sampler import BalancedMixedIndexSampler


def build_balanced_sampler(rows, args):
    if args.distributed:
        raise ValueError('Balanced Mixed currently supports single-GPU training only')
    if not args.sampler_mining_cache:
        raise ValueError('Balanced Mixed requires --sampler-mining-cache; build it first')
    index, _ = validate_cache(args.sampler_mining_cache, rows)
    return BalancedMixedIndexSampler(
        rows, args.batch_size, args.positive_pairs, negative_index=index, seed=args.seed,
        exposure_mode='coverage', hard_fraction=0.25, semi_hard_fraction=0.25,
        hard_top_k=10)
