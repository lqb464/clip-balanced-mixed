from sampler import BalancedMixedIndexSampler


def build_balanced_mixed_sampler(args, rows):
    if getattr(args, "distributed", False):
        raise ValueError("Balanced Mixed currently supports single-GPU training")
    return BalancedMixedIndexSampler(
        rows,
        batch_size=args.batch_size,
        positive_pairs_per_batch=args.positive_pairs,
        negative_index=None,
        exposure_mode="coverage",
        hard_fraction=0.25,
        semi_hard_fraction=0.25,
        hard_top_k=10,
        seed=args.seed,
    )
