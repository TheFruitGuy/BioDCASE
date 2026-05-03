"""
Phase 1e: Cross-Site Noise Mixing
==================================

The most ambitious of the Phase 1 augmentations. Single-axis change
to the F1=0.474 baseline (``train.py``): for each positive training
segment, with some probability, mix in a no-call audio clip from a
*different* training site at controlled SNR.

This directly addresses the cross-site failure mode diagnosed in
Phase 0a (training on kerguelen2005 → eval on casey2017 = F1=0.000)
and the residual generalization gap visible in Phase 0n's per-site
breakdown (we underperform paper most on bmabz at out-of-distribution
sites).

Mechanism
---------
The model is trained to recognize calls from training site A, but
sees them embedded in the noise texture of training site B. Over
the course of training, every training-site-pair noise combination
appears in the augmentation distribution. The model is forced to
learn "what makes this a call" independently of "what kind of
ambient noise surrounds it" — i.e. site-invariant call features.

The held-out validation sites (casey2017, kerguelen2014, kerguelen2015)
are not in the noise pool, but the *same kind of distributional shift*
(unseen ambient texture surrounding a known call) is exactly what
the model trained on. Generalization should improve.

Hyperparameters
---------------
``mix_prob = 0.5``: half of positive samples get cross-site noise mixed.
``snr_min_db = -6.0``, ``snr_max_db = 6.0``: SNR (call energy / noise
energy) drawn uniformly. -6 dB is a quite noisy mix, +6 dB is mild.

For negative samples (no calls), the mixed-in noise is added at SNR=0
dB equivalent (matched to the original signal energy). Negatives still
contribute to "this kind of background isn't a call" learning.

Prerequisites
-------------
Run ``precompute_no_call_pool.py`` once before launching this script.
It dumps a ~6 MB ``no_call_pool.pt`` file with 50 30-second no-call
clips per training site.

Usage
-----
::

    # One-time:
    python precompute_no_call_pool.py

    # Then:
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase1e.py
"""

import argparse
from pathlib import Path

from augmentations import apply_cross_site_mix
from phase1_baseline import NoCallPool, run_phase1_training


PHASE1E_CONFIG = {
    "mix_prob": 0.5,
    "snr_min_db": -6.0,
    "snr_max_db":  6.0,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool", type=str, default="no_call_pool.pt",
        help="Path to the precomputed no-call pool (default: no_call_pool.pt)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pool_path = Path(args.pool)
    if not pool_path.exists():
        raise SystemExit(
            f"No-call pool not found at {pool_path}. Run "
            f"`python precompute_no_call_pool.py` first."
        )
    print(f"Loading no-call pool from {pool_path}...")
    pool = NoCallPool.load(pool_path)
    n_sites = len(pool.by_site)
    n_clips = sum(t.shape[0] for t in pool.by_site.values())
    print(f"  {n_sites} sites, {n_clips} clips, "
          f"clip length = {pool.clip_samples} samples "
          f"({pool.clip_samples / pool.sample_rate:.1f}s @ {pool.sample_rate} Hz)")

    config_with_meta = {
        **PHASE1E_CONFIG,
        "pool_n_sites": n_sites,
        "pool_n_clips": n_clips,
        "pool_clip_seconds": pool.clip_samples / pool.sample_rate,
    }

    run_phase1_training(
        phase_name="1e",
        augmentation_fn=apply_cross_site_mix,
        augmentation_config=config_with_meta,
        augmentation_state=pool,
    )
