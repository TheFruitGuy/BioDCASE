"""
Phase 13a — Conformer + Schedule-Free RAdam
===========================================

The first and highest-priority rung of the Conformer ladder, and the optimiser
fix for the early oscillation seen in phase 13. Defaults to **Schedule-Free
RAdam**: RAdam's variance rectification is an automatic warmup, and schedule-free
iterate-averaging replaces the decay schedule *and* folds in the weight-averaging
that 13b's EMA would otherwise add — so this single rung subsumes 13a+13b. No
warmup-epoch or schedule tuning; constant LR.

Fallback (lower-variance, Miyazaki-style): ``--optimizer radam`` (or ``adamw``)
uses warmup + cosine decay; pair with ``train_phase13b.py`` for the cleanly
separable A→B ladder if you'd rather measure warmup and EMA independently.

Everything else (data, weighted-BCE loss, per-epoch resampling, fixed-0.3
validation, macro_paper selection) is the final pipeline, reused verbatim via
conformer_core. Requires ``pip install schedulefree`` in the env.

Usage
-----
    CUDA_VISIBLE_DEVICES=5 python train_phase13a.py                       # sf-radam (default)
    CUDA_VISIBLE_DEVICES=5 python train_phase13a.py --peak-lr 2.5e-3      # sf often likes higher LR
    CUDA_VISIBLE_DEVICES=5 python train_phase13a.py --optimizer radam --warmup-epochs 1  # fallback
"""

import argparse

import config_final as cfg
from conformer_core import run_training, build_conformer, add_arch_args, add_opt_args, opt_kwargs_from


def main():
    p = argparse.ArgumentParser(description="Phase 13a: Conformer + LR warmup/cosine schedule")
    add_arch_args(p)
    add_opt_args(p)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13a", build_model=build_conformer, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "sf_radam"],
    )


if __name__ == "__main__":
    main()
