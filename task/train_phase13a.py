"""
Phase 13a — Conformer + LR warmup / schedule
=============================================

The first and highest-priority rung of the Conformer ladder. The final recipe's
fixed 5e-5 was tuned for the BiLSTM; the Conformer underfits and oscillates early
without warmup (seen in phase 13). This swaps to RAdam (or AdamW) with a higher
peak LR, linear warmup, and cosine decay. Everything else (data, weighted-BCE
loss, per-epoch resampling, fixed-0.3 validation, macro_paper selection) is the
final pipeline, reused verbatim via conformer_core.

Usage
-----
    CUDA_VISIBLE_DEVICES=5 python train_phase13a.py
    CUDA_VISIBLE_DEVICES=5 python train_phase13a.py --optimizer adamw --peak-lr 3e-4
    CUDA_VISIBLE_DEVICES=5 python train_phase13a.py --warmup-epochs 1 --epochs 30
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
        use_ema=False, extra_tags=["conformer", "warmup"],
    )


if __name__ == "__main__":
    main()
