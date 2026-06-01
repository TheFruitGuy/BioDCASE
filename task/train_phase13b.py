"""
Phase 13b — 13a (warmup) + EMA weight averaging
===============================================

Builds on 13a by maintaining an exponential moving average of the weights and
evaluating / checkpointing the EMA model. Smooths the noisy fixed-threshold
validation curve and usually nets a small gain (the mean-teacher "teacher" is
exactly a weight EMA). Inherits 13a's warmup/schedule defaults.

Usage
-----
    CUDA_VISIBLE_DEVICES=5 python train_phase13b.py
    CUDA_VISIBLE_DEVICES=5 python train_phase13b.py --ema-decay 0.9995
"""

import argparse

import config_final as cfg
from conformer_core import run_training, build_conformer, add_arch_args, add_opt_args, opt_kwargs_from


def main():
    p = argparse.ArgumentParser(description="Phase 13b: Conformer + warmup + EMA")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.add_argument("--ema-decay", type=float, default=0.999)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13b", build_model=build_conformer, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=True, ema_decay=a.ema_decay,
        extra_tags=["conformer", "warmup", "ema"],
    )


if __name__ == "__main__":
    main()
