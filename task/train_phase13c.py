"""
Phase 13c — capacity & kernel sweep
===================================

Same stabilised recipe as 13b (warmup + EMA); the experiment is varying the
Conformer's capacity. Phase 13 used conv kernel 31 / 4 blocks / d_model 128;
Miyazaki's DCASE2020 Conformer-SED used conv kernel 7 and 2–10 blocks, so the
optimum may be elsewhere. Run several configs and pick the best macro_paper as
the new base for 13d.

Suggested sweep
---------------
    python train_phase13c.py --conv-kernel 7
    python train_phase13c.py --conv-kernel 15
    python train_phase13c.py --layers 6
    python train_phase13c.py --layers 8 --d-model 192

(Each run tags as phase 13c in W&B; distinguish them by the arch config logged.)
"""

import argparse

import config_final as cfg
from conformer_core import run_training, build_conformer, add_arch_args, add_opt_args, opt_kwargs_from


def main():
    p = argparse.ArgumentParser(description="Phase 13c: Conformer capacity/kernel sweep (warmup + EMA)")
    add_arch_args(p)
    add_opt_args(p)
    p.add_argument("--ema-decay", type=float, default=0.999)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13c", build_model=build_conformer, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=True, ema_decay=a.ema_decay,
        extra_tags=["conformer", "warmup", "ema", "arch_sweep"],
    )


if __name__ == "__main__":
    main()
