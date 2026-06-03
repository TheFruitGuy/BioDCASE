"""
Phase 13h - FDY + FilterAugment  (single-axis ablation)
=======================================================

Isolates the FilterAugment half of the 13f stack - the suspected cause of 13f's
regression. Blue-whale BMABZ calls are narrowband tonals (~18-28 Hz); FilterAugment's
random per-band gains (+/- `--fa-db`, default 4.5 dB) can attenuate the very band the
call sits in, which would explain BMABZ's precision collapse at equal recall. This
rung turns on **FilterAugment only** to test that directly.

Identical to 13d (FDY stem, RAdam / const-5e-5 / no warmup) with `--filteraug`
defaulted on and EMA / dual-res off. A bare ::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase13h.py

If this alone reproduces the 13f collapse, FilterAugment is mismatched to the
narrowband regime -> documented negative, drop it. Salvage knobs before giving up:
`--fa-db 1.5` (gentler band gains) or `--no-freq-warp` (drop the warp component, keep
only band gains). FilterAugment is train-only, so eval needs no handling.
"""

import argparse

from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)
from train_phase13d import build_conformer_fdy
import config_final as cfg


def main():
    p = argparse.ArgumentParser(
        description="Phase 13h: FDY + FilterAugment (FilterAugment-only ablation)")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4)
    p.add_argument("--fdy-temp", type=float, default=1.0)
    # 13h recipe: FDY near-base + FilterAugment on; EMA/dual-res stay OFF so
    # FilterAugment is the only change versus 13d. --fa-db / --no-freq-warp (from
    # add_arch_args) tune the augmentation for the salvage runs.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   filteraug=True)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13h", build_model=build_conformer_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", "13h", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
