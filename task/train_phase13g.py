"""
Phase 13g - FDY + EMA  (single-axis ablation)
=============================================

Isolates the EMA half of the 13f stack. 13f layered FilterAugment *and* EMA on the
FDY stem at once and regressed hard (fixed-0.3 macro_paper 0.29 vs FDY-alone's 0.43,
with BMABZ precision collapsing at equal recall). This rung turns on **EMA only** so
its contribution is attributable on its own.

Identical to 13d (FDY stem, RAdam / const-5e-5 / no warmup) with `--ema` defaulted
on and FilterAugment / dual-res off. A bare ::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase13g.py

Expectation: EMA averages over the per-epoch negative-resampling noise that drove
the 13d swings, so this should land at FDY-alone's level or a touch better (a higher
floor / a steadier selected checkpoint), not worse. If 13g is fine and 13h
(FilterAugment-only) reproduces the regression, FilterAugment is the culprit.

The saved checkpoint stores the EMA-averaged weights (validate() runs on them too),
so `eval_conformer.py` evaluates them directly with no eval-side handling.
"""

import argparse

from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)
from train_phase13d import build_conformer_fdy
import config_final as cfg


def main():
    p = argparse.ArgumentParser(
        description="Phase 13g: FDY + EMA (EMA-only ablation)")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4)
    p.add_argument("--fdy-temp", type=float, default=1.0)
    # 13g recipe: FDY near-base + EMA on; FilterAugment/dual-res stay OFF so EMA is
    # the only change versus 13d.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const", ema=True)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13g", build_model=build_conformer_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=a.ema, extra_tags=["conformer", "fdy", "13g", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
