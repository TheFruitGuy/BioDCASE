"""
Phase 13j - FDY + larger batch  (single-axis: the larger-batch recipe)
======================================================================

Tests the batch axis on the FDY stem at 30 s, isolated from 13i's segment change.
WhaleVAD's 0.440 used a "very large batch"; a bigger batch also averages over more
of the per-epoch negative-resampling noise that drives your val swings.

IMPORTANT - batch and LR are coupled. A larger batch at the *same* LR mostly just
means fewer gradient updates per epoch, so it tends to look neutral-or-worse on a
fixed epoch budget rather than testing "does a larger batch help". This rung
therefore scales peak_lr up with the batch (sqrt rule: 32 -> 96 is ~1.7x, so
5e-5 -> ~1e-4) so the comparison reflects the *recipe*, not just an update-count
change. If you want the strict same-LR batch ablation instead, pass --peak-lr 5e-5.

bf16 AMP is on so batch 96 fits at 30 s. To go beyond what physical VRAM allows
(WhaleVAD-scale "very large" batch), prefer gradient accumulation over a bigger
physical batch -- accumulation is free on memory; keep the physical batch >=16 so
BatchNorm stats stay healthy. EMA / FilterAugment / dual-res stay off.

A bare ::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase13j.py          # batch 96, lr 1e-4
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase13j.py --batch-size 64
"""

import argparse

from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)
from train_phase13d import build_conformer_fdy
import config_final as cfg


def main():
    p = argparse.ArgumentParser(
        description="Phase 13j: FDY + larger batch (LR scaled with batch)")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4)
    p.add_argument("--fdy-temp", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=96,
                   help="Mini-batch size; overrides cfg.BATCH_SIZE for this run "
                        "(default 96 vs the 32 baseline).")
    # 13j recipe: FDY near-base, 30 s segments, larger batch with peak_lr scaled up
    # (sqrt rule). AMP on so batch 96 fits at 30 s. EMA/FilterAugment/dual-res off.
    p.set_defaults(peak_lr=1e-4, warmup_epochs=0.0, lr_schedule="const", amp=True)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    cfg.BATCH_SIZE = int(a.batch_size)
    run_training(
        "13j", build_model=build_conformer_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", "13j", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
