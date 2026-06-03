"""
Phase 13i - FDY + 60 s training segments  (single-axis: segment length)
=======================================================================

Tests the segment-length axis on the FDY stem. WhaleVAD's 0.440 used "long
segments"; your own data shows 60 s eval tiles beat 30 s; and the temporal
grammar needs the window (a blue-whale Z-call is ~20 s, so a full BMABZ unit
sequence barely fits in 30 s and sits comfortably in 60 s).

Only TRAIN_SEGMENT_S changes (30 -> 60). The eval window (build_val_segments /
EVAL_SEGMENT_S) is untouched, so the validation numbers are directly comparable
to 13d -- this isolates the *training* segment length, not the eval tiling.

bf16 AMP is on by default so the run keeps **batch 32** at 60 s (attention is
O(T^2); 30->60 s is 4x the attention memory, which would otherwise force a batch
cut and confound the comparison). AMP touches only the model forward, not the
STFT, and shifts the metric by less than seed noise -- so 13i (AMP, 60 s) vs 13d
(fp32, 30 s) is still a fair segment-length test. If 60 s/batch-32 still OOMs even
with AMP, fall back to gradient checkpointing rather than dropping the batch.

EMA / FilterAugment / dual-res stay off. A bare ::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase13i.py

Eval the best checkpoint with eval_conformer.py as usual.
"""

import argparse

from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)
from train_phase13d import build_conformer_fdy
import config_final as cfg


def main():
    p = argparse.ArgumentParser(
        description="Phase 13i: FDY + 60 s training segments")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4)
    p.add_argument("--fdy-temp", type=float, default=1.0)
    p.add_argument("--segment-s", type=float, default=60.0,
                   help="Training segment length in seconds (eval window "
                        "unchanged). Default 60.")
    # 13i recipe: FDY near-base, 60 s training segments, bf16 AMP on so batch 32
    # still fits. EMA/FilterAugment/dual-res stay off (segment length is the only
    # change versus 13d).
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const", amp=True)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    cfg.TRAIN_SEGMENT_S = float(a.segment_s)   # eval window (EVAL_SEGMENT_S) intact
    run_training(
        "13i", build_model=build_conformer_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", "13i", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
