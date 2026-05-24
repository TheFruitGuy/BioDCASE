"""
Phase 12a: Frequency Dynamic Convolution in the Aggregation Block
==================================================================

Single-axis change to the F1=0.474 baseline (``train.py``): replace
the first and third depthwise 3x3 convolutions in the residual
aggregation block with ``FDConv2d`` (K=4 basis kernels, attention
reduction=4 — both paper standard). Same data, same loss, no
augmentation.

Why this matters
----------------
Standard 2D convs assume "the same pattern at 27 Hz means the same
thing as at 100 Hz" — wrong for whale data, where calls are *defined*
by their frequency band (Z-calls at 27 Hz, D-calls above 20 Hz).
Diagnostic ablations (Phases 1-2) confirmed the CNN frontend is the
bottleneck; this phase attacks the wrong frequency-equivariance
inductive bias directly.

Reference: Nam et al. 2022 (FDY-SED). Reported +7-9% on DESED. Whale
data has more frequency-disjoint classes than DESED, so the prior
should bite at least as hard; on the other hand we have less training
data and the frequency axis is already heavily downsampled by the
time the aggregation block runs, which could limit gains.

Standard hyperparameters
------------------------
- ``K = 4`` basis kernels (paper default; range 4-8)
- ``reduction = 4`` in the attention head (paper default)
- Replace 2 of 3 depthwise convs (first at index 1, last at index 7
  of the aggregation Sequential; middle stays standard). Matches the
  conservative "one or two of the depthwise convs" recommendation.

If 12a > baseline outside the seed-noise band (±0.005 macro F1),
phase 12b can sweep K, replace all 3 convs, or extend FDConv into the
bottleneck.

Parameter cost
--------------
~8k parameters per FDConv layer; ~16k total. On a ~1M-parameter
WhaleVAD this is well below 2% overhead. See the self-test in
``model_fdconv_whalevad.py`` for the exact delta.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase12a.py
"""

import torch.nn as nn

from model_fdconv_whalevad import (
    WhaleVAD_FDConv,
    DEFAULT_FDCONV_POSITIONS,
)
from phase1_baseline import run_phase1_training


# Standard FDY values. Documented as module-level constants so they
# appear verbatim in PHASE12A_CONFIG and on the wandb run config.
PHASE12A_K = 4
PHASE12A_REDUCTION = 4
PHASE12A_POSITIONS = DEFAULT_FDCONV_POSITIONS  # (1, 7) — first and last


PHASE12A_CONFIG = {
    "arch_change":         "fdconv_aggregation",
    "fdconv_K":            PHASE12A_K,
    "fdconv_reduction":    PHASE12A_REDUCTION,
    "fdconv_positions":    list(PHASE12A_POSITIONS),
    "fdconv_layer_count":  len(PHASE12A_POSITIONS),
    "fdconv_total_layers": 3,  # there are 3 depthwise convs in the block
}


def _factory(num_classes: int) -> nn.Module:
    """
    Construct the FDConv variant. Closure over the phase 12a
    hyperparameters so callers don't need to know them.
    """
    return WhaleVAD_FDConv(
        num_classes=num_classes,
        K=PHASE12A_K,
        reduction=PHASE12A_REDUCTION,
        positions=PHASE12A_POSITIONS,
    )


# Readable name in wandb logs (otherwise the closure name "_factory"
# would surface).
_factory.__name__ = "WhaleVAD_FDConv_factory"


if __name__ == "__main__":
    print(
        f"Phase 12a: FDConv aggregation "
        f"K={PHASE12A_K}, reduction={PHASE12A_REDUCTION}, "
        f"replacing {len(PHASE12A_POSITIONS)}/3 depthwise convs "
        f"(positions {PHASE12A_POSITIONS})"
    )
    run_phase1_training(
        phase_name="12a",
        augmentation_fn=None,
        augmentation_config=None,
        model_factory=_factory,
        model_factory_config=PHASE12A_CONFIG,
    )
