"""
Phase 1b: Narrowband Frequency Masking (Safe)
=============================================

Single-axis change to the F1=0.474 baseline (``train.py``): the
training loop applies per-sample narrowband frequency masking,
deliberately *avoiding* the 15-50 Hz band where blue whale and
fin whale call energy lives.

The protected region is defined as bins ``[13, 53]`` (≈12-52 Hz at
sr=250, n_fft=256) — slightly wider than the actual call band to
provide a guard margin. The mask is drawn from the regions outside:
``[0, 13)`` (DC + sub-call low frequencies) or ``[53, 129)`` (above
the call band).

Why this might help
-------------------
The held-out validation sites have different ambient noise profiles
than the training sites. In particular, casey2017 (our worst site)
is dominated by ice-related transient sounds and lacks the persistent
shipping-noise band typical of other sites. By erasing random
narrowband regions outside the call band during training, we force
the model to not rely on any specific narrowband cue from the
background — only the 15-50 Hz call structure should matter.

What's the failure mode?
------------------------
If the protected band is too narrow, freq masking erases call
content and F1 craters. We've set the guard ±2 bins beyond the
literal call band to be safe.

If the protected band is too wide, almost no frequencies are
augmentable and the regularization is too weak. The F-bin range
[0, 13) ∪ [53, 129) leaves 89 of 129 bins available — more than
enough.

Hyperparameters
---------------
``mask_prob = 0.5``: half of training samples get masked per epoch.
``min_bins = 5`` / ``max_bins = 12``: 5-12 freq bins ≈ 5-12 Hz wide.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase1b.py
"""

from augmentations import (
    apply_freq_mask_safe,
    PROTECTED_FREQ_BIN_LO, PROTECTED_FREQ_BIN_HI,
)
from phase1_baseline import run_phase1_training


PHASE1B_CONFIG = {
    "mask_prob": 0.5,
    "min_bins": 5,
    "max_bins": 12,
    "protected_lo_bin": PROTECTED_FREQ_BIN_LO,
    "protected_hi_bin": PROTECTED_FREQ_BIN_HI,
    "protected_lo_hz": PROTECTED_FREQ_BIN_LO * 125 / 128,  # ≈ 12.7 Hz
    "protected_hi_hz": PROTECTED_FREQ_BIN_HI * 125 / 128,  # ≈ 51.8 Hz
}


if __name__ == "__main__":
    run_phase1_training(
        phase_name="1b",
        augmentation_fn=apply_freq_mask_safe,
        augmentation_config=PHASE1B_CONFIG,
    )
