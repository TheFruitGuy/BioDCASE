"""
Phase 1a: Time Masking on Spectrograms
=======================================

Single-axis change to the F1=0.474 baseline (``train.py``): the
training loop applies per-sample time masking to the spectrogram
between extraction and the model forward pass.

What we expect to see
---------------------
- Train loss should track or be slightly higher than the no-aug
  baseline (1a is a regularizer, not a free win).
- Val F1 should be similar in early epochs and start to diverge
  positively in later epochs as the regularization compounds.
- Per-site breakdown: improvement should be most visible on
  out-of-distribution sites (casey2017 in particular) where
  overfitting is most painful.

If F1 plateaus below the 0.474 baseline by epoch 50, the mask is
too aggressive — try ``mask_prob=0.3`` or ``max_frames=40``.

Hyperparameters
---------------
``mask_prob = 0.5``: half of training samples get masked per epoch.
``min_frames = 10`` / ``max_frames = 75``: 0.2-1.5s mask width.
Whale calls are 5-30s, so a 1.5s mask leaves plenty of context.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase1a.py
"""

from augmentations import apply_time_mask
from phase1_baseline import run_phase1_training


# Augmentation hyperparameters logged to wandb config so dashboards
# can split runs by setting if we sweep later.
PHASE1A_CONFIG = {
    "mask_prob": 0.5,
    "min_frames": 10,
    "max_frames": 75,
    "min_seconds": 10 * 5 / 250,   # ≈ 0.2s
    "max_seconds": 75 * 5 / 250,   # ≈ 1.5s
}


if __name__ == "__main__":
    run_phase1_training(
        phase_name="1a",
        augmentation_fn=apply_time_mask,
        augmentation_config=PHASE1A_CONFIG,
    )
