"""
Phase 1c: Whole-Segment Volume Scaling
=======================================

Single-axis change to the F1=0.474 baseline (``train.py``): the
training loop applies per-sample volume scaling to the audio
waveform before STFT. Both call and background are scaled together
— this is volume augmentation, not SNR augmentation.

Why this might help
-------------------
Different hydrophones have different gain calibrations and capture
calls at different distances from the source. Even within a single
hydrophone, the apparent loudness of a call depends on the whale's
distance. The model should be invariant to absolute volume — only
relative spectro-temporal structure should matter.

Per-frequency normalization in the spectrogram extractor already
provides some invariance, but it's not perfect for low-energy or
saturated samples. Direct volume augmentation gives the model
explicit exposure to the full range of plausible volumes.

Hyperparameters
---------------
``scale_prob = 0.5``: half of training samples get scaled per epoch.
``scale_min = 0.5``, ``scale_max = 2.0``: 1 octave (-6 dB to +6 dB)
of variation. Sampled in log-space so amplification and attenuation
are equally likely.

This is conservative — broader ranges would test invariance more
aggressively but might saturate or null-out the input. Half-octave
scaling is a safe starting point.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase1c.py
"""

from augmentations import apply_volume_scaling
from phase1_baseline import run_phase1_training


PHASE1C_CONFIG = {
    "scale_prob": 0.5,
    "scale_min": 0.5,
    "scale_max": 2.0,
    "scale_db_min": -6.0,  # 20 * log10(0.5)
    "scale_db_max":  6.0,  # 20 * log10(2.0)
}


if __name__ == "__main__":
    run_phase1_training(
        phase_name="1c",
        augmentation_fn=apply_volume_scaling,
        augmentation_config=PHASE1C_CONFIG,
    )
