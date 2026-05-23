"""
Audio and Spectrogram Augmentation Functions for Phase 1
=========================================================

Phase 1 of the WhaleVAD project explores audio augmentation, an axis
Geldenhuys does not investigate in either of his papers. The motivating
hypothesis: existing detectors overfit to training-site-specific noise
textures, which limits cross-site generalisation (Phase 0n's per-site
breakdown shows we underperform the paper most on bmabz at the held-out
sites — a generalisation gap, not a learning gap).

This module provides four augmentation functions, one per Phase 1
sub-experiment:

  - ``apply_time_mask`` (Phase 1a)
      Zero a random 0.2-1.5s window of the spectrogram. Forces the model
      to recognise calls from partial information.

  - ``apply_freq_mask_safe`` (Phase 1b)
      Zero a narrow (5-10 Hz) frequency band, deliberately *avoiding*
      the 15-50 Hz region where blue/fin whale call energy lives.
      Models robustness to narrowband interference (shipping noise,
      etc.) without erasing the discriminative signal.

  - ``apply_volume_scaling`` (Phase 1c)
      Scale the entire audio segment by a random factor in [0.5, 2.0].
      Models recording-distance and gain variation across hydrophones.

  - ``apply_cross_site_mix`` (Phase 1e)
      For each positive sample, with some probability mix in a no-call
      audio clip from a *different* training site at controlled SNR.
      Directly addresses the diagnosed cross-site failure mode: the
      model sees the same calls embedded in noise textures from sites
      it would otherwise overfit to.

Design conventions
------------------
- All functions are stateless and side-effect-free. Tensors are cloned
  before modification so callers can keep the originals.
- Augmentations declare via their ``DOMAIN`` attribute whether they
  operate in the audio domain (before STFT) or the spectrogram domain
  (after STFT). The Phase 1 training loop reads this to decide where
  to call them.
- Mask handling: time-domain augmentations (1a) zero the per-frame loss
  mask in the masked region — the model is *not* penalised for missing
  calls in regions we erased. Frequency-domain augmentations (1b) leave
  the loss mask untouched (the loss is per-frame, not per-frequency).
- Augmentations take ``targets`` only when they need to know where the
  call is (for SNR-style operations). The targets are not modified —
  augmentation should never silently shift labels.
- All randomness is local to each call (uses ``torch.rand``); the
  caller's seeding determines reproducibility.

The augmentation hook signature in phase1_baseline is::

    spec, mask, audio = aug_fn(
        spec=spec, mask=mask, audio=audio, targets=targets,
        metas=metas, state=state,
    )

Audio-domain augmentations modify ``audio`` (and the caller re-extracts
``spec`` afterwards). Spectrogram-domain augmentations modify ``spec``
(and ignore ``audio``). All four return the full tuple regardless, so
the call site is uniform.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import torch


# Spectrogram framing constants — must match config.py / spectrogram.py
# settings. Hard-coded here rather than imported because augmentations
# should be usable from contexts where cfg isn't available (e.g.
# unit tests, sweeps).
SAMPLE_RATE = 250          # Hz
HOP_LENGTH = 5             # samples between STFT frames
N_FFT = 256                # STFT window size (informational only)
N_FREQ = N_FFT // 2 + 1    # 129 frequency bins
NYQUIST = SAMPLE_RATE / 2  # 125 Hz

# The frequency band where blue and fin whale calls live. Bin index is
# computed as freq_hz / (sample_rate / 2) * (N_FFT // 2). At
# sr=250, N_FFT=256: bin n covers [(n - 0.5)/128 * 125, (n + 0.5)/128 * 125] Hz.
# 15 Hz ≈ bin 15, 50 Hz ≈ bin 51. We add a 2-bin guard on each side so
# 1b's frequency mask never touches discriminative content even at the
# edges.
PROTECTED_FREQ_BIN_LO = 13
PROTECTED_FREQ_BIN_HI = 53


# ======================================================================
# Phase 1a: Time masking
# ======================================================================

def apply_time_mask(
    *,
    spec: torch.Tensor,           # (B, C, F, T_spec)
    mask: torch.Tensor,           # (B, T_target) bool/float
    audio: torch.Tensor,          # ignored
    targets: torch.Tensor,        # ignored (we don't shift labels)
    metas: list[dict],            # ignored
    state: Optional[Any] = None,  # ignored
    mask_prob: float = 0.5,
    min_frames: int = 10,         # 0.2s at hop=5
    max_frames: int = 75,         # 1.5s at hop=5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample time masking on the spectrogram.

    For each sample with probability ``mask_prob``:
      1. Pick width w uniform in ``[min_frames, max_frames]``.
      2. Pick start s uniform in the sample's *valid* frame range
         (i.e. excluding already-padded frames).
      3. Zero spec[b, :, :, s:s+w] across all channels and freq bins.
      4. Zero mask[b, s:s+w] so loss ignores those frames.

    Whale calls last 5-30 seconds, so a 1.5-second mask leaves plenty
    of context. This is the conservative end of SpecAugment-style
    masking.

    Returns ``(spec, mask, audio)``. ``audio`` is returned unmodified
    so the call site can be uniform with audio-domain augmentations.
    """
    B = spec.size(0)
    T_spec = spec.size(-1)
    T_mask = mask.size(-1)
    T_eff = min(T_spec, T_mask)
    if T_eff <= min_frames:
        return spec, mask, audio

    spec = spec.clone()
    mask = mask.clone()

    for b in range(B):
        if torch.rand(1, device=spec.device).item() >= mask_prob:
            continue

        # Find the sample's valid (non-padded) range so we don't
        # "mask" frames that are already padding.
        valid_idx = mask[b, :T_eff].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= min_frames:
            continue
        valid_start = int(valid_idx[0].item())
        valid_end = int(valid_idx[-1].item()) + 1
        valid_len = valid_end - valid_start
        if valid_len <= min_frames:
            continue

        w = int(torch.randint(
            min_frames, min(max_frames, valid_len) + 1,
            (1,), device=spec.device,
        ).item())
        s_max = valid_end - w
        if s_max <= valid_start:
            continue
        s = int(torch.randint(
            valid_start, s_max + 1, (1,), device=spec.device,
        ).item())
        e = s + w

        spec[b, :, :, s:e] = 0.0
        mask[b, s:e] = 0.0

    return spec, mask, audio


apply_time_mask.DOMAIN = "spectrogram"


# ======================================================================
# Phase 1b: Narrowband frequency masking, avoiding the 15-50 Hz band
# ======================================================================

def apply_freq_mask_safe(
    *,
    spec: torch.Tensor,           # (B, C, F, T)
    mask: torch.Tensor,
    audio: torch.Tensor,
    targets: torch.Tensor,        # ignored
    metas: list[dict],            # ignored
    state: Optional[Any] = None,  # ignored
    mask_prob: float = 0.5,
    min_bins: int = 5,            # ≈ 5 Hz wide
    max_bins: int = 12,           # ≈ 12 Hz wide
    protected_lo: int = PROTECTED_FREQ_BIN_LO,
    protected_hi: int = PROTECTED_FREQ_BIN_HI,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample narrowband frequency masking on the spectrogram.

    With probability ``mask_prob``, zero a contiguous frequency band of
    width ``[min_bins, max_bins]`` chosen to lie *outside* the
    protected band ``[protected_lo, protected_hi]`` where whale call
    energy lives.

    Two valid regions exist: below the protected band and above it.
    For each masked sample we pick one region, then a band within it.

    The loss mask is NOT modified — loss is per-frame, not
    per-frequency, so erasing a freq bin is just an input perturbation,
    not a label-shift.

    If the mask region is too small to fit ``min_bins`` (very short
    spectrograms or wide protected bands), the sample is left
    unmasked.
    """
    B, C, F, T = spec.shape
    if F <= protected_hi + min_bins:
        # Almost no headroom above the protected band. Use only the
        # below-band region.
        if protected_lo <= min_bins:
            return spec, mask, audio  # nowhere to mask

    spec = spec.clone()

    # Two candidate regions: [0, protected_lo) and [protected_hi, F).
    # Each region is valid if its length >= min_bins.
    regions = []
    if protected_lo >= min_bins:
        regions.append((0, protected_lo))
    if F - protected_hi >= min_bins:
        regions.append((protected_hi, F))
    if not regions:
        return spec, mask, audio

    for b in range(B):
        if torch.rand(1, device=spec.device).item() >= mask_prob:
            continue

        # Pick a region (uniform over available regions).
        r_lo, r_hi = regions[
            int(torch.randint(0, len(regions), (1,), device=spec.device).item())
        ]
        region_len = r_hi - r_lo

        w_max = min(max_bins, region_len)
        if w_max < min_bins:
            continue
        w = int(torch.randint(
            min_bins, w_max + 1, (1,), device=spec.device,
        ).item())
        s_max = r_hi - w
        if s_max < r_lo:
            continue
        s = int(torch.randint(
            r_lo, s_max + 1, (1,), device=spec.device,
        ).item())
        e = s + w

        # Zero this freq band across all channels and time frames for
        # this sample.
        spec[b, :, s:e, :] = 0.0

    return spec, mask, audio


apply_freq_mask_safe.DOMAIN = "spectrogram"


# ======================================================================
# Phase 1c: Whole-segment volume scaling
# ======================================================================

def apply_volume_scaling(
    *,
    spec: torch.Tensor,           # ignored
    mask: torch.Tensor,
    audio: torch.Tensor,          # (B, n_samples)
    targets: torch.Tensor,        # ignored
    metas: list[dict],            # ignored
    state: Optional[Any] = None,  # ignored
    scale_prob: float = 0.5,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample whole-segment volume scaling in the audio domain.

    With probability ``scale_prob``, multiply the entire audio segment
    by a random factor in ``[scale_min, scale_max]``. Both call and
    background are scaled together — this is volume augmentation, not
    SNR augmentation. (True SNR augmentation would scale only the call
    samples, but identifying call boundaries at audio resolution
    requires upsampling the targets; whole-segment scaling is
    simpler and still gives the model exposure to gain variation
    across hydrophones.)

    The spectrogram is NOT modified here — it's recomputed by the
    caller after this augmentation runs in the audio domain.
    """
    B = audio.size(0)
    audio = audio.clone()

    for b in range(B):
        if torch.rand(1, device=audio.device).item() >= scale_prob:
            continue
        # Sample uniformly in log-space so 0.5x and 2x are equally
        # likely (otherwise linear sampling biases toward amplification).
        log_min = torch.log(torch.tensor(scale_min))
        log_max = torch.log(torch.tensor(scale_max))
        log_factor = log_min + torch.rand(1) * (log_max - log_min)
        factor = float(torch.exp(log_factor).item())
        audio[b] = audio[b] * factor

    return spec, mask, audio


apply_volume_scaling.DOMAIN = "audio"


# ======================================================================
# Phase 1e: Cross-site noise mixing
# ======================================================================

def apply_cross_site_mix(
    *,
    spec: torch.Tensor,           # ignored
    mask: torch.Tensor,
    audio: torch.Tensor,          # (B, n_samples)
    targets: torch.Tensor,        # (B, T_target, C)
    metas: list[dict],            # batch metadata (provides each sample's site)
    state: Any,                   # required: NoCallPool instance
    mix_prob: float = 0.5,
    snr_min_db: float = -6.0,
    snr_max_db: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample cross-site noise mixing in the audio domain.

    For each sample with probability ``mix_prob``:
      1. Identify the source site from ``metas[b]["dataset"]``.
      2. Draw a no-call audio clip from a *different* site via
         ``state.sample(exclude_site=...)``.
      3. Pick a target SNR uniformly in ``[snr_min_db, snr_max_db]``.
         SNR is measured as call_energy / noise_energy, where the call
         portion is identified by frames with any non-zero target.
      4. Scale the noise so the resulting SNR matches the target.
      5. Add scaled noise to the original audio.

    Targets are NOT modified — the call is still where it was, just
    embedded in cross-site noise.

    Negative samples (no positive frames in targets) get the noise
    added at SNR=0 dB equivalent — i.e. matching the original signal
    energy. They contribute to learning that "this kind of background
    isn't a call" regardless of site.

    Parameters
    ----------
    state : NoCallPool
        Provides ``state.sample(exclude_site, n_samples_needed,
        device)`` returning a 1-D audio tensor of length
        ``n_samples_needed`` from a site other than ``exclude_site``.
        See ``precompute_no_call_pool.py`` for the precomputation step
        and ``phase1_baseline.NoCallPool`` for the runtime container.
    """
    if state is None:
        raise ValueError(
            "apply_cross_site_mix requires a NoCallPool in 'state'. "
            "Pre-compute it with precompute_no_call_pool.py and pass "
            "it via the run_phase1_training augmentation_state arg."
        )

    B, n_samples = audio.shape
    audio = audio.clone()

    for b in range(B):
        if torch.rand(1, device=audio.device).item() >= mix_prob:
            continue

        src_site = metas[b].get("dataset", None)
        if src_site is None:
            continue

        # Sample a no-call clip from a *different* site. NoCallPool
        # raises if no other site has any clips, but in practice we
        # always have 8 training sites with 50+ clips each.
        try:
            noise = state.sample(
                exclude_site=src_site,
                n_samples=n_samples,
                device=audio.device,
            )
        except RuntimeError:
            continue
        if noise is None or noise.numel() != n_samples:
            continue

        # SNR computation. Identify call frames from targets: a frame
        # is call-positive if any class is active. For SNR scaling we
        # need the audio samples corresponding to those frames; we
        # use the relationship n_samples_per_frame = HOP_LENGTH.
        any_class = (targets[b].sum(dim=-1) > 0).float()  # (T_target,)
        if any_class.sum() == 0:
            # Negative segment — use whole-segment energy, target SNR 0 dB.
            sig_energy = (audio[b] ** 2).mean().clamp(min=1e-12)
            target_snr_db = 0.0
        else:
            # Compute call energy from frames with positive targets.
            n_call_frames = int(any_class.sum().item())
            n_call_samples = n_call_frames * HOP_LENGTH
            # Find first/last call frame to extract a contiguous-ish
            # call region. Calls are typically contiguous within a
            # segment thanks to the collar mechanism.
            call_idx = any_class.nonzero(as_tuple=False).squeeze(-1)
            f0 = int(call_idx[0].item())
            f1 = int(call_idx[-1].item()) + 1
            s0 = f0 * HOP_LENGTH
            s1 = min(f1 * HOP_LENGTH, n_samples)
            sig_energy = (audio[b, s0:s1] ** 2).mean().clamp(min=1e-12)
            target_snr_db = float(
                (snr_min_db + torch.rand(1).item() * (snr_max_db - snr_min_db))
            )

        noise_energy = (noise ** 2).mean().clamp(min=1e-12)
        # SNR_dB = 10 * log10(sig / noise); solve for scale s.t.
        # sig / (s^2 * noise) = 10^(SNR/10)
        # => s = sqrt(sig / (noise * 10^(SNR/10)))
        scale = torch.sqrt(
            sig_energy / (noise_energy * (10.0 ** (target_snr_db / 10.0)))
        )
        audio[b] = audio[b] + scale * noise

    return spec, mask, audio


apply_cross_site_mix.DOMAIN = "audio"
