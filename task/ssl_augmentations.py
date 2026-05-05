"""
SSL Augmentations
=================

Positive-pair augmentations for contrastive pretraining. The supervised
augmentations in ``augmentations.py`` are designed around a (spec, mask,
audio, targets, metas) protocol that doesn't map cleanly to SSL — there
are no labels, no padding mask, and we want two independent views per
clip. The functions here are stripped-down equivalents that operate
directly on tensors.

Three augmentations:

- ``ssl_volume_scaling`` (audio domain)
    Per-sample log-uniform volume scaling.

- ``ssl_freq_mask`` (spec domain)
    Narrowband freq mask outside the protected [13, 53] bin range
    (whale-call energy band — same band as ``apply_freq_mask_safe``).

- ``ssl_cross_site_mix`` (audio domain)
    Mix in a no-call clip from a different site at random SNR. Only
    used in 3β. Treats every SSL clip as a "negative" for SNR purposes
    (whole-segment energy reference) since labels are unavailable.

All three are applied independently to each view, so the two views of
the same source clip differ in their realised augmentation parameters.
"""

from __future__ import annotations

from typing import Optional

import torch


# Constants — must match config.py / spectrogram.py. Hardcoded so this
# module is independent of cfg.
HOP_LENGTH = 5
N_FREQ = 129                  # n_fft // 2 + 1 with n_fft=256

# Whale-call energy band (bin indices). Outside this range it's safe to
# zero-mask without erasing discriminative content.
PROTECTED_FREQ_BIN_LO = 13
PROTECTED_FREQ_BIN_HI = 53


# ======================================================================
# Audio-domain augmentations
# ======================================================================

def ssl_volume_scaling(
    audio: torch.Tensor,                # (B, n_samples)
    p: float = 0.5,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> torch.Tensor:
    """Per-sample log-uniform volume scaling, returns a new tensor."""
    B = audio.size(0)
    out = audio.clone()
    log_min = torch.log(torch.tensor(scale_min))
    log_max = torch.log(torch.tensor(scale_max))
    for b in range(B):
        if torch.rand(1, device=audio.device).item() >= p:
            continue
        log_factor = log_min + torch.rand(1).item() * (log_max - log_min)
        factor = float(torch.exp(log_factor).item())
        out[b] = out[b] * factor
    return out


def ssl_cross_site_mix(
    audio: torch.Tensor,                # (B, n_samples)
    sites: list[str],
    no_call_pool,                       # NoCallPool instance
    p: float = 0.5,
    snr_min_db: float = -6.0,
    snr_max_db: float = 6.0,
) -> torch.Tensor:
    """
    Per-sample cross-site noise mixing.

    For each sample with probability ``p``, draw a no-call clip from a
    site other than ``sites[b]`` and mix it in at SNR ∈ [snr_min_db,
    snr_max_db]. SNR is computed against whole-segment energy, since
    we have no labels to identify "call frames" — equivalent to
    treating each SSL clip as if it were a negative segment.

    This is intentionally weaker than the supervised ``apply_cross_site_mix``
    which uses call-frame energy. The encoder will see clips at a wider
    range of effective SNRs, which is fine for pretraining: the goal is
    to push the encoder toward site-noise invariance, not to perfectly
    preserve call SNR.
    """
    B, n_samples = audio.shape
    out = audio.clone()

    for b in range(B):
        if torch.rand(1, device=audio.device).item() >= p:
            continue
        try:
            noise = no_call_pool.sample(
                exclude_site=sites[b],
                n_samples=n_samples,
                device=audio.device,
            )
        except RuntimeError:
            continue
        if noise is None or noise.numel() != n_samples:
            continue

        sig_energy = (out[b] ** 2).mean().clamp(min=1e-12)
        noise_energy = (noise ** 2).mean().clamp(min=1e-12)
        snr_db = float(snr_min_db + torch.rand(1).item() * (snr_max_db - snr_min_db))
        scale = torch.sqrt(sig_energy / (noise_energy * (10.0 ** (snr_db / 10.0))))
        out[b] = out[b] + scale * noise

    return out


# ======================================================================
# Spectrogram-domain augmentation
# ======================================================================

def ssl_freq_mask(
    spec: torch.Tensor,                  # (B, C, F, T)
    p: float = 0.5,
    min_bins: int = 5,
    max_bins: int = 12,
    protected_lo: int = PROTECTED_FREQ_BIN_LO,
    protected_hi: int = PROTECTED_FREQ_BIN_HI,
) -> torch.Tensor:
    """
    Per-sample narrowband frequency masking, avoiding the protected
    [protected_lo, protected_hi] bin range.

    Two valid mask regions: below the protected band and above it. For
    each masked sample we pick one region uniformly, then a random
    band of width ``[min_bins, max_bins]`` within it.
    """
    B, _, F, _ = spec.shape
    out = spec.clone()

    regions: list[tuple[int, int]] = []
    if protected_lo >= min_bins:
        regions.append((0, protected_lo))
    if protected_hi <= F - min_bins:
        regions.append((protected_hi, F))
    if not regions:
        return out

    for b in range(B):
        if torch.rand(1, device=spec.device).item() >= p:
            continue
        r_lo, r_hi = regions[
            int(torch.randint(0, len(regions), (1,), device=spec.device).item())
        ]
        region_len = r_hi - r_lo
        w_max = min(max_bins, region_len)
        if w_max < min_bins:
            continue
        w = int(torch.randint(min_bins, w_max + 1, (1,), device=spec.device).item())
        s_max = r_hi - w
        if s_max < r_lo:
            continue
        s = int(torch.randint(r_lo, s_max + 1, (1,), device=spec.device).item())
        out[b, :, s:s + w, :] = 0.0

    return out


# ======================================================================
# View pipeline
# ======================================================================

def make_view(
    audio: torch.Tensor,
    sites: list[str],
    spec_extractor,
    *,
    use_volume: bool = True,
    use_freq_mask: bool = True,
    use_cross_site: bool = False,
    no_call_pool=None,
    volume_p: float = 0.5,
    freq_mask_p: float = 0.5,
    cross_site_p: float = 0.5,
) -> torch.Tensor:
    """
    Apply the SSL augmentation pipeline once to produce a single view.

    Audio-domain augmentations (volume, optionally cross-site mix) run
    before spectrogram extraction; spectrogram-domain augmentations
    (freq mask) run after. Returns the augmented spectrogram of shape
    ``(B, 3, F, T_spec)`` ready to feed into the encoder.

    Call this twice (with different RNG state) per source batch to get
    the (view_a, view_b) contrastive pair.
    """
    a = audio
    if use_volume:
        a = ssl_volume_scaling(a, p=volume_p)
    if use_cross_site:
        if no_call_pool is None:
            raise ValueError("use_cross_site=True requires no_call_pool")
        a = ssl_cross_site_mix(a, sites, no_call_pool, p=cross_site_p)
    spec = spec_extractor(a)
    if use_freq_mask:
        spec = ssl_freq_mask(spec, p=freq_mask_p)
    return spec
