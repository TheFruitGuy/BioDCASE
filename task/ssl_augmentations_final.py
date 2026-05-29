"""
SSL Augmentations — _final
==========================

Identical to ``ssl_augmentations.py`` but the spectrogram constants are
sourced from ``config_final`` (rather than hardcoded), so they can never
drift from the _final STFT front end:

    HOP_LENGTH, SAMPLE_RATE  <- config_final
    N_FREQ = N_FFT // 2 + 1  (one-sided STFT bin count)

The protected band [13, 53] is the physical whale-call energy band in bin
indices for SR=250 / N_FFT=256 and is unchanged. The module imports
config_final but uses no class/label logic, so there is no USE_3CLASS hazard.
"""

from __future__ import annotations

from typing import Optional

import torch

import config_final as cfg

# Constants from config_final (single source of truth with the _final spectrogram).
HOP_LENGTH = cfg.HOP_LENGTH
N_FREQ = cfg.N_FFT // 2 + 1
SAMPLE_RATE = cfg.SAMPLE_RATE

# Whale-call energy band (bin indices). Outside this range it's safe to
# zero-mask without erasing discriminative content.
PROTECTED_FREQ_BIN_LO = 13
PROTECTED_FREQ_BIN_HI = 53


# ======================================================================
# Audio-domain augmentations
# ======================================================================

def ssl_volume_scaling(
    audio: torch.Tensor,                # (B, n_samples)
    p: float = 1.0,
    scale_min: float = 0.3,
    scale_max: float = 3.0,
) -> torch.Tensor:
    """Per-sample log-uniform volume scaling."""
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


def ssl_time_mask(
    audio: torch.Tensor,                # (B, n_samples)
    p: float = 1.0,
    min_seconds: float = 5.0,
    max_seconds: float = 12.0,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Per-sample audio-domain time masking: zero a random 5-12s window.

    On a 30s clip this masks 17-40% of the duration — large enough to
    force the encoder to reason about content rather than memorize the
    waveform.
    """
    B, n_samples = audio.shape
    out = audio.clone()
    min_samp = int(min_seconds * sample_rate)
    max_samp = int(max_seconds * sample_rate)
    for b in range(B):
        if torch.rand(1, device=audio.device).item() >= p:
            continue
        w = int(torch.randint(min_samp, max_samp + 1, (1,)).item())
        if n_samples - w <= 0:
            continue
        s = int(torch.randint(0, n_samples - w + 1, (1,)).item())
        out[b, s:s + w] = 0.0
    return out


def ssl_add_noise(
    audio: torch.Tensor,                # (B, n_samples)
    p: float = 1.0,
    snr_min_db: float = 5.0,
    snr_max_db: float = 20.0,
) -> torch.Tensor:
    """Add Gaussian noise at random per-sample SNR."""
    B = audio.size(0)
    out = audio.clone()
    for b in range(B):
        if torch.rand(1, device=audio.device).item() >= p:
            continue
        sig_energy = (out[b] ** 2).mean().clamp(min=1e-12)
        snr_db = float(snr_min_db + torch.rand(1).item() * (snr_max_db - snr_min_db))
        noise_energy = sig_energy / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(out[b]) * torch.sqrt(noise_energy)
        out[b] = out[b] + noise
    return out


def ssl_cross_site_mix(
    audio: torch.Tensor,                # (B, n_samples)
    sites: list[str],
    no_call_pool,                       # NoCallPool / ExtendedNoCallPool
    p: float = 0.5,
    snr_min_db: float = -6.0,
    snr_max_db: float = 6.0,
) -> torch.Tensor:
    """Per-sample cross-site noise mixing (3β only)."""
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
    p: float = 1.0,
    min_bins: int = 8,
    max_bins: int = 20,
    protected_lo: int = PROTECTED_FREQ_BIN_LO,
    protected_hi: int = PROTECTED_FREQ_BIN_HI,
) -> torch.Tensor:
    """Per-sample narrowband freq mask outside the [13, 53] protected band."""
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
    use_time_mask: bool = True,
    use_noise: bool = True,
    use_freq_mask: bool = True,
    use_cross_site: bool = False,
    no_call_pool=None,
    volume_p: float = 1.0,
    time_mask_p: float = 1.0,
    noise_p: float = 1.0,
    freq_mask_p: float = 1.0,
    cross_site_p: float = 0.5,
) -> torch.Tensor:
    """
    Apply the SSL augmentation pipeline once to produce a single view.

    Order:
        audio  ->  volume  ->  time_mask  ->  noise  ->  cross_site
              ->  STFT
              ->  freq_mask
              ->  spec
    """
    a = audio
    if use_volume:
        a = ssl_volume_scaling(a, p=volume_p)
    if use_time_mask:
        a = ssl_time_mask(a, p=time_mask_p)
    if use_noise:
        a = ssl_add_noise(a, p=noise_p)
    if use_cross_site:
        if no_call_pool is None:
            raise ValueError("use_cross_site=True requires no_call_pool")
        a = ssl_cross_site_mix(a, sites, no_call_pool, p=cross_site_p)

    spec = spec_extractor(a)

    if use_freq_mask:
        spec = ssl_freq_mask(spec, p=freq_mask_p)

    return spec
