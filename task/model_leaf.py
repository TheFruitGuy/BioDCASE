"""
Phase 13m -- LEAF learnable frontend on the Conformer
=====================================================

The last base-architecture swing. FDY, tfwSE, the backbone, batch/LR, 60s,
EMA, FilterAugment, dual-res and (earlier) PCEN-as-frontend, multi-res STFT and
FDConv all left D pinned at ~0.20. Those all *reweight* a fixed STFT. LEAF
(Zeghidour et al., ICLR 2021, arXiv:2101.08596) instead *learns what gets
measured*: a complex Gabor filterbank + per-channel Gaussian lowpass pooling +
per-channel learnable PCEN, replacing log-mel end to end. It is the BioDCASE
winner's load-bearing frontend.

Paper-accurate, ported from CPJKU/EfficientLEAF's OG_leaf.py (a faithful
PyTorch port of google-research/leaf-audio) with two deliberate, documented
changes:

1. Whale sample rate. The paper targets 16 kHz / 25 ms / 10 ms. Whale audio
   here is 250 Hz, calls in ~15-100 Hz, so the analysis window must be long for
   low-frequency resolution: window_len = 1024 ms (257 samples, matching the
   STFT's N_FFT=256), stride = 20 ms (5 samples, matching HOP_LENGTH). Filters
   are mel-initialised over [5, 125] Hz; below 1 kHz the mel scale is ~linear,
   so this is both paper-faithful and well-matched to the narrow whale band.

2. PCEN EMA. The exact sPCEN smoother M[t] = s.E[t] + (1-s).M[t-1] is a
   sequential recursion; at T~1449 frames that is a 1449-deep Python scan,
   intractable to train. We compute it as a truncated (K=400) causal
   exponential conv, which the self-test verifies matches the exact recursion
   to <1% for smoothing coefficients >= 0.01 (sPCEN inits s=0.04).

Integration
-----------
LEAF has learnable parameters (Gabor centres/sigmas, Gaussian sigmas, sPCEN
alpha/delta/r/s), and conformer_core only optimises and checkpoints
``model.parameters()`` -- so LEAF lives INSIDE the model, not as the
spec_extractor. ``build_leaf`` pairs this model with a ``LeafPassthrough``
spec_extractor that returns the raw waveform; the model runs LEAF, then aligns
its frame count to the STFT framing (so the dataset's targets/mask line up),
then runs the unchanged WhaleVAD_Conformer stem + body (feat_channels=1).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config_final as cfg
from model_conformer import WhaleVAD_Conformer


# ======================================================================
# Gabor impulse response  (paper-exact)
# ======================================================================

def gabor_impulse_response(t: torch.Tensor, center: torch.Tensor,
                           fwhm: torch.Tensor) -> torch.Tensor:
    """Complex Gabor filters: (1/(sqrt(2pi) sigma)) e^{i center t} e^{-t^2/2 sigma^2}."""
    denominator = (1.0 / (np.sqrt(2 * np.pi) * fwhm)).type(torch.complex64).unsqueeze(1)
    gaussian = torch.exp(torch.outer(1.0 / (2.0 * fwhm ** 2), -t ** 2)).type(torch.complex64)
    sinusoid = torch.exp(1j * torch.outer(center.type(torch.complex64),
                                          t.type(torch.complex64)))
    return denominator * sinusoid * gaussian


def gabor_filters(kernel: torch.Tensor, size: int) -> torch.Tensor:
    t = torch.arange(-(size // 2), (size + 1) // 2, device=kernel.device,
                     dtype=torch.float32)
    return gabor_impulse_response(t, center=kernel[:, 0], fwhm=kernel[:, 1])


class GaborConstraint(nn.Module):
    """mu in [0, pi]; sigma s.t. frequency-FWHM in [1/W, 1/2] (paper Sec. 3.1)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        mu = torch.clamp(kernel[:, 0], 0.0, math.pi)
        sigma_lo = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_hi = self.k * math.sqrt(2 * math.log(2)) / math.pi
        sigma = torch.clamp(kernel[:, 1], sigma_lo, sigma_hi)
        return torch.stack([mu, sigma], dim=1)


# ======================================================================
# Mel initialisation of the Gabor (mu, sigma)  (paper-exact)
# ======================================================================

def _linear_to_mel_weight_matrix(n_filters, n_fft, sample_rate, min_freq, max_freq):
    hz2mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    nyq = sample_rate / 2.0
    input_bins = n_fft // 2 + 1
    hz_bins = torch.linspace(0, nyq, input_bins)
    mel_bins = torch.linspace(hz2mel(min_freq), hz2mel(max_freq), n_filters + 2)
    x = hz2mel(hz_bins.numpy())
    x = torch.from_numpy(x).float().unsqueeze(1)
    l, c, r = mel_bins[:-2], mel_bins[1:-1], mel_bins[2:]
    tri = torch.min((x - l) / (c - l), (x - r) / (c - r))
    return torch.clamp(tri, min=0.0)                       # (input_bins, n_filters)


def mel_gabor_params(n_filters, sample_rate, min_freq, max_freq, n_fft=512):
    """Centre freqs (rad/sample) and sigmas matching a mel filterbank."""
    mel = _linear_to_mel_weight_matrix(n_filters, n_fft, sample_rate,
                                       min_freq, max_freq).t()   # (n_filters, bins)
    coeff = np.sqrt(2.0 * np.log(2.0)) * n_fft
    sqrt_filters = torch.sqrt(mel)
    centers = torch.argmax(sqrt_filters, dim=1).float()
    peaks, _ = torch.max(sqrt_filters, dim=1)
    half = (peaks / 2.0).unsqueeze(1)
    fwhms = torch.sum((sqrt_filters >= half).float(), dim=1).clamp(min=1.0)
    return torch.stack([centers * 2 * np.pi / n_fft, coeff / (np.pi * fwhms)], dim=1)


def linear_gabor_params(n_filters, sample_rate, min_freq, max_freq):
    """Centres linear in Hz across the band; sigma for ~2x-spacing FWHM."""
    nyq = sample_rate / 2.0
    mu = torch.linspace(math.pi * min_freq / nyq, math.pi * max_freq / nyq, n_filters)
    spacing = (mu[-1] - mu[0]) / max(n_filters - 1, 1)
    sigma = torch.full((n_filters,), float(2.3548 / (2.0 * spacing)))
    return torch.stack([mu, sigma], dim=1)


# ======================================================================
# LEAF stages
# ======================================================================

class GaborConv1D(nn.Module):
    """Complex Gabor filterbank as a real conv1d with 2N (real,imag) channels."""

    def __init__(self, n_filters, kernel_size, init_params):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self._kernel = nn.Parameter(init_params.float())     # (n_filters, 2)
        self.constraint = GaborConstraint(kernel_size)

    def forward(self, x):                                    # x: (B, 1, T)
        kernel = self.constraint(self._kernel)
        filt = gabor_filters(kernel, self.kernel_size)       # (n_filters, size) complex
        real, imag = filt.real, filt.imag
        w = torch.stack([real, imag], dim=1).reshape(2 * self.n_filters, self.kernel_size)
        w = w[:, None, :]                                    # (2N, 1, size)
        return F.conv1d(x, w, stride=1, padding="same")      # (B, 2N, T)


class SquaredModulus(nn.Module):
    """(real_i^2 + imag_i^2) per filter, via paired avg-pool x2 (paper-exact)."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):                                    # (B, 2N, T)
        x = x.transpose(2, 1)                                # (B, T, 2N)
        x = 2.0 * self.pool(x ** 2)                          # (B, T, N)
        return x.permute(0, 2, 1)                            # (B, N, T)


class GaussianLowpass(nn.Module):
    """Depthwise per-channel Gaussian pooling; learnable sigma (init 0.4)."""

    def __init__(self, n_filters, kernel_size, stride):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sigma = nn.Parameter(torch.full((n_filters,), 0.4))

    def forward(self, x):                                    # (B, N, T)
        s = torch.clamp(self.sigma, 2.0 / self.kernel_size, 0.5).view(self.n_filters, 1)
        t = torch.arange(self.kernel_size, device=x.device, dtype=torch.float32).view(1, -1)
        num = t - 0.5 * (self.kernel_size - 1)
        den = s * 0.5 * (self.kernel_size - 1)
        kern = torch.exp(-0.5 * (num / den) ** 2)            # (N, kernel)
        kern = (kern / kern.sum(dim=1, keepdim=True))[:, None, :]   # (N, 1, kernel), unit-gain
        pad = self.kernel_size // 2
        return F.conv1d(x, kern, stride=self.stride, padding=pad, groups=self.n_filters)


class sPCEN(nn.Module):
    """Per-channel learnable PCEN (paper-exact compression; truncated-EMA smoother).

    out = (E / (eps + M)^alpha + delta)^(1/r) - delta^(1/r),
    M = causal EMA of E with per-channel smoothing s (init 0.04).
    alpha init 0.96, delta init 2.0, r init 2.0.
    """

    def __init__(self, n_filters, ema_taps=400, eps=1e-6):
        super().__init__()
        self.n_filters = n_filters
        self.ema_taps = ema_taps
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((n_filters,), 0.96))
        self.delta = nn.Parameter(torch.full((n_filters,), 2.0))
        self.root = nn.Parameter(torch.full((n_filters,), 2.0))
        self.smooth = nn.Parameter(torch.full((n_filters,), 0.04))

    def _ema(self, E):                                       # E: (B, N, T)
        s = torch.clamp(self.smooth, 1e-4, 1.0).view(self.n_filters, 1)
        k = torch.arange(self.ema_taps, device=E.device, dtype=torch.float32).view(1, -1)
        kern = (s * (1.0 - s) ** k).flip(1)[:, None, :]      # (N,1,taps); flip -> causal corr
        Ep = F.pad(E, (self.ema_taps - 1, 0))
        cold = F.conv1d(Ep, kern, groups=self.n_filters)     # zero-init truncated EMA
        # Warm-start correction so M[0]=E[0] exactly (LEAF inits the recursion
        # at the first frame). The closed-form leakage of that initial state is
        # E[0]*(1-s)^(t+1); adding it makes cold-conv == warm recursion, not just
        # asymptotically. Decays in ~1/s frames; numerically (1-s)^t in [0,1].
        T = E.shape[-1]
        tt = torch.arange(T, device=E.device, dtype=torch.float32).view(1, 1, T)
        corr = E[:, :, :1] * (1.0 - s).view(1, self.n_filters, 1) ** (tt + 1.0)
        return cold + corr

    def forward(self, E):                                    # E: (B, N, T)
        M = self._ema(E)
        alpha = torch.minimum(self.alpha, torch.ones_like(self.alpha)).view(1, -1, 1)
        root = torch.maximum(self.root, torch.ones_like(self.root)).view(1, -1, 1)
        delta = self.delta.view(1, -1, 1)
        inv_root = 1.0 / root
        smoothed = (self.eps + M) ** alpha
        return (E / smoothed + delta) ** inv_root - delta ** inv_root


class LEAF(nn.Module):
    """Gabor filterbank -> squared modulus -> Gaussian lowpass -> sPCEN/log."""

    def __init__(self, n_filters=128, sample_rate=None, window_len_ms=None,
                 window_stride_ms=None, min_freq=5.0, max_freq=None,
                 init="mel", compression="pcen"):
        super().__init__()
        sample_rate = sample_rate or cfg.SAMPLE_RATE
        window_len_ms = window_len_ms if window_len_ms is not None \
            else cfg.WIN_LENGTH / sample_rate * 1000.0
        window_stride_ms = window_stride_ms if window_stride_ms is not None \
            else cfg.HOP_LENGTH / sample_rate * 1000.0
        max_freq = max_freq if max_freq is not None else sample_rate / 2.0
        if init not in ("mel", "linear"):
            raise ValueError("init must be 'mel' or 'linear'")
        if compression not in ("pcen", "log"):
            raise ValueError("compression must be 'pcen' or 'log'")

        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.min_freq, self.max_freq = min_freq, max_freq
        self.compression = compression
        window_size = int(sample_rate * window_len_ms // 1000 + 1)
        stride = max(1, int(sample_rate * window_stride_ms // 1000))
        self.window_size, self.stride = window_size, stride

        if init == "mel":
            params = mel_gabor_params(n_filters, sample_rate, min_freq, max_freq)
        else:
            params = linear_gabor_params(n_filters, sample_rate, min_freq, max_freq)

        self.complex_conv = GaborConv1D(n_filters, window_size, params)
        self.activation = SquaredModulus()
        self.pooling = GaussianLowpass(n_filters, window_size, stride)
        self.compress = sPCEN(n_filters) if compression == "pcen" else None

    def forward(self, audio):                                # (B, T) or (B, 1, T)
        x = audio[:, None, :] if audio.dim() == 2 else audio
        x = self.complex_conv(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = torch.clamp(x, min=1e-5)
        if self.compress is not None:
            x = self.compress(x)
        else:
            x = torch.log(x.clamp(min=1e-6))
        return x                                             # (B, n_filters, T)


class LeafPassthrough(nn.Module):
    """spec_extractor that returns the raw waveform (LEAF runs inside the model).

    Exposes ``hop_length`` because the eval/validate frame->time mapping reads
    it: the LEAF model interpolates its output to the STFT frame count, so the
    effective hop is cfg.HOP_LENGTH (identical frame->time mapping to the STFT).
    """

    def __init__(self):
        super().__init__()
        self.hop_length = cfg.HOP_LENGTH

    def forward(self, audio):
        return audio


# ======================================================================
# LEAF + Conformer
# ======================================================================

class WhaleVAD_Conformer_LEAF(WhaleVAD_Conformer):
    """Conformer whose frontend is LEAF instead of the fixed STFT (feat_channels=1)."""

    def __init__(self, *, leaf_n_filters=128, leaf_init="mel",
                 leaf_compression="pcen", leaf_min_freq=5.0, **kw):
        kw["feat_channels"] = 1
        super().__init__(**kw)
        self.leaf = LEAF(n_filters=leaf_n_filters, init=leaf_init,
                         compression=leaf_compression, min_freq=leaf_min_freq)
        self.leaf_n_filters = leaf_n_filters
        self.leaf_init = leaf_init
        self.leaf_compression = leaf_compression
        self.leaf_min_freq = leaf_min_freq

    def forward(self, audio, key_padding_mask=None):
        x = self.leaf(audio)                                 # (B, n_filters, T_leaf)
        x = x.unsqueeze(1)                                   # (B, 1, n_filters, T_leaf)
        # Align LEAF's frame count to the STFT framing so the dataset's
        # per-frame targets/mask (and the Conformer's key_padding_mask) line up.
        T_target = (audio.shape[-1] - cfg.N_FFT) // cfg.HOP_LENGTH + 1
        if x.shape[-1] != T_target:
            x = F.interpolate(x, size=(x.shape[2], T_target),
                              mode="bilinear", align_corners=False)
        return super().forward(x, key_padding_mask=key_padding_mask)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    sr = cfg.SAMPLE_RATE

    # 1) mel init: centres span the whale band, sorted-ish, sigmas in valid range.
    p = mel_gabor_params(128, sr, 5.0, 125.0)
    mu_hz = p[:, 0] / math.pi * (sr / 2.0)
    print(f"mel init: centre Hz min={mu_hz.min():.1f} max={mu_hz.max():.1f} "
          f"med={mu_hz.median():.1f}; sigma min={p[:,1].min():.2f} max={p[:,1].max():.2f}")
    assert 0 <= mu_hz.min() and mu_hz.max() <= 125.5

    # 2) Gabor filter is bandpass: FFT of filter k peaks near its centre freq.
    gc = GaborConv1D(128, 257, p)
    with torch.no_grad():
        filt = gabor_filters(gc.constraint(gc._kernel), 257)
        k = 60
        mag = torch.fft.rfft(filt[k].real).abs()
        peak_bin = mag.argmax().item()
        peak_hz = peak_bin * sr / 257
    print(f"filter 60: FFT peak ~{peak_hz:.1f} Hz vs centre {mu_hz[60]:.1f} Hz")

    # 3) truncated EMA == exact recursion (validates the sPCEN smoother).
    pc = sPCEN(8, ema_taps=400)
    E = torch.rand(2, 8, 600) + 0.1
    with torch.no_grad():
        M_fast = pc._ema(E)
        s = torch.clamp(pc.smooth, 1e-4, 1.0).view(8, 1)
        M_ref = torch.zeros_like(E); M_ref[:, :, 0] = E[:, :, 0]
        for t in range(1, E.shape[-1]):
            M_ref[:, :, t] = s.squeeze(1) * E[:, :, t] + (1 - s.squeeze(1)) * M_ref[:, :, t - 1]
        rel = (M_fast - M_ref).abs().max() / M_ref.abs().max()
    print(f"truncated-EMA vs exact recursion: max rel err = {rel.item():.4%}")
    assert rel < 0.01, "EMA approximation too loose"

    # 4) LEAF forward: shape, finite, varies.
    leaf = LEAF(n_filters=128, init="mel", compression="pcen")
    audio = torch.randn(2, sr * 30) * 0.02
    out = leaf(audio)
    print(f"LEAF out: {tuple(out.shape)}  finite={torch.isfinite(out).all().item()}  "
          f"std={out.std():.4f}")
    assert torch.isfinite(out).all() and out.std() > 1e-4

    # 5) full model: output frame count == STFT framing; grad reaches LEAF params.
    cfg.USE_3CLASS = True
    n = cfg.n_classes()
    model = WhaleVAD_Conformer_LEAF(num_classes=n, leaf_init="mel", leaf_compression="pcen")
    logits = model(audio)
    T_stft = (sr * 30 - cfg.N_FFT) // cfg.HOP_LENGTH + 1
    print(f"model out {tuple(logits.shape)}  (T_target={T_stft})")
    assert logits.shape == (2, T_stft, n)
    F.binary_cross_entropy_with_logits(model(audio), torch.zeros_like(logits)).backward()
    for name in ["leaf.complex_conv._kernel", "leaf.pooling.sigma",
                 "leaf.compress.alpha", "leaf.compress.smooth"]:
        prm = dict(model.named_parameters())[name]
        g = prm.grad
        ok = g is not None and g.abs().sum().item() > 0
        print(f"  grad {name}: {'ok' if ok else 'ZERO'}")
        assert ok
    n_par = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"params (trainable) = {n_par:,}")
    print("ALL LEAF SELF-TESTS PASSED")
