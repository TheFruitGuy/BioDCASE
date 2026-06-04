"""
Phase 13n -- PCEN as an extra input channel (not as the frontend)
=================================================================

Every prior D-lever either reweighted a fixed STFT (FDY, tfwSE) or REPLACED the
frontend with a learnable filterbank whose AGC flattened the signal (LEAF,
PCEN-as-frontend 11a/11b/13m, all of which killed D). This is different: keep
the phase-aware STFT exactly as 13d uses it, and APPEND one channel of fixed
PCEN computed on the raw STFT magnitude.

Why this targets D specifically. D misses are a 71.5% background/sensitivity
problem: weak ~20-100 Hz downsweeps buried in slowly-varying noise. PCEN's
per-bin AGC divides the energy by a running EMA of itself, so stationary
background is normalised toward unity while a transient that rises faster than
the EMA pops above it. The model keeps the demeaned STFT (channels 0-2) for the
loud, easy BMABZ/BP and gets a transient-enhanced view (channel 3) where a weak
D event is no longer buried. Unlike PCEN-as-frontend, nothing is thrown away.

Design
------
- Channels 0..2 are byte-identical to ``SpectrogramExtractor`` (clean A/B vs
  13d): this module simply calls it.
- Channel 3 is fixed PCEN on the RAW (non-demeaned) magnitude. Demeaning is the
  wrong preprocessing for PCEN -- its EMA already removes the background, more
  adaptively. Fixed parameters (librosa defaults) keep the module param-less, so
  it stays a drop-in extractor (conformer_core only trains/saves the model).
- The EMA smoother reuses the warm-start truncated-conv from ``model_leaf``
  (verified there to match the exact recursion), so it is fully vectorised.

PCEN(E) = (E / (eps + M)^alpha + delta)^power - delta^power,
  M[t] = (1-s) M[t-1] + s E[t]   (per frequency bin, shared s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_final as cfg
from spectrogram_final import SpectrogramExtractor


class PCENChannelExtractor(nn.Module):
    """SpectrogramExtractor channels + one fixed-PCEN transient-enhanced channel."""

    def __init__(self, alpha: float = 0.98, delta: float = 2.0, power: float = 0.5,
                 smooth: float = 0.025, eps: float = 1e-6, ema_taps: int = 512):
        super().__init__()
        self.base = SpectrogramExtractor()
        self.hop_length = self.base.hop_length          # eval/validate read this
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.register_buffer("window", torch.hann_window(self.win_length))
        self.alpha, self.delta, self.power = alpha, delta, power
        self.smooth, self.eps, self.ema_taps = smooth, eps, ema_taps
        base_ch = 4 if cfg.DUAL_RESOLUTION else 3
        self.n_out_channels = base_ch + 1

    def _ema(self, E: torch.Tensor) -> torch.Tensor:     # E: (B, F, T)
        Fr, T = E.shape[1], E.shape[2]
        s = self.smooth
        taps = min(self.ema_taps, T)
        k = torch.arange(taps, device=E.device, dtype=E.dtype)
        kern = (s * (1.0 - s) ** k).flip(0).view(1, 1, taps).expand(Fr, 1, taps).contiguous()
        cold = F.conv1d(F.pad(E, (taps - 1, 0)), kern, groups=Fr)
        # Warm-start correction (M[0]=E[0]); E[0]*(1-s)^(t+1) decaying leakage.
        tt = torch.arange(T, device=E.device, dtype=E.dtype).view(1, 1, T)
        return cold + E[:, :, :1] * (1.0 - s) ** (tt + 1.0)

    def _pcen(self, E: torch.Tensor) -> torch.Tensor:
        M = self._ema(E)
        return (E / (self.eps + M) ** self.alpha + self.delta) ** self.power \
            - self.delta ** self.power

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        base = self.base(audio)                          # (B, base_ch, F, T), == 13d
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=False, return_complex=True,
        )
        pcen = self._pcen(stft.abs()).unsqueeze(1)       # (B, 1, F, T), raw-mag PCEN
        return torch.cat([base, pcen], dim=1)            # (B, base_ch+1, F, T)


if __name__ == "__main__":
    torch.manual_seed(0)
    sr = cfg.SAMPLE_RATE
    audio = torch.randn(2, sr * 30) * 0.02

    ext = PCENChannelExtractor()
    out = ext(audio)
    base = SpectrogramExtractor()(audio)
    T_stft = (sr * 30 - cfg.N_FFT) // cfg.HOP_LENGTH + 1
    print(f"base channels : {tuple(base.shape)}")
    print(f"pcen-aug out  : {tuple(out.shape)}  (expect (2, {base.shape[1]+1}, 129, {T_stft}))")
    assert out.shape[1] == base.shape[1] + 1 == ext.n_out_channels
    assert out.shape[-1] == T_stft

    # channels 0..2 must be byte-identical to 13d
    same = torch.allclose(out[:, :base.shape[1]], base, atol=0, rtol=0)
    print(f"base channels identical to 13d: {same}")
    assert same

    pc = out[:, -1]
    print(f"PCEN channel : finite={torch.isfinite(pc).all().item()} "
          f"min={pc.min():.3f} max={pc.max():.3f} mean={pc.mean():.3f} std={pc.std():.3f}")
    assert torch.isfinite(pc).all() and pc.std() > 1e-4

    # EMA correctness vs exact recursion (fixed s)
    E = torch.rand(2, 129, 400) + 0.1
    Mf = ext._ema(E)
    s = ext.smooth
    Mr = torch.zeros_like(E); Mr[:, :, 0] = E[:, :, 0]
    for t in range(1, E.shape[-1]):
        Mr[:, :, t] = s * E[:, :, t] + (1 - s) * Mr[:, :, t - 1]
    rel = (Mf - Mr).abs().max() / Mr.abs().max()
    print(f"EMA vs exact recursion: max rel err = {rel.item():.4%}")
    assert rel < 0.01

    n_par = sum(p.numel() for p in ext.parameters() if p.requires_grad)
    print(f"trainable params = {n_par} (must be 0)")
    assert n_par == 0
    print("ALL PCEN-CHANNEL SELF-TESTS PASSED")
