"""
NAVE front end -- phase-aware STFT with a PCEN transient-enhanced channel.
=========================================================================
Parameter-free ``nn.Module`` (no trainable weights; the Hann window is a
buffer, so it moves with ``.to(device)``). For a waveform ``(B, n_samples)``
it returns ``(B, 4, n_freq, n_frames)``:

    ch 0  demeaned STFT magnitude        (per-frequency complex mean removed)
    ch 1  cos(phase)
    ch 2  sin(phase)
    ch 3  fixed PCEN on the RAW magnitude (per-bin AGC: stationary background
          normalised toward unity, transients that outrun the EMA pop above it)

Channels 0-2 are the phase-aware front end; channel 3 is the "Normalized" in
NAVE -- the lever that recovers weak, background-buried D-call downsweeps.
PCEN(E) = (E / (eps + M)^alpha + delta)^power - delta^power, with the per-bin
EMA  M[t] = (1-s) M[t-1] + s E[t]  computed as a vectorised truncated conv with
an exact warm-start correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import nave_config as cfg


class NAVEFeatureExtractor(nn.Module):
    """4-channel phase-aware STFT + fixed-PCEN extractor (no trainable params)."""

    def __init__(self):
        super().__init__()
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.hop_length = cfg.HOP_LENGTH
        self.register_buffer("window", torch.hann_window(self.win_length))

        self.alpha = cfg.PCEN_ALPHA
        self.delta = cfg.PCEN_DELTA
        self.power = cfg.PCEN_POWER
        self.smooth = cfg.PCEN_SMOOTH
        self.eps = cfg.PCEN_EPS
        self.ema_taps = cfg.PCEN_EMA_TAPS
        self.n_out_channels = cfg.FEAT_CHANNELS  # 4

    # -- per-bin EMA smoother, vectorised with exact warm start -----------
    def _ema(self, E: torch.Tensor) -> torch.Tensor:        # E: (B, F, T)
        Fr, T = E.shape[1], E.shape[2]
        s = self.smooth
        taps = min(self.ema_taps, T)
        k = torch.arange(taps, device=E.device, dtype=E.dtype)
        kern = (s * (1.0 - s) ** k).flip(0).view(1, 1, taps).expand(Fr, 1, taps).contiguous()
        cold = F.conv1d(F.pad(E, (taps - 1, 0)), kern, groups=Fr)
        tt = torch.arange(T, device=E.device, dtype=E.dtype).view(1, 1, T)
        return cold + E[:, :, :1] * (1.0 - s) ** (tt + 1.0)

    def _pcen(self, E: torch.Tensor) -> torch.Tensor:
        M = self._ema(E)
        return (E / (self.eps + M) ** self.alpha + self.delta) ** self.power \
            - self.delta ** self.power

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=False, return_complex=True,
        )                                                   # (B, F, T) complex

        # channels 0-2: demeaned magnitude + trig phase
        demeaned = stft - stft.mean(dim=-1, keepdim=True) \
            if cfg.NORM_FEATURES == "demean" else stft
        angle = demeaned.angle()
        base = torch.stack([demeaned.abs(), torch.cos(angle), torch.sin(angle)], dim=1)

        # channel 3: fixed PCEN on the RAW (non-demeaned) magnitude
        pcen = self._pcen(stft.abs()).unsqueeze(1)          # (B, 1, F, T)
        return torch.cat([base, pcen], dim=1)               # (B, 4, F, T)
