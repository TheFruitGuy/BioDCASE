"""
Delta / delta-delta channel front-end (phase 13s).

Wraps the canonical PCEN-channel extractor and APPENDS temporal-derivative
channels (regression deltas, torchaudio convention) computed along the TIME
axis of one source channel.

Motivation (D-call = blue-whale downsweep 90->25 Hz, 1-4 s, high FM variability;
71.5% of D misses are background-sensitivity): a stationary noise floor has a
near-zero temporal derivative, while the moving energy of an FM downsweep does
not -- so a delta channel is intrinsically a high-pass-in-time filter that
suppresses the stationary background and emphasises the moving call contour.
This is the same discriminative cue the classic low-frequency detectors
exploited via pitch-tracking, expressed as cheap extra channels.

The transform is PARAMETER-LESS (fixed linear regression-delta kernel), so step-0
behaviour is identical regardless of seed and channels 0..base-1 are byte-for-byte
the base extractor's output. Output layout matches the base extractor:
(B, C, F, T), with the delta channels appended on the channel axis.

Channels (default, source="pcen", order=2):
    [mag, cos, sin, PCEN, delta(PCEN), delta-delta(PCEN)]  -> 6 channels
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectrogram_pcen import PCENChannelExtractor


def _regression_delta(x: torch.Tensor, win: int) -> torch.Tensor:
    """torchaudio-convention regression delta along the LAST axis (time).

    d_t = ( sum_{n=1..win} n * (x_{t+n} - x_{t-n}) ) / ( 2 * sum_{n=1..win} n^2 )

    Edges use replicate padding so the output keeps the input length. ``x`` is
    (B, C, F, T); the derivative is taken over T (dim=-1).
    """
    if win < 1:
        raise ValueError(f"delta window must be >= 1, got {win}")
    T = x.shape[-1]
    # replicate-pad only the time axis; F.pad needs a 3D tensor to pad one dim
    x3 = x.reshape(-1, x.shape[-2], x.shape[-1])         # (N, F, T)
    xp = F.pad(x3, (win, win), mode="replicate")         # (N, F, T+2win)
    xp = xp.reshape(*x.shape[:-1], xp.shape[-1])         # (..., T+2win)
    denom = 2.0 * float(sum(n * n for n in range(1, win + 1)))
    out = torch.zeros_like(x)
    for n in range(1, win + 1):
        plus = xp[..., win + n: win + n + T]             # x_{t+n}
        minus = xp[..., win - n: win - n + T]            # x_{t-n}
        out = out + float(n) * (plus - minus)
    return out / denom


class DeltaChannelExtractor(nn.Module):
    """PCEN-channel extractor + appended temporal-derivative channels.

    Args:
        smooth:      PCEN smoothing coefficient passed to PCENChannelExtractor
                     (0.025 = the canonical 13n value).
        order:       1 -> append delta only; 2 -> append delta and delta-delta.
        source:      which base channel to differentiate, "pcen" (index 3, the
                     last channel) or "mag" (index 0, the demeaned magnitude).
        win:         regression-delta window (frames); 2 = torchaudio default.
    """

    def __init__(self, smooth: float = 0.025, order: int = 2,
                 source: str = "pcen", win: int = 2):
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"delta order must be 1 or 2, got {order}")
        if source not in ("pcen", "mag"):
            raise ValueError(f"delta source must be 'pcen' or 'mag', got {source}")
        self.base = PCENChannelExtractor(smooth=smooth)
        self.order = int(order)
        self.source = source
        self.win = int(win)
        # PCEN extractor channel order is [mag, cos, sin, PCEN].
        self._src_idx = 3 if source == "pcen" else 0
        self._n_out = self.base.n_out_channels + self.order

    @property
    def n_out_channels(self) -> int:
        return self._n_out

    @property
    def hop_length(self) -> int:
        return self.base.hop_length

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.base(audio)                          # (B, base, F, T)
        src = feats[:, self._src_idx: self._src_idx + 1, :, :]   # (B, 1, F, T)
        chans = [feats]
        d1 = _regression_delta(src, self.win)
        chans.append(d1)
        if self.order >= 2:
            chans.append(_regression_delta(d1, self.win))
        return torch.cat(chans, dim=1)                    # (B, base+order, F, T)
