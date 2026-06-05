"""
Phase 13o -- learnable per-band PCEN as an extra input channel
==============================================================

The ambitious version of 13n. 13n appended a FIXED PCEN channel (alpha 0.98,
delta 2.0, power 0.5, smooth 0.025) and lifted tuned D 0.206 -> 0.336. Here the
PCEN alpha / delta / power / smooth are LEARNABLE, one value PER FREQUENCY BIN,
so the network tunes the per-band AGC to exactly what surfaces the weak,
background-buried D downsweeps.

Why this is safe where LEAF / PCEN-as-frontend was not. Those learned an sPCEN
that REPLACED the spectrogram; its AGC could flatten the whole signal and killed
D (11a/11b/13m). Here PCEN is an ADDED channel: channels 0..2 are the untouched
phase-aware STFT, so whatever the PCEN params do, the loud BMABZ/BP information
survives. The flattening failure mode cannot recur.

Where the params live. ``conformer_core``'s optimiser only steps
``model.parameters()``, so the PCEN cannot live in the (param-less) extractor --
it lives HERE, inside the model. The extractor
(``spectrogram_lpcen.RawMagChannelExtractor``) hands over the raw STFT magnitude
as the last channel; ``forward`` replaces that channel with
``LearnablePCEN(raw_mag)`` before the FDY stem sees it.

Init. Every band starts at 13n's fixed values, so at step 0 this model's input
is byte-identical to 13n's -- it can only learn AWAY from a known-good config,
never start from a degenerate one.

PCEN(E) = (E / (eps + M)^alpha + delta)^power - delta^power,   M = EMA_s(E).
alpha, power, smooth in (0, 1) via sigmoid; delta >= 0 via softplus; all per-bin.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_final as cfg
from model_conformer_fdy import WhaleVAD_Conformer_FDY


def _logit(x: float) -> float:
    return math.log(x / (1.0 - x))


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LearnablePCEN(nn.Module):
    """Per-frequency-bin trainable PCEN over a raw-magnitude spectrogram (B,F,T).

    Parameters are unconstrained and squashed to valid ranges in :meth:`params`:
    alpha / power / smooth -> sigmoid -> (0,1); delta -> softplus -> [0, inf).
    Per-band (default) keeps one value per frequency bin; ``per_band=False``
    shares one scalar across all bins (4 trainable scalars total).
    """

    def __init__(self, n_freq: int, per_band: bool = True, eps: float = 1e-6,
                 ema_taps: int = 512, init_alpha: float = 0.98,
                 init_delta: float = 2.0, init_power: float = 0.5,
                 init_smooth: float = 0.025):
        super().__init__()
        self.eps, self.ema_taps = eps, ema_taps
        shape = (n_freq,) if per_band else (1,)
        self.z_alpha = nn.Parameter(torch.full(shape, _logit(init_alpha)))
        self.z_power = nn.Parameter(torch.full(shape, _logit(init_power)))
        self.z_smooth = nn.Parameter(torch.full(shape, _logit(init_smooth)))
        self.z_delta = nn.Parameter(torch.full(shape, _inv_softplus(init_delta)))

    def params(self):
        alpha = torch.sigmoid(self.z_alpha)
        power = torch.sigmoid(self.z_power)
        smooth = torch.sigmoid(self.z_smooth).clamp(1e-4, 1.0 - 1e-4)
        delta = F.softplus(self.z_delta)
        return alpha, power, smooth, delta

    def _ema(self, E: torch.Tensor, s: torch.Tensor) -> torch.Tensor:   # E:(B,F,T)
        """Per-band warm-started EMA via a truncated causal conv (vectorised).

        Row f of the kernel is ``s_f * (1-s_f)^k`` (k=0..taps-1), and a warm-start
        term restores M[0]=E[0]; matches the exact recursion to <1% (see test).
        """
        Fr, T = E.shape[1], E.shape[2]
        sb = s if s.numel() == Fr else s.expand(Fr)         # (Fr,)
        taps = min(self.ema_taps, T)
        k = torch.arange(taps, device=E.device, dtype=E.dtype)         # (taps,)
        kern = sb.view(Fr, 1) * (1.0 - sb).view(Fr, 1) ** k.view(1, taps)
        kern = kern.flip(1).view(Fr, 1, taps)               # causal, per band
        cold = F.conv1d(F.pad(E, (taps - 1, 0)), kern, groups=Fr)
        tt = torch.arange(T, device=E.device, dtype=E.dtype).view(1, 1, T)
        warm = E[:, :, :1] * (1.0 - sb).view(1, Fr, 1) ** (tt + 1.0)
        return cold + warm

    def forward(self, E: torch.Tensor) -> torch.Tensor:     # E:(B,F,T) raw magnitude
        alpha, power, smooth, delta = self.params()
        M = self._ema(E, smooth)
        a = alpha.view(1, -1, 1)
        p = power.view(1, -1, 1)
        d = delta.view(1, -1, 1)
        return (E / (self.eps + M) ** a + d) ** p - d ** p


class WhaleVAD_Conformer_FDY_LPCEN(WhaleVAD_Conformer_FDY):
    """FDY-Conformer whose LAST input channel is a LEARNABLE per-band PCEN.

    The extractor supplies the raw magnitude as the final channel; this model
    transforms it with :class:`LearnablePCEN` (a real submodule, so its params
    are in ``model.parameters()`` and the optimiser trains them) before the FDY
    stem. ``feat_channels`` is unchanged by the transform (channel count is
    preserved), so the stem / projection / Conformer body are exactly 13n's.
    """

    def __init__(self, *, lpcen_global: bool = False, lpcen_eps: float = 1e-6,
                 lpcen_taps: int = 512, **kw):
        super().__init__(**kw)
        n_freq = cfg.N_FFT // 2 + 1
        self.lpcen = LearnablePCEN(
            n_freq=n_freq, per_band=not lpcen_global,
            eps=lpcen_eps, ema_taps=lpcen_taps,
        )

    def forward(self, spec: torch.Tensor, key_padding_mask=None):
        base, rawmag = spec[:, :-1], spec[:, -1]            # (B,C-1,F,T), (B,F,T)
        pcen = self.lpcen(rawmag).unsqueeze(1)              # (B,1,F,T)
        spec = torch.cat([base, pcen], dim=1)               # restore channel count
        return super().forward(spec, key_padding_mask)


if __name__ == "__main__":
    from spectrogram_pcen import PCENChannelExtractor
    from spectrogram_lpcen import RawMagChannelExtractor

    torch.manual_seed(0)
    cfg.USE_3CLASS = True
    sr = cfg.SAMPLE_RATE
    n_freq = cfg.N_FFT // 2 + 1

    # 1) At init, LearnablePCEN must reproduce 13n's FIXED PCEN exactly.
    E = torch.rand(2, n_freq, 400) + 0.05
    lp = LearnablePCEN(n_freq=n_freq)                       # per-band, fixed inits
    fixed = PCENChannelExtractor()                          # alpha .98/delta 2/power .5/s .025
    with torch.no_grad():
        got = lp(E)
        ref = fixed._pcen(E)
    rel = (got - ref).abs().max() / ref.abs().clamp_min(1e-6).max()
    print(f"learnable-init vs fixed PCEN: max rel err = {rel.item():.4%}")
    assert rel < 0.01, "init does not match the fixed 13n PCEN"

    # 2) Per-band EMA matches the exact recursion (per-band smooth).
    s = torch.sigmoid(lp.z_smooth).clamp(1e-4, 1 - 1e-4)
    Mf = lp._ema(E, s)
    Mr = torch.zeros_like(E); Mr[:, :, 0] = E[:, :, 0]
    sb = s.view(1, -1)
    for t in range(1, E.shape[-1]):
        Mr[:, :, t] = sb * E[:, :, t] + (1 - sb) * Mr[:, :, t - 1]
    erel = (Mf - Mr).abs().max() / Mr.abs().max()
    print(f"per-band EMA vs exact recursion: max rel err = {erel.item():.4%}")
    assert erel < 0.01

    # 3) Full model: forward shape, and gradients reach the PCEN params.
    ext = RawMagChannelExtractor()
    base_model = WhaleVAD_Conformer_FDY(num_classes=cfg.n_classes(), feat_channels=4)
    model = WhaleVAD_Conformer_FDY_LPCEN(num_classes=cfg.n_classes(), feat_channels=4)
    audio = torch.randn(2, sr * 30) * 0.02
    spec = ext(audio)                                       # (2,4,F,T), last = raw mag
    logits = model(spec)
    print(f"logits shape : {tuple(logits.shape)}  (expect (2, T, 3))")
    assert logits.shape[0] == 2 and logits.shape[-1] == 3

    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    g = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
         for n, p in model.lpcen.named_parameters()}
    print(f"lpcen grads  : {g}")
    assert all(v > 0 for v in g.values()), "a PCEN param got no gradient"

    with torch.no_grad():                                  # materialise base's lazy proj too
        base_model(spec)                                   # 4-channel; content irrelevant here
    nb = sum(p.numel() for p in base_model.parameters())
    nl = sum(p.numel() for p in model.parameters())
    n_lpcen = sum(p.numel() for p in model.lpcen.parameters())
    print(f"params  base={nb:,}   lpcen-model={nl:,}   (+{nl - nb:,})   "
          f"lpcen submodule={n_lpcen} (expect {4 * n_freq})")
    assert n_lpcen == 4 * n_freq                            # 4 per-band fields
    assert nl - nb == 4 * n_freq                            # only the PCEN params differ
    print("ALL LEARNABLE-PCEN SELF-TESTS PASSED")
