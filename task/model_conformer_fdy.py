"""
Phase 13d — Frequency-Dynamic Conv stem for the Conformer (final pipeline)
==========================================================================

The optimiser ladder (13a–13c) converged the Conformer at ~0.40 macro_paper
with D-class F1 stuck at ~0.15 even after threshold tuning. D is a frontend
problem, not an optimiser or a threshold problem: the CNN stem applies the
*same* learned kernels to every frequency bin, but D-calls are narrow-band,
non-stationary downsweeps whose discriminative structure lives in a specific
band. An ordinary conv's translation-invariance along frequency is exactly
the wrong inductive bias there.

**Frequency-Dynamic Convolution** (Nam et al., Interspeech 2022; +7.6 % F1 on
DESED, "especially effective on non-stationary events") fixes this: it keeps
``K`` basis kernels and mixes them *per output frequency bin* with a softmax
attention, so the effective kernel varies along frequency. Different bands get
different kernels — the band-specific structure D needs.

Where it goes
-------------
FDY only helps where there are still many frequency bins to differentiate. In
this stem frequency is strided down hard and early
(129 → 41 → 14 → 10 → 5 → 3), so the only worthwhile targets are the two
*early* convs:

  * ``filterbank``          Conv2d(3→64,  k(7,1), s(3,1))   freq 129 → 41
  * ``feat_extractor[0]``   Conv2d(64→128, k(5,5), s(3,1))  freq  41 → 14

Both are strided over frequency. The alignment trick (from the official
implementation): give the attention branch the **same** frequency
kernel/stride/padding as the main conv, so its per-bin attention output count
equals the conv's output frequency count automatically — no manual pooling.

Everything downstream is untouched: FDY preserves each conv's output shape, so
``C*Fr = 384`` at the flatten and the wide projection / Conformer body / head
are exactly ``WhaleVAD_Conformer``'s. This class therefore just *subclasses*
it and swaps the two convs in place, reusing the entire bottleneck /
aggregation / residual stack and the whole encoder verbatim.

Notes
-----
* Temperature is a fixed softmax temperature (default 1.0 = plain softmax).
  The paper anneals it 31 → 1 over training for stability; that needs the
  epoch threaded into ``forward`` and is omitted here for a clean drop-in.
  At init each basis kernel is an independent Kaiming conv, so the
  attention-weighted output already behaves like a sensible conv regardless
  of attention sharpness — early training is stable without annealing. Sweep
  ``--fdy-temp`` if needed.
* Param cost: each FDY conv is ~K× its dense conv. ``feat_extractor[0]`` is
  the 5×5/64→128 conv, so K=4 there dominates (~+0.64 M); the filterbank is
  negligible. Net model ≈ 2.6 M (vs 2.0 M). Use ``--fdy-targets filterbank``
  for a near-free variant if 2.6 M is too heavy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import config_final as cfg
from model_conformer import WhaleVAD_Conformer


# ======================================================================
# Frequency-Dynamic 2-D convolution
# ======================================================================

class FDYConv2d(nn.Module):
    """Frequency-Dynamic Conv2d — a drop-in for ``nn.Conv2d`` over (B,C,F,T).

    ``K`` basis kernels are mixed *per output frequency bin* by a softmax
    attention computed from the time-averaged input. Time-shared: one
    kernel-mix per frequency bin, applied at every frame.

    The attention branch uses the same frequency kernel/stride/padding as the
    main conv, so its output frequency length matches the conv's even when the
    conv is strided over frequency.

    Parameters mirror ``nn.Conv2d`` (groups is fixed to 1 — the FDY targets in
    this stem are not grouped). ``n_basis`` is K; ``temperature`` softens the
    attention softmax (1.0 = plain softmax).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride=1,
                 padding=0, n_basis: int = 4, temperature: float = 1.0,
                 att_hidden: int | None = None, bias: bool = True):
        super().__init__()
        kf, kt = _pair(kernel_size)
        sf, st = _pair(stride)
        pf, pt = _pair(padding)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kf, self.kt = kf, kt
        self.sf, self.st = sf, st
        self.pf, self.pt = pf, pt
        self.n_basis = int(n_basis)
        self.temperature = float(temperature)

        # K basis kernels stacked on a leading axis: (K, out, in, kf, kt).
        self.weight = nn.Parameter(torch.empty(self.n_basis, out_ch, in_ch, kf, kt))
        self.bias = nn.Parameter(torch.zeros(self.n_basis, out_ch)) if bias else None
        self.reset_parameters()

        # Frequency attention: collapse time, conv over frequency with the SAME
        # (kf, sf, pf) so the output bin count equals the main conv's, then a
        # 1×1 conv to K logits, softmax over the basis axis.
        hidden = att_hidden or max(in_ch, self.n_basis * 4)
        self.att = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kf, stride=sf, padding=pf),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, self.n_basis, kernel_size=1),
        )

    def reset_parameters(self) -> None:
        for k in range(self.n_basis):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_ch * self.kf * self.kt
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, n_basis: int = 4,
                  temperature: float = 1.0) -> "FDYConv2d":
        """Build an FDY conv matching an existing ``nn.Conv2d``'s shape."""
        if conv.groups != 1:
            raise ValueError("FDYConv2d only supports groups=1 convs")
        return cls(
            conv.in_channels, conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, n_basis=n_basis, temperature=temperature,
            bias=conv.bias is not None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, C, F, T)
        B, C, _, T = x.shape

        # --- per-frequency basis attention (time-averaged input) ---
        a = x.mean(dim=3)                                      # (B, C, F)
        a = self.att(a)                                        # (B, K, F')
        a = torch.softmax(a / self.temperature, dim=1)         # over basis
        Fp = a.shape[-1]

        # --- basis convolution batched as extra output channels ---
        w = self.weight.reshape(self.n_basis * self.out_ch, self.in_ch,
                                self.kf, self.kt)
        b = self.bias.reshape(-1) if self.bias is not None else None
        y = F.conv2d(x, w, bias=b, stride=(self.sf, self.st),
                     padding=(self.pf, self.pt))               # (B, K*out, F', T')
        Tp = y.shape[-1]
        y = y.view(B, self.n_basis, self.out_ch, y.shape[-2], Tp)

        # attention output-bin count must line up with the conv's
        if y.shape[-2] != Fp:
            raise RuntimeError(
                f"FDY freq mismatch: conv F'={y.shape[-2]} vs attention F'={Fp}"
            )

        a = a.view(B, self.n_basis, 1, Fp, 1)                  # broadcast C, T
        return (y * a).sum(dim=1)                              # (B, out, F', T')

    def extra_repr(self) -> str:
        return (f"in={self.in_ch}, out={self.out_ch}, "
                f"k=({self.kf},{self.kt}), s=({self.sf},{self.st}), "
                f"p=({self.pf},{self.pt}), K={self.n_basis}, "
                f"T={self.temperature:g}")


# ======================================================================
# Conformer with an FDY stem
# ======================================================================

_FDY_TARGETS = ("filterbank", "feat0")


class WhaleVAD_Conformer_FDY(WhaleVAD_Conformer):
    """``WhaleVAD_Conformer`` with the early stem convs replaced by FDY convs.

    Reuses the full base model and swaps in place after construction, so the
    bottleneck / aggregation / residual stack and the entire Conformer body /
    projection / head are identical and the trained checkpoints are directly
    comparable to the plain Conformer.

    Extra parameters
    ----------------
    fdy_targets : tuple   any of ("filterbank", "feat0"); default both.
    fdy_basis   : int     K basis kernels (default 4).
    fdy_temp    : float   attention softmax temperature (default 1.0).
    """

    def __init__(self, *, fdy_targets=_FDY_TARGETS, fdy_basis: int = 4,
                 fdy_temp: float = 1.0, **kw):
        super().__init__(**kw)
        targets = tuple(fdy_targets)
        unknown = set(targets) - set(_FDY_TARGETS)
        if unknown:
            raise ValueError(f"unknown fdy_targets {sorted(unknown)}; "
                             f"choose from {_FDY_TARGETS}")
        self.fdy_targets = targets
        self.fdy_basis = fdy_basis
        self.fdy_temp = fdy_temp

        if "filterbank" in targets:
            self._inner.filterbank = FDYConv2d.from_conv(
                self._inner.filterbank, n_basis=fdy_basis, temperature=fdy_temp)
        if "feat0" in targets:
            self._inner.feat_extractor[0] = FDYConv2d.from_conv(
                self._inner.feat_extractor[0], n_basis=fdy_basis, temperature=fdy_temp)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    from spectrogram_final import SpectrogramExtractor

    torch.manual_seed(0)

    # 1) FDYConv2d output shape == the plain conv it replaces, on both targets.
    for desc, conv in [
        ("filterbank", nn.Conv2d(3, 64, (7, 1), stride=(3, 1), padding=0)),
        ("feat0",      nn.Conv2d(64, 128, (5, 5), stride=(3, 1), padding=(2, 2))),
    ]:
        fdy = FDYConv2d.from_conv(conv, n_basis=4, temperature=1.0)
        x = torch.randn(2, conv.in_channels, 129 if desc == "filterbank" else 41, 50)
        yc, yf = conv(x), fdy(x)
        assert yc.shape == yf.shape, (desc, yc.shape, yf.shape)
        print(f"{desc:11s} conv {tuple(yc.shape)} == fdy {tuple(yf.shape)}  OK")

    # 2) gradients reach both the basis kernels and the attention.
    fdy = FDYConv2d(64, 128, (5, 5), stride=(3, 1), padding=(2, 2), n_basis=4)
    out = fdy(torch.randn(2, 64, 41, 40)).sum()
    out.backward()
    assert fdy.weight.grad is not None and fdy.weight.grad.abs().sum() > 0, "no grad to basis"
    att_grad = sum(p.grad.abs().sum().item() for p in fdy.att.parameters() if p.grad is not None)
    assert att_grad > 0, "no grad to attention"
    print("backward: basis-kernel grad OK, attention grad OK")

    # 3) full model forward + backward + param count (3-class head).
    cfg.USE_3CLASS = True
    extractor = SpectrogramExtractor()
    base = WhaleVAD_Conformer(num_classes=cfg.n_classes())
    fdy_model = WhaleVAD_Conformer_FDY(num_classes=cfg.n_classes())
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    with torch.no_grad():
        lb = base(spec)
        lf = fdy_model(spec)
    assert lb.shape == lf.shape, (lb.shape, lf.shape)
    # one real backward through the FDY model
    loss = F.binary_cross_entropy_with_logits(fdy_model(spec), torch.zeros_like(lf))
    loss.backward()
    nb = sum(p.numel() for p in base.parameters() if p.requires_grad)
    nf = sum(p.numel() for p in fdy_model.parameters() if p.requires_grad)
    fb_is_fdy = isinstance(fdy_model._inner.filterbank, FDYConv2d)
    f0_is_fdy = isinstance(fdy_model._inner.feat_extractor[0], FDYConv2d)
    print(f"logits base {tuple(lb.shape)} == fdy {tuple(lf.shape)}  OK")
    print(f"filterbank is FDY: {fb_is_fdy}   feat_extractor[0] is FDY: {f0_is_fdy}")
    print(f"params  base={nb:,}   fdy={nf:,}   (+{nf - nb:,})")
    print("ALL FDY SELF-TESTS PASSED")
