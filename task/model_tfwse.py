"""
Phase 13l -- Time-frame frequency-wise SE on the FDY-Conformer stem
===================================================================

13k settled the backbone question: under a matched recipe the CNN-BiLSTM and
the Conformer tie, so the gap is not the sequence model. Every variant stalls
at D ~0.20, and the D-diagnosis is a frontend-sensitivity miss (71.5 %
background), not inter-class confusion. tfwSE is a frequency lever aimed at
exactly that.

tfwSE (time-frame frequency-wise Squeeze-and-Excitation; Nam et al., DCASE2023,
arXiv:2306.11277): at each time frame, squeeze the channel axis to a
per-frequency descriptor, excite it through a frequency bottleneck, and gate
the input frequencies by the resulting per-(frequency, frame) weights. Because
the gate is recomputed every frame -- unlike fwSE, which averages over time and
emits one static per-frequency gain -- the emphasised band can move over time,
matching a D-call's FM downsweep. It is also per-frame by construction, so
(unlike FDY's time-averaged attention) padded frames cannot contaminate the
gate of a real frame.

Where it goes
-------------
The stem strides frequency down hard (129 -> 41 -> 14 -> 10 -> 5 -> 3), so the
only point with real frequency resolution is right after the (FDY) filterbank,
at F=41. We insert one tfwSE there. The Conformer's forward calls
``self._inner.filterbank(spec)``, so wrapping that attribute in a
``Sequential(filterbank, tfwSE)`` inserts tfwSE with no forward changes; tfwSE
is shape-preserving so the rest of the stem (and the C*F=384 flatten) is
untouched and checkpoints stay comparable to 13d.

Caveats (from the paper)
------------------------
* The paper's strongest config is SE *then* tfwSE; tfwSE alone is the weaker
  half. We test tfwSE alone (single axis on top of FDY); channel-SE is the
  documented follow-up if this is neutral.
* tfwSE is framed as a lighter *alternative* to FDY, so FDY+tfwSE may be
  redundant (two frequency mechanisms). This run measures whether the per-frame
  gate adds anything the FDY stem does not already capture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_conformer_fdy import WhaleVAD_Conformer_FDY


class TimeFrameFreqSE(nn.Module):
    """Time-frame frequency-wise SE over (B, C, F, T) -> (B, C, F, T).

    Squeeze channels to a per-(frequency, frame) descriptor, excite the
    frequency axis through an ``F -> F/r -> F`` bottleneck applied independently
    per frame (a kernel-1 Conv1d over time with frequency as the channel axis),
    sigmoid-gate, and scale. The frequency count is inferred lazily on the first
    forward (mirrors the stem's lazy ``feat_proj``), so it adapts to the
    post-filterbank F without hard-coding it.
    """

    def __init__(self, reduction: int = 4, min_hidden: int = 4):
        super().__init__()
        self.reduction = int(reduction)
        self.min_hidden = int(min_hidden)
        self.excite: nn.Module | None = None  # built on first forward

    def _build(self, n_freq: int, device: torch.device) -> None:
        hidden = max(n_freq // self.reduction, self.min_hidden)
        self.excite = nn.Sequential(
            nn.Conv1d(n_freq, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, n_freq, kernel_size=1),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, F, T)
        if self.excite is None:
            self._build(x.shape[2], x.device)
        z = x.mean(dim=1)                      # squeeze channels -> (B, F, T)
        s = torch.sigmoid(self.excite(z))      # per-frame freq gate -> (B, F, T)
        return x * s.unsqueeze(1)              # broadcast over channels


class WhaleVAD_Conformer_FDY_tfwSE(WhaleVAD_Conformer_FDY):
    """FDY-Conformer with a tfwSE block after the (FDY) filterbank, at F=41."""

    def __init__(self, *, tfwse_reduction: int = 4, **kw):
        super().__init__(**kw)             # builds inner WhaleVAD + Conformer, swaps FDY
        self.tfwse_reduction = int(tfwse_reduction)
        # Wrap the (already FDY) filterbank so the inherited forward runs
        # filterbank -> tfwSE transparently. Named under "_inner.filterbank.*",
        # so it is NOT caught by the base model's lstm/classifier freeze filter
        # and stays trainable.
        self._inner.filterbank = nn.Sequential(
            self._inner.filterbank,
            TimeFrameFreqSE(reduction=tfwse_reduction),
        )


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    import config_final as cfg
    from spectrogram_final import SpectrogramExtractor

    torch.manual_seed(0)

    # 1) tfwSE module: shape-preserving, grad flows, gate VARIES across frames.
    se = TimeFrameFreqSE(reduction=4)
    x = torch.randn(2, 64, 41, 50)
    y = se(x)
    assert y.shape == x.shape, (y.shape, x.shape)
    # gate should differ frame-to-frame for a time-varying input
    xv = torch.zeros(1, 64, 41, 3)
    xv[:, :, 3, 0] = 5.0      # energy in low freq at frame 0
    xv[:, :, 30, 1] = 5.0     # energy in high freq at frame 1
    with torch.no_grad():
        zz = xv.mean(dim=1)
        ss = torch.sigmoid(se.excite(zz))         # (1, 41, 3)
    per_frame_std = ss.std(dim=2).mean().item()
    assert per_frame_std > 1e-4, f"gate not per-frame (std={per_frame_std})"
    se.zero_grad(); se(x).sum().backward()
    g = sum(p.grad.abs().sum().item() for p in se.parameters() if p.grad is not None)
    assert g > 0
    print(f"tfwSE module: shape {tuple(y.shape)} OK, per-frame gate std={per_frame_std:.3f}, grad OK")

    # 2) full model: same output shape as the FDY-Conformer, filterbank wrapped,
    #    param delta tiny, grad reaches tfwSE.
    cfg.USE_3CLASS = True
    n = cfg.n_classes()
    extractor = SpectrogramExtractor()
    base = WhaleVAD_Conformer_FDY(num_classes=n)
    mod = WhaleVAD_Conformer_FDY_tfwSE(num_classes=n)
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    with torch.no_grad():
        lb = base(spec)
        lf = mod(spec)
    assert lb.shape == lf.shape == (2, lf.shape[1], n), (lb.shape, lf.shape)

    wrapped = isinstance(mod._inner.filterbank, nn.Sequential) and \
        isinstance(mod._inner.filterbank[1], TimeFrameFreqSE)
    print(f"filterbank wrapped with tfwSE: {wrapped}")
    assert wrapped

    nb = sum(p.numel() for p in base.parameters() if p.requires_grad)
    nf = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    print(f"params  base={nb:,}  +tfwSE={nf:,}  (+{nf - nb:,})")

    mod.zero_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(
        mod(spec), torch.zeros_like(lf)).backward()
    se_grad = sum(p.grad.abs().sum().item()
                  for p in mod._inner.filterbank[1].parameters()
                  if p.grad is not None)
    print("grad reaches tfwSE:", se_grad > 0)
    assert se_grad > 0
    print("ALL TFWSE SELF-TESTS PASSED")
