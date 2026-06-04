"""
FDY-CRNN: WhaleVAD CNN-BiLSTM with a Frequency-Dynamic Conv stem
================================================================

The backbone test. The recheck confirmed WhaleVAD is a CNN-BiLSTM (CRNN) and it
sits at ~0.443 macro_paper -- above the FDY-Conformer's 0.429 on the same split
-- consistent with the SED literature that Conformers over-smooth in time
relative to CRNNs. This model puts the *one frontend lever that worked* (the FDY
stem from 13d) onto the *proven-stronger backbone* (the BiLSTM), isolating the
backbone: identical FDY stem, BiLSTM instead of Conformer downstream.

It subclasses the base ``WhaleVAD`` and swaps the same two early convs as
``WhaleVAD_Conformer_FDY`` -- the learnable filterbank and ``feat_extractor[0]``,
the only stem convs with enough frequency resolution to matter -- for
``FDYConv2d``. Everything else (bottleneck, aggregation, residual stack, lazy
projection, BiLSTM, classifier) is the untouched WhaleVAD, so a trained
checkpoint is directly comparable to both the plain WhaleVAD and the FDY-Conformer.

The forward accepts (and ignores) a ``key_padding_mask`` kwarg so the model
drops into ``conformer_core.run_training`` unchanged: ``_train_epoch`` passes the
mask (for the Conformer's attention), ``validate`` calls ``model(spec)`` without
it. The BiLSTM consumes the full padded sequence and padded frames are excluded
by the loss mask downstream, exactly as in the native WhaleVAD training. This
keeps the recipe (RAdam / const-5e-5 / weighted-BCE / macro_paper selection)
identical to the FDY-Conformer for a clean A/B on the backbone.
"""

from __future__ import annotations

import torch

from model_final import WhaleVAD
from model_conformer_fdy import FDYConv2d

_FDY_TARGETS = ("filterbank", "feat0")


class WhaleVAD_FDY(WhaleVAD):
    """``WhaleVAD`` (CNN-BiLSTM) with the early stem convs replaced by FDY convs."""

    def __init__(self, *, num_classes: int = 7, feat_channels: int = 3,
                 fdy_targets=_FDY_TARGETS, fdy_basis: int = 4,
                 fdy_temp: float = 1.0):
        super().__init__(num_classes=num_classes, feat_channels=feat_channels)
        targets = tuple(fdy_targets)
        unknown = set(targets) - set(_FDY_TARGETS)
        if unknown:
            raise ValueError(f"unknown fdy_targets {sorted(unknown)}; "
                             f"choose from {_FDY_TARGETS}")
        self.fdy_targets = targets
        self.fdy_basis = fdy_basis
        self.fdy_temp = fdy_temp

        # Swap in place after construction, exactly as WhaleVAD_Conformer_FDY does
        # for the inner stem, so the output shapes (and therefore the lazy
        # projection dim, the BiLSTM, and the head) are identical to base WhaleVAD.
        if "filterbank" in targets:
            self.filterbank = FDYConv2d.from_conv(
                self.filterbank, n_basis=fdy_basis, temperature=fdy_temp)
        if "feat0" in targets:
            self.feat_extractor[0] = FDYConv2d.from_conv(
                self.feat_extractor[0], n_basis=fdy_basis, temperature=fdy_temp)

    def forward(self, spec, key_padding_mask=None):  # noqa: D401
        # key_padding_mask is accepted for drop-in compatibility with
        # conformer_core's train loop and ignored: the BiLSTM runs over the full
        # sequence and padded frames are handled by the loss mask downstream,
        # matching the native WhaleVAD training.
        return super().forward(spec)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    import config_final as cfg

    n = cfg.n_classes()
    base = WhaleVAD(num_classes=n, feat_channels=3)
    fdy = WhaleVAD_FDY(num_classes=n, feat_channels=3)

    base.eval(); fdy.eval()                   # disable dropout so the check is deterministic
    x = torch.randn(2, 3, 129, 60)
    with torch.no_grad():
        y_base = base(x)                      # base: forward(spec)
        y1 = fdy(x)                           # FDY: no mask
        y2 = fdy(x, key_padding_mask=torch.zeros(2, 60, dtype=torch.bool))
    print("base out :", tuple(y_base.shape))
    print("fdy  out :", tuple(y1.shape), " (mask-arg ignored:",
          torch.allclose(y1, y2), ")")
    assert y1.shape == y_base.shape == (2, 60, n)
    assert torch.allclose(y1, y2)

    fb = isinstance(fdy.filterbank, FDYConv2d)
    f0 = isinstance(fdy.feat_extractor[0], FDYConv2d)
    print(f"filterbank is FDY: {fb}   feat_extractor[0] is FDY: {f0}")
    assert fb and f0

    pb = sum(p.numel() for p in base.parameters())
    pf = sum(p.numel() for p in fdy.parameters())
    print(f"params  base={pb:,}  fdy={pf:,}  (+{pf - pb:,})")

    # grad flows to FDY basis kernels + attention branch
    fdy.zero_grad()
    fdy(x).sum().backward()
    fb_mod = fdy.filterbank
    has_basis_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in fb_mod.parameters())
    print("grad reaches FDY filterbank:", has_basis_grad)
    assert has_basis_grad
    print("OK")
