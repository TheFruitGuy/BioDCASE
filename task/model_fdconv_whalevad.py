"""
WhaleVAD Variant with Frequency Dynamic Convolution
====================================================

Phase 12a's architecture: take the F1=0.474 baseline ``WhaleVAD`` and
swap two of the three depthwise 3x3 convolutions in the residual
aggregation block for ``FDConv2d`` layers. Everything else — learnable
filterbank, feature extractor, bottleneck, projection, BiLSTM,
classifier — is identical to the baseline.

Why the aggregation block?
--------------------------
Diagnostic ablations (Phase 1 augmentation + Phase 2 capacity tests)
showed the bottleneck is in the CNN frontend's learned representation,
not in augmentation richness or sequence model capacity. The
aggregation block is the last CNN stage before the BiLSTM, so the
representation it produces is what shapes everything downstream.

Why two out of three depthwise convs?
-------------------------------------
- **First conv (index 1)**: receives the most direct mix of bottleneck
  features. If frequency-specific processing helps anywhere, it's here.
- **Last conv (index 7)**: aggregates a wider temporal receptive field.
  Frequency-aware aggregation at the end most directly shapes the
  representation handed off to the BiLSTM.
- **Middle conv (index 4)**: kept as standard ``nn.Conv2d``. A
  sandwich layout (FD - standard - FD) avoids the failure mode where
  consecutive FDConvs over-specialise per-band kernels in correlated
  ways, and keeps parameter overhead modest.

If phase 12a beats baseline, phase 12b can sweep:
  - replace all 3 depthwise convs
  - increase K (4 -> 6 -> 8)
  - apply FDConv to the bottleneck 3x3 conv as well

Implementation strategy
-----------------------
Subclass ``WhaleVAD`` to inherit the entire architecture, then mutate
the aggregation block in ``__init__`` to swap in ``FDConv2d`` at the
chosen positions. State_dict keys differ from the baseline only at
the swapped positions; everything else loads cleanly.
"""

from __future__ import annotations

import torch.nn as nn

from model import WhaleVAD
from model_fdconv import FDConv2d


# Indices of the depthwise 3x3 convolutions inside the aggregation
# Sequential. See model.py: aggregation = nn.Sequential(
#     Dropout2d,           # 0
#     Conv2d (depthwise),  # 1  <- candidate for FDConv
#     BatchNorm2d,         # 2
#     GELU,                # 3
#     Conv2d (depthwise),  # 4  <- candidate for FDConv
#     BatchNorm2d,         # 5
#     GELU,                # 6
#     Conv2d (depthwise),  # 7  <- candidate for FDConv
#     BatchNorm2d,         # 8
#     GELU,                # 9
# )
DEPTHWISE_INDICES = (1, 4, 7)

# Phase 12a default: replace first and last, keep middle standard.
DEFAULT_FDCONV_POSITIONS = (1, 7)


class WhaleVAD_FDConv(WhaleVAD):
    """
    ``WhaleVAD`` with selected depthwise 3x3 convs swapped for
    ``FDConv2d``.

    Parameters
    ----------
    num_classes : int, default 3
        Same as ``WhaleVAD``.
    feat_channels : int, default 3
        Same as ``WhaleVAD``.
    K : int, default 4
        Number of basis kernels per FDConv layer. Paper standard is
        4-8; 4 is the conservative starting value.
    reduction : int, default 4
        Channel-reduction ratio inside the FDConv attention head.
        Paper standard.
    positions : tuple of int, default (1, 7)
        Indices within the aggregation ``nn.Sequential`` where convs
        will be replaced with ``FDConv2d``. Must be a subset of
        ``DEPTHWISE_INDICES = (1, 4, 7)``.

    Notes
    -----
    The constructor performs a structural check on the aggregation
    block: if ``model.py``'s layout drifts (e.g. someone adds a layer
    inside the aggregation block), this class fails loudly at
    construction rather than silently swapping the wrong module.
    """

    def __init__(
        self,
        num_classes: int = 3,
        feat_channels: int = 3,
        K: int = 4,
        reduction: int = 4,
        positions: tuple[int, ...] = DEFAULT_FDCONV_POSITIONS,
    ):
        super().__init__(num_classes=num_classes, feat_channels=feat_channels)

        # Validate caller-supplied positions.
        for idx in positions:
            if idx not in DEPTHWISE_INDICES:
                raise ValueError(
                    f"position {idx} is not one of the depthwise conv "
                    f"indices {DEPTHWISE_INDICES}. FDConv only makes "
                    f"sense at the depthwise 3x3 positions."
                )

        self.K = K
        self.reduction = reduction
        self.fdconv_positions = tuple(positions)

        # Aggregation block lives at residual_stack.blocks[1]; see
        # WhaleVAD.__init__ for the ResidualBlock(bottleneck, aggregation)
        # construction.
        agg: nn.Sequential = self.residual_stack.blocks[1]

        # Sanity-check the layout. If model.py's aggregation Sequential
        # ever changes (e.g. extra normalisation layers added), this
        # block raises rather than silently swapping the wrong module.
        for idx in DEPTHWISE_INDICES:
            mod = agg[idx]
            if not isinstance(mod, nn.Conv2d):
                raise RuntimeError(
                    f"Expected nn.Conv2d at aggregation[{idx}], got "
                    f"{type(mod).__name__}. model.py's aggregation "
                    f"layout has changed; update WhaleVAD_FDConv."
                )
            if mod.groups != mod.in_channels:
                raise RuntimeError(
                    f"Conv at aggregation[{idx}] is not depthwise "
                    f"(groups={mod.groups}, in_channels={mod.in_channels}). "
                    f"FDConv assumes depthwise aggregation."
                )

        # Swap chosen positions with shape-matched FDConv2d layers.
        # We copy every relevant attribute from the source conv so the
        # replacement is genuinely drop-in.
        for idx in self.fdconv_positions:
            src: nn.Conv2d = agg[idx]
            agg[idx] = FDConv2d(
                in_channels=src.in_channels,
                out_channels=src.out_channels,
                kernel_size=src.kernel_size,
                padding=src.padding,
                stride=src.stride,
                groups=src.groups,
                bias=(src.bias is not None),
                K=K,
                reduction=reduction,
            )


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Quick end-to-end sanity check. Run:
    #   python model_fdconv_whalevad.py
    import torch

    import config as cfg
    from spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor()
    model = WhaleVAD_FDConv(
        num_classes=cfg.n_classes(), K=4, reduction=4,
    )

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    logits = model(spec)

    print(f"Audio:   {audio.shape}")
    print(f"Spec:    {spec.shape}")
    print(f"Logits:  {logits.shape}")

    # Parameter accounting vs. the baseline WhaleVAD. Force the
    # baseline through a forward pass so its lazy projection layer is
    # materialised; otherwise the comparison undercounts the baseline.
    baseline = WhaleVAD(num_classes=cfg.n_classes())
    _ = baseline(spec)
    n_fd = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"Baseline params:  {n_base:,}")
    print(f"FDConv params:    {n_fd:,}")
    print(f"Delta:            +{n_fd - n_base:,} "
          f"({100 * (n_fd - n_base) / n_base:.2f}%)")
    print(f"FDConv positions: {model.fdconv_positions}")
    print(f"K, reduction:     K={model.K}, reduction={model.reduction}")
