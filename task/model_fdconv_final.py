"""
Final Pipeline - WhaleVAD with Frequency Dynamic Convolution
=============================================================

Phase 12a's architecture: take the canonical ``WhaleVAD`` from
``model_final`` (7-class default, paper BiLSTM, used by the consolidated
``train_final.py`` recipe) and swap two of the three depthwise 3x3
convolutions in the residual aggregation block for ``FDConv2d`` layers.
Everything else — learnable filterbank, feature extractor, bottleneck,
projection, BiLSTM, classifier — is identical to the canonical model.

Why subclass model_final.WhaleVAD rather than model.WhaleVAD?
-------------------------------------------------------------
``model.py``/``phase1_baseline.py`` is the older 3-class arch-experiment
pattern (used by phase 1/2 ablations). The current canonical recipe
lives in the ``*_final.py`` modules: 8-site training, paper BiLSTM,
7 fine call types collapsed to 3 coarse classes at evaluation,
weighted BCE with segment-count-normalised pos_weight, per-epoch
negative resampling. Phase 12a tests FDConv *on top of that recipe*
so the comparison is against ``final`` rather than ``baseline``.

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

If phase 12a beats final, phase 12b can sweep:
  - replace all 3 depthwise convs
  - increase K (4 -> 6 -> 8)
  - apply FDConv to the bottleneck 3x3 conv as well
  - swap FDY conv for the dilated (DFD) or multi-dilated (MDFD) variants

Implementation strategy
-----------------------
Subclass ``model_final.WhaleVAD`` to inherit the full canonical
architecture, then mutate the aggregation block in ``__init__`` to swap
in ``FDConv2d`` at the chosen positions. State_dict keys differ from
the ``final`` baseline only at the swapped positions; everything else
loads cleanly via ``strict=False`` if you ever want to warm-start from
a converged ``final`` checkpoint.
"""

from __future__ import annotations

import torch.nn as nn

from model_final import WhaleVAD
from model_fdconv import FDConv2d


# Indices of the depthwise 3x3 convolutions inside the aggregation
# Sequential. See model_final.py: aggregation = nn.Sequential(
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
    ``model_final.WhaleVAD`` with selected depthwise 3x3 convs swapped
    for ``FDConv2d``.

    Parameters
    ----------
    num_classes : int, default 7
        Same default as ``model_final.WhaleVAD``: 7 fine call types,
        collapsed to 3 at evaluation by the post-processing pipeline.
    feat_channels : int, default 3
        Same as the canonical model.
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
    block: if ``model_final.py``'s layout drifts (e.g. someone adds a
    layer inside the aggregation block), this class fails loudly at
    construction rather than silently swapping the wrong module.
    """

    def __init__(
        self,
        num_classes: int = 7,
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
        # model_final.WhaleVAD.__init__ for the
        # ResidualBlock(bottleneck, aggregation) construction.
        agg: nn.Sequential = self.residual_stack.blocks[1]

        # Sanity-check the layout. If model_final.py's aggregation
        # Sequential ever changes (e.g. extra normalisation layers
        # added), this block raises rather than silently swapping the
        # wrong module.
        for idx in DEPTHWISE_INDICES:
            mod = agg[idx]
            if not isinstance(mod, nn.Conv2d):
                raise RuntimeError(
                    f"Expected nn.Conv2d at aggregation[{idx}], got "
                    f"{type(mod).__name__}. model_final.py's "
                    f"aggregation layout has changed; update "
                    f"WhaleVAD_FDConv."
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
    #   python model_fdconv_final.py
    import torch

    import config_final as cfg
    from spectrogram_final import SpectrogramExtractor

    extractor = SpectrogramExtractor()
    # Default num_classes=7, matching the canonical recipe.
    model = WhaleVAD_FDConv(K=4, reduction=4)

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    logits = model(spec)

    print(f"Audio:   {audio.shape}")
    print(f"Spec:    {spec.shape}")
    print(f"Logits:  {logits.shape}  (B, T, num_classes=7)")

    # Parameter accounting vs. the canonical WhaleVAD. Force both
    # through a forward pass so their lazy projection layers are
    # materialised; otherwise the baseline param count would be
    # under-reported.
    baseline = WhaleVAD()  # default 7-class
    _ = baseline(spec)
    n_fd = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"Baseline (model_final.WhaleVAD): {n_base:,}")
    print(f"FDConv variant:                  {n_fd:,}")
    print(f"Delta:                           +{n_fd - n_base:,} "
          f"({100 * (n_fd - n_base) / n_base:.2f}%)")
    print(f"FDConv positions:                {model.fdconv_positions}")
    print(f"K, reduction:                    K={model.K}, "
          f"reduction={model.reduction}")
