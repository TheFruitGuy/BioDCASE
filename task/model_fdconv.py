"""
Frequency Dynamic Convolution (FDConv) Layer
============================================

Drop-in replacement for ``nn.Conv2d`` that breaks the frequency-
equivariance assumption of standard convolutions. Standard 2D conv
applies the same kernel at every frequency bin; FDConv learns ``K``
basis kernels and combines them with per-frequency attention weights,
so the effective kernel varies along the frequency axis.

This is exactly what whale-call detection needs: Z-calls live around
27 Hz, D-calls above 20 Hz, and these frequency bands are *defined* by
their location. A frequency-equivariant conv treats a pattern at 27 Hz
the same as the same pattern at 100 Hz, which is provably wrong here.

Reference
---------
Nam, Kim, Park & Park (2022). "Frequency Dynamic Convolution:
Frequency-Adaptive Pattern Recognition for Sound Event Detection."
INTERSPEECH 2022. arXiv:2203.15296.

This layer implements the original FDY conv ("FDY" for Frequency
DYnamic), which reports +7.56% on DESED over a standard-conv CRNN
baseline. The name "FDConv" is overloaded across follow-up work in
this family — Dilated FDY (DFD, +9.27%), Multi-Dilated FDY (MDFD,
+10.98%), Full-Frequency Dynamic Conv (FFDConv, arXiv:2401.04976),
and a 2025 survey/extension paper (arXiv:2506.12785) all use related
terminology. This file is the 2022 original; the dilated and full-
frequency variants are candidates for a follow-up phase 12b if 12a
moves the needle.

Usage
-----
Replace any ``nn.Conv2d`` in a CNN with ``FDConv2d``, keeping the same
in/out channels, kernel size, padding, stride, and groups. The
frequency dimension is assumed to be axis 2 of the input ``(B, C, F, T)``
— same convention as the rest of WhaleVAD.

::

    # Standard
    self.conv = nn.Conv2d(128, 128, 3, padding=1, groups=128)
    # FDConv (same shape semantics, K basis kernels)
    self.conv = FDConv2d(128, 128, 3, padding=1, groups=128, K=4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FDConv2d(nn.Module):
    """
    Per-band-attention over K basis kernels.

    For each frequency band ``f``, the effective conv kernel is a
    weighted combination of K basis kernels, where the weights come
    from an attention head conditioned on the time-pooled input
    features at band ``f``. This breaks translation equivariance along
    the frequency axis while preserving it along time — appropriate
    for whale calls and other narrowband sound events.

    Parameters
    ----------
    in_channels, out_channels : int
        Same semantics as ``nn.Conv2d``.
    kernel_size : int or tuple
        Convolution kernel size. ``int`` is broadcast to ``(int, int)``.
    padding : int or tuple, default 0
    stride : int or tuple, default 1
    groups : int, default 1
        Standard ``nn.Conv2d`` groups. For depthwise: ``groups=in_channels``.
    bias : bool, default True
        Whether each basis kernel has a per-band-weighted bias term.
    K : int, default 4
        Number of basis kernels. Paper uses 4-8; ``K=4`` is the
        standard starting value.
    reduction : int, default 4
        Channel-reduction ratio in the attention bottleneck. Hidden
        dim = ``max(in_channels // reduction, 4)``.

    Notes
    -----
    The attention is **input-dependent** (not just position-dependent),
    so the per-band kernel can adapt to the actual acoustic context.
    The paper shows this matters: pure positional weights without
    input conditioning underperformed by ~3 percentage points on
    DESED.

    Memory cost: ``K`` parallel conv outputs ``(B, K, out_C, F, T)``
    are stacked before the weighted sum. For depthwise convs at 128
    channels and ``K=4``, the intermediate is 4x the original feature
    map — negligible at ``K=4``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        padding: int | tuple = 0,
        stride: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        K: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by "
                f"groups ({groups})."
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"groups ({groups})."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.K = K
        self.use_bias = bias

        # ---- K basis kernels ----------------------------------------------
        # Shape mirrors ``nn.Conv2d.weight`` with an extra leading K dim.
        # Each basis kernel is independently Kaiming-initialised — matches
        # what a vanilla Conv2d would do at construction. Per-basis init
        # (rather than one shared init replicated K times) breaks symmetry,
        # so attention has something to specialise toward from epoch 1.
        weight = torch.empty(
            K, out_channels, in_channels // groups, *kernel_size
        )
        for k in range(K):
            nn.init.kaiming_normal_(
                weight[k], a=0, mode="fan_in", nonlinearity="relu",
            )
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels))
        else:
            self.register_parameter("bias", None)

        # ---- Per-band attention head --------------------------------------
        # Reduces channels, then maps to K attention logits per frequency
        # band. Time is mean-pooled before this head, so the attention is
        # conditioned on each band's time-averaged feature vector. This is
        # the standard FDY formulation.
        hidden = max(in_channels // reduction, 4)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, K, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FDConv to ``(B, C, F, T)``.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map, shape ``(B, in_channels, F, T)``.

        Returns
        -------
        torch.Tensor
            Output feature map, shape ``(B, out_channels, F', T')``,
            where ``F'`` and ``T'`` depend on kernel/stride/padding
            the same way ``nn.Conv2d`` would produce them.
        """
        B, C, Fr, T = x.shape

        # ---- Per-band attention weights -----------------------------------
        # Mean-pool over time to get one feature vector per (B, F) cell.
        # Mean rather than max so attention sees typical energy in each
        # band, not just the loudest moment.
        pooled = x.mean(dim=3, keepdim=True)            # (B, C, F, 1)
        attn_logits = self.attn(pooled)                  # (B, K, F, 1)
        attn = F.softmax(attn_logits, dim=1)             # softmax over K
        attn = attn.squeeze(-1)                          # (B, K, F)

        # ---- Apply each basis kernel, combine per band --------------------
        # K parallel convs. At K=4 this is 4 GPU launches; for depthwise
        # convs the cost is negligible. The alternative (single grouped
        # conv with reshaped weights) is marginally faster but much
        # harder to follow when debugging.
        outs = []
        for k in range(self.K):
            out_k = F.conv2d(
                x,
                self.weight[k],
                bias=None,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )
            outs.append(out_k)
        out_stack = torch.stack(outs, dim=1)             # (B, K, out_C, F', T')

        # If padding/stride changed F, resample attention along F to match.
        # This rarely fires for the WhaleVAD aggregation block (stride=1,
        # padding=1, F preserved), but the layer should be generic enough
        # to drop in anywhere a Conv2d would go.
        F_out = out_stack.shape[3]
        if F_out != Fr:
            attn = F.interpolate(
                attn.unsqueeze(-1), size=(F_out, 1),
                mode="bilinear", align_corners=False,
            ).squeeze(-1)

        # Broadcast attn to (B, K, 1, F', 1) and weighted sum over K.
        attn_b = attn.unsqueeze(2).unsqueeze(-1)
        out = (out_stack * attn_b).sum(dim=1)            # (B, out_C, F', T')

        # ---- Per-band-weighted bias ---------------------------------------
        if self.bias is not None:
            # attn: (B, K, F_out); bias: (K, out_C)
            # Result: (B, out_C, F_out), broadcast along T.
            weighted_bias = torch.einsum("bkf,ko->bof", attn, self.bias)
            out = out + weighted_bias.unsqueeze(-1)

        return out

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, K={self.K}"
        )


def replicate_from_conv(
    fdconv: FDConv2d,
    source_conv: nn.Conv2d,
    noise_std: float = 0.01,
) -> None:
    """
    Warm-start an ``FDConv2d`` layer from a trained ``nn.Conv2d``.

    Replicates the source kernel into all K basis kernels with a small
    additive perturbation, and copies the source bias if present. The
    perturbation prevents the K kernels from being exact duplicates at
    init, so attention has gradient signal to specialise from.

    Useful when warm-starting from a baseline checkpoint: the model
    starts with FDConv layers that behave identically to the original
    convs in expectation (uniform attention over identical kernels),
    then learns to differentiate them. Not used by phase 12a (which
    trains from scratch), but kept here for later warm-start
    experiments.

    Parameters
    ----------
    fdconv : FDConv2d
        Target layer. Basis shape must match the source conv.
    source_conv : nn.Conv2d
        Source layer to copy from.
    noise_std : float, default 0.01
        Std of Gaussian noise added to each replica. Set to 0 for
        exact replication (attention becomes a dead head until
        training breaks the symmetry).
    """
    if fdconv.weight.shape[1:] != source_conv.weight.shape:
        raise ValueError(
            f"Shape mismatch: FDConv basis is "
            f"{tuple(fdconv.weight.shape[1:])}, source conv is "
            f"{tuple(source_conv.weight.shape)}."
        )
    with torch.no_grad():
        for k in range(fdconv.K):
            fdconv.weight[k].copy_(source_conv.weight)
            if noise_std > 0:
                fdconv.weight[k].add_(
                    torch.randn_like(source_conv.weight) * noise_std
                )
        if fdconv.bias is not None and source_conv.bias is not None:
            for k in range(fdconv.K):
                fdconv.bias[k].copy_(source_conv.bias)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Quick shape and parameter-count sanity check. Run:
    #   python model_fdconv.py
    torch.manual_seed(0)

    # Depthwise FDConv mirroring the WhaleVAD aggregation conv.
    fd = FDConv2d(
        in_channels=128, out_channels=128, kernel_size=3,
        padding=1, groups=128, K=4, reduction=4,
    )
    x = torch.randn(2, 128, 12, 1500)  # (B, C, F, T) — typical agg input
    y = fd(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    n = sum(p.numel() for p in fd.parameters())
    print(f"Depthwise FDConv (C=128, K=4):  output {tuple(y.shape)}, "
          f"params {n:,}")

    # Compare to the standard depthwise conv it would replace.
    std = nn.Conv2d(128, 128, 3, padding=1, groups=128)
    n_std = sum(p.numel() for p in std.parameters())
    print(f"Standard depthwise Conv2d:      params {n_std:,} "
          f"(FDConv adds {n - n_std:,})")

    # Warm-start replication sanity check.
    fd2 = FDConv2d(128, 128, 3, padding=1, groups=128, K=4)
    replicate_from_conv(fd2, std, noise_std=0.0)
    # With noise_std=0 and uniform attention init, the first forward
    # pass should approximately equal the standard conv's output on
    # the same input. Attention is not literally uniform (the head has
    # random init), but the K weighted kernels are identical.
    y2 = fd2(x)
    y_std = std(x)
    # The mean is what matters: weighted_sum over identical kernels =
    # the same kernel, regardless of weights. So y2 ≈ y_std up to the
    # per-band bias term (which we copied, plus zero-mean attention).
    rel = (y2 - y_std).abs().mean() / y_std.abs().mean().clamp_min(1e-6)
    print(f"Warm-start equivalence check:   relative diff = {rel:.4f} "
          f"(should be small; nonzero from bias attention reweighting)")
