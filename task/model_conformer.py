"""
Conformer-SED Variant of WhaleVAD (final pipeline)
==================================================

A *new* standalone architecture for BioDCASE Task 2 whale-call SED, built
on the **final** pipeline (``*_final`` modules, ``config_final``). The
WhaleVAD CNN stem (learnable filterbank + feature extractor + residual/
depthwise stack, from ``model_final``) feeds a stack of Conformer blocks
instead of the 2-layer BiLSTM. No recurrence anywhere.

Like the rest of the final pipeline, the classifier head defaults to the
7 fine call types (``cfg.USE_3CLASS=False``); probabilities are collapsed
to the 3 coarse classes at evaluation time by ``postprocess_final``.

How this differs from Phase 2b (and why it should clear parity)
---------------------------------------------------------------
Phase 2b tested attention on this task and landed in the seed-noise band.
Two structural reasons, both fixed here:

  1. **Width.** Phase 2b reused WhaleVAD's existing 64-dim projection and
     ran attention at d_model=64 / 4 heads = 16 dims per head. Here we add
     our *own* projection from the flattened CNN features straight to
     ``d_model`` (default 128); the inner model's 64-dim projection is left
     unused.
  2. **No conv module.** Phase 2b was a *plain* Transformer encoder. A
     Conformer interleaves a depthwise-separable convolution module with
     self-attention in every block — local spectro-temporal modelling plus
     global attention. That conv module is the main reason this is worth
     trying when the plain Transformer was not.

Block structure (canonical Conformer, Gulati et al. 2020):

    x → x + ½·FFN(x)
      → x + SelfAttention(LN(x))         (RoPE relative positions)
      → x + ConvModule(x)                (pointwise→GLU→depthwise→BN→SiLU→pointwise)
      → x + ½·FFN(x)
      → LayerNorm(x)

Interface
---------
``forward(spec, key_padding_mask=None) -> (B, T, num_classes)``

The ``spec`` contract matches ``model_final.WhaleVAD.forward`` — a
``(B, 3, F, T)`` phase-aware spectrogram from ``spectrogram_final``. So
this is drop-in for the final training / validation machinery.
``key_padding_mask`` (``(B, T)`` bool, ``True`` = padded) is optional:
``train_conformer.py`` passes it during training (variable-length 30 s
segments are zero-padded to the batch max), while ``train_final.validate``
calls the model without it — keeping validation identical to the baseline.
The stem preserves the time axis exactly, so the model's internal sequence
length equals the STFT frame count; a length guard reconciles any
off-by-one between an externally supplied mask and that count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_final as cfg
from model_final import WhaleVAD


# ======================================================================
# Rotary positional embedding (relative positions for Q/K)
# ======================================================================

def _rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    """Return (cos, sin) of shape ``(seq_len, head_dim // 2)``."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # (T, hd/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(B, H, T, head_dim)`` (interleaved)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None]                                  # (1,1,T,hd/2)
    sin = sin[None, None]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)     # (B,H,T,hd)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                   # (3,B,H,T,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = _rope_cache(T, self.head_dim, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA bool mask: True = participate. key_padding_mask True = pad,
            # so keep = ~mask. Broadcast over heads and query positions.
            attn_mask = (~key_padding_mask)[:, None, None, :]   # (B,1,1,T)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                  # (B,H,T,hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


# ======================================================================
# Conformer sub-modules
# ======================================================================

class FeedForward(nn.Module):
    """Macaron feed-forward module (applied with a ½ residual weight)."""

    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """
    Conformer convolution module:
    LayerNorm → pointwise(→2d) → GLU → depthwise(kernel) → BN → SiLU →
    pointwise(→d) → dropout. Operates on (B, T, d_model).
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "conv kernel must be odd for 'same' padding"
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)                   # (B, D, T)
        x = self.pw1(x)
        x = F.glu(x, dim=1)                                # (B, D, T)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)                           # (B, T, D)


class ConformerBlock(nn.Module):
    """A single Conformer block (macaron-FFN / MHSA / conv / macaron-FFN)."""

    def __init__(self, d_model, nhead, ffn_mult, conv_kernel, dropout):
        super().__init__()
        self.ffn1 = FeedForward(d_model, ffn_mult, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, nhead, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForward(d_model, ffn_mult, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x), key_padding_mask))
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


# ======================================================================
# Full model
# ======================================================================

class WhaleVAD_Conformer(nn.Module):
    """
    WhaleVAD CNN stem (from model_final) + Conformer encoder, per-frame head.

    Parameters
    ----------
    num_classes, feat_channels
        Same meaning as ``model_final.WhaleVAD``. Defaults to 7 (the final
        pipeline trains on the fine call types and collapses to 3 at eval).
    d_model
        Conformer width. Default 128 (vs. Phase 2b's 64). The flattened CNN
        features (C*F, typically ~384) are projected to this.
    nhead, num_layers, ffn_mult, conv_kernel, dropout
        Standard Conformer hyperparameters.
    """

    def __init__(
        self,
        num_classes: int = 7,
        feat_channels: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_mult: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Reuse the final pipeline's CNN stem by composition. The inner
        # model's BiLSTM, classifier and 64-dim projection are dead here.
        self._inner = WhaleVAD(num_classes=num_classes, feat_channels=feat_channels)
        for p in self._inner.lstm.parameters():
            p.requires_grad = False
        for p in self._inner.classifier.parameters():
            p.requires_grad = False

        self.d_model = d_model

        # Our own wide projection (lazy: C*F depends on STFT settings).
        self._proj: nn.Linear | None = None
        self.input_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, ffn_mult, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    # ------------------------------------------------------------------
    def _init_proj(self, in_dim: int, device: torch.device):
        if self._proj is None or self._proj.in_features != in_dim:
            self._proj = nn.Linear(in_dim, self.d_model).to(device)

    def parameters(self, recurse: bool = True):
        """Yield trainable params, excluding the inner model's dead BiLSTM/classifier."""
        for name, p in self.named_parameters(recurse=recurse):
            if "_inner.lstm." in name or "_inner.classifier." in name:
                continue
            yield p

    # ------------------------------------------------------------------
    def forward(self, spec: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # CNN stem (reuse inner modules directly; never calls _inner.forward,
        # so the inner 64-dim projection stays uninitialised/unused).
        x = self._inner.filterbank(spec)
        x = self._inner.feat_extractor(x)
        x = self._inner.residual_stack(x)

        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)

        self._init_proj(C * Fr, x.device)
        x = self.input_drop(self._proj(x))                 # (B, T, d_model)

        # Reconcile an externally supplied mask with the model's frame count.
        if key_padding_mask is not None and key_padding_mask.size(1) != T:
            kpm = key_padding_mask
            if kpm.size(1) > T:
                kpm = kpm[:, :T]
            else:
                pad = torch.ones(B, T - kpm.size(1), dtype=torch.bool, device=kpm.device)
                kpm = torch.cat([kpm, pad], dim=1)
            key_padding_mask = kpm

        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        return self.classifier(x)                          # (B, T, num_classes)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    from spectrogram_final import SpectrogramExtractor

    extractor = SpectrogramExtractor()
    model = WhaleVAD_Conformer(num_classes=cfg.n_classes())

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    logits = model(spec)
    print(f"Spec:    {tuple(spec.shape)}")
    print(f"Logits:  {tuple(logits.shape)}   (num_classes={cfg.n_classes()})")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:  {n:,}")
