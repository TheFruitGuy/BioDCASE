"""
Verifier Model (Stage 2)
========================

Multi-head binary classifier that re-scores candidate events from the
stage-1 CNN-BiLSTM. Architecture:

    audio crop                          raw waveform
        │
        ▼  SpectrogramExtractor         3-channel phase-aware spec
        ▼  Conv2d 3→32 (7×7, s=2)       time/freq downsample
        ▼  Conv2d 32→64 (3×3) + pool
        ▼  Conv2d 64→128 (3×3) + pool
        ▼  Conv2d 128→128 (3×3) + pool
        ▼  AdaptiveAvgPool2d → flatten  (B, 128)
        │
        ▼  concat with [stage1_score]   (B, 129)
        │
        ▼  3 parallel heads (one per class)
        ▼  pick head[class_idx]         per-sample routing
        │
    sigmoid logit → P(real call of this class)

v3 (2026-05-08): forward() optionally returns the pre-head feature
vector so train_verifier.py can compute a supervised contrastive
(SupCon) auxiliary loss alongside BCE. No breaking changes — the
default forward signature is unchanged.

Why fresh backbone instead of reusing stage-1's
-----------------------------------------------
The candidates the verifier must reject are exactly the patterns that
fooled stage-1's feature extractor. A frozen stage-1 frontend would
hand the verifier the same representation that already failed to
separate them. A fresh backbone trained on the binary TP-vs-FP task has
different inductive biases (no per-frame supervision, balanced classes,
focused windows) and is more likely to learn complementary features.

Why shared backbone + per-class heads
-------------------------------------
Low-level audio features are class-agnostic. Sharing them lets the
small per-class data (especially D's 99 TPs in val) benefit from the
larger per-class data (BMABZ's 1000+ TPs). The per-class heads handle
task-specific calibration on top.

Stage-1 score as auxiliary input
--------------------------------
Concatenated to the pooled descriptor before the head. The verifier
learns a *correction* on top of the stage-1 score rather than
re-deriving the answer from scratch — mechanically the same trick
boosting uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ======================================================================
# Backbone
# ======================================================================

class VerifierBackbone(nn.Module):
    """
    Small 4-block CNN producing a fixed-size descriptor via global
    average pooling.

    Input  : (B, 3, F, T)
    Output : (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.block1(spec)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return x


# ======================================================================
# Per-class binary head
# ======================================================================

class VerifierHead(nn.Module):
    """2-layer MLP that maps (descriptor + aux) → 1 logit."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ======================================================================
# Full verifier model
# ======================================================================

class WhaleVerifier(nn.Module):
    """
    Multi-head verifier: shared backbone + one binary head per class.

    Forward signature::

        # default (matches v1/v2 callers — no breaking change):
        logit = model(spec, class_idx, aux)

        # v3 — also return the pre-head descriptor for SupCon:
        logit, feats = model(spec, class_idx, aux, return_features=True)

    Parameters
    ----------
    n_classes : int, default 3
    embed_dim : int, default 128
    n_aux : int, default 1
        Number of auxiliary scalar features concatenated to the
        descriptor before the head. Currently 1 = stage-1 score.
    backbone_dropout : float, default 0.5
    head_dropout : float, default 0.3
    """

    def __init__(
        self,
        n_classes: int = 3,
        embed_dim: int = 128,
        n_aux: int = 1,
        backbone_dropout: float = 0.5,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux = n_aux
        self.embed_dim = embed_dim

        self.backbone = VerifierBackbone(
            embed_dim=embed_dim, dropout=backbone_dropout,
        )
        head_in_dim = embed_dim + n_aux
        self.heads = nn.ModuleList([
            VerifierHead(head_in_dim, dropout=head_dropout)
            for _ in range(n_classes)
        ])

    def forward(
        self,
        spec: torch.Tensor,
        class_idx: torch.Tensor,
        aux: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Run a forward pass with per-sample head routing.

        Parameters
        ----------
        spec : torch.Tensor, shape (B, 3, F, T)
        class_idx : torch.Tensor, shape (B,) long
        aux : torch.Tensor, shape (B, n_aux) float
        return_features : bool
            If True, also return the pre-head descriptor (B, embed_dim).
            Useful for adding a SupCon auxiliary loss in the trainer.

        Returns
        -------
        logits : torch.Tensor, shape (B,) float
            Apply ``sigmoid`` for probability.
        feats : torch.Tensor, shape (B, embed_dim) — only if return_features.
            Raw, un-normalized; the caller L2-normalizes if doing
            cosine-based contrastive losses.
        """
        feats = self.backbone(spec)
        x = torch.cat([feats, aux], dim=-1)
        out = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for c in range(self.n_classes):
            mask = (class_idx == c)
            if mask.any():
                out[mask] = self.heads[c](x[mask])
        if return_features:
            return out, feats
        return out

    @torch.no_grad()
    def predict_proba(
        self,
        spec: torch.Tensor,
        class_idx: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper returning sigmoid probabilities, no grad."""
        return torch.sigmoid(self.forward(spec, class_idx, aux))


# ======================================================================
# Parameter count helper
# ======================================================================

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_params, trainable_params)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ======================================================================
# Sanity check
# ======================================================================

if __name__ == "__main__":
    model = WhaleVerifier()
    total, trainable = count_parameters(model)
    print(f"WhaleVerifier params: total={total:,}  trainable={trainable:,}")

    # Dummy phase-aware spectrogram. F=N_FFT//2+1, T~700 for 15 s @ 250 Hz.
    dummy_spec = torch.randn(4, 3, cfg.N_FFT // 2 + 1, 700)
    dummy_class = torch.tensor([0, 1, 2, 0])
    dummy_aux = torch.tensor([[0.31], [0.85], [0.42], [0.99]])

    out = model(dummy_spec, dummy_class, dummy_aux)
    print(f"Default forward: {tuple(out.shape)}  (expected (4,))")

    out2, feats = model(dummy_spec, dummy_class, dummy_aux,
                         return_features=True)
    print(f"With features:   logits {tuple(out2.shape)}, "
          f"feats {tuple(feats.shape)}")
    assert torch.allclose(out, out2, atol=1e-6), "feature path diverged"
    print("OK")
