"""
Verifier Model (Stage 2)
========================

Multi-head binary classifier that re-scores candidate events from the
stage-1 CNN-BiLSTM. Architecture:

    audio crop (30 s)                     raw waveform
        │
        ▼  SpectrogramExtractor           same as stage-1 (3-channel,
        │                                  phase-aware, demeaned)
        ▼  Conv2d 3→32 (7×7, s=2)         time/freq downsample
        ▼  Conv2d 32→64 (3×3) + pool      mid-level features
        ▼  Conv2d 64→128 (3×3) + pool     deeper features
        ▼  Conv2d 128→128 (3×3) + pool
        ▼  AdaptiveAvgPool2d → flatten    (B, 128) global descriptor
        │
        ▼  concat with [stage1_score]     (B, 129)
        │
        ▼  3 parallel heads (one per class)
        │  each: Linear(129, 64) → GELU → Dropout → Linear(64, 1)
        │
        ▼  pick head[class_idx]           per-sample routing
        │
    sigmoid logit → P(real call of this class)

Why fresh backbone instead of reusing stage-1's
-----------------------------------------------
The candidates the verifier must reject are *exactly* the patterns that
fooled stage-1's feature extractor in the first place. A frozen stage-1
frontend would hand the verifier the same representation that already
failed to separate them. A fresh backbone trained on the binary TP-vs-FP
task has different inductive biases (no per-frame supervision, balanced
classes, focused windows) and is more likely to learn complementary
features.

Why shared backbone instead of three independent models
-------------------------------------------------------
Low-level audio features (spectro-temporal edges, envelopes) are class-
agnostic. Sharing them lets the small per-class data (especially D's 403
TPs) benefit from the larger per-class data (BMABZ's 5438 TPs). The
per-class binary heads handle the task-specific calibration on top.

Stage-1 score as auxiliary input
--------------------------------
The candidate's stage-1 confidence is concatenated to the pooled feature
vector before the head. The verifier learns a *correction* on top of the
stage-1 score rather than re-deriving the answer from scratch. This is
mechanically the same trick boosting uses (each weak learner refines the
previous prediction).
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
    Small 4-block CNN that turns a 3-channel phase-aware spectrogram into a
    fixed-size descriptor via global average pooling.

    Input  shape: ``(B, 3, F, T)`` — typically F=129, T~1450 for 30 s @ 250 Hz
    Output shape: ``(B, embed_dim)``

    Parameters
    ----------
    embed_dim : int, default 128
        Final descriptor dimensionality.
    dropout : float, default 0.3
        Dropout applied after the backbone, before the head.
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        # Block 1: aggressive downsample on both axes — the input is huge
        # (~129 × 1450) and we want to compress quickly.
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: same channels along, downsample 2× more.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: deeper features, less downsampling now (we don't want to
        # collapse time too aggressively — temporal extent matters for SED).
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: refinement, no further pooling.
        self.block4 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Global pool to a fixed-length descriptor regardless of input T.
        # We average over BOTH frequency and time — the verifier doesn't
        # need explicit time localization; it just needs to decide
        # "does this crop contain a real call somewhere?".
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spec : torch.Tensor, shape (B, 3, F, T)

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        x = self.block1(spec)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)              # (B, embed_dim)
        x = self.dropout(x)
        return x


# ======================================================================
# Per-class binary head
# ======================================================================

class VerifierHead(nn.Module):
    """
    2-layer MLP that maps (backbone descriptor + aux features) to a single
    logit. Used as one of three parallel heads in the multi-head verifier.

    Parameters
    ----------
    in_dim : int
        Input dimensionality (embed_dim + n_aux_features).
    hidden_dim : int, default 64
    dropout : float, default 0.2
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # (B,)


# ======================================================================
# Full verifier model
# ======================================================================

class WhaleVerifier(nn.Module):
    """
    Multi-head verifier: shared backbone + one binary head per class.

    Forward signature::

        logit = model(spec, class_idx, aux_features)

    where ``aux_features[:, 0]`` is the candidate's stage-1 score (and
    additional aux columns can be added without changing the head's
    forward signature — just bump ``n_aux``).

    Parameters
    ----------
    n_classes : int, default 3
        Number of binary heads. One per coarse whale-call class.
    embed_dim : int, default 128
    n_aux : int, default 1
        Number of auxiliary scalar features concatenated to the
        descriptor before the head. Currently 1 = stage-1 score.
    backbone_dropout : float, default 0.3
    head_dropout : float, default 0.2
    """

    def __init__(
        self,
        n_classes: int = 3,
        embed_dim: int = 128,
        n_aux: int = 1,
        backbone_dropout: float = 0.3,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux = n_aux

        self.backbone = VerifierBackbone(
            embed_dim=embed_dim, dropout=backbone_dropout,
        )
        head_in_dim = embed_dim + n_aux
        # ModuleList so each head is a registered submodule with its own
        # state-dict entry; cleaner than wrapping in a single big Linear.
        self.heads = nn.ModuleList([
            VerifierHead(head_in_dim, dropout=head_dropout)
            for _ in range(n_classes)
        ])

    def forward(
        self,
        spec: torch.Tensor,
        class_idx: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass with per-sample head routing.

        Parameters
        ----------
        spec : torch.Tensor, shape (B, 3, F, T)
        class_idx : torch.Tensor, shape (B,) long
            Which head to apply per sample.
        aux : torch.Tensor, shape (B, n_aux) float
            Auxiliary scalar features (stage-1 score, etc.).

        Returns
        -------
        torch.Tensor, shape (B,) float
            Per-sample binary logit (apply ``sigmoid`` for probability).
        """
        feats = self.backbone(spec)             # (B, embed_dim)
        x = torch.cat([feats, aux], dim=-1)     # (B, embed_dim + n_aux)

        # Per-sample routing: run each head on its own subset and scatter
        # the results back. Avoids running every head on every sample,
        # which would 3× the head FLOPs.
        out = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for c in range(self.n_classes):
            mask = (class_idx == c)
            if mask.any():
                out[mask] = self.heads[c](x[mask])
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
    """
    Return ``(total_params, trainable_params)``. Useful for logging.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick sanity check: build the model, run a dummy forward pass, print
    # parameter count and output shape.
    model = WhaleVerifier()
    total, trainable = count_parameters(model)
    print(f"WhaleVerifier params: total={total:,}  trainable={trainable:,}")

    # Dummy 30 s phase-aware spectrogram. F=129 = N_FFT//2+1, T~1450 for
    # 7500 audio samples at HOP_LENGTH=5.
    dummy_spec = torch.randn(4, 3, cfg.N_FFT // 2 + 1, 1450)
    dummy_class = torch.tensor([0, 1, 2, 0])
    dummy_aux = torch.tensor([[0.31], [0.85], [0.42], [0.99]])

    out = model(dummy_spec, dummy_class, dummy_aux)
    print(f"Output shape: {tuple(out.shape)}  (expected (4,))")
    print(f"Output sample: {out.tolist()}")
