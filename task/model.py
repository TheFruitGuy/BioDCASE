"""
Whale-VAD model architecture.

Exact reproduction of the paper (Geldenhuys et al., DCASE 2025) and
their github.com/CMGeldenhuys/Whale-VAD/blob/master/whalevad/model.py

Architecture (Figure 2, Table 2):
  Spectrogram(3ch) → Filterbank(7x1 Conv) → FeatureExtractor(2 Conv+MaxPool)
    → Residual(Bottleneck + DepthwiseConv)
    → Projection(Linear → 64)
    → BiLSTM(2 layers, hidden=128, dropout=0.5)
    → Classifier(Linear → num_classes)
    → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ----------------------------------------------------------------------
# Building block: Residual (sum-connected) — matches their ResidualBlock
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Applies each block in series with sum-style residual connections:
        X = X + block(X)   for each block
    """
    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


# ----------------------------------------------------------------------
# Main classifier — Table 2 in paper
# ----------------------------------------------------------------------

class WhaleVAD(nn.Module):
    """
    Inputs:
        spec: (B, 3, F, T) phase-aware spectrogram from SpectrogramExtractor

    Outputs:
        logits: (B, T, num_classes) — per-frame call logits
    """

    def __init__(self, num_classes: int = 3, feat_channels: int = 3):
        super().__init__()

        # ── Learnable filterbank (Table 2) ──────────────────────────
        # Conv2d: kernel (7,1), stride (3,1), in=3, out=64
        # Only convolves across frequency, not time
        self.filterbank = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=cfg.FILTERBANK_OUT_CH,
            kernel_size=(7, 1),
            stride=(3, 1),
            padding=0,
        )

        # ── Feature extractor (Table 2) ─────────────────────────────
        # Two Conv2d layers with BN + GELU + MaxPool
        self.feat_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                cfg.FILTERBANK_OUT_CH,
                cfg.FEAT_EXTRACTOR_CH,
                kernel_size=(5, 5),
                stride=(3, 1),
                padding=(2, 2),                # keep time resolution
            ),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(5, 1), stride=1, padding=0),
            # Layer 2
            nn.Conv2d(
                cfg.FEAT_EXTRACTOR_CH,
                cfg.FEAT_EXTRACTOR_CH,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=0),
        )

        # ── Bottleneck network (Table 2) ────────────────────────────
        # 128 → 64 → 64 → 128, residually connected later
        self.bottleneck = nn.Sequential(
            nn.Conv2d(cfg.FEAT_EXTRACTOR_CH, cfg.BOTTLENECK_CH,
                      kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.GELU(),
            nn.Dropout(cfg.BOTTLENECK_DROPOUT),
            nn.Conv2d(cfg.BOTTLENECK_CH, cfg.BOTTLENECK_CH,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
            nn.Dropout(cfg.BOTTLENECK_DROPOUT),
            nn.Conv2d(cfg.BOTTLENECK_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.Dropout(cfg.BOTTLENECK_DROPOUT),
        )

        # ── Depthwise convolutional aggregation (Table 2) ───────────
        # Three 3×3 depthwise convs (groups=128) with BN+GELU
        self.aggregation = nn.Sequential(
            nn.Dropout2d(cfg.AGG_DROPOUT),                    # spatial dropout
            nn.Conv2d(cfg.FEAT_EXTRACTOR_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(3, 3), stride=(1, 1), padding=1,
                      groups=cfg.FEAT_EXTRACTOR_CH),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.Conv2d(cfg.FEAT_EXTRACTOR_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(3, 3), stride=(1, 1), padding=1,
                      groups=cfg.FEAT_EXTRACTOR_CH),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.Conv2d(cfg.FEAT_EXTRACTOR_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(3, 3), stride=(1, 1), padding=1,
                      groups=cfg.FEAT_EXTRACTOR_CH),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
        )

        # Combine bottleneck + aggregation with residual connections
        self.residual_stack = ResidualBlock(self.bottleneck, self.aggregation)

        # ── Projection: (channels * freq) → 64 ──────────────────────
        # The frequency dim after filterbank + feature extractor depends on
        # the input. We compute it once and create the projection lazily.
        self._proj_in_dim = None
        self.feat_proj = None

        # ── BiLSTM (Section 5.4, dropout from Section 5.3 = 0.5) ────
        self.lstm = nn.LSTM(
            input_size=cfg.PROJECTION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT,
        )

        # ── Classifier ──────────────────────────────────────────────
        self.classifier = nn.Linear(
            cfg.LSTM_HIDDEN * 2, num_classes   # bidirectional → *2
        )

    def _init_projection(self, in_dim: int, device: torch.device):
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (B, 3, F, T) phase-aware spectrogram
        Returns:
            logits: (B, T, num_classes)
        """
        # Filterbank
        x = self.filterbank(spec)                             # (B, 64, F', T)

        # Feature extractor
        x = self.feat_extractor(x)                            # (B, 128, F'', T)

        # Residual stack (bottleneck + depthwise aggregation)
        x = self.residual_stack(x)                            # (B, 128, F'', T)

        # Flatten (channels × freq) → features for linear projection
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()                # (B, T, C, F)
        x = x.view(B, T, C * Fr)                               # (B, T, C*F)

        # Lazy init projection (depends on F at first forward pass)
        self._init_projection(C * Fr, x.device)
        x = self.feat_proj(x)                                 # (B, T, 64)

        # BiLSTM temporal processing
        x, _ = self.lstm(x)                                   # (B, T, 256)

        # Per-frame classifier
        logits = self.classifier(x)                           # (B, T, num_classes)
        return logits


# ----------------------------------------------------------------------
# Loss: weighted BCE + focal (Section 5.6)
# ----------------------------------------------------------------------

class WhaleVADLoss(nn.Module):
    """
    Weighted BCE + Focal loss, exactly as in the paper.
    
    Paper Section 5.6:
        - weighted BCE with w_c = N / P_c (negative/positive segments)
        - Focal loss: alpha=0.25, gamma=2 (on top of BCE)
    """
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.alpha = cfg.FOCAL_ALPHA
        self.gamma = cfg.FOCAL_GAMMA
        self.use_focal = cfg.USE_FOCAL_LOSS

    def forward(
        self,
        logits: torch.Tensor,                   # (B, T, C)
        targets: torch.Tensor,                  # (B, T, C)
        padding_mask: torch.Tensor | None = None,  # (B, T) bool, True=valid
    ) -> torch.Tensor:
        pw = self.pos_weight.view(1, 1, -1) if self.pos_weight is not None else None

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )

        if self.use_focal:
            # Focal loss modulation: alpha * (1 - p_t)^gamma
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.gamma
            loss = focal_mod * bce
        else:
            loss = bce

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()          # (B, T, 1)
            loss = loss * mask
            denom = mask.sum() * logits.size(-1) + 1e-8
            return loss.sum() / denom

        return loss.mean()


# ----------------------------------------------------------------------
# Weight computation (Section 5.6): w_c = N / P_c (segment-based)
# ----------------------------------------------------------------------

def compute_class_weights() -> torch.Tensor:
    """
    Paper Section 5.6: w_c = N / P_c
    Interpreted as: N = files NOT containing class c, P_c = files containing class c.
    This gives each class weight proportional to its rarity across files.
    """
    from dataset import load_annotations

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    total_files = annotations.groupby(["dataset", "filename"]).ngroups

    class_names = cfg.class_names()
    weights = []
    for c_name in class_names:
        if cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]

        p_c = max(class_annots.groupby(["dataset", "filename"]).ngroups, 1)
        n_neg = max(total_files - p_c, 1)
        w = n_neg / p_c
        weights.append(w)

    result = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights (w_c = N/P_c, file-level):")
    for name, w in zip(class_names, result.tolist()):
        print(f"  {name}: {w:.3f}")
    return result


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor()
    model = WhaleVAD(num_classes=cfg.n_classes())

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    logits = model(spec)

    print(f"Audio:   {audio.shape}")
    print(f"Spec:    {spec.shape}")
    print(f"Logits:  {logits.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:  {n:,}")
