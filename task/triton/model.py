"""
Triton — Baseline model
=======================

Per-frame whale-call classifier. CNN frontend (learnable filterbank →
feature extractor → bottleneck-and-depthwise residual stack) followed
by a per-frame linear projection and a BiLSTM-based temporal head.

The math is identical to the WhaleVAD baseline from Geldenhuys et al.
(DCASE 2025); only the naming differs. Comparable F1 to WhaleVAD's
0.443 reference (our reproduction lands at ~0.469±0.005 with focal
loss across seeds, ~0.464 without).

Forward output
--------------
Returns a dict (not a bare tensor) so the training loop is symmetric
with ``TritonTIDE``::

    {
        "logits": (B, T, num_classes),  # raw per-frame logits
        "probs":  (B, T, num_classes),  # sigmoid(logits)
    }

Where "T" is the model's output time dimension, which may differ by ±1
from the naive ``n_samples // hop_length`` count due to boundary
arithmetic in the convolutions. The training loop's ``align_lengths``
helper handles the reconciliation.

Lazy projection
---------------
The projection layer from flattened CNN features to the BiLSTM input is
created lazily on the first forward pass: the input frequency dim
depends on the STFT settings, which is cleaner to discover at runtime
than to compute analytically. Callers must perform a dummy forward
pass *before* calling ``load_state_dict`` on a saved checkpoint so the
projection layer exists and can receive its stored weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg


# ======================================================================
# Building block: sum-connected residual container
# ======================================================================

class ResidualBlock(nn.Module):
    """
    Sum-connected residual container.

    Applies each sub-block in sequence and adds its output back to the
    running tensor:

        x → x + block_1(x) → (x + block_1(x)) + block_2(...) → ...

    Each sub-module must preserve the channel/spatial dimensions of its
    input so the additive skip is shape-compatible.
    """

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


# ======================================================================
# Main classifier
# ======================================================================

class Triton(nn.Module):
    """
    Per-frame whale-call classifier.

    Parameters
    ----------
    num_classes : int, default = 3
        Number of output classes. Use 3 for the coarse challenge task
        or 7 for fine-grained classification.
    feat_channels : int, default = 3
        Number of input spectrogram channels (3 for the trig
        representation: magnitude, cos-phase, sin-phase).
    """

    def __init__(self, num_classes: int = 3, feat_channels: int = 3):
        super().__init__()

        # ----- 1. Learnable filterbank --------------------------------
        # Conv2d with (7, 1) kernel acting only along the frequency
        # axis. Stride 3 along freq compresses spectral resolution by
        # 3× while keeping all time frames intact. Learnable analogue
        # to a mel filterbank — discovers whale-call-specific
        # groupings during training.
        self.filterbank = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=cfg.FILTERBANK_OUT_CH,
            kernel_size=(7, 1),
            stride=(3, 1),
            padding=0,
        )

        # ----- 2. Feature extractor -----------------------------------
        # Two Conv-BN-GELU-MaxPool blocks. Frequency resolution is
        # further compressed; time resolution is preserved throughout
        # via stride-1 pooling, so the output still has one frame per
        # 20 ms.
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(cfg.FILTERBANK_OUT_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(5, 5), stride=(3, 1), padding=(2, 2)),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(5, 1), stride=1, padding=0),
            nn.Conv2d(cfg.FEAT_EXTRACTOR_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=0),
        )

        # ----- 3. Bottleneck (squeeze-expand) -------------------------
        # 128 → 64 → 64 → 128 with 1×1, 3×3, 1×1 convolutions.
        bottleneck = nn.Sequential(
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

        # ----- 4. Depthwise aggregation -------------------------------
        # Three consecutive depthwise 3×3 convolutions. Each channel
        # processes its own spatial context independently — large
        # temporal receptive field with few parameters.
        aggregation = nn.Sequential(
            nn.Dropout2d(cfg.AGG_DROPOUT),  # spatial (channel-wise) dropout
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

        # Chain bottleneck and aggregation with skip connections. Kept
        # in a single ResidualBlock attribute rather than as separate
        # children so each weight appears in the state_dict once.
        self.residual_stack = ResidualBlock(bottleneck, aggregation)

        # ----- 5. Projection (lazy) -----------------------------------
        # Discovered on first forward pass; see module docstring.
        self._proj_in_dim: int | None = None
        self.feat_proj: nn.Linear | None = None

        # ----- 6. BiLSTM temporal model -------------------------------
        self.lstm = nn.LSTM(
            input_size=cfg.PROJECTION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT,
        )

        # ----- 7. Classifier head -------------------------------------
        self.classifier = nn.Linear(cfg.LSTM_HIDDEN * 2, num_classes)

    def _init_projection(self, in_dim: int, device: torch.device) -> None:
        """
        Create the projection layer on the first forward pass. Safe to
        call repeatedly: only acts when the input dim changes.
        """
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Run the forward pass.

        Parameters
        ----------
        spec : torch.Tensor
            Phase-aware spectrogram of shape ``(B, 3, F, T)``, typically
            produced by ``SpectrogramExtractor``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys ``"logits"`` and ``"probs"``, both shape
            ``(B, T_m, num_classes)``. ``T_m`` may differ by ±1 from
            the naive frame count due to CNN boundary arithmetic; the
            training loop's ``align_lengths`` reconciles this against
            the target tensor.
        """
        # CNN front end
        x = self.filterbank(spec)                   # (B, 64, F', T)
        x = self.feat_extractor(x)                  # (B, 128, F'', T)
        x = self.residual_stack(x)                  # (B, 128, F'', T)

        # Permute to (B, T, C, F), flatten C × F for the projection.
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, T, C, F)
        x = x.view(B, T, C * Fr)                    # (B, T, C*F)

        # Lazy projection initialisation.
        self._init_projection(C * Fr, x.device)
        x = self.feat_proj(x)                       # (B, T, 64)

        # Temporal model
        x, _ = self.lstm(x)                         # (B, T, 256)

        # Per-frame classifier
        logits = self.classifier(x)                 # (B, T, num_classes)
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probs": probs}


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Smoke test: build the model, forward a dummy batch, print shapes
    # and parameter count. Run as:  python model.py
    from spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor()
    model = Triton(num_classes=cfg.n_classes())

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    out = model(spec)

    print(f"Audio:    {audio.shape}")
    print(f"Spec:     {spec.shape}")
    print(f"Logits:   {out['logits'].shape}")
    print(f"Probs:    {out['probs'].shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:   {n:,}")
