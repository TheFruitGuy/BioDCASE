"""
Final Pipeline - Whale-VAD Model
================================

CNN-BiLSTM per-frame classifier from Geldenhuys et al. (DCASE 2025):

    Input spectrogram (B, 3, F, T)
        -> Filterbank (Conv2d 7x1, stride 3x1)
        -> FeatureExtractor (2x Conv-BN-GELU-MaxPool)
        -> ResidualBlock (bottleneck + depthwise aggregation)
        -> Flatten (C*F) -> Linear projection (per-frame, 64-dim)
        -> BiLSTM (paper config: 2 layers, hidden 128)
        -> Linear classifier
    Output logits (B, T, num_classes)

Every 20 ms of audio receives its own multi-label classification output,
enabling event-level localisation downstream.
"""

import torch
import torch.nn as nn

import config_final as cfg


class ResidualBlock(nn.Module):
    """
    Sum-connected residual container.

    Applies each sub-block in sequence, adding its output to the running
    tensor: ``x -> x + block_1(x) -> ... ``. Each sub-block must preserve its
    input's channel and spatial dimensions.
    """

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class WhaleVAD(nn.Module):
    """
    Whale-VAD per-frame classifier.

    Consumes a three-channel phase-aware spectrogram and outputs per-frame
    multi-label logits.

    Parameters
    ----------
    num_classes : int, default 7
        Number of output classes. The final pipeline uses 7 (fine call types).
    feat_channels : int, default 3
        Number of input spectrogram channels (magnitude, cos-phase, sin-phase).

    Notes
    -----
    The projection layer from flattened CNN features to the BiLSTM input is
    created lazily on the first forward pass, because its input dimension
    depends on the STFT settings. Callers loading a checkpoint must therefore
    run one dummy forward pass before ``load_state_dict`` so the projection
    layer exists; the training entry point handles this automatically.
    """

    def __init__(self, num_classes: int = 7, feat_channels: int = 3):
        super().__init__()

        # 1. Learnable filterbank: a (7, 1) kernel over frequency, stride 3,
        #    compressing spectral resolution while keeping all time frames.
        self.filterbank = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=cfg.FILTERBANK_OUT_CH,
            kernel_size=(7, 1),
            stride=(3, 1),
            padding=0,
        )

        # 2. Feature extractor: two Conv-BN-GELU-MaxPool blocks pooling along
        #    frequency only, preserving the per-20ms time resolution.
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

        # 3. Bottleneck: 128 -> 64 -> 64 -> 128 squeeze-expand block.
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

        # 4. Depthwise aggregation: three depthwise 3x3 convolutions enlarging
        #    the temporal receptive field at low parameter cost.
        aggregation = nn.Sequential(
            nn.Dropout2d(cfg.AGG_DROPOUT),
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

        self.residual_stack = ResidualBlock(bottleneck, aggregation)

        # 5. Projection (lazy-initialised on first forward pass).
        self._proj_in_dim = None
        self.feat_proj = None

        # 6. BiLSTM temporal model.
        self.lstm = nn.LSTM(
            input_size=cfg.PROJECTION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT,
        )

        # 7. Per-frame classifier head.
        self.classifier = nn.Linear(cfg.LSTM_HIDDEN * 2, num_classes)

    def _init_projection(self, in_dim: int, device: torch.device):
        """Create the projection layer once the CNN output width is known."""
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spec : torch.Tensor
            Phase-aware spectrogram of shape ``(B, 3, F, T)``.

        Returns
        -------
        torch.Tensor
            Per-frame logits of shape ``(B, T, num_classes)``.
        """
        x = self.filterbank(spec)
        x = self.feat_extractor(x)
        x = self.residual_stack(x)

        # (B, C, F, T) -> (B, T, C*F) for the per-frame linear projection.
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * Fr)

        self._init_projection(C * Fr, x.device)
        x = self.feat_proj(x)

        x, _ = self.lstm(x)
        return self.classifier(x)
