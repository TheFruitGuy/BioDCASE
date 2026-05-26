"""
WhaleVAD-BPN Model
==================

Boundary Proposal Network (BPN) extension of the Whale-VAD classifier, after
Geldenhuys et al., "WhaleVAD-BPN" (arXiv:2510.21280v2). Built fresh on top of
the clean ``*_final`` modules.

Two pieces, both controlled by the ``use_bpn`` flag so a single class can serve
the whole experimental ladder:

1. **Modified backbone.** The original depthwise aggregation block (three
   plain depthwise convolutions) is replaced by a residual depthwise block
   with increasing dilation (2, 4, 8) and spatial dropout. Its three layers
   expose intermediate feature maps ("taps" h0, h1, h2).

2. **BPN branch** (``use_bpn=True``). Each tap is passed through its own
   projection head; the proposal network expands every head into R ROI feature
   vectors via two conv-transposes; each of the H*R ROIs is scored over time by
   a shared BiLSTM (or linear head) to a single gate logit; the H*R sigmoid
   gates are combined by a learned weighted mean into one per-frame **mask** in
   (0, 1). The single-channel mask multiplies the classifier's posterior
   probabilities (broadcast over classes), gating out false positives.

Output convention
-----------------
- ``use_bpn=False``: ``forward`` returns per-frame **logits** ``(B, T, C)``.
- ``use_bpn=True`` : ``forward`` returns gated **probabilities** ``(B, T, C)``
  (already in (0, 1); train with :func:`focal_loss_with_probs`).

The ``returns_probs`` attribute records which convention is active.

Faithful reproduction (arXiv:2510.21280v2, Fig. 2 + Table III)
--------------------------------------------------------------
This is the paper's BPN-multi configuration:

- The number of ROIs per head, R, is **not** a free hyperparameter. It is fixed
  by the architecture: the projection head's ``MaxPool(3, 1)`` collapses the
  tap's frequency axis (3 bins for the default N_FFT=256 backbone) to 1, and the
  two conv-transposes (kernels (4, 1) then (5, 1)) expand it ``1 -> 4 -> 8``, so
  R = 8. The proposal output is ``(C_bpn, H, R, T)``. R is derived at the first
  forward pass rather than configured, so it tracks the spectrogram/backbone
  config automatically.
- The learned weighted mean is the global form: one weight vector over the H*R
  ROIs, applied identically to every frame and fixed at inference.
- The mask is single-channel ``(B, T, 1)`` and gates every class equally
  (Fig. 2: ``probs(T, K) * mask(T, 1)``).
- The BPN gate is trained purely through the gradient from the gated output
  (no auxiliary per-frame target), initialised to pass-through.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_final as cfg
from model_final import ResidualBlock


# ======================================================================
# Focal loss
# ======================================================================

def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = cfg.FOCAL_ALPHA,
    gamma: float = cfg.FOCAL_GAMMA,
) -> torch.Tensor:
    """
    Element-wise focal loss on raw logits (``reduction="none"``).

    Use for the backbone-only model whose head emits logits. The caller is
    responsible for masking padded frames and reducing.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t) ** gamma * bce


def focal_loss_with_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = cfg.FOCAL_ALPHA,
    gamma: float = cfg.FOCAL_GAMMA,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Element-wise focal loss on probabilities already in (0, 1)
    (``reduction="none"``).

    Use for the gated BPN output, which is a probability (classifier prob x
    mask) rather than a logit.
    """
    p = probs.clamp(eps, 1.0 - eps)
    bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t) ** gamma * bce


# ======================================================================
# Dilated residual depthwise block (exposes taps)
# ======================================================================

class DilatedDepthwiseBlock(nn.Module):
    """
    Three residual depthwise 3x3 convolutions with increasing dilation.

    Each layer computes ``x = x + GELU(BN(DWConv_dilated(SpatialDropout(x))))``
    and the output after each layer is collected as a tap. Padding equals the
    dilation so the spatial size is preserved.
    """

    def __init__(self, ch: int, dilations=cfg.BPN_DILATIONS,
                 dropout: float = cfg.AGG_DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(nn.Sequential(
                nn.Dropout2d(dropout),
                nn.Conv2d(ch, ch, kernel_size=(3, 3), padding=(d, d),
                          dilation=(d, d), groups=ch),
                nn.BatchNorm2d(ch),
                nn.GELU(),
            ))

    def forward(self, x: torch.Tensor):
        taps = []
        for layer in self.layers:
            x = x + layer(x)
            taps.append(x)
        return x, taps


# ======================================================================
# WhaleVAD-BPN
# ======================================================================

class WhaleVADBPN(nn.Module):
    """
    Whale-VAD classifier with the BPN backbone change and an optional gating
    branch.

    Parameters
    ----------
    num_classes : int, default 7
    feat_channels : int, default 3
    use_bpn : bool, default False
        If False, behaves as the dilated-backbone classifier (returns logits).
        If True, adds the BPN branch and returns gated probabilities.
    bpn_taps : tuple of int, default (0, 1, 2)
        Which depthwise taps (0, 1, 2) feed the BPN branch (H = len(taps)).
    bpn_channels : int, default cfg.BPN_CHANNELS
        ROI feature dimensionality (C_bpn).
    bpn_lstm_hidden : int, default cfg.BPN_LSTM_HIDDEN
    bpn_use_bilstm : bool, default True
        Process ROI vectors with a BiLSTM (True) or a per-frame linear head
        (False).
    bpn_gate_init_bias : float, default cfg.BPN_GATE_INIT_BIAS
        Initial bias of the gate output so the mask starts near 1.
    """

    def __init__(self, num_classes: int = 7, feat_channels: int = 3, *,
                 use_bpn: bool = False,
                 bpn_taps=(0, 1, 2),
                 bpn_channels: int = cfg.BPN_CHANNELS,
                 bpn_lstm_hidden: int = cfg.BPN_LSTM_HIDDEN,
                 bpn_use_bilstm: bool = True,
                 bpn_gate_init_bias: float = cfg.BPN_GATE_INIT_BIAS):
        super().__init__()
        self.num_classes = num_classes
        self.use_bpn = use_bpn
        self.returns_probs = use_bpn
        self.bpn_taps = tuple(bpn_taps)
        self.bpn_n_roi = None   # H*R, set lazily at first forward

        # --- backbone front end (identical to model_final) ---
        self.filterbank = nn.Conv2d(
            feat_channels, cfg.FILTERBANK_OUT_CH,
            kernel_size=(7, 1), stride=(3, 1), padding=0,
        )
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

        # --- bottleneck (spatial dropout per the BPN paper) ---
        bottleneck = nn.Sequential(
            nn.Conv2d(cfg.FEAT_EXTRACTOR_CH, cfg.BOTTLENECK_CH,
                      kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout2d(cfg.BOTTLENECK_DROPOUT),
            nn.Conv2d(cfg.BOTTLENECK_CH, cfg.BOTTLENECK_CH,
                      kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.Dropout2d(cfg.BOTTLENECK_DROPOUT),
            nn.Conv2d(cfg.BOTTLENECK_CH, cfg.FEAT_EXTRACTOR_CH,
                      kernel_size=(1, 1)),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.Dropout2d(cfg.BOTTLENECK_DROPOUT),
        )
        self.bottleneck = ResidualBlock(bottleneck)

        # --- dilated residual depthwise block (exposes taps) ---
        self.depthwise = DilatedDepthwiseBlock(cfg.FEAT_EXTRACTOR_CH)

        # --- classification path (lazy projection + BiLSTM + head) ---
        self._proj_in_dim = None
        self.feat_proj = None
        self.lstm = nn.LSTM(
            input_size=cfg.PROJECTION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT,
        )
        self.classifier = nn.Linear(cfg.LSTM_HIDDEN * 2, num_classes)

        if use_bpn:
            self._build_bpn(bpn_channels, bpn_lstm_hidden, bpn_use_bilstm,
                            bpn_gate_init_bias)

    # ------------------------------------------------------------------
    # BPN branch construction
    # ------------------------------------------------------------------

    def _build_bpn(self, bpn_channels, bpn_lstm_hidden, use_bilstm,
                   gate_init_bias):
        ch = cfg.FEAT_EXTRACTOR_CH
        self._gate_init_bias = gate_init_bias

        # Intermediate projection head per tap (Table III: Conv2D 1x1 128->128,
        # BN, GELU, MaxPool(3,1)). Shared architecture, separate weights.
        self.bpn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(1, 1)),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1)),
            )
            for _ in self.bpn_taps
        ])

        # Proposal network (Table III): two conv-transposes over the
        # (pooled-frequency, time) plane, shared across heads. They expand the
        # pooled-frequency axis to R ROIs and reduce channels 128 -> 128 -> C_bpn.
        # Both layers carry BN, GELU, and spatial dropout.
        self.bpn_proposal = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=(4, 1)),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Dropout2d(cfg.AGG_DROPOUT),
            nn.ConvTranspose2d(ch, bpn_channels, kernel_size=(5, 1)),
            nn.BatchNorm2d(bpn_channels),
            nn.GELU(),
            nn.Dropout2d(cfg.AGG_DROPOUT),
        )

        # ROI processor: shared across all H*R ROIs, emits one gate logit per
        # ROI per frame.
        self.bpn_use_bilstm = use_bilstm
        if use_bilstm:
            self.bpn_lstm = nn.LSTM(
                input_size=bpn_channels, hidden_size=bpn_lstm_hidden,
                num_layers=1, batch_first=True, bidirectional=True,
            )
            self.bpn_out = nn.Linear(bpn_lstm_hidden * 2, 1)
        else:
            self.bpn_lstm = None
            self.bpn_out = nn.Linear(bpn_channels, 1)

        # Initialise the gate so the mask starts ~1 (sigmoid(bias)) and does
        # not suppress the classifier before it has learned anything.
        nn.init.zeros_(self.bpn_out.weight)
        nn.init.constant_(self.bpn_out.bias, gate_init_bias)

        # Learned weighted mean over the H*R ROIs (global; sized lazily at the
        # first forward once R is known from the data).
        self.roi_weights = None

    # ------------------------------------------------------------------
    # Forward sub-paths
    # ------------------------------------------------------------------

    def _init_projection(self, in_dim: int, device: torch.device):
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone feature map -> per-frame class logits (B, T, C)."""
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)
        self._init_projection(C * Fr, x.device)
        x = self.feat_proj(x)
        x, _ = self.lstm(x)
        return self.classifier(x)

    def _roi_gate(self, roi: torch.Tensor) -> torch.Tensor:
        """ROI vectors (N, T, C_bpn) -> one gate logit per ROI/frame (N, T, 1)."""
        if self.bpn_use_bilstm:
            roi, _ = self.bpn_lstm(roi)
        return self.bpn_out(roi)

    def _bpn_mask(self, taps: list[torch.Tensor]) -> torch.Tensor:
        """Selected taps -> single-channel per-frame mask in (0, 1), shape (B, T)."""
        # Each head: project + collapse frequency via MaxPool(3,1) -> (B, ch, Dh, T).
        heads = [self.bpn_heads[hi](taps[ti])
                 for hi, ti in enumerate(self.bpn_taps)]
        x = torch.stack(heads, dim=1)              # (B, H, ch, Dh, T)
        B, H, C, Dh, T = x.shape

        # Proposal network shared across heads (fold H into batch). The
        # conv-transposes expand the Dh axis to R; channels go ch -> C_bpn.
        x = x.reshape(B * H, C, Dh, T)
        x = self.bpn_proposal(x)                   # (B*H, C_bpn, R, T)
        _, Cb, R, T = x.shape

        # H*R ROI vectors, each a C_bpn-dim sequence over time.
        x = x.permute(0, 2, 3, 1).reshape(B * H * R, T, Cb)
        gate = torch.sigmoid(self._roi_gate(x))    # (B*H*R, T, 1)
        gate = gate.view(B, H * R, T)              # (B, H*R, T)

        # Learned global weighted mean over the H*R ROIs (lazily sized).
        if self.roi_weights is None or self.roi_weights.numel() != H * R:
            self.roi_weights = nn.Parameter(
                torch.zeros(H * R, device=gate.device))
            self.bpn_n_roi = H * R
        w = torch.softmax(self.roi_weights, dim=0).view(1, H * R, 1)
        return (gate * w).sum(dim=1)               # (B, T)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.filterbank(spec)
        x = self.feat_extractor(x)
        x = self.bottleneck(x)
        x, taps = self.depthwise(x)

        cls_logits = self._classify(x)
        if not self.use_bpn:
            return cls_logits

        mask = self._bpn_mask(taps)                # (B, T)
        T = min(cls_logits.size(1), mask.size(1))
        # Single-channel mask gates every class equally (broadcast over C).
        return torch.sigmoid(cls_logits[:, :T]) * mask[:, :T].unsqueeze(-1)
