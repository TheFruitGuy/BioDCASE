"""
WhaleVAD-BPN Model
==================

Reproduction of the WhaleVAD-BPN architecture from Geldenhuys et al.
(arxiv:2510.21280v2, "Improving Baleen Whale Call Detection with Boundary
Proposal Networks and Post-processing Optimisation").

The paper introduces two architectural changes on top of the baseline
WhaleVAD model:

1. **Dilated depthwise aggregation block**: the original 3-layer depthwise
   convolution stack is augmented with residual connections between each
   layer, time-axis dilations of (2, 4, 8), and spatial (channel-wise)
   dropout in place of standard dropout.

2. **Boundary Proposal Network (BPN)**: a separate gating module that
   uses intermediate feature maps from the dilated block to produce a
   per-frame "is-call" mask. This mask multiplies the classifier's
   sigmoid outputs and acts as a soft gate suppressing false positives.

Several design choices in the paper are under-specified (number of
intermediate taps, ROI count R, weighted-mean form, gate initialization,
etc.). Those are exposed via ``BPNConfig`` so they can be ablated from
the training script without code changes.

This file is fully self-contained — it duplicates the ``Filterbank`` and
``FeatureExtractor`` definitions from ``model.py`` rather than importing
them, so that the BPN experiment is structurally isolated from the
baseline. The cost is ~80 lines of duplicated module definitions; the
benefit is that ``model.py`` stays bit-identical to its baseline form
and ablations against it are unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ======================================================================
# BPN Configuration
# ======================================================================

@dataclass
class BPNConfig:
    """
    Tunables for the Boundary Proposal Network.

    These are the design choices the paper does not pin down. Each has
    a defensible default but is exposed for ablation. Pass into
    ``WhaleVADBPN(num_classes=..., bpn_cfg=BPNConfig(...))``.

    Attributes
    ----------
    enabled : bool
        Master switch. When False, the BPN gate is bypassed and the
        model behaves like the dilated-depthwise variant of WhaleVAD
        without any gating. Useful for ablating "architecture changes
        only" vs "architecture + BPN".
    n_taps : int
        Number of intermediate feature maps fed into the BPN. The
        dilated depthwise block has 3 layers; with n_taps=3 we tap
        after each. With n_taps=1 we tap only the deepest layer.
    n_rois : int
        R from the paper. Number of ROI vectors produced per head.
    dim : int
        C_bpn from the paper. Internal channel dimension of the BPN.
    pool_mode : str
        How per-head ROI scores are combined into a single per-frame
        gating mask. Options:
          "softmax"  — softmax over R ROIs (default; matches paper text)
          "sigmoid"  — independent sigmoid weights, then renormalise
          "mean"     — uniform average (no learned weighting)
    pool_scope : str
        Whether the learned weighting is per-frame or global.
          "framewise" — one set of weights per (B, T) position
          "global"    — one set of weights for the whole sequence
    init_mode : str
        Gate initialization. With "near_one" the gate starts close to
        1.0 everywhere (so the model behaves like vanilla WhaleVAD at
        epoch 1 and learns to gate FPs over training). With "random"
        the gate is initialised by default Kaiming. Critical for
        training stability.
          "near_one" — bias the final BPN linear toward outputting 1
          "random"   — default initialization
    temporal : str
        Temporal model applied to ROI vectors before the gate.
          "bilstm" — BiLSTM (paper's choice, performs better in their
                    ablation than logistic regression)
          "lr"     — single linear layer with sigmoid (paper's
                    alternative, kept for ablation)
          "none"   — skip temporal modelling, sigmoid the projection
                    network output directly
    lstm_hidden : int
        Hidden size per direction for the BPN BiLSTM. Has no effect
        when ``temporal != "bilstm"``.
    lstm_layers : int
        Number of BPN BiLSTM layers. As above.
    spatial_dropout_p : float
        Dropout probability inside the dilated depthwise block. The
        paper says "spatial dropout" but does not give a value; 0.2
        matches the existing baseline's AGG_DROPOUT.
    bpn_dropout_p : float
        Spatial dropout inside the proposal network.
    aux_loss : bool
        If True, add an auxiliary BCE loss on the gate against
        per-frame "any-class-active" targets. Encourages the gate
        to learn directly from the labels rather than relying on
        gradient flow through the classifier head.
    aux_weight : float
        Weight of the auxiliary loss. Total loss is
        ``main_loss + aux_weight * gate_loss``.
    dilations : tuple[int, ...]
        Per-layer dilation factors in the depthwise block. Default
        (2, 4, 8) matches the paper. Use (1, 2, 4) or (2, 4, 8, 16)
        for ablation. Must be all-time-axis.
    """
    enabled: bool = True
    n_taps: int = 3
    n_rois: int = 4
    dim: int = 64
    pool_mode: str = "softmax"
    pool_scope: str = "framewise"
    init_mode: str = "near_one"
    temporal: str = "bilstm"
    lstm_hidden: int = 64
    lstm_layers: int = 1
    spatial_dropout_p: float = 0.2
    bpn_dropout_p: float = 0.2
    aux_loss: bool = False
    aux_weight: float = 0.1
    dilations: tuple[int, ...] = (2, 4, 8)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        return d


# ======================================================================
# Building blocks (duplicated from model.py for full isolation)
# ======================================================================

class _ResidualBlock(nn.Module):
    """
    Sum-connected residual container — verbatim copy of model.ResidualBlock.

    Kept as a private duplicate so ``model_bpn.py`` does not depend on
    ``model.py``. The cost is ~10 lines of repeated code; the benefit
    is that any change to model.py cannot accidentally alter BPN
    training dynamics, which would confuse ablations.
    """

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


# ======================================================================
# Dilated depthwise aggregation block (the architecture change)
# ======================================================================

class DilatedDepthwiseBlock(nn.Module):
    """
    Modified version of WhaleVAD's depthwise aggregation block.

    Differences from the baseline aggregation block:
      1. Each depthwise conv has a *time-axis dilation* increasing by
         layer (default 2, 4, 8). Frequency-axis dilation stays at 1.
      2. Each conv is wrapped in its own residual: the layer's output
         is added to its input, so gradient can flow through any layer.
      3. Spatial (channel-wise) dropout is applied at the entry of
         every layer instead of only at the entry of the block.
      4. Padding is set to keep both freq and time dimensions exactly
         preserved so that downstream code can still rely on a fixed
         number of frames per input window.

    The block exposes intermediate feature maps for the BPN: setting
    ``return_intermediates=True`` in the forward pass returns the list
    of post-residual activations from every layer, which the BPN taps
    via its projection heads.

    Parameters
    ----------
    channels : int
        Channel count, passed through unchanged across all layers.
    dilations : tuple[int, ...]
        Time-axis dilation factor per layer. The number of layers in
        the block equals ``len(dilations)``.
    spatial_dropout_p : float
        Probability for the per-layer ``Dropout2d`` (spatial dropout).
    """

    def __init__(
        self, channels: int,
        dilations: tuple[int, ...] = (2, 4, 8),
        spatial_dropout_p: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            # Time-axis dilation only: kernel (3, 3), dilation (1, d),
            # padding (1, d) keeps spatial dims preserved.
            block = nn.Sequential(
                nn.Dropout2d(spatial_dropout_p),
                nn.Conv2d(
                    channels, channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, d),
                    dilation=(1, d),
                    groups=channels,
                ),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            self.layers.append(block)

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        intermediates = []
        for layer in self.layers:
            # Per-layer residual: x_{l+1} = x_l + f_l(x_l).
            x = x + layer(x)
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x


# ======================================================================
# Boundary Proposal Network components
# ======================================================================

class IntermediateProjectionHead(nn.Module):
    """
    Per-tap projection head from Table III of the paper.

    Each intermediate feature map is processed by an independent
    instance of this module (separate weights, identical architecture).
    The output collapses the frequency dimension to 1, leaving a
    per-tap sequence of (B, C_bpn, T) features that the proposal
    network ingests.

    Layer composition:
      Conv2D(1×1) → BatchNorm → GELU → MaxPool(F, 1) → squeeze freq dim

    Parameters
    ----------
    c_in : int
        Number of input channels (matches backbone feature channels).
    c_out : int
        Number of output channels (= BPNConfig.dim).
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1),
                              stride=(1, 1), padding=0)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
        # MaxPool kernel size along F is set lazily on the first forward
        # pass once we know the actual frequency dimension after the
        # backbone CNN. (The backbone has lazy projection; we mirror.)
        self.maxpool = None

    def _ensure_pool(self, freq_dim: int) -> None:
        if self.maxpool is None or self.maxpool.kernel_size[0] != freq_dim:
            self.maxpool = nn.MaxPool2d(kernel_size=(freq_dim, 1),
                                        stride=(freq_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, F, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        self._ensure_pool(x.size(2))
        x = self.maxpool(x)              # (B, C_out, 1, T)
        x = x.squeeze(2)                 # (B, C_out, T)
        return x


class ProposalNetwork(nn.Module):
    """
    Maps stacked intermediate features (B, C, H, T) to ROI vectors
    (B, C_out, R, T) via two transposed convolutions along the H axis.

    The paper's Table III specifies ConvTranspose2d layers with kernels
    (4, 1) and (5, 1) at 128 → 128 → 64 channels. With H input heads,
    the natural output is H + 7 along the H axis. We then optionally
    pool to exactly R ROIs using adaptive average pooling. This makes
    the (n_taps, n_rois) combination flexible without changing the
    paper's per-layer kernel sizes.

    Spatial dropout is applied between the two conv layers, mirroring
    the BPN architecture description.
    """

    def __init__(self, c_in: int, c_out: int, n_rois: int,
                 dropout_p: float = 0.2):
        super().__init__()
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_in, kernel_size=(4, 1),
                               stride=(1, 1), padding=0),
            nn.BatchNorm2d(c_in),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.ConvTranspose2d(c_in, c_out, kernel_size=(5, 1),
                               stride=(1, 1), padding=0),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
        )
        self.target_rois = n_rois

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, T) — stacked head outputs along H axis
        x = self.expand(x)               # (B, C_out, H+7, T)
        # Adapt H' → R via average pooling along the H axis.
        if x.size(2) != self.target_rois:
            x = F.adaptive_avg_pool2d(x, (self.target_rois, x.size(3)))
        return x                         # (B, C_out, R, T)


class BPNGate(nn.Module):
    """
    Full Boundary Proposal Network gate.

    Pipeline:
      1. Project each intermediate feature map via its own projection head.
      2. Stack the per-tap outputs along a new dim H → (B, C, H, T).
      3. Run the proposal network → (B, C, R, T) ROI vectors.
      4. Apply the temporal model (BiLSTM / LR / none) to each ROI to
         produce per-frame, per-ROI scores → (B, R, T).
      5. Combine R ROI scores into a single (B, T) gating mask via the
         configured pool_mode and pool_scope.

    The output mask is in [0, 1] and is meant to multiply the classifier's
    sigmoid probabilities elementwise.

    Parameters
    ----------
    backbone_channels : int
        Channel count of the intermediate feature maps from the backbone.
        Used to size the projection heads' input.
    bpn_cfg : BPNConfig
        All design knobs.
    """

    def __init__(self, backbone_channels: int, bpn_cfg: BPNConfig):
        super().__init__()
        self.cfg = bpn_cfg

        # One projection head per tap — each has its own weights.
        self.heads = nn.ModuleList([
            IntermediateProjectionHead(backbone_channels, bpn_cfg.dim)
            for _ in range(bpn_cfg.n_taps)
        ])

        self.proposal = ProposalNetwork(
            c_in=bpn_cfg.dim,
            c_out=bpn_cfg.dim,
            n_rois=bpn_cfg.n_rois,
            dropout_p=bpn_cfg.bpn_dropout_p,
        )

        # Temporal model. The BiLSTM operates on (B*R, T, C_bpn) so all
        # ROIs share weights but have independent hidden states.
        if bpn_cfg.temporal == "bilstm":
            self.temporal = nn.LSTM(
                input_size=bpn_cfg.dim,
                hidden_size=bpn_cfg.lstm_hidden,
                num_layers=bpn_cfg.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.0 if bpn_cfg.lstm_layers == 1 else 0.3,
            )
            score_in_dim = 2 * bpn_cfg.lstm_hidden
        elif bpn_cfg.temporal == "lr":
            self.temporal = None
            score_in_dim = bpn_cfg.dim
        elif bpn_cfg.temporal == "none":
            self.temporal = None
            score_in_dim = bpn_cfg.dim
        else:
            raise ValueError(f"Unknown temporal mode: {bpn_cfg.temporal!r}")

        # Per-ROI scoring head: maps ROI features → scalar per (frame, ROI).
        self.score_head = nn.Linear(score_in_dim, 1)

        # Learned weighted-mean parameters for combining R ROI scores.
        if bpn_cfg.pool_mode == "mean":
            # No learnable params in mean mode.
            self.head_weights = None
        else:
            # Per-ROI logit weights. With "framewise" pool_scope these
            # are still scalars (broadcast across T); to truly vary per
            # frame we would need a much larger param tensor that
            # depends on the input length. The paper's "weighted mean
            # over R" reads as a single learned vector of length R, so
            # framewise here means "applied to every frame identically",
            # not "different per frame". We keep the option to gate
            # globally for ablation.
            self.head_weights = nn.Parameter(torch.zeros(bpn_cfg.n_rois))

        # Initialise the score head to produce values that, after
        # sigmoid + pool, give a gate close to 1.0 everywhere. This
        # lets the model start as if the gate were absent and learn
        # to gate FPs over training. The bias is set so that
        # sigmoid(score) ≈ 0.95.
        if bpn_cfg.init_mode == "near_one":
            nn.init.constant_(self.score_head.bias, 3.0)
            nn.init.zeros_(self.score_head.weight)
        # else: leave default initialization

    def forward(
        self, intermediates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        Run the BPN gate forward.

        Parameters
        ----------
        intermediates : list[torch.Tensor]
            The K most recent feature maps from the dilated depthwise
            block, each of shape (B, C, F, T). Length must equal
            ``cfg.n_taps``.

        Returns
        -------
        mask : torch.Tensor, shape (B, T)
            Per-frame gating values in [0, 1].
        extras : dict
            Diagnostic tensors keyed by name. Useful for sanity checks
            during ablation runs:
              - ``per_roi_scores``: (B, R, T) the pre-pool scores
              - ``head_weights_resolved``: (R,) or (R, T) the actual
                normalised weights used in the pool
        """
        if len(intermediates) != self.cfg.n_taps:
            raise RuntimeError(
                f"BPN expects {self.cfg.n_taps} intermediate feature maps, "
                f"got {len(intermediates)}"
            )

        # Step 1: project each tap. Outputs: list of (B, C_bpn, T).
        head_outs = [h(im) for h, im in zip(self.heads, intermediates)]

        # Step 2: stack along H axis. (B, C_bpn, H, T)
        stacked = torch.stack(head_outs, dim=2)

        # Step 3: proposal network. (B, C_bpn, R, T)
        rois = self.proposal(stacked)

        B, C, R, T = rois.shape

        # Step 4: temporal modelling per ROI. We reshape to (B*R, T, C),
        # run through BiLSTM (or LR, or skip), then reshape back.
        x = rois.permute(0, 2, 3, 1).contiguous()    # (B, R, T, C)
        x = x.view(B * R, T, C)                      # (B*R, T, C)
        if isinstance(self.temporal, nn.LSTM):
            x, _ = self.temporal(x)                  # (B*R, T, 2*hidden)
        # For "lr" and "none", x stays as is — the score_head is the LR.

        scores = self.score_head(x)                  # (B*R, T, 1)
        scores = scores.view(B, R, T)                # (B, R, T)

        # Step 5: combine R ROI scores into a per-frame gate.
        if self.cfg.pool_mode == "mean":
            # Equal weighting across ROIs. Sigmoid each, then average.
            mask_per_roi = torch.sigmoid(scores)     # (B, R, T)
            mask = mask_per_roi.mean(dim=1)          # (B, T)
            weights_resolved = torch.full(
                (R,), 1.0 / R, device=scores.device,
            )
        elif self.cfg.pool_mode == "softmax":
            # Learned weighted mean over ROIs. Softmax(weights) gives
            # a normalised distribution. Apply to per-ROI sigmoids.
            assert self.head_weights is not None
            w = F.softmax(self.head_weights, dim=0)  # (R,)
            mask_per_roi = torch.sigmoid(scores)
            # (B, R, T) * (R, 1) broadcast → (B, R, T) → sum over R
            mask = (mask_per_roi * w.view(1, R, 1)).sum(dim=1)
            weights_resolved = w
        elif self.cfg.pool_mode == "sigmoid":
            # Independent per-ROI gates with renormalisation. More
            # capacity than softmax — each weight can grow/shrink
            # independently without competition.
            assert self.head_weights is not None
            w = torch.sigmoid(self.head_weights)     # (R,) in [0,1]
            w = w / (w.sum() + 1e-8)
            mask_per_roi = torch.sigmoid(scores)
            mask = (mask_per_roi * w.view(1, R, 1)).sum(dim=1)
            weights_resolved = w
        else:
            raise ValueError(f"Unknown pool_mode: {self.cfg.pool_mode!r}")

        return mask, {
            "per_roi_scores": scores,
            "head_weights_resolved": weights_resolved,
        }


# ======================================================================
# Full WhaleVAD-BPN model
# ======================================================================

class WhaleVADBPN(nn.Module):
    """
    Full WhaleVAD-BPN per-frame classifier.

    Mirrors the original WhaleVAD layer-for-layer up to the depthwise
    aggregation block, replaces that block with ``DilatedDepthwiseBlock``,
    and wires the BPN gate onto the intermediate outputs of that block.

    Output of forward is a dict with three tensors:
      - ``probs``  : (B, T, num_classes) gated probabilities, in [0, 1]
      - ``logits`` : (B, T, num_classes) raw classifier logits
      - ``mask``   : (B, T) BPN gate values, in [0, 1]

    The training loop consumes ``probs`` and (optionally) ``mask`` for
    the auxiliary loss. The validation loop reads ``probs`` directly
    without applying sigmoid.

    Parameters
    ----------
    num_classes : int
        Number of output classes (3, 4, or 7 depending on cfg flags).
    bpn_cfg : BPNConfig
        BPN design knobs. If ``bpn_cfg.enabled=False``, the BPN gate
        is bypassed and probabilities equal sigmoid(logits).
    feat_channels : int
        Number of input spectrogram channels (3 for trig representation).
    """

    def __init__(
        self, num_classes: int = 3,
        bpn_cfg: BPNConfig | None = None,
        feat_channels: int = 3,
    ):
        super().__init__()
        self.bpn_cfg = bpn_cfg or BPNConfig()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Backbone — identical to model.WhaleVAD up through the bottleneck
        # ------------------------------------------------------------------
        self.filterbank = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=cfg.FILTERBANK_OUT_CH,
            kernel_size=(7, 1),
            stride=(3, 1),
            padding=0,
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

        # Bottleneck — copied from baseline.
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

        # The new dilated depthwise aggregation. We hold it as an explicit
        # attribute (rather than wrapping in a Sequential) so we can call
        # it with return_intermediates=True for the BPN.
        self.dilated_depthwise = DilatedDepthwiseBlock(
            channels=cfg.FEAT_EXTRACTOR_CH,
            dilations=self.bpn_cfg.dilations,
            spatial_dropout_p=self.bpn_cfg.spatial_dropout_p,
        )

        # The bottleneck stays in a residual wrapper, but the depthwise
        # block has its own per-layer residuals. So the residual stack
        # here is just the bottleneck.
        self.bottleneck_stack = _ResidualBlock(bottleneck)

        # ------------------------------------------------------------------
        # BPN gate (taps the dilated depthwise intermediates)
        # ------------------------------------------------------------------
        if self.bpn_cfg.enabled:
            # Sanity: number of dilated layers must >= n_taps so we
            # can actually tap that many.
            n_layers = len(self.bpn_cfg.dilations)
            if self.bpn_cfg.n_taps > n_layers:
                raise ValueError(
                    f"BPNConfig.n_taps={self.bpn_cfg.n_taps} but the "
                    f"dilated block has only {n_layers} layers."
                )
            self.bpn = BPNGate(
                backbone_channels=cfg.FEAT_EXTRACTOR_CH,
                bpn_cfg=self.bpn_cfg,
            )
        else:
            self.bpn = None

        # ------------------------------------------------------------------
        # Lazy projection (same pattern as baseline)
        # ------------------------------------------------------------------
        self._proj_in_dim = None
        self.feat_proj = None

        # BiLSTM temporal model — identical to baseline.
        self.lstm = nn.LSTM(
            input_size=cfg.PROJECTION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT,
        )

        # Classifier head — identical to baseline.
        self.classifier = nn.Linear(cfg.LSTM_HIDDEN * 2, num_classes)

    def _init_projection(self, in_dim: int, device: torch.device) -> None:
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor]:
        # CNN front-end (filterbank + feat extractor + bottleneck residual)
        x = self.filterbank(spec)
        x = self.feat_extractor(x)
        x = self.bottleneck_stack(x)

        # Dilated depthwise — return intermediates iff BPN is enabled.
        if self.bpn is not None:
            x, intermediates = self.dilated_depthwise(
                x, return_intermediates=True,
            )
            # Pick the last n_taps intermediates (deepest features last).
            taps = intermediates[-self.bpn_cfg.n_taps:]
        else:
            x = self.dilated_depthwise(x, return_intermediates=False)
            taps = None

        # Flatten + project to BiLSTM input.
        B, C, Fr, T = x.shape
        x_lstm = x.permute(0, 3, 1, 2).contiguous()      # (B, T, C, F)
        x_lstm = x_lstm.view(B, T, C * Fr)
        self._init_projection(C * Fr, x.device)
        x_lstm = self.feat_proj(x_lstm)

        x_lstm, _ = self.lstm(x_lstm)                    # (B, T, 256)
        logits = self.classifier(x_lstm)                  # (B, T, C)

        # Apply BPN gate (if enabled).
        if self.bpn is not None:
            mask, bpn_extras = self.bpn(taps)            # (B, T)
            # mask[:, :, None] * sigmoid(logits) gives the gated probs.
            # We clamp probs to a tiny epsilon range to avoid log(0)
            # in any downstream loss.
            probs_raw = torch.sigmoid(logits)            # (B, T, C)
            mask_expanded = mask.unsqueeze(-1)           # (B, T, 1)
            # Align time dims defensively — the BPN's mask T can be
            # off-by-one from logits T due to pool/conv arithmetic.
            T_logits = logits.size(1)
            T_mask = mask_expanded.size(1)
            if T_logits != T_mask:
                m = min(T_logits, T_mask)
                probs_raw = probs_raw[:, :m]
                logits = logits[:, :m]
                mask_expanded = mask_expanded[:, :m]
                mask = mask[:, :m]
            probs = probs_raw * mask_expanded
        else:
            probs = torch.sigmoid(logits)
            mask = torch.ones(logits.size(0), logits.size(1),
                              device=logits.device)
            bpn_extras = {}

        return {
            "probs": probs,
            "logits": logits,
            "mask": mask,
            "bpn_extras": bpn_extras,
        }


# ======================================================================
# Loss
# ======================================================================

class WhaleVADBPNLoss(nn.Module):
    """
    BPN-aware loss = weighted/focal BCE on gated probabilities + optional
    auxiliary loss on the gate.

    Differences from ``model.WhaleVADLoss``:
      - Operates on gated probabilities, not raw logits, since the BPN
        has already applied the sigmoid + multiplicative gate.
      - Adds an optional auxiliary BCE on the gate against per-frame
        "any-class-active" targets, controlled by BPNConfig.

    Parameters
    ----------
    pos_weight : torch.Tensor, optional
        Per-class positive weights, shape ``(num_classes,)``.
    bpn_cfg : BPNConfig
        Used to read aux_loss / aux_weight / focal flags.
    """

    def __init__(
        self, pos_weight: torch.Tensor | None = None,
        bpn_cfg: BPNConfig | None = None,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.bpn_cfg = bpn_cfg or BPNConfig()
        self.alpha = cfg.FOCAL_ALPHA
        self.gamma = cfg.FOCAL_GAMMA
        self.use_focal = cfg.USE_FOCAL_LOSS
        self.eps = 1e-7

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = outputs["probs"].clamp(self.eps, 1 - self.eps)
        gate = outputs["mask"]

        # Element-wise BCE in probability space.
        bce = -(targets * torch.log(probs)
                + (1 - targets) * torch.log(1 - probs))

        # Focal modulation (in probability space).
        if self.use_focal:
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.gamma
            bce = focal_mod * bce

        # Per-class positive weighting on top.
        if self.pos_weight is not None:
            pw = self.pos_weight.view(1, 1, -1)
            bce = bce * (targets * pw + (1 - targets))

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            bce = bce * mask
            denom = mask.sum() * bce.size(-1) + 1e-8
            main_loss = bce.sum() / denom
        else:
            main_loss = bce.mean()

        total = main_loss

        # Auxiliary loss: gate should track "any class active".
        if self.bpn_cfg.aux_loss:
            # any_call: 1 wherever any class label is positive.
            any_call = targets.max(dim=-1).values.float()  # (B, T)
            gate_clamped = gate.clamp(self.eps, 1 - self.eps)
            aux = -(any_call * torch.log(gate_clamped)
                    + (1 - any_call) * torch.log(1 - gate_clamped))
            if padding_mask is not None:
                aux = aux * padding_mask.float()
                aux_loss = aux.sum() / (padding_mask.sum() + 1e-8)
            else:
                aux_loss = aux.mean()
            total = total + self.bpn_cfg.aux_weight * aux_loss

        return total


# ======================================================================
# Class weights (re-export from model.py to keep BPN script self-contained)
# ======================================================================

def compute_class_weights() -> torch.Tensor:
    """
    Per-class positive weights using the paper's formula.

    Identical to ``model.compute_class_weights``; re-implemented here
    so ``train_bpn.py`` does not import from ``model.py``. Kept verbatim
    so any future refactor to the formula in ``model.py`` does not
    silently affect BPN runs.
    """
    from dataset import load_annotations

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    total_files = annotations.groupby(["dataset", "filename"]).ngroups

    class_names = cfg.class_names()
    weights = []
    for c_name in class_names:
        if getattr(cfg, "USE_4CLASS_D_SPLIT", False):
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP_4.items()
                           if v == c_name]
            class_annots = annotations[
                annotations["annotation"].isin(orig_labels)
            ]
        elif cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items()
                           if v == c_name]
            class_annots = annotations[
                annotations["annotation"].isin(orig_labels)
            ]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]

        p_c = max(class_annots.groupby(["dataset", "filename"]).ngroups, 1)
        n_neg = max(total_files - p_c, 1)
        weights.append(n_neg / p_c)

    result = torch.tensor(weights, dtype=torch.float32)
    result = result / result.min()

    print(f"Class weights (w_c = N/P_c, normalized to min=1):")
    for name, w in zip(class_names, result.tolist()):
        print(f"  {name}: {w:.3f}")
    return result


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    """
    Smoke test: build the BPN model, push a dummy input through, print
    output shapes and parameter counts. Run as:

        python model_bpn.py
    """
    from spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor()
    bpn_cfg = BPNConfig(
        enabled=True, n_taps=3, n_rois=4, dim=64,
        pool_mode="softmax", init_mode="near_one", temporal="bilstm",
    )
    print(f"BPNConfig: {bpn_cfg.to_dict()}")
    model = WhaleVADBPN(num_classes=cfg.n_classes(), bpn_cfg=bpn_cfg)

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    out = model(spec)

    print(f"Audio:    {audio.shape}")
    print(f"Spec:     {spec.shape}")
    print(f"Probs:    {out['probs'].shape}    range=[{out['probs'].min():.3f},"
          f"{out['probs'].max():.3f}]")
    print(f"Logits:   {out['logits'].shape}")
    print(f"Mask:     {out['mask'].shape}     range=[{out['mask'].min():.3f},"
          f"{out['mask'].max():.3f}]")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params (with BPN):    {n:,}")

    # Compare to BPN-disabled (architecture-changes-only run).
    bpn_cfg_off = BPNConfig(enabled=False)
    model_off = WhaleVADBPN(num_classes=cfg.n_classes(), bpn_cfg=bpn_cfg_off)
    _ = model_off(spec)
    n_off = sum(p.numel() for p in model_off.parameters() if p.requires_grad)
    print(f"Params (no BPN):      {n_off:,}")
    print(f"BPN module overhead:  {n - n_off:,}")
