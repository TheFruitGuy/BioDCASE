"""
WhaleVAD-BPN Model — Paper-Faithful Version (v2)
================================================

Paper-faithful reproduction of the WhaleVAD-BPN architecture from
Geldenhuys et al. (arxiv:2510.21280v2). Replaces ``model_bpn.py`` for
ablation; v1 checkpoints will not load into v2.

Architectural fixes vs. ``model_bpn.py`` (v1)
---------------------------------------------
1. **Macro-block taps.** Paper Fig. 2 taps three sequential macro-blocks
   (post feature extractor, post bottleneck, post depthwise stack).
   v1 tapped three layers *inside* the dilated depthwise stack — i.e.
   features at three closely related depths instead of three different
   semantic levels. v2 taps the macro-blocks via the
   ``BPNConfigV2.taps`` field; default is the paper's three.

2. **Frequency preservation in projection heads.** Paper Table III says
   the projection-head maxpool has kernel ``(3, 1)`` stride ``(1, 1)``,
   which trims freq by 2. v1 used kernel ``(F, 1)`` stride ``(F, 1)``,
   collapsing freq to 1. v2 preserves freq, so the proposal network's
   H axis carries spatial-frequency information.

3. **Channel widths 128 → 128 → 64.** Paper Table III is explicit.
   v1 ran the whole BPN at 64 channels by default (``BPNConfig.dim``).
   v2 uses ``proj_dim=128`` (heads + first proposal layer) and
   ``out_dim=64`` (BiLSTM input).

4. **H axis carries frequency content.** Paper concatenates per-head
   outputs along an axis that aggregates frequency from multiple taps.
   v1 stacked along a brand-new ``n_taps``-sized axis with no freq
   content. v2 concatenates per-head ``(C, F-2, T)`` along the freq
   axis to produce ``(C, n_taps × (F-2), T)``, which the proposal
   network then expands meaningfully.

5. **Proposal network expansion is meaningful.** With v2 the
   ConvTranspose ``(4,1)(5,1)`` layers grow a freq-bearing axis by 7 as
   the paper intends, rather than expanding a degenerate ``n_taps=3``
   axis from 3 to 10.

Auxiliary-loss fixes
--------------------
v1's aux loss collapsed the gate to ~0 because vanilla BCE on a sparse
positive target ("any class active") drives the mask toward 0 on the
~95% negative frames. v2 adds:

- ``aux_pos_weight``: per-frame BCE positive weight (default 19.0).
  Counters the positive-frame imbalance so the gate isn't dragged
  toward zero by the negative majority.
- ``aux_warmup_epochs``: aux loss kicks in only after this many epochs.
  The BPN first learns implicit gating through classifier gradients,
  then gets explicit supervision once the classifier is competent.
- Soft Gaussian smoothing of the aux target (``aux_target_sigma``).
  Calls don't have crisp on/off in the spectrogram; a hard step-function
  target makes the mask fight reality at boundaries. ``sigma=2.0``
  frames is enough to soften.
- Asymmetric weighting (``aux_recall_weight``) — penalize "mask low on
  call frame" more than "mask high on no-call frame". The BPN's job
  is suppressing FPs *without* killing recall, so the loss should
  reflect that asymmetry.

Engineering choices preserved from v1
-------------------------------------
- ``init_mode="near_one"``: bias the score head so the gate starts at
  ~0.95. The model behaves like the BPN-disabled variant at epoch 1
  and learns gating gradually. Stability win, paper doesn't specify.
- Dilated depthwise block with per-layer residuals + dilations (2, 4, 8)
  + spatial dropout. These match the paper.

Usage
-----
    from model_bpn_v2 import WhaleVADBPNV2, BPNConfigV2, WhaleVADBPNLossV2

    bpn_cfg = BPNConfigV2(
        enabled=True,
        taps=("feat_ext", "bottleneck", "depthwise"),
        n_rois=32,
        aux_loss=True,
        aux_pos_weight=19.0,
        aux_warmup_epochs=5,
    )
    model = WhaleVADBPNV2(num_classes=cfg.n_classes(), bpn_cfg=bpn_cfg)
    criterion = WhaleVADBPNLossV2(pos_weight=class_weights, bpn_cfg=bpn_cfg)

    # Trainer must call this each epoch for aux warmup to work:
    for epoch in range(num_epochs):
        criterion.set_epoch(epoch)
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ======================================================================
# BPN Configuration (v2)
# ======================================================================

@dataclass
class BPNConfigV2:
    """
    Configuration for the paper-faithful BPN v2.

    Attributes
    ----------
    enabled : bool
        Master switch. When False, the BPN is bypassed and the model
        behaves like the dilated-depthwise WhaleVAD without gating.
    taps : tuple[str, ...]
        Which macro-block outputs to feed into the BPN. Each entry must
        be one of:
          - ``"feat_ext"``  : output of the feature-extractor stack
          - ``"bottleneck"``: output of the bottleneck residual block
          - ``"depthwise"`` : output of the dilated depthwise block
        Default ``("feat_ext", "bottleneck", "depthwise")`` matches
        Fig. 2 of the paper.
    proj_dim : int
        Projection-head output channels (also the proposal network's
        first-layer output). Paper Table III: 128.
    out_dim : int
        Proposal network's final output channels (also the BiLSTM
        input dimension). Paper Table III: 64.
    n_rois : int
        R from the paper. After the proposal network expands the H
        axis by 7 via ConvTranspose, an adaptive avg-pool collapses
        H' → R so the BiLSTM and weighted-mean parameters have a
        fixed shape. Default 32; the paper does not pin this down,
        but ablations in the BPN paper text used "multiple ROIs per
        head" without naming a specific R.
    pool_mode : str
        How per-frame R-vector scores are combined into a single
        gating mask. Options:
          - ``"softmax"``: softmax over learnable weights (paper text)
          - ``"sigmoid"``: independent sigmoids, then renormalise
          - ``"mean"``   : uniform average (no learned weighting)
    init_mode : str
        Score-head initialization. ``"near_one"`` biases the gate to
        ~0.95 at start so the model behaves like the BPN-disabled
        variant in epoch 1 and learns gating gradually.
    temporal : str
        Temporal model applied to per-ROI sequences:
          - ``"bilstm"``: paper's choice, dominant in their ablation
          - ``"lr"``    : single linear layer
          - ``"none"``  : skip temporal modelling
    lstm_hidden : int
        BiLSTM hidden size per direction.
    lstm_layers : int
        BiLSTM layer count.
    spatial_dropout_p : float
        Spatial dropout in the dilated depthwise block.
    bpn_dropout_p : float
        Spatial dropout inside the proposal network.
    aux_loss : bool
        Enable the auxiliary loss on the gate against an "any-class-
        active" target. False by default — see notes in the loss class.
    aux_weight : float
        Weight of the auxiliary loss in the total. Total =
        ``main_loss + aux_weight × aux_loss``.
    aux_pos_weight : float
        Per-frame positive weight for the aux BCE. With ~5% positive
        frames, set to ≈ 19.0 to balance gradients. Critical to avoid
        the v1 "mask collapses to 0" failure mode.
    aux_warmup_epochs : int
        Aux loss is multiplied by 0 for this many epochs, then linearly
        ramped to ``aux_weight`` over the next ``aux_ramp_epochs``.
        Lets the classifier reach a sensible state before the BPN
        gets pushed.
    aux_ramp_epochs : int
        Number of epochs over which aux_weight ramps from 0 to its
        configured value after warmup ends.
    aux_target_sigma : float
        Standard deviation (in frames) of a Gaussian smoothing kernel
        applied to the aux target. Softens the hard 0/1 boundary so the
        mask isn't fighting the spectrogram's gradual call onsets.
        Set to 0 to disable.
    aux_recall_weight : float
        Extra multiplier on the "mask should be high on call frames"
        half of the aux BCE, **on top of** ``aux_pos_weight``. Default
        1.0 means class-balanced BCE only (no extra recall bias). Set
        to 2-5 if you want to lean further toward recall preservation
        ("reduce FPs while preserving recall", per the paper's framing).
    dilations : tuple[int, ...]
        Per-layer dilation factors in the depthwise block. Default
        (2, 4, 8) matches the paper.
    """
    enabled: bool = True
    taps: tuple[str, ...] = ("feat_ext", "bottleneck", "depthwise")
    proj_dim: int = 128
    out_dim: int = 64
    n_rois: int = 32
    pool_mode: str = "softmax"
    init_mode: str = "near_one"
    temporal: str = "bilstm"
    lstm_hidden: int = 64
    lstm_layers: int = 1
    spatial_dropout_p: float = 0.2
    bpn_dropout_p: float = 0.2
    aux_loss: bool = False
    aux_weight: float = 0.1
    aux_pos_weight: float = 19.0
    aux_warmup_epochs: int = 5
    aux_ramp_epochs: int = 5
    aux_target_sigma: float = 2.0
    aux_recall_weight: float = 1.0
    dilations: tuple[int, ...] = (2, 4, 8)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        d["taps"] = list(self.taps)
        return d


# ======================================================================
# Building blocks (duplicated for structural isolation, as in v1)
# ======================================================================

class _ResidualBlock(nn.Module):
    """Sum-connected residual container (verbatim copy of model.ResidualBlock)."""

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class DilatedDepthwiseBlock(nn.Module):
    """
    Modified WhaleVAD aggregation block per Section IV.A of the paper.

    Per-layer residuals, time-axis dilations (default 2, 4, 8), spatial
    dropout. Identical to v1's version, but no longer needs to expose
    intermediates because the v2 BPN taps macro-blocks instead of
    per-layer.
    """

    def __init__(
        self, channels: int,
        dilations: tuple[int, ...] = (2, 4, 8),
        spatial_dropout_p: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


# ======================================================================
# BPN components (v2: paper-faithful)
# ======================================================================

class IntermediateProjectionHeadV2(nn.Module):
    """
    Per-tap projection head from Table III, paper-faithful.

    Layer composition:
      Conv2D (1×1) → BatchNorm → GELU → MaxPool kernel (3, 1) stride (1, 1)

    The maxpool **preserves** the frequency dimension (trims by 2 only).
    This is the v1 → v2 fix: v1's maxpool collapsed freq to 1.

    Parameters
    ----------
    c_in : int
        Input channels — matches the macro-block tap channel count
        (FEAT_EXTRACTOR_CH = 128 for all three macro-blocks).
    c_out : int
        Output channels (= BPNConfigV2.proj_dim).
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1),
                              stride=(1, 1), padding=0)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
        # Paper Table III: kernel (3, 1), stride (1, 1) — preserves freq
        # except for the kernel-3 trim (-2).
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, F, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)              # (B, C_out, F-2, T)
        return x


class ProposalNetworkV2(nn.Module):
    """
    Proposal network from Table III, applied to the freq-bearing H axis.

    Per Table III:
      ConvTranspose2D (4, 1) (1, 1) 128 → 128
      ConvTranspose2D (5, 1) (1, 1) 128 → 64

    With stride (1, 1) and padding 0, kernel (4, 1) grows H by 3 and
    kernel (5, 1) grows H by 4 — total +7. The output is then collapsed
    to ``n_rois`` along the H axis via adaptive average pooling so the
    downstream BiLSTM and weighted-mean parameters have fixed shape.

    Spatial dropout sits between the two transposed convs, as the
    paper architecture description specifies.
    """

    def __init__(
        self, c_in: int, c_mid: int, c_out: int,
        n_rois: int, dropout_p: float = 0.2,
    ):
        super().__init__()
        self.conv_t1 = nn.ConvTranspose2d(
            c_in, c_mid, kernel_size=(4, 1),
            stride=(1, 1), padding=0,
        )
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(dropout_p)

        self.conv_t2 = nn.ConvTranspose2d(
            c_mid, c_out, kernel_size=(5, 1),
            stride=(1, 1), padding=0,
        )
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout2d(dropout_p)

        self.target_rois = n_rois

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, T) — H = sum of per-head freq dims
        x = self.conv_t1(x)              # (B, C_mid, H+3, T)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv_t2(x)              # (B, C_out, H+7, T)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        # Collapse expanded H axis → fixed R via adaptive pool.
        # The paper does not specify R explicitly; adaptive pool
        # makes R independent of the input freq dim.
        if x.size(2) != self.target_rois:
            x = F.adaptive_avg_pool2d(x, (self.target_rois, x.size(3)))
        return x                         # (B, C_out, R, T)


class BPNGateV2(nn.Module):
    """
    Full Boundary Proposal Network, paper-faithful.

    Pipeline:
      1. Project each macro-block tap via its own projection head.
         Per-head output: (B, proj_dim, F-2, T).
      2. Concatenate along the **freq** axis (not a new axis):
         (B, proj_dim, n_taps × (F-2), T).
      3. Proposal network: 2 ConvTranspose layers expand H by 7,
         then adaptive pool to ``n_rois`` → (B, out_dim, R, T).
      4. BiLSTM (or LR / none) over (B*R, T, out_dim).
      5. Score head per (frame, ROI), then weighted mean over R
         → (B, T) gating mask in [0, 1].
    """

    def __init__(self, backbone_channels: int, bpn_cfg: BPNConfigV2):
        super().__init__()
        self.cfg = bpn_cfg
        n_taps = len(bpn_cfg.taps)

        # One projection head per tap. All heads share architecture
        # but have independent weights (per the paper).
        self.heads = nn.ModuleList([
            IntermediateProjectionHeadV2(backbone_channels, bpn_cfg.proj_dim)
            for _ in range(n_taps)
        ])

        self.proposal = ProposalNetworkV2(
            c_in=bpn_cfg.proj_dim,
            c_mid=bpn_cfg.proj_dim,           # paper: 128 → 128
            c_out=bpn_cfg.out_dim,            # paper: 128 → 64
            n_rois=bpn_cfg.n_rois,
            dropout_p=bpn_cfg.bpn_dropout_p,
        )

        # Temporal model.
        if bpn_cfg.temporal == "bilstm":
            self.temporal = nn.LSTM(
                input_size=bpn_cfg.out_dim,
                hidden_size=bpn_cfg.lstm_hidden,
                num_layers=bpn_cfg.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.0 if bpn_cfg.lstm_layers == 1 else 0.3,
            )
            score_in_dim = 2 * bpn_cfg.lstm_hidden
        elif bpn_cfg.temporal in ("lr", "none"):
            self.temporal = None
            score_in_dim = bpn_cfg.out_dim
        else:
            raise ValueError(f"Unknown temporal mode: {bpn_cfg.temporal!r}")

        # Per-(frame, ROI) score head.
        self.score_head = nn.Linear(score_in_dim, 1)

        # Learned weighting over R ROIs (paper: "trainable weighted mean").
        if bpn_cfg.pool_mode == "mean":
            self.head_weights = None
        else:
            self.head_weights = nn.Parameter(torch.zeros(bpn_cfg.n_rois))

        # Bias the gate to ~0.95 at start (engineering choice — not in paper).
        if bpn_cfg.init_mode == "near_one":
            nn.init.constant_(self.score_head.bias, 3.0)
            nn.init.zeros_(self.score_head.weight)

    def forward(
        self, taps_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        Run the BPN gate forward.

        Parameters
        ----------
        taps_list : list[torch.Tensor]
            One tensor per configured tap, each of shape (B, C, F, T).
            Length must equal ``len(cfg.taps)``. The freq dim ``F`` may
            differ across taps — they are independently projected before
            being concatenated.

        Returns
        -------
        mask : torch.Tensor, shape (B, T) in [0, 1]
        extras : dict with diagnostic tensors
        """
        if len(taps_list) != len(self.cfg.taps):
            raise RuntimeError(
                f"BPN expects {len(self.cfg.taps)} taps "
                f"({list(self.cfg.taps)}), got {len(taps_list)}"
            )

        # Step 1: project each tap. Outputs: list of (B, proj_dim, F-2, T).
        head_outs = [h(t) for h, t in zip(self.heads, taps_list)]

        # Step 2: concatenate along the freq axis (paper-faithful).
        # The freq dims must match for cat to work; in this codebase
        # all three macro-blocks preserve freq, so they do match.
        # If they don't (e.g. someone changes the backbone), align via
        # interpolation to the smallest freq dim.
        min_freq = min(h.size(2) for h in head_outs)
        if any(h.size(2) != min_freq for h in head_outs):
            head_outs = [
                F.adaptive_avg_pool2d(h, (min_freq, h.size(3)))
                for h in head_outs
            ]
        stacked = torch.cat(head_outs, dim=2)
        # stacked: (B, proj_dim, n_taps × (F-2), T)

        # Step 3: proposal network — expand H, then collapse to R.
        rois = self.proposal(stacked)             # (B, out_dim, R, T)
        B, C, R, T = rois.shape

        # Step 4: temporal modelling. Reshape to (B*R, T, C).
        x = rois.permute(0, 2, 3, 1).contiguous()  # (B, R, T, C)
        x = x.view(B * R, T, C)
        if isinstance(self.temporal, nn.LSTM):
            x, _ = self.temporal(x)               # (B*R, T, 2*hidden)

        scores = self.score_head(x)               # (B*R, T, 1)
        scores = scores.view(B, R, T)             # (B, R, T)

        # Step 5: weighted mean over R → per-frame mask.
        if self.cfg.pool_mode == "mean":
            mask_per_roi = torch.sigmoid(scores)
            mask = mask_per_roi.mean(dim=1)
            weights_resolved = torch.full(
                (R,), 1.0 / R, device=scores.device,
            )
        elif self.cfg.pool_mode == "softmax":
            assert self.head_weights is not None
            w = F.softmax(self.head_weights, dim=0)
            mask_per_roi = torch.sigmoid(scores)
            mask = (mask_per_roi * w.view(1, R, 1)).sum(dim=1)
            weights_resolved = w
        elif self.cfg.pool_mode == "sigmoid":
            assert self.head_weights is not None
            w = torch.sigmoid(self.head_weights)
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
# Full WhaleVAD-BPN model (v2)
# ======================================================================

class WhaleVADBPNV2(nn.Module):
    """
    Full WhaleVAD-BPN per-frame classifier, paper-faithful version.

    The backbone is identical to the original WhaleVAD up to the
    depthwise stack. Three macro-block outputs (post-FE, post-bottleneck,
    post-depthwise) are exposed to the BPN. The BPN produces a per-frame
    gating mask that multiplies the classifier's sigmoid outputs.

    Output of forward is a dict with three tensors:
      - ``probs``  : (B, T, num_classes) gated probabilities, in [0, 1]
      - ``logits`` : (B, T, num_classes) raw classifier logits
      - ``mask``   : (B, T) BPN gate values, in [0, 1]
    """

    _TAP_NAMES: tuple[str, ...] = ("feat_ext", "bottleneck", "depthwise")

    def __init__(
        self, num_classes: int = 3,
        bpn_cfg: BPNConfigV2 | None = None,
        feat_channels: int = 3,
    ):
        super().__init__()
        self.bpn_cfg = bpn_cfg or BPNConfigV2()
        self.num_classes = num_classes

        # Validate tap names.
        unknown = set(self.bpn_cfg.taps) - set(self._TAP_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown taps {sorted(unknown)}; "
                f"valid: {list(self._TAP_NAMES)}"
            )

        # ------------------------------------------------------------------
        # Backbone
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
        self.bottleneck_stack = _ResidualBlock(bottleneck)

        self.dilated_depthwise = DilatedDepthwiseBlock(
            channels=cfg.FEAT_EXTRACTOR_CH,
            dilations=self.bpn_cfg.dilations,
            spatial_dropout_p=self.bpn_cfg.spatial_dropout_p,
        )

        # ------------------------------------------------------------------
        # BPN gate (taps the macro-blocks, paper Fig. 2)
        # ------------------------------------------------------------------
        if self.bpn_cfg.enabled:
            self.bpn = BPNGateV2(
                backbone_channels=cfg.FEAT_EXTRACTOR_CH,
                bpn_cfg=self.bpn_cfg,
            )
        else:
            self.bpn = None

        # Lazy projection (same as baseline).
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

    def _init_projection(self, in_dim: int, device: torch.device) -> None:
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor]:
        # Backbone forward — capture intermediates as we go.
        x = self.filterbank(spec)
        x = self.feat_extractor(x)
        feat_ext_out = x                          # macro-block tap 1

        x = self.bottleneck_stack(x)
        bottleneck_out = x                        # macro-block tap 2

        x = self.dilated_depthwise(x)
        depthwise_out = x                         # macro-block tap 3

        # Collect taps in the order specified by config.
        tap_lookup = {
            "feat_ext": feat_ext_out,
            "bottleneck": bottleneck_out,
            "depthwise": depthwise_out,
        }
        taps = (
            [tap_lookup[name] for name in self.bpn_cfg.taps]
            if self.bpn is not None else None
        )

        # Classifier head: flatten + project + BiLSTM + linear.
        B, C, Fr, T = x.shape
        x_lstm = x.permute(0, 3, 1, 2).contiguous()      # (B, T, C, F)
        x_lstm = x_lstm.view(B, T, C * Fr)
        self._init_projection(C * Fr, x.device)
        x_lstm = self.feat_proj(x_lstm)

        x_lstm, _ = self.lstm(x_lstm)                    # (B, T, 2*hidden)
        logits = self.classifier(x_lstm)                  # (B, T, num_classes)

        if self.bpn is not None:
            mask, bpn_extras = self.bpn(taps)            # (B, T)
            probs_raw = torch.sigmoid(logits)            # (B, T, C)
            mask_expanded = mask.unsqueeze(-1)           # (B, T, 1)

            # Defensive temporal alignment: BPN's mask T can differ
            # from logits' T due to pool/conv arithmetic in edge cases.
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
# Loss (v2: aux-loss collapse fixed)
# ======================================================================

class WhaleVADBPNLossV2(nn.Module):
    """
    BPN-aware loss with the v1 aux-loss collapse fix.

    Main loss: focal BCE on gated probabilities, identical to v1.

    Auxiliary loss (when enabled): BCE on the gate against per-frame
    "any-class-active" targets, with three additions over v1:
      1. ``pos_weight`` rebalancing to counter the ~95% negative-frame
         majority that drove v1's mask to 0.
      2. Linear warmup over ``aux_warmup_epochs + aux_ramp_epochs``
         so the BPN doesn't get pushed before the classifier is
         competent.
      3. Optional Gaussian smoothing of the binary target so the mask
         doesn't fight the spectrogram's gradual call onsets.
      4. Asymmetric weighting toward recall preservation
         (``aux_recall_weight``).

    Trainer integration
    -------------------
    The trainer must call ``loss_fn.set_epoch(epoch)`` before each
    epoch's loop; otherwise warmup is fixed at 0.
    """

    def __init__(
        self, pos_weight: torch.Tensor | None = None,
        bpn_cfg: BPNConfigV2 | None = None,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.bpn_cfg = bpn_cfg or BPNConfigV2()
        self.alpha = cfg.FOCAL_ALPHA
        self.gamma = cfg.FOCAL_GAMMA
        self.use_focal = cfg.USE_FOCAL_LOSS
        self.eps = 1e-7
        self._epoch = 0

        # Pre-build a Gaussian smoothing kernel if requested.
        if self.bpn_cfg.aux_target_sigma > 0:
            sigma = self.bpn_cfg.aux_target_sigma
            kernel_size = max(3, int(round(sigma * 6)) | 1)  # odd, ≥3
            half = kernel_size // 2
            xs = torch.arange(-half, half + 1, dtype=torch.float32)
            kernel = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            self.register_buffer(
                "_smooth_kernel", kernel.view(1, 1, -1)
            )
        else:
            self._smooth_kernel = None

    def set_epoch(self, epoch: int) -> None:
        """Trainer should call this once per epoch for aux warmup."""
        self._epoch = int(epoch)

    def _aux_weight_now(self) -> float:
        """Compute the effective aux weight at the current epoch."""
        if not self.bpn_cfg.aux_loss:
            return 0.0
        warmup = self.bpn_cfg.aux_warmup_epochs
        ramp = max(self.bpn_cfg.aux_ramp_epochs, 1)
        if self._epoch < warmup:
            return 0.0
        progress = min(1.0, (self._epoch - warmup) / ramp)
        return self.bpn_cfg.aux_weight * progress

    def _smooth_target(self, any_call: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to a binary (B, T) target."""
        if self._smooth_kernel is None:
            return any_call
        x = any_call.unsqueeze(1)                # (B, 1, T)
        pad = self._smooth_kernel.size(-1) // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        x = F.conv1d(x, self._smooth_kernel)
        return x.squeeze(1).clamp(0.0, 1.0)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = outputs["probs"].clamp(self.eps, 1 - self.eps)
        gate = outputs["mask"]

        # ----- Main loss: focal BCE on gated probabilities. -----
        bce = -(targets * torch.log(probs)
                + (1 - targets) * torch.log(1 - probs))

        if self.use_focal:
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.gamma
            bce = focal_mod * bce

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

        # ----- Aux loss: gate-vs-(any-call), with all four fixes. -----
        aux_w = self._aux_weight_now()
        if aux_w > 0:
            # any_call: 1 wherever any class label is positive.
            any_call = targets.max(dim=-1).values.float()  # (B, T)
            target = self._smooth_target(any_call)         # (B, T) in [0,1]

            # Trim to match the gate's T (it can be slightly shorter).
            T_gate = gate.size(1)
            if target.size(1) != T_gate:
                m = min(target.size(1), T_gate)
                target = target[:, :m]
                gate_t = gate[:, :m]
                if padding_mask is not None:
                    pad_t = padding_mask[:, :m]
                else:
                    pad_t = None
            else:
                gate_t = gate
                pad_t = padding_mask

            gate_clamped = gate_t.clamp(self.eps, 1 - self.eps)

            # Asymmetric BCE: pos_weight on the positive term
            # (recall-preserving), 1.0 on the negative term (FP-suppressing).
            recall_w = self.bpn_cfg.aux_recall_weight
            pos_w = self.bpn_cfg.aux_pos_weight
            pos_term = -target * torch.log(gate_clamped) * (pos_w * recall_w)
            neg_term = -(1 - target) * torch.log(1 - gate_clamped)
            aux = pos_term + neg_term

            if pad_t is not None:
                aux = aux * pad_t.float()
                aux_loss = aux.sum() / (pad_t.sum() + 1e-8)
            else:
                aux_loss = aux.mean()

            total = total + aux_w * aux_loss

        return total


# ======================================================================
# Class weights (re-exported for self-containment)
# ======================================================================

def compute_class_weights() -> torch.Tensor:
    """Per-class positive weights using the paper's formula."""
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
    Smoke test: build the v2 BPN model, push a dummy input through,
    print output shapes and parameter counts, and compare to the
    BPN-disabled variant.

        python model_bpn_v2.py
    """
    from spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor()

    # Paper-faithful default config.
    bpn_cfg = BPNConfigV2(
        enabled=True,
        taps=("feat_ext", "bottleneck", "depthwise"),
        proj_dim=128,
        out_dim=64,
        n_rois=32,
        pool_mode="softmax",
        init_mode="near_one",
        temporal="bilstm",
        aux_loss=True,
    )
    print(f"BPNConfigV2: {bpn_cfg.to_dict()}")
    model = WhaleVADBPNV2(num_classes=cfg.n_classes(), bpn_cfg=bpn_cfg)

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    out = model(spec)

    print(f"\n=== Forward pass shapes ===")
    print(f"Audio:    {audio.shape}")
    print(f"Spec:     {spec.shape}")
    print(f"Probs:    {out['probs'].shape}    "
          f"range=[{out['probs'].min():.3f}, {out['probs'].max():.3f}]")
    print(f"Logits:   {out['logits'].shape}")
    print(f"Mask:     {out['mask'].shape}     "
          f"range=[{out['mask'].min():.3f}, {out['mask'].max():.3f}]")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParams (with BPN v2): {n:,}")

    # Compare to BPN-disabled (architecture-changes-only run).
    bpn_cfg_off = BPNConfigV2(enabled=False)
    model_off = WhaleVADBPNV2(num_classes=cfg.n_classes(), bpn_cfg=bpn_cfg_off)
    _ = model_off(spec)
    n_off = sum(p.numel() for p in model_off.parameters() if p.requires_grad)
    print(f"Params (no BPN):      {n_off:,}")
    print(f"BPN v2 module overhead: {n - n_off:,}")

    # Sanity-check the loss with aux warmup behavior.
    print(f"\n=== Loss aux warmup schedule ===")
    criterion = WhaleVADBPNLossV2(bpn_cfg=bpn_cfg)
    targets = torch.zeros_like(out["probs"])
    targets[..., 0] = (torch.rand_like(targets[..., 0]) > 0.95).float()
    for ep in [0, 4, 5, 7, 10, 15]:
        criterion.set_epoch(ep)
        loss = criterion(out, targets)
        w = criterion._aux_weight_now()
        print(f"  epoch={ep:2d}  aux_weight_eff={w:.4f}  total_loss={loss.item():.4f}")
