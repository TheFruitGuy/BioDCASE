"""
Triton-TIDE — Tap-Integrated Detection Enhancer
================================================

A novel false-positive-suppression module for Triton. Like the BPN of
Geldenhuys et al. (arXiv:2510.21280), TIDE taps intermediate feature
maps from the backbone CNN and uses them to produce a per-frame gating
mask in ``[0, 1]`` that multiplies the classifier's sigmoid output.
Unlike the BPN, TIDE deliberately strips out the ROI machinery:

    BPN  : taps → per-tap projection → stack → ConvTranspose proposal
           → R ROI vectors → per-ROI BiLSTM → learned weighted mean → gate
    TIDE : taps → per-tap projection → stack → cross-tap 2D conv
           → 1D temporal smoothing → linear → sigmoid → gate

Concretely the differences are:
  - **No ROI count R.** TIDE produces a single per-frame gate directly,
    rather than R intermediate ROI scores combined by a learned weighted
    mean. Removes one hyperparameter (n_rois) and one design choice
    (pool_mode).
  - **No ConvTranspose expansion.** The cross-tap mixing is a single
    2D conv that reduces the tap axis to 1.
  - **No BiLSTM in the gate.** Temporal context inside the gate comes
    from a small Conv1d (kernel=5, ~100 ms receptive field per layer).
    Lighter, fewer parameters, easier to interpret.
  - **Same backbone as Triton.** The Triton-TIDE variant changes the
    backbone *nothing*: bottleneck and depthwise aggregation are the
    same as ``model.Triton``. This isolates the contribution to the
    gate alone, so ablation against the Triton baseline is clean.

Forward output
--------------
::

    {
        "logits": (B, T, num_classes),   # raw classifier logits
        "probs":  (B, T, num_classes),   # gated probabilities, in [0, 1]
        "gate":   (B, T),                # the TIDE gate, in [0, 1]
    }

The training loop (``train.py``) reads ``probs`` directly for both
loss and validation; the loss class (``TritonTIDELoss``) optionally
also supervises ``gate`` against the per-frame "any-class-active"
target.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn

import config as cfg


# ======================================================================
# TIDE Configuration
# ======================================================================

@dataclass
class TIDEConfig:
    """
    Knobs for the TIDE gate.

    Attributes
    ----------
    enabled : bool
        Master switch. When False, the gate is bypassed and the model
        produces ``probs = sigmoid(logits)`` (and ``gate = 1`` everywhere).
        Useful for ablating "model output dict only" vs "model output
        dict + actual gating".
    n_taps : int
        How many intermediate feature maps to tap from the depthwise
        aggregation stack. The stack has 3 layers; the deepest
        ``n_taps`` outputs are used. Range 1–3.
    dim : int
        Internal channel dimension of the gate. Same role as
        ``BPNConfig.dim`` (paper's C_bpn).
    use_dilated_backbone : bool
        When True, the depthwise aggregation uses dilations (2, 4, 8)
        with a per-layer residual structure, matching the WhaleVAD-BPN
        paper's modified backbone. When False, the plain depthwise
        stack of the Triton baseline is used (with a single outer
        residual). The plain version isolates TIDE's contribution to
        the gate alone; the dilated version lets you ablate the
        backbone change separately from the gate.
    temporal : str
        Temporal modelling inside the gate:
          ``"conv1d"`` — Conv1d kernel=temporal_kernel (default 5,
                         ~100 ms RF). Default; lightweight.
          ``"bilstm"`` — BiLSTM (input_size=dim, hidden=
                         temporal_lstm_hidden), with a Linear back to
                         dim. Matches the BPN paper's "BPN BiLSTM"
                         option. Heavier but unbounded RF.
          ``"none"``   — Identity. Score head sees the cross-tap
                         output directly.
    temporal_kernel : int
        Conv1d kernel width when ``temporal == "conv1d"``. Must be odd.
    temporal_lstm_hidden : int
        BiLSTM hidden size when ``temporal == "bilstm"``. Output is
        ``2 * temporal_lstm_hidden`` after concat, projected back to
        ``dim`` by a Linear.
    init_mode : str
        Gate initialisation:
          ``"near_one"`` — bias the output toward 1.0 everywhere so the
                           model starts behaving like Triton without
                           gating, and learns to suppress FPs over
                           training.
          ``"random"``   — default Kaiming init.
    aux_loss : bool
        If True, ``TritonTIDELoss`` adds an auxiliary BCE on the gate
        against per-frame "any class active" targets. Has no effect on
        the model itself — just a flag the loss reads.
    aux_weight : float
        Weight of the auxiliary loss when ``aux_loss=True``.
    """
    enabled: bool = True
    n_taps: int = 3
    dim: int = 64
    use_dilated_backbone: bool = False
    temporal: str = "conv1d"
    temporal_kernel: int = 5
    temporal_lstm_hidden: int = 64
    init_mode: str = "near_one"
    aux_loss: bool = False
    aux_weight: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================
# Building blocks (duplicated from model.py for full isolation)
# ======================================================================

class _ResidualBlock(nn.Module):
    """Sum-connected residual container — verbatim copy of model.ResidualBlock."""

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class _DepthwiseStack(nn.Module):
    """
    Three-layer depthwise aggregation block in two modes.

    ``dilations=None`` — plain mode. Each layer is a depthwise Conv2d
    (3×3, dilation=1) followed by BN and GELU; **no** per-layer
    residual is added inside the stack. The outer ``TritonTIDE.forward``
    wraps the whole stack in a single residual ``y = x + stack(x)``,
    which is mathematically identical to the Triton baseline's
    aggregation block.

    ``dilations=(d1, d2, d3)`` — dilated mode. Each layer uses the
    corresponding dilation factor (with matching padding so the time
    dimension is preserved) and a **per-layer residual is added
    inside** the stack: ``x = x + layer(x)`` for each layer. The outer
    ``TritonTIDE.forward`` skips the outer residual in this mode. This
    matches the WhaleVAD-BPN paper's modified depthwise block
    (dilations 2, 4, 8 with residuals between each layer).

    Spatial dropout is applied once at the entry in either mode, as in
    the baseline.
    """

    def __init__(
        self, channels: int, dilations: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.dilations = dilations
        self.entry_drop = nn.Dropout2d(cfg.AGG_DROPOUT)

        ds = dilations if dilations is not None else (1, 1, 1)
        if len(ds) != 3:
            raise ValueError(
                f"_DepthwiseStack expects 3 dilation factors, got {ds!r}"
            )

        self.layers = nn.ModuleList()
        for d in ds:
            # padding=d keeps the spatial dims the same for a (3, 3) kernel.
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(3, 3),
                          stride=(1, 1), padding=(d, d),
                          dilation=(d, d), groups=channels),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ))

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.entry_drop(x)
        intermediates: list[torch.Tensor] = []
        for layer in self.layers:
            if self.dilations is not None:
                # Dilated mode: per-layer residual inside the stack.
                x = x + layer(x)
            else:
                # Plain mode: no internal residual (the outer model adds
                # a single residual around the whole stack).
                x = layer(x)
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x


# ======================================================================
# TIDE gate
# ======================================================================

class _TapHead(nn.Module):
    """
    Per-tap projection head: collapses (B, C, F, T) → (B, dim, T).

    Conv2d (1×1) + BN + GELU + max-pool along the frequency axis. The
    freq-axis pool kernel is sized lazily on the first forward pass —
    the backbone has a lazy feature-projection layer for the same
    reason (frequency dim depends on STFT settings).
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
        self.freq_pool: nn.MaxPool2d | None = None

    def _ensure_freq_pool(self, freq_dim: int) -> None:
        if self.freq_pool is None or self.freq_pool.kernel_size[0] != freq_dim:
            self.freq_pool = nn.MaxPool2d(
                kernel_size=(freq_dim, 1), stride=(freq_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, F, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        self._ensure_freq_pool(x.size(2))
        x = self.freq_pool(x)            # (B, C_out, 1, T)
        return x.squeeze(2)              # (B, C_out, T)


class TIDEGate(nn.Module):
    """
    Tap-Integrated Detection Enhancer.

    Pipeline:
      1. Project each intermediate feature map via its own ``_TapHead``.
         Outputs: ``n_taps`` × (B, dim, T).
      2. Stack along a new axis → (B, dim, n_taps, T).
      3. Cross-tap mix: Conv2d kernel=(n_taps, 1) reduces the tap
         axis to 1 → (B, dim, 1, T) → squeeze → (B, dim, T).
      4. Temporal modelling (configurable via ``tide_cfg.temporal``):
           - ``"conv1d"``: Conv1d kernel=temporal_kernel (default),
                          padding to preserve T → (B, dim, T).
           - ``"bilstm"``: permute to (B, T, dim), BiLSTM, Linear back
                          to dim, permute back. Matches the BPN paper's
                          BiLSTM-in-the-gate option.
           - ``"none"``:   Identity. Score head sees the cross-tap
                          output directly.
      5. Output linear (Conv1d 1×1, dim → 1) → (B, 1, T) → squeeze.
      6. Sigmoid → gate in [0, 1].

    The cross-tap mixing (step 3) is the structural distinction from
    the BPN. The BPN uses ConvTranspose2d to expand the H axis then
    adaptive-pools to R ROIs, producing multiple "views" of the
    temporal info that are then combined by a learned weighted mean
    over per-ROI sigmoids. TIDE collapses the tap axis to one in a
    single step — no R, no weighted mean. Step 4's BiLSTM mode brings
    TIDE's temporal capacity in line with the BPN's, isolating the
    cross-tap-mixing difference for ablation.

    Parameters
    ----------
    backbone_channels : int
        Channel count of the intermediate feature maps.
    tide_cfg : TIDEConfig
        Knobs.
    """

    def __init__(self, backbone_channels: int, tide_cfg: TIDEConfig):
        super().__init__()
        self.cfg = tide_cfg

        # 1. Per-tap projection heads — one independent set of weights
        #    per tap, all sharing the same architecture.
        self.heads = nn.ModuleList([
            _TapHead(backbone_channels, tide_cfg.dim)
            for _ in range(tide_cfg.n_taps)
        ])

        # 3. Cross-tap mixing conv. Reduces the tap axis to 1 in a
        #    single step.
        self.cross_tap_mix = nn.Sequential(
            nn.Conv2d(
                tide_cfg.dim, tide_cfg.dim,
                kernel_size=(tide_cfg.n_taps, 1),
                stride=(1, 1), padding=0,
            ),
            nn.BatchNorm2d(tide_cfg.dim),
            nn.GELU(),
        )

        # 4. Temporal modelling — three modes, configured at construction.
        #    The branches set the same set of attributes so that
        #    ``forward`` can dispatch on ``cfg.temporal`` cleanly.
        self.temporal_conv = None
        self.temporal_lstm = None
        self.temporal_proj = None

        if tide_cfg.temporal == "conv1d":
            k = tide_cfg.temporal_kernel
            if k % 2 == 0:
                raise ValueError(
                    f"TIDEConfig.temporal_kernel must be odd, got {k}."
                )
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(tide_cfg.dim, tide_cfg.dim,
                          kernel_size=k, stride=1, padding=k // 2),
                nn.BatchNorm1d(tide_cfg.dim),
                nn.GELU(),
            )
        elif tide_cfg.temporal == "bilstm":
            self.temporal_lstm = nn.LSTM(
                input_size=tide_cfg.dim,
                hidden_size=tide_cfg.temporal_lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            # Project the 2*hidden BiLSTM output back to ``dim`` so the
            # score head sees the same input size as in the other modes.
            self.temporal_proj = nn.Linear(
                2 * tide_cfg.temporal_lstm_hidden, tide_cfg.dim,
            )
        elif tide_cfg.temporal == "none":
            pass  # Identity — score head sees the cross-tap output.
        else:
            raise ValueError(
                f"TIDEConfig.temporal must be one of "
                f"'conv1d' | 'bilstm' | 'none', got {tide_cfg.temporal!r}."
            )

        # 5. Output linear (Conv1d 1×1, dim → 1).
        self.score = nn.Conv1d(tide_cfg.dim, 1, kernel_size=1)

        # 6. Near-one init: pin the output conv's bias to +3 so
        #    σ(score) ≈ 0.95 everywhere at t=0. The model starts
        #    behaving like Triton (gate is essentially the identity)
        #    and learns to gate FPs over training.
        if tide_cfg.init_mode == "near_one":
            nn.init.zeros_(self.score.weight)
            nn.init.constant_(self.score.bias, 3.0)
        # else: default (Kaiming) init.

    def forward(self, intermediates: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute the gate from ``n_taps`` intermediate feature maps.

        Parameters
        ----------
        intermediates : list[torch.Tensor]
            Each shape (B, C, F, T). ``len`` must equal ``cfg.n_taps``.

        Returns
        -------
        gate : torch.Tensor, shape (B, T)
            Per-frame gating values in [0, 1].
        """
        if len(intermediates) != self.cfg.n_taps:
            raise RuntimeError(
                f"TIDE expects {self.cfg.n_taps} intermediate feature maps, "
                f"got {len(intermediates)}"
            )

        # 1. Project each tap. List of (B, dim, T).
        head_outs = [h(im) for h, im in zip(self.heads, intermediates)]

        # 2. Stack along a new "tap" axis → (B, dim, n_taps, T).
        stacked = torch.stack(head_outs, dim=2)

        # 3. Cross-tap mix → (B, dim, 1, T) → (B, dim, T).
        x = self.cross_tap_mix(stacked).squeeze(2)

        # 4. Temporal modelling — dispatch on cfg.temporal.
        if self.cfg.temporal == "conv1d":
            x = self.temporal_conv(x)                # (B, dim, T)
        elif self.cfg.temporal == "bilstm":
            # BiLSTM wants (B, T, dim); we have (B, dim, T).
            x = x.transpose(1, 2)                    # (B, T, dim)
            x, _ = self.temporal_lstm(x)             # (B, T, 2*hidden)
            x = self.temporal_proj(x)                # (B, T, dim)
            x = x.transpose(1, 2)                    # (B, dim, T)
        # "none" → leave x unchanged.

        # 5. Score head + sigmoid.
        score = self.score(x).squeeze(1)             # (B, T)
        return torch.sigmoid(score)


# ======================================================================
# Full Triton-TIDE model
# ======================================================================

class TritonTIDE(nn.Module):
    """
    Triton + TIDE gate.

    Same backbone as ``model.Triton`` — filterbank, feature extractor,
    bottleneck, plain depthwise aggregation. The only architectural
    difference is that the depthwise aggregation here is the
    ``_DepthwiseStack`` module (which can return its intermediates)
    instead of the Sequential used in the baseline. The math through
    the backbone is identical to ``model.Triton`` in either case.

    The TIDE gate taps the deepest ``n_taps`` outputs of the depthwise
    stack and produces a per-frame mask in ``[0, 1]``. The mask
    multiplies ``sigmoid(logits)`` to produce the final ``probs``.

    Parameters
    ----------
    num_classes : int
    tide_cfg : TIDEConfig
        If ``tide_cfg.enabled=False``, the gate is bypassed and
        ``probs = sigmoid(logits)``, ``gate = 1``. The model output
        dict still has all three keys so the training loop can stay
        oblivious to the toggle.
    feat_channels : int
    """

    def __init__(
        self,
        num_classes: int = 3,
        tide_cfg: TIDEConfig | None = None,
        feat_channels: int = 3,
    ):
        super().__init__()
        self.tide_cfg = tide_cfg or TIDEConfig()
        self.num_classes = num_classes

        # ----- Backbone (matches model.Triton exactly) ----------------
        self.filterbank = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=cfg.FILTERBANK_OUT_CH,
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

        # Depthwise aggregation that exposes per-layer outputs. Two
        # modes, controlled by ``tide_cfg.use_dilated_backbone``:
        #   - plain   (default): no dilations, no per-layer residuals;
        #                        outer residual added in ``forward``.
        #   - dilated: dilations (2, 4, 8) with residuals between each
        #                        layer (matches the WhaleVAD-BPN paper).
        self.depthwise = _DepthwiseStack(
            channels=cfg.FEAT_EXTRACTOR_CH,
            dilations=(2, 4, 8) if self.tide_cfg.use_dilated_backbone else None,
        )

        # The bottleneck stays under a residual wrapper. The depthwise
        # block's residual is handled differently per mode (outer for
        # plain, internal for dilated — see ``forward``).
        self.bottleneck_stack = _ResidualBlock(bottleneck)

        # ----- TIDE gate (the only architectural addition) ------------
        if self.tide_cfg.enabled:
            if not 1 <= self.tide_cfg.n_taps <= 3:
                raise ValueError(
                    f"TIDEConfig.n_taps must be in [1, 3], "
                    f"got {self.tide_cfg.n_taps} (depthwise stack has 3 layers)."
                )
            self.tide = TIDEGate(
                backbone_channels=cfg.FEAT_EXTRACTOR_CH,
                tide_cfg=self.tide_cfg,
            )
        else:
            self.tide = None

        # ----- Lazy projection (same pattern as baseline) -------------
        self._proj_in_dim: int | None = None
        self.feat_proj: nn.Linear | None = None

        # ----- BiLSTM + classifier (identical to baseline) ------------
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
        # CNN front end: filterbank, feature extractor, bottleneck-with-residual.
        x = self.filterbank(spec)
        x = self.feat_extractor(x)
        x = self.bottleneck_stack(x)

        # Depthwise aggregation. Residual wiring depends on mode:
        #   - plain   (no dilations): the stack itself has no internal
        #                             residuals, so we add a single
        #                             outer one: ``y = x + dw(x)``.
        #   - dilated:                the stack already added per-layer
        #                             residuals internally, so the
        #                             outer residual is skipped (its
        #                             output already mixes with the
        #                             input via the internal skips).
        # ``return_intermediates=True`` exposes per-layer outputs so
        # TIDE can tap them.
        use_outer_residual = not self.tide_cfg.use_dilated_backbone
        if self.tide is not None:
            dw_out, intermediates = self.depthwise(x, return_intermediates=True)
            x = (x + dw_out) if use_outer_residual else dw_out
            taps = intermediates[-self.tide_cfg.n_taps:]
        else:
            dw_out = self.depthwise(x, return_intermediates=False)
            x = (x + dw_out) if use_outer_residual else dw_out
            taps = None

        # Flatten + project to BiLSTM input.
        B, C, Fr, T = x.shape
        x_lstm = x.permute(0, 3, 1, 2).contiguous()      # (B, T, C, F)
        x_lstm = x_lstm.view(B, T, C * Fr)
        self._init_projection(C * Fr, x.device)
        x_lstm = self.feat_proj(x_lstm)

        x_lstm, _ = self.lstm(x_lstm)                    # (B, T, 256)
        logits = self.classifier(x_lstm)                  # (B, T, num_classes)

        # Apply TIDE gate (if enabled).
        if self.tide is not None:
            gate = self.tide(taps)                       # (B, T)
            # Defensive time-axis alignment: the gate's T can be ±1
            # from logits T because of cumulative arithmetic in the
            # pool/conv stack. Take the prefix-min so both align.
            T_logits = logits.size(1)
            T_gate = gate.size(1)
            if T_logits != T_gate:
                m = min(T_logits, T_gate)
                logits = logits[:, :m]
                gate = gate[:, :m]
            probs = torch.sigmoid(logits) * gate.unsqueeze(-1)
        else:
            gate = torch.ones(
                logits.size(0), logits.size(1), device=logits.device,
            )
            probs = torch.sigmoid(logits)

        return {"logits": logits, "probs": probs, "gate": gate}


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Smoke test: build with TIDE enabled, then disabled. Forward a
    # dummy batch each way, print shapes and parameter counts.
    # Run as:  python model_tide.py
    from spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor()

    print("--- TIDE enabled ---")
    tide_cfg = TIDEConfig(enabled=True, n_taps=3, dim=64, temporal_kernel=5)
    print(f"TIDEConfig: {tide_cfg.to_dict()}")
    model = TritonTIDE(num_classes=cfg.n_classes(), tide_cfg=tide_cfg)

    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    spec = extractor(audio)
    out = model(spec)
    print(f"Audio:    {audio.shape}")
    print(f"Spec:     {spec.shape}")
    print(f"Logits:   {out['logits'].shape}")
    print(f"Probs:    {out['probs'].shape}   "
          f"range=[{out['probs'].min():.3f}, {out['probs'].max():.3f}]")
    print(f"Gate:     {out['gate'].shape}    "
          f"range=[{out['gate'].min():.3f}, {out['gate'].max():.3f}]")
    n_on = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params (with TIDE): {n_on:,}")

    print("\n--- TIDE disabled (ablation) ---")
    model_off = TritonTIDE(
        num_classes=cfg.n_classes(),
        tide_cfg=TIDEConfig(enabled=False),
    )
    _ = model_off(spec)
    n_off = sum(p.numel() for p in model_off.parameters() if p.requires_grad)
    print(f"Params (no TIDE):   {n_off:,}")
    print(f"TIDE module cost:   {n_on - n_off:,}")
