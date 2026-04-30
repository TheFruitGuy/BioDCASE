"""
Whale-VAD Model Architecture
============================

Reproduction of the CNN-BiLSTM classifier from Geldenhuys et al. (DCASE 2025).
The architecture follows Table 2 of the paper exactly, with the module layout
summarized below:

    Input spectrogram (B, 3, F, T)              phase-aware STFT
        │
        ▼  Filterbank (Conv2d 7×1, stride 3×1)  learnable frequency filterbank
        │
        ▼  FeatureExtractor                     2× (Conv → BN → GELU → MaxPool)
        │
        ▼  ResidualBlock                        bottleneck + depthwise aggregation
        │
        ▼  Flatten (C × F) → Linear             per-frame projection to 64-dim
        │
        ▼  BiLSTM (2 layers, hidden=128)        temporal context modelling
        │
        ▼  Linear classifier                    num_classes logits per frame
        │
    Output logits (B, T, num_classes)           per-frame predictions

The model is designed for per-frame detection: every 20 ms of input audio
receives its own classification output, enabling precise event-level
localization via the post-processing pipeline.

Also defined here:
    - ``WhaleVADLoss``: weighted BCE + optional focal loss (Section 5.6)
    - ``compute_class_weights``: file-level class weighting formula
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ======================================================================
# Building blocks
# ======================================================================

class ResidualBlock(nn.Module):
    """
    Sum-connected residual container.

    Unlike classical ResNet blocks, this module applies each sub-block in
    sequence and adds its output back to the running tensor:

        x → x + block_1(x) → (x + block_1(x)) + block_2(x + block_1(x)) → ...

    This matches the reference implementation and provides a simple way to
    chain arbitrary blocks with residual connections without hand-wiring
    each skip connection.

    Parameters
    ----------
    *blocks : nn.Module
        One or more sub-modules to be applied with residual connections.
        Each sub-module must preserve the channel/spatial dimensions of its
        input so that the additive skip is shape-compatible.
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

class WhaleVAD(nn.Module):
    """
    Whale-VAD per-frame classifier (Table 2 of the paper).

    The model consumes a three-channel phase-aware spectrogram and outputs
    per-frame multi-label logits for the requested number of classes. The
    architecture is deliberately lightweight (~1M parameters) to enable
    training on modest hardware while remaining expressive enough for the
    low-frequency, temporally structured whale-call detection task.

    Parameters
    ----------
    num_classes : int, default=3
        Number of output classes. Use 3 for the coarse challenge task,
        7 for fine-grained call-type classification, or 1 for binary
        per-class models (see ``train_binary.py``).
    feat_channels : int, default=3
        Number of input spectrogram channels. Three is the standard setting
        (magnitude, cos-phase, sin-phase).

    Notes
    -----
    The projection layer from flattened CNN features to the BiLSTM input is
    created lazily on the first forward pass. This is because the input
    frequency dimension after the filterbank depends on the STFT settings,
    which it is cleaner to discover at runtime than to compute analytically.
    As a consequence, callers must perform a dummy forward pass *before*
    calling ``load_state_dict`` on a saved checkpoint so that the projection
    layer exists and can receive its stored weights. The training and
    inference entry points handle this automatically.
    """

    def __init__(self, num_classes: int = 3, feat_channels: int = 3):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Learnable filterbank (Table 2, row 1)
        # ------------------------------------------------------------------
        # A single Conv2d with a (7, 1) kernel that operates only along the
        # frequency axis. The stride of 3 along frequency compresses the
        # spectral resolution by 3× while keeping all time frames intact.
        # This acts as a learnable analogue to a mel filterbank, discovering
        # whale-call-specific frequency groupings during training.
        self.filterbank = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=cfg.FILTERBANK_OUT_CH,
            kernel_size=(7, 1),
            stride=(3, 1),
            padding=0,
        )

        # ------------------------------------------------------------------
        # 2. Feature extractor (Table 2, rows 2-3)
        # ------------------------------------------------------------------
        # Two Conv-BN-GELU-MaxPool blocks, each pooling further along the
        # frequency axis. Time resolution is preserved throughout via
        # stride-1 pooling, so the output still has one frame per 20 ms.
        self.feat_extractor = nn.Sequential(
            # Block 1: 64 → 128 channels, ~3× frequency downsampling
            nn.Conv2d(
                cfg.FILTERBANK_OUT_CH,
                cfg.FEAT_EXTRACTOR_CH,
                kernel_size=(5, 5),
                stride=(3, 1),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(cfg.FEAT_EXTRACTOR_CH),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(5, 1), stride=1, padding=0),
            # Block 2: 128 → 128 channels, further frequency smoothing
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

        # ------------------------------------------------------------------
        # 3. Bottleneck block (Table 2, rows 4-6)
        # ------------------------------------------------------------------
        # Squeeze-expand pattern: 128 → 64 → 64 → 128 with 1×1, 3×3, 1×1
        # convolutions. Reduces parameter count and acts as an information
        # bottleneck that forces the network to compact its spectral
        # representation.
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

        # ------------------------------------------------------------------
        # 4. Depthwise convolutional aggregation (Table 2, rows 7-9)
        # ------------------------------------------------------------------
        # Three consecutive depthwise 3×3 convolutions (groups=channels).
        # Each channel processes its own spatial context independently,
        # which dramatically reduces parameters relative to a full Conv2d
        # while still enlarging the temporal receptive field.
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

        # Chain bottleneck and aggregation with skip connections. Note that
        # we don't keep bottleneck/aggregation as separate attributes — they
        # would be registered twice in the state_dict (once under their
        # attribute name, once under residual_stack.blocks.N), bloating
        # checkpoints and confusing key-remap logic when loading external
        # checkpoints.
        self.residual_stack = ResidualBlock(bottleneck, aggregation)

        # ------------------------------------------------------------------
        # 5. Projection (lazy-initialized)
        # ------------------------------------------------------------------
        # After the CNN stack, we flatten the (channels × frequency) axes and
        # project down to PROJECTION_DIM. The input dimension depends on the
        # frequency resolution after all previous layers, which we discover
        # on the first forward pass.
        self._proj_in_dim = None
        self.feat_proj = None

        # ------------------------------------------------------------------
        # 6. BiLSTM temporal model
        # ------------------------------------------------------------------
        # Two stacked bidirectional LSTM layers add temporal context to each
        # frame. The paper specifies dropout of 50% between layers (Section
        # 5.3). Output dim is 2 × LSTM_HIDDEN because of bidirectionality.
        self.lstm = nn.LSTM(
            input_size=cfg.PROJECTION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT,
        )

        # ------------------------------------------------------------------
        # 7. Classifier head
        # ------------------------------------------------------------------
        # Linear projection from the BiLSTM output to per-class logits.
        self.classifier = nn.Linear(cfg.LSTM_HIDDEN * 2, num_classes)

        # Prior-matched bias initialization. With a bias of -3.0, the
        # sigmoid output starts at σ(-3) ≈ 0.047, which is close to the
        # ~5% prior probability of a call being present at any given frame.
        # Without this correction, the model wastes many epochs just
        # learning to predict "no call most of the time" before it can
        # start learning useful features. This trick is not in the paper
        # but is standard practice for imbalanced binary classification.
        # nn.init.constant_(self.classifier.bias, -3.0)

    def _init_projection(self, in_dim: int, device: torch.device):
        """
        Create the projection layer on the first forward pass, once the
        CNN output dimensionality is known. Safe to call repeatedly: only
        acts when the current layer's input dim doesn't match ``in_dim``.
        """
        if self.feat_proj is None or self.feat_proj.in_features != in_dim:
            self.feat_proj = nn.Linear(in_dim, cfg.PROJECTION_DIM).to(device)
            self._proj_in_dim = in_dim

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Run the full forward pass.

        Parameters
        ----------
        spec : torch.Tensor
            Phase-aware spectrogram of shape ``(B, 3, F, T)``, typically
            produced by ``SpectrogramExtractor``.

        Returns
        -------
        torch.Tensor
            Per-frame logits of shape ``(B, T, num_classes)``. Apply
            ``sigmoid`` to convert to probabilities.
        """
        # CNN front end
        x = self.filterbank(spec)                   # (B, 64, F', T)
        x = self.feat_extractor(x)                  # (B, 128, F'', T)
        x = self.residual_stack(x)                  # (B, 128, F'', T)

        # Rearrange to (B, T, C, F) and flatten to (B, T, C*F) for linear
        # projection. We permute so that the time dimension becomes the
        # sequence axis expected by the BiLSTM.
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, T, C, F)
        x = x.view(B, T, C * Fr)                    # (B, T, C*F)

        # Lazy projection initialization (first forward pass only).
        self._init_projection(C * Fr, x.device)
        x = self.feat_proj(x)                       # (B, T, 64)

        # Temporal model
        x, _ = self.lstm(x)                         # (B, T, 256)

        # Per-frame classifier
        logits = self.classifier(x)                 # (B, T, num_classes)
        return logits


# ======================================================================
# Loss function
# ======================================================================

class WhaleVADLoss(nn.Module):
    """
    Weighted BCE with optional focal modulation.

    The paper (Section 5.6) uses per-class positive weights to counteract
    the severe class imbalance (most frames are negative), optionally
    combined with focal loss to further emphasize hard examples. In our
    reproduction, enabling both simultaneously produced unstable training,
    so focal loss is disabled by default via the ``USE_FOCAL_LOSS`` config
    flag.

    Parameters
    ----------
    pos_weight : torch.Tensor, optional
        Per-class positive weights, shape ``(num_classes,)``. Computed by
        ``compute_class_weights``. If ``None``, standard BCE is used.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        if pos_weight is not None:
            # Register as buffer so it moves with .to(device)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.alpha = cfg.FOCAL_ALPHA
        self.gamma = cfg.FOCAL_GAMMA
        self.use_focal = cfg.USE_FOCAL_LOSS

    def forward(
        self,
        logits: torch.Tensor,                           # (B, T, C)
        targets: torch.Tensor,                          # (B, T, C)
        padding_mask: torch.Tensor | None = None,       # (B, T) bool
    ) -> torch.Tensor:
        """
        Compute the loss for a mini-batch.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs, shape ``(B, T, C)``.
        targets : torch.Tensor
            Binary per-frame labels, shape ``(B, T, C)``.
        padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(B, T)`` where ``True`` marks valid
            frames. Padded frames (after collation to max length) are
            excluded from the loss. If ``None``, all frames contribute
            equally.

        Returns
        -------
        torch.Tensor
            Scalar loss averaged over all valid frames and classes.
        """
        pw = self.pos_weight.view(1, 1, -1) if self.pos_weight is not None else None

        # Element-wise BCE (no reduction) so we can apply focal modulation
        # and masking afterwards.
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )

        if self.use_focal:
            # Focal loss reweights each sample by (1 - p_t)^γ, where p_t is
            # the probability assigned to the correct class. Easy, confident
            # predictions receive a smaller weight; hard, wrong predictions
            # receive a larger one.
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.gamma
            loss = focal_mod * bce
        else:
            loss = bce

        if padding_mask is not None:
            # Exclude padded frames from the mean. Expand the mask to match
            # the class dimension, then normalize by the count of valid
            # frame-class elements.
            mask = padding_mask.unsqueeze(-1).float()
            loss = loss * mask
            denom = mask.sum() * logits.size(-1) + 1e-8
            return loss.sum() / denom

        return loss.mean()


# ======================================================================
# Class weight computation
# ======================================================================

def compute_class_weights() -> torch.Tensor:
    """
    Compute per-class positive weights using the paper's formula.

    The paper defines ``w_c = N / P_c`` where ``N`` is the number of negative
    segments and ``P_c`` is the number of positive segments for class c.
    We interpret "segment" here as a distinct audio file containing at least
    one call of class c, which yields sensible weights for the BioDCASE
    dataset: classes appearing in fewer files receive proportionally higher
    weight, so the loss does not collapse to "predict bmabz everywhere".

    The raw weights are normalized so that the minimum weight equals 1,
    preserving the ratios between classes while ensuring no class is
    *actively* downweighted.

    Returns
    -------
    torch.Tensor
        Per-class weight tensor of shape ``(num_classes,)``, with the
        minimum weight normalized to 1.
    """
    # Local import to avoid circular dependency (dataset imports model
    # indirectly through its own typing annotations).
    from dataset import load_annotations

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    total_files = annotations.groupby(["dataset", "filename"]).ngroups

    class_names = cfg.class_names()
    weights = []
    for c_name in class_names:
        # Map coarse name back to the set of fine-grained labels it contains,
        # so we can count the relevant annotations in the raw CSV.
        if cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]

        # Count positive files (at least one annotation of class c) and
        # treat the rest as negatives for weight computation purposes.
        p_c = max(class_annots.groupby(["dataset", "filename"]).ngroups, 1)
        n_neg = max(total_files - p_c, 1)
        w = n_neg / p_c
        weights.append(w)

    result = torch.tensor(weights, dtype=torch.float32)
    # Normalize so the minimum weight is 1.0. Without this, the common class
    # (bmabz) would be actively downweighted (weight < 1), slowing down
    # learning on the class with the most training signal.
    result = result / result.min()

    print(f"Class weights (w_c = N/P_c, normalized to min=1):")
    for name, w in zip(class_names, result.tolist()):
        print(f"  {name}: {w:.3f}")
    return result


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Quick shape and parameter-count sanity check. Run:
    #   python model.py
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
