"""
Transformer-Encoder Variant of WhaleVAD for Phase 2b
=====================================================

Phase 2b tests whether self-attention beats the 2-layer BiLSTM as the
temporal model on top of the WhaleVAD CNN frontend. The CNN frontend
is unchanged — we only swap rows 6-7 of the architecture (LSTM +
classifier) for a Transformer encoder + classifier.

Why this might help
-------------------
The BiLSTM has a fixed receptive field that grows with depth.
Capturing dependencies across distant parts of a 30-second segment
(1500 frames) requires the gradient to backpropagate through many
recurrent steps, which can lose long-range information.

Self-attention has constant path length between any two frames,
making it well-suited to long, slow phenomena like blue whale Z-calls
that span 20+ seconds. The model can directly attend from the start
of a call to its end.

Why it might NOT help
---------------------
1. Whale calls have local structure (frequency sweep, harmonics) that
   convolutions and recurrence already capture well. Attention is
   most useful when the discriminative information is in long-range
   relationships, which may not be the dominant pattern here.
2. With only 1M-ish parameters, attention can underfit on 1500-frame
   sequences if positional encoding isn't well-chosen.
3. Self-attention is O(T²) in memory; with T=1500 and batch=32, this
   is fine on a 2080 Ti (about 230 MB for the attention map alone)
   but won't scale to longer windows.

Architecture choices
--------------------
- ``num_layers=4``: matches the typical "small transformer" depth for
  audio tasks. Going deeper without more data risks overfitting.
- ``d_model=64`` matches ``cfg.PROJECTION_DIM``, so the CNN→temporal
  interface is dimension-preserving (no extra projection needed).
- ``nhead=4``: 16 dim per head, a reasonable sweet spot for d=64.
- ``dim_feedforward=256``: 4× ``d_model``, the canonical ratio.
- Sinusoidal absolute positional encoding (added to the projected
  features before the encoder). Learned positional embeddings are
  riskier for sequences this long without careful initialization.
- Classifier head: ``Linear(d_model, num_classes)`` — different shape
  than the BiLSTM variant (which had ``Linear(LSTM_HIDDEN*2,
  num_classes)``). Checkpoints are not interchangeable between the
  two models.

Implementation note
-------------------
Rather than copy WhaleVAD's CNN frontend code (~100 lines, brittle
to keep in sync), we instantiate the original ``WhaleVAD``, then
swap out its ``.lstm`` and ``.classifier`` attributes in-place. The
forward pass is identical up to the BiLSTM — we just intercept after
the projection layer and run the Transformer instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import config as cfg
from model import WhaleVAD


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You
    Need". Adds a fixed, deterministic position-dependent bias to each
    frame's feature vector, so the Transformer's order-invariant
    attention can recover frame ordering.

    For a 30-second segment at 50 fps, T=1500. The standard ``10000^(2i/d)``
    period scaling covers up to ~10^4 positions before periods repeat,
    which comfortably exceeds 1500.
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        # Precompute (max_len, d_model) lookup once. Stored as a non-trainable
        # buffer so it follows the model to GPU.
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, d_model)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, d_model)`` with positional bias added.
        """
        T = x.size(1)
        if T > self.max_len:
            raise RuntimeError(
                f"Sequence length {T} exceeds positional encoding max_len "
                f"{self.max_len}. Increase max_len in the constructor."
            )
        return x + self.pe[:, :T, :]


class WhaleVAD_Transformer(nn.Module):
    """
    WhaleVAD with the BiLSTM replaced by a Transformer encoder.

    Constructs a vanilla ``WhaleVAD`` to inherit the CNN frontend and
    projection layer, then ignores the original ``.lstm`` and
    ``.classifier`` attributes during forward pass — instead routing
    through a fresh Transformer encoder + classifier.

    Note: the original ``.lstm`` and ``.classifier`` modules still
    exist as attributes but receive no gradient (no forward pass goes
    through them). They could be deleted for parameter accounting, but
    keeping them avoids breaking any debug/inspection code that
    expects the standard WhaleVAD attribute layout. The training loop
    only optimises ``trainable=True`` parameters, which we set
    explicitly to exclude the dead attributes.
    """

    def __init__(
        self,
        num_classes: int = 3,
        feat_channels: int = 3,
        d_model: int = None,        # default: cfg.PROJECTION_DIM
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # CNN frontend: instantiate the original WhaleVAD and use its
        # CNN modules. We keep a reference to the inner model rather
        # than copying its modules so any future changes to the
        # CNN frontend in model.py are picked up automatically here.
        self._inner = WhaleVAD(
            num_classes=num_classes, feat_channels=feat_channels,
        )

        # Disable gradients on the dead BiLSTM and classifier so they
        # don't appear in the optimizer's parameter list. They'll be
        # excluded from ``parameters()`` by overriding it below.
        for p in self._inner.lstm.parameters():
            p.requires_grad = False
        for p in self._inner.classifier.parameters():
            p.requires_grad = False

        d_model = d_model or cfg.PROJECTION_DIM

        # Standard PyTorch transformer encoder. ``batch_first=True`` so
        # input shape matches the ``(B, T, d_model)`` convention used
        # by the rest of the model.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — more stable for deep transformers
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=2048)

        # Classifier maps from d_model directly. Smaller than the
        # BiLSTM variant (which had hidden_size * 2 = 256 input).
        self.classifier = nn.Linear(d_model, num_classes)

        # Match WhaleVAD's prior-matched bias init for the prevalence
        # ~5% of calls per frame. Exact constant matches model.py.
        nn.init.constant_(self.classifier.bias, -3.0)

    def parameters(self, recurse: bool = True):
        """
        Override ``parameters()`` so the optimizer sees only the
        modules we actually use: the CNN frontend, the projection
        layer (lazy-init'd inside ``_inner``), the transformer, the
        positional encoding (no learnable params), and the classifier.

        Excludes the dead BiLSTM and the original WhaleVAD classifier
        — they have ``requires_grad=False`` but listing them in
        ``parameters()`` would still let them eat optimizer state
        memory.
        """
        # Include all CNN frontend modules from the inner WhaleVAD
        # (filterbank, feat_extractor, residual_stack) plus its lazy
        # projection layer, but NOT its lstm/classifier.
        for name, p in self._inner.named_parameters(recurse=recurse):
            if name.startswith("lstm.") or name.startswith("classifier."):
                continue
            yield p
        for p in self.transformer.parameters(recurse=recurse):
            yield p
        for p in self.classifier.parameters(recurse=recurse):
            yield p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Mirrors ``WhaleVAD.forward`` up through the
        projection layer, then routes through the Transformer instead
        of the BiLSTM.
        """
        # CNN frontend (reuse the inner model's modules directly).
        x = self._inner.filterbank(spec)
        x = self._inner.feat_extractor(x)
        x = self._inner.residual_stack(x)

        # Permute to (B, T, C, F), flatten C×F.
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * Fr)

        # Lazy projection initialisation. The inner WhaleVAD does this
        # in its forward; since we bypass that forward, we need to
        # invoke it manually here.
        self._inner._init_projection(C * Fr, x.device)
        x = self._inner.feat_proj(x)            # (B, T, d_model)

        # Add sinusoidal positional encoding before attention.
        x = self.pos_enc(x)

        # Transformer encoder. No mask — we don't have causal
        # constraints here; bidirectional context is fine for offline
        # detection. If/when we move to streaming inference, add a
        # causal mask here.
        x = self.transformer(x)                 # (B, T, d_model)

        logits = self.classifier(x)             # (B, T, num_classes)
        return logits
