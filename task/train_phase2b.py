"""
Phase 2b: Transformer Encoder Replacement for the BiLSTM
=========================================================

Single-axis change to the F1=0.474 baseline (``train.py``):
the 2-layer BiLSTM is replaced by a 4-layer Transformer encoder
(d_model=64, nhead=4, dim_feedforward=256) with sinusoidal
positional encoding. Same data, same loss, same CNN frontend, no
augmentation.

What this tests
---------------
Whether self-attention with constant-length attention paths beats
the BiLSTM's growing-receptive-field recurrence on 30-second
sequences (1500 frames). If 2b > 2a at similar parameter counts,
the win is from attention specifically, not from raw capacity.

Architecture summary
--------------------
  CNN frontend (unchanged from WhaleVAD)
    ↓ (B, T, 64)        ← matches cfg.PROJECTION_DIM
  Sinusoidal PE
    ↓
  TransformerEncoder × 4 layers
    - d_model = 64
    - nhead = 4              (16 dim per head)
    - dim_feedforward = 256  (4× d_model, canonical)
    - dropout = 0.1
    - GELU activation
    - pre-norm (more stable than post-norm for deep transformers)
    ↓ (B, T, 64)
  Linear classifier → (B, T, num_classes)

Parameter accounting (rough)
----------------------------
- CNN frontend (shared): ~750k
- Transformer (4 × ~50k): ~200k
- Classifier: ~200 (3-class) or ~450 (7-class)
- Total: ~950k
  vs. WhaleVAD BiLSTM (~1.03M) — slightly smaller, different
  computation pattern. If 2b matches 2a's F1 with fewer parameters,
  attention is more efficient than recurrence here.

Implementation note
-------------------
The model lives in ``model_transformer.py`` as
``WhaleVAD_Transformer``. It instantiates a vanilla ``WhaleVAD``
internally to inherit the CNN frontend and lazy-projection layer,
then routes the projected features through the Transformer encoder
instead of the BiLSTM. The dead BiLSTM and original classifier
attributes still exist but are excluded from the optimizer's
parameter list (see ``WhaleVAD_Transformer.parameters``).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase2b.py
"""

import torch.nn as nn

from model_transformer import WhaleVAD_Transformer
from phase1_baseline import run_phase1_training


PHASE2B_CONFIG = {
    "arch_change": "transformer_encoder",
    "tx_d_model": 64,           # matches cfg.PROJECTION_DIM
    "tx_nhead": 4,              # 16 dim per head
    "tx_num_layers": 4,
    "tx_dim_feedforward": 256,  # 4× d_model
    "tx_dropout": 0.1,
    "tx_norm_first": True,  # pre-norm
    "tx_pos_encoding": "sinusoidal_absolute",
    "lstm_hidden_baseline": 128,
    "lstm_layers_baseline": 2,
}


def _factory(num_classes: int) -> nn.Module:
    """
    Construct the Transformer-encoder variant. Closure over the
    Phase 2b hyperparameters so callers don't need to know them.
    """
    return WhaleVAD_Transformer(
        num_classes=num_classes,
        d_model=PHASE2B_CONFIG["tx_d_model"],
        nhead=PHASE2B_CONFIG["tx_nhead"],
        num_layers=PHASE2B_CONFIG["tx_num_layers"],
        dim_feedforward=PHASE2B_CONFIG["tx_dim_feedforward"],
        dropout=PHASE2B_CONFIG["tx_dropout"],
    )


# A wrapped name so wandb logs something readable (otherwise it'd see
# the closure name "_factory").
_factory.__name__ = "WhaleVAD_Transformer_factory"


if __name__ == "__main__":
    print(f"Phase 2b: Transformer encoder "
          f"d_model={PHASE2B_CONFIG['tx_d_model']}, "
          f"layers={PHASE2B_CONFIG['tx_num_layers']}, "
          f"nhead={PHASE2B_CONFIG['tx_nhead']}")
    run_phase1_training(
        phase_name="2b",
        augmentation_fn=None,
        augmentation_config=None,
        model_factory=_factory,
        model_factory_config=PHASE2B_CONFIG,
    )
