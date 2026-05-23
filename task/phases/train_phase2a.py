"""
Phase 2a: Bigger BiLSTM Capacity Test
======================================

Single-axis change to the F1=0.474 baseline (``train.py``):
the BiLSTM is enlarged from (hidden=128, layers=2) to
(hidden=256, layers=3). Same data, same loss, same augmentation
(none — this is an architecture experiment).

Why this matters
----------------
Before swapping to a fundamentally different temporal model
(Transformer, Conformer, Mamba), we need to know whether the
existing architecture is just too small. If a bigger BiLSTM hits
F1 > 0.49, capacity is the bottleneck and there's still room in
the recurrent paradigm. If a bigger BiLSTM does *not* help, the
LSTM family is saturated and an attention-based model has a real
shot at a clean improvement.

This is the cheapest-to-implement Phase 2 test — no new code, just
config overrides at startup. Same training time as the baseline.

Why these specific values
-------------------------
- ``LSTM_HIDDEN = 256`` doubles per-direction capacity. Pushing to
  512 would risk overfitting on this dataset size (~58k positive
  segments), and parameter count would roughly quadruple.
- ``LSTM_LAYERS = 3`` adds one layer. Going to 4+ on 1500-frame
  sequences without warm-up is unstable in our experience and
  Geldenhuys' too.
- These changes roughly 2.7× the BiLSTM parameter count
  (≈260K → ≈700K), bringing the total model from ~1M to ~1.4M
  parameters. Still small by modern standards, well within
  capacity of a 2080 Ti.

The classifier head input dim auto-scales because it reads
``cfg.LSTM_HIDDEN * 2`` from the config — no other code changes
needed.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase2a.py
"""

import config as cfg
from phase1_baseline import run_phase1_training


# Override LSTM hyperparameters BEFORE any model construction.
# These values become the new defaults for any code path that reads
# cfg.LSTM_HIDDEN / cfg.LSTM_LAYERS — including model.py's WhaleVAD
# constructor and the parameter-count print in run_phase1_training.
PHASE2A_LSTM_HIDDEN = 256
PHASE2A_LSTM_LAYERS = 3

# Module-level config flips. These have to happen before any module
# that imports cfg attempts to use the LSTM hyperparameters. Doing
# the override at import time of this script is sufficient because
# phase1_baseline.run_phase1_training does the WhaleVAD construction
# inside its own scope.
cfg.LSTM_HIDDEN = PHASE2A_LSTM_HIDDEN
cfg.LSTM_LAYERS = PHASE2A_LSTM_LAYERS


PHASE2A_CONFIG = {
    "arch_change": "bigger_bilstm",
    "lstm_hidden": PHASE2A_LSTM_HIDDEN,
    "lstm_layers": PHASE2A_LSTM_LAYERS,
    "lstm_hidden_baseline": 128,
    "lstm_layers_baseline": 2,
}


if __name__ == "__main__":
    print(f"Phase 2a: LSTM hidden {PHASE2A_LSTM_HIDDEN} "
          f"({2 * PHASE2A_LSTM_HIDDEN} bidirectional output dim), "
          f"layers={PHASE2A_LSTM_LAYERS}")
    run_phase1_training(
        phase_name="2a",
        augmentation_fn=None,
        augmentation_config=None,
        model_factory_config=PHASE2A_CONFIG,
    )
