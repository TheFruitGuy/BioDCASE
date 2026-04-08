"""
Whale-Conformer — Central Configuration
========================================
Edit this single file to change paths, splits, model, training, and postprocessing.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT       = Path("/home/matthias-nagl/BioDCASE/task/2026_BioDCASE_development_set/")
OUTPUT_DIR      = Path("./runs")
SUBMISSION_PATH = Path("./submission.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset splits
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_DATASETS = [
    "ballenyislands2015",
    "casey2014",
    "elephantisland2013",
    "elephantisland2014",
    "greenwich2015",
    "kerguelen2005",
    "maudrise2014",
    "rosssea2014",
]

VAL_DATASETS = [
    "casey2017",
    "kerguelen2014",
    "kerguelen2015",
]

EVAL_DATASETS = [
    "kerguelen2020",
    "ddu2021",
]


# ──────────────────────────────────────────────────────────────────────────────
# Audio & labels
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 250
FRAME_STRIDE_S  = 0.02      # 20 ms classification resolution
N_FFT           = 256
WIN_LENGTH      = 250       # ~1 s window
HOP_LENGTH      = 5         # 20 ms at 250 Hz

CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
CALL_TYPES_3 = ["bmabz", "d", "bp"]
COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d",
    "bpd": "bp", "bp20": "bp", "bp20plus": "bp",
}

USE_3CLASS = True


# ──────────────────────────────────────────────────────────────────────────────
# Segment extraction
# ──────────────────────────────────────────────────────────────────────────────

COLLAR_MIN_S        = 1.0
COLLAR_MAX_S        = 5.0
EVAL_SEGMENT_S      = 30.0
EVAL_OVERLAP_S      = 2.0
MIN_CALL_DURATION_S = 0.5
MAX_CALL_DURATION_S = 30.0
NEG_RATIO           = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Model — Conformer
# ──────────────────────────────────────────────────────────────────────────────

D_MODEL     = 256
N_HEADS     = 4
D_FF        = 1024
N_LAYERS    = 4
CONV_KERNEL = 15
DROPOUT     = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

EPOCHS        = 60
BATCH_SIZE    = 16
LR            = 3e-4
WEIGHT_DECAY  = 0.001
WARMUP_EPOCHS = 3
GRAD_CLIP     = 1.0

# Loss — CRITICAL FIXES from previous version:
#
# 1) pos_weight: upweights the positive (call-present) term in BCE.
#    With ~5% positive frames, pos_weight ≈ 19 balances the classes.
#    Set to None to auto-compute from actual training data.
POS_WEIGHT    = 15

# 2) focal_alpha > 0.5 upweights positives (minority class).
#    NOTE: alpha=0.25 DOWNWEIGHTS positives — was backwards before!
FOCAL_ALPHA   = 0.75        # was 0.25 — that killed positive predictions
FOCAL_GAMMA   = 2.0
FOCAL_WEIGHT  = 0

NUM_WORKERS   = 24
SEED          = 42


# ──────────────────────────────────────────────────────────────────────────────
# Postprocessing
# ──────────────────────────────────────────────────────────────────────────────

SMOOTH_KERNEL_MS = 1000
MERGE_GAP_S      = 2.0
POST_MIN_DUR_S   = 0.5
POST_MAX_DUR_S   = 30.0

# Lower default — the model's probabilities are low early in training
DEFAULT_THRESHOLDS = [0.1, 0.1, 0.1]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def n_classes() -> int:
    return 3 if USE_3CLASS else 7

def class_names() -> list[str]:
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)

def class_to_idx() -> dict[str, int]:
    return {c: i for i, c in enumerate(class_names())}
