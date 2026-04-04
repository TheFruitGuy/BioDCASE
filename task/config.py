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

DATA_ROOT       = Path("/home/matthias-nagl/BioDCASE/task/2026_BioDCASE_development_set/")           # site-year folders with .wav + annotation.csv
OUTPUT_DIR      = Path("./runs")                # training checkpoints & logs
SUBMISSION_PATH = Path("./submission.csv")       # inference output


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

# Evaluation set — released June 1 2026
EVAL_DATASETS = [
    "kerguelen2020",
    "ddu2021",
]


# ──────────────────────────────────────────────────────────────────────────────
# Audio & labels
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 250       # Hz — fixed by ATBFL
FRAME_STRIDE_S  = 0.02      # 20 ms classification resolution
N_FFT           = 256       # → 129 frequency bins
WIN_LENGTH      = 250       # ~1 s window (matches low-freq whale calls)
HOP_LENGTH      = 5         # samples between frames → 20 ms at 250 Hz

CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
CALL_TYPES_3 = ["bmabz", "d", "bp"]
COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d",
    "bpd": "bp", "bp20": "bp", "bp20plus": "bp",
}

# Train on 3-class collapsed problem (recommended — +15 % F1 in Whale-VAD)
USE_3CLASS = True


# ──────────────────────────────────────────────────────────────────────────────
# Segment extraction
# ──────────────────────────────────────────────────────────────────────────────

COLLAR_MIN_S        = 1.0       # min random collar around each annotation
COLLAR_MAX_S        = 5.0       # max random collar
EVAL_SEGMENT_S      = 30.0      # fixed segment length at inference
EVAL_OVERLAP_S      = 2.0       # overlap between inference segments
MIN_CALL_DURATION_S = 0.5
MAX_CALL_DURATION_S = 30.0
NEG_RATIO           = 1.0       # neg-to-pos segments per training epoch


# ──────────────────────────────────────────────────────────────────────────────
# Model — Conformer
# ──────────────────────────────────────────────────────────────────────────────

D_MODEL     = 256       # embedding dimension
N_HEADS     = 4         # attention heads
D_FF        = 1024      # feed-forward inner dim
N_LAYERS    = 4         # Conformer blocks  (try 4–8)
CONV_KERNEL = 15        # depthwise conv kernel (try 15–31)
DROPOUT     = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

EPOCHS        = 60
BATCH_SIZE    = 16
LR            = 1e-4
WEIGHT_DECAY  = 0.001
WARMUP_EPOCHS = 5
GRAD_CLIP     = 1.0

FOCAL_ALPHA   = 0.25
FOCAL_GAMMA   = 2.0
FOCAL_WEIGHT  = 1.0        # weight of focal term relative to BCE

NUM_WORKERS   = 4
SEED          = 42


# ──────────────────────────────────────────────────────────────────────────────
# Postprocessing
# ──────────────────────────────────────────────────────────────────────────────

SMOOTH_KERNEL_MS = 500      # median filter kernel
MERGE_GAP_S      = 0.5      # merge detections closer than this
POST_MIN_DUR_S   = 0.5      # discard shorter
POST_MAX_DUR_S   = 30.0     # discard longer

DEFAULT_THRESHOLDS = [0.5, 0.5, 0.5]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def n_classes() -> int:
    return 3 if USE_3CLASS else 7

def class_names() -> list[str]:
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)

def class_to_idx() -> dict[str, int]:
    return {c: i for i, c in enumerate(class_names())}
