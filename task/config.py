"""
Whale-VAD Reproduction — Configuration
======================================
All hyperparameters match the paper (Geldenhuys et al., DCASE 2025)
and their public GitHub code: github.com/CMGeldenhuys/Whale-VAD

Paper sections referenced in comments:
  - 5.1 Preprocessing
  - 5.2 Feature extractor
  - 5.4 Whale-VAD model
  - 5.5 Stochastic negative undersampling
  - 5.6 Loss function
  - 5.8 Postprocessing
"""

from pathlib import Path


# ----------------------------------------------------------------------
# Paths — EDIT THESE FOR YOUR SETUP
# ----------------------------------------------------------------------

DATA_ROOT       = Path("/home/matthias-nagl/BioDCASE/task/2026_BioDCASE_development_set/")
OUTPUT_DIR      = Path("./runs")
SUBMISSION_PATH = Path("./submission.csv")

# Optional: path to contrastive-pretrained encoder weights.
# If None, train from scratch (paper reproduction).
PRETRAINED_PATH = None

# Optional: path to unlabeled audio for self-supervised pretraining
UNLABELED_DATA_DIR = Path("./data_unlabeled/combined")


# ----------------------------------------------------------------------
# Dataset splits (Table 1 in paper)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Audio & spectrogram (Section 5.2)
# ----------------------------------------------------------------------

SAMPLE_RATE     = 250
FRAME_STRIDE_S  = 0.02                        # 20 ms classification resolution

# Paper: 256-point FFT, ~1 s frame length, 20 ms stride
# Their spectrogram.py: n_fft=256, win_length=256, hop_length=5
N_FFT           = 256
WIN_LENGTH      = 256                         # paper: ~1 s at 250 Hz
HOP_LENGTH      = 5                           # 20 ms at 250 Hz

# Paper Section 5.2: "mean spectral and cepstral subtraction... mean computed
# independently for each frequency bin over the duration of the segment."
# Their weights.py: norm_features="demean", complex_repr="trig"
NORM_FEATURES   = "demean"                    # per-frequency mean subtraction
COMPLEX_REPR    = "trig"                      # [mag, cos θ, sin θ] → 3 channels


# ----------------------------------------------------------------------
# Classes (Section 4: 7 call types → 3-class evaluation)
# ----------------------------------------------------------------------

CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
CALL_TYPES_3 = ["bmabz", "d", "bp"]

COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d",
    "bpd": "bp", "bp20": "bp", "bp20plus": "bp",
}

# Paper: "+ Three class" gave +15.2% F1 improvement
USE_3CLASS = True


# ----------------------------------------------------------------------
# Segmentation (Section 5.1)
# ----------------------------------------------------------------------

# Training: center call in segment with random collar before/after
COLLAR_MIN_S        = 1.0
COLLAR_MAX_S        = 5.0

# Evaluation: fixed 30s windows with 2s overlap
EVAL_SEGMENT_S      = 30.0
EVAL_OVERLAP_S      = 2.0

# Discard unreasonable annotations
MIN_CALL_DURATION_S = 0.5
MAX_CALL_DURATION_S = 30.0


# ----------------------------------------------------------------------
# Negative undersampling (Section 5.5)
# ----------------------------------------------------------------------

# Paper: "approximately as many negative as positive segments per mini-batch"
# Each epoch samples a different subset of negatives
NEG_RATIO = 1.0


# ----------------------------------------------------------------------
# Model architecture (Section 5.4 + Table 2)
# ----------------------------------------------------------------------

# These are from their model.py, kept identical for reproduction
FILTERBANK_OUT_CH   = 64
FEAT_EXTRACTOR_CH   = 128
BOTTLENECK_CH       = 64
PROJECTION_DIM      = 64
LSTM_HIDDEN         = 128
LSTM_LAYERS         = 2
LSTM_DROPOUT        = 0.5                     # Section 5.3: "20%-50%"
BOTTLENECK_DROPOUT  = 0.1
AGG_DROPOUT         = 0.2                     # Section 5.4: "Dropout2d"


# ----------------------------------------------------------------------
# Training (Section 5.6)
# ----------------------------------------------------------------------

EPOCHS        = 60
BATCH_SIZE    = 32
LR            = 1e-5                          # paper: "1 × 10⁻⁵"
WEIGHT_DECAY  = 0.001                         # paper: "weight decay factor of 0.001"
BETA1         = 0.9                           # paper momentum term
BETA2         = 0.999                         # paper momentum term
GRAD_CLIP     = 1.0


# ----------------------------------------------------------------------
# Loss (Section 5.6)
# ----------------------------------------------------------------------

# Paper uses BOTH weighted BCE AND focal loss (focal on top of BCE)
# Weights: w_c = N / P_c where N = # negative segments, P_c = # positive for class c
USE_WEIGHTED_BCE = True
USE_FOCAL_LOSS   = True

# Paper: "class imbalance term to 0.25 and focus term to 2"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


# ----------------------------------------------------------------------
# Postprocessing (Section 5.8)
# ----------------------------------------------------------------------

SMOOTH_KERNEL_MS = 500                        # paper: "500 ms kernel"
MERGE_GAP_S      = 0.5                        # paper: "joining calls separated by less than 500ms"
POST_MIN_DUR_S   = 0.5                        # paper: "shorter than 500ms were discarded"
POST_MAX_DUR_S   = 30.0                       # paper: "longer than 30s... discarded"

# Per-class thresholds tuned on validation (paper Section 5.8)
# Default to 0.5; use tune_thresholds() after training to optimize
DEFAULT_THRESHOLDS = [0.5, 0.5, 0.5]


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------

NUM_WORKERS = 16
SEED        = 42


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def n_classes() -> int:
    return 3 if USE_3CLASS else 7

def class_names() -> list[str]:
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)

def class_to_idx() -> dict[str, int]:
    return {c: i for i, c in enumerate(class_names())}
