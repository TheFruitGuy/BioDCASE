"""
Final Pipeline - Configuration
==============================

Self-contained configuration for the clean Whale-VAD training pipeline
(``train_final.py`` and its sibling ``*_final.py`` modules). Values follow
Geldenhuys et al. (DCASE 2025) and reproduce the recipe converged on through
the phase-0 ablation ladder: 8-site training, paper-sized BiLSTM, 7 fine call
types collapsed to 3 coarse classes at evaluation, weighted BCE, and per-epoch
negative resampling.

This module is imported as ``cfg`` by every other ``*_final`` module. The
``USE_3CLASS`` flag is intentionally a mutable module global: the validation
path flips it to ``True`` for the duration of post-processing (so detections
are labelled in the coarse space) and restores it afterwards. Because all
modules share this single module object, the toggle is observed consistently
everywhere.
"""

from pathlib import Path


# ======================================================================
# Filesystem paths
# ======================================================================
# Machine-specific; edit DATA_ROOT for your environment. The expected layout:
#     DATA_ROOT/
#       train/      annotations/{dataset}.csv   audio/{dataset}/*.wav
#       validation/ annotations/{dataset}.csv   audio/{dataset}/*.wav

#: Root of the BioDCASE development dataset.
DATA_ROOT = Path("/home/matthias-nagl/BioDCASE/task/2026_BioDCASE_development_set/")

#: Directory under which each training run writes a timestamped subdirectory
#: of checkpoints.
OUTPUT_DIR = Path("./runs")


# ======================================================================
# Dataset splits
# ======================================================================

#: Training sites (labels available). The final recipe trains on all eight.
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

#: Official validation sites (labels available, used for model selection).
VAL_DATASETS = [
    "casey2017",
    "kerguelen2014",
    "kerguelen2015",
]


# ======================================================================
# Audio and spectrogram parameters
# ======================================================================

#: Audio sample rate in Hz. All ATBFL recordings are resampled to 250 Hz.
SAMPLE_RATE = 250

#: Temporal stride between consecutive output frames (seconds): one
#: prediction every 20 ms.
FRAME_STRIDE_S = 0.02

#: STFT window / FFT size in samples (~1 s analysis window at 250 Hz).
N_FFT = 256
WIN_LENGTH = 256

#: STFT hop length in samples (= SAMPLE_RATE * FRAME_STRIDE_S = 5).
HOP_LENGTH = 5

#: Per-frequency normalisation mode. "demean" subtracts, per frequency bin,
#: the temporal mean of the complex STFT before magnitude/phase decomposition.
NORM_FEATURES = "demean"


# ======================================================================
# Classes and label collapsing
# ======================================================================
# The corpus provides seven fine-grained call types; the challenge evaluates
# three coarse classes. The final pipeline trains on the 7 fine types and
# collapses model outputs to the 3 coarse classes at evaluation time.

#: Fine-grained call-type labels as they appear in the annotation CSVs.
CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]

#: Coarse evaluation labels used by the challenge.
CALL_TYPES_3 = ["bmabz", "d", "bp"]

#: Maps each fine-grained label to its coarse class.
COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d", "bpd": "d",
    "bp20": "bp", "bp20plus": "bp",
}

#: When True the model space is the 3 coarse classes; when False it is the 7
#: fine classes. The final pipeline trains with this False (7-class targets)
#: and the validation path toggles it to True only around post-processing.
USE_3CLASS = False


# ======================================================================
# Segmentation
# ======================================================================

#: Range of random silence padding ("collar") added on each side of a call
#: when building a positive training segment.
COLLAR_MIN_S = 1.0
COLLAR_MAX_S = 5.0

#: Length to which every training segment is extended, matching the fixed
#: validation window length so the model sees the same context at train and
#: eval time.
TRAIN_SEGMENT_S = 30.0

#: Fixed validation/evaluation window length and overlap (seconds).
EVAL_SEGMENT_S = 30.0
EVAL_OVERLAP_S = 2.0

#: Annotations shorter than MIN or longer than MAX (seconds) are filtered out
#: during positive-segment construction.
MIN_CALL_DURATION_S = 0.5
MAX_CALL_DURATION_S = 30.0


# ======================================================================
# Negative undersampling
# ======================================================================

#: Ratio of negative-only segments to positive segments drawn per epoch. A
#: fresh negative subset is sampled at the start of every epoch.
NEG_RATIO = 1.0


# ======================================================================
# Model architecture
# ======================================================================

#: Channels in the learnable filterbank (first convolution).
FILTERBANK_OUT_CH = 64

#: Channels in the main feature extractor.
FEAT_EXTRACTOR_CH = 128

#: Bottleneck channel count inside the residual block.
BOTTLENECK_CH = 64

#: Per-frame feature dimension fed into the BiLSTM.
PROJECTION_DIM = 64

#: Hidden size per BiLSTM direction (output dim is 2 * LSTM_HIDDEN). The final
#: recipe uses the paper configuration of 128 hidden units over 2 layers.
LSTM_HIDDEN = 128
LSTM_LAYERS = 2

#: Dropout between BiLSTM layers.
LSTM_DROPOUT = 0.5

#: Dropout inside the residual bottleneck.
BOTTLENECK_DROPOUT = 0.1

#: Spatial dropout at the entry to the depthwise aggregation block.
AGG_DROPOUT = 0.2


# ======================================================================
# Training hyperparameters
# ======================================================================

#: Number of training epochs.
EPOCHS = 30

#: Mini-batch size (segments per gradient step).
BATCH_SIZE = 32

#: AdamW learning rate.
LR = 5e-5

#: AdamW weight decay.
WEIGHT_DECAY = 0.001

#: AdamW moment-decay terms.
BETA1 = 0.9
BETA2 = 0.999

#: Maximum L2 norm for gradient clipping.
GRAD_CLIP = 1.0

#: Single per-class probability threshold used for in-training validation.
THRESHOLD = 0.3


# ======================================================================
# WhaleVAD-BPN: focal loss + training recipe (arXiv 2510.21280v2)
# ======================================================================
# The BPN follow-on model trains with focal loss (not weighted BCE) and a
# different optimiser configuration. Christiaan confirmed by email that focal
# loss "drastically stabilises" BPN training.

#: Focal loss class-imbalance and focusing terms (Lin et al. 2018 defaults,
#: as used by the paper).
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

#: BPN training recipe (paper Section V-B5): AdamW, fixed LR, ~30 s segments.
BPN_LR = 1e-3
BPN_WEIGHT_DECAY = 0.01
BPN_BATCH_SIZE = 48
BPN_EPOCHS = 32

# ----------------------------------------------------------------------
# BPN architecture. Several of these are values the paper does not state
# and that we determine experimentally through the BPN phase ladder; the
# defaults here are the starting points for that search.
# ----------------------------------------------------------------------

#: Dilations of the three residual depthwise layers in the modified
#: aggregation block. These layers also expose the intermediate taps.
BPN_DILATIONS = (2, 4, 8)

#: ROI vector dimensionality (C_bpn), the proposal network output channels.
BPN_CHANNELS = 64

#: Number of ROIs per frame produced by the proposal network (R). The paper
#: leaves R unstated; with its two conv-transpose layers expanding a singleton
#: ROI axis the implied value is 8, which we use as the default and sweep.
BPN_R = 8

#: Hidden size per direction of the BPN ROI BiLSTM (paper does not state it).
BPN_LSTM_HIDDEN = 128

#: Bias used to initialise the BPN gate output so that sigmoid(gate) ~ 0.98 at
#: the start of training, i.e. the mask passes the classifier through almost
#: unchanged and only learns to suppress over time.
BPN_GATE_INIT_BIAS = 4.0


# ======================================================================
# Post-processing
# ======================================================================

#: Width of the median filter applied to per-frame probabilities (ms).
SMOOTH_KERNEL_MS = 500

#: Maximum gap (seconds) between two same-class detections that are merged.
MERGE_GAP_S = 0.5

#: Duration bounds (seconds) for a post-merge detection to be kept.
POST_MIN_DUR_S = 0.5
POST_MAX_DUR_S = 30.0


# ======================================================================
# Runtime
# ======================================================================

#: Parallel worker processes for PyTorch DataLoaders.
NUM_WORKERS = 16

#: Master random seed for ``random``, ``numpy``, and ``torch``.
SEED = 42


# ======================================================================
# Convenience helpers
# ======================================================================

def n_classes() -> int:
    """Return the number of output classes the model should produce."""
    return 3 if USE_3CLASS else 7


def class_names() -> list[str]:
    """Return the ordered list of class label strings currently in use."""
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)


def class_to_idx() -> dict[str, int]:
    """Return a mapping from class name to zero-based output index."""
    return {c: i for i, c in enumerate(class_names())}
