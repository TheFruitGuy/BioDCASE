"""
Triton — Configuration
======================

Single source of truth for hyperparameters, paths, and constants used across
the pipeline. Imported as ``cfg`` everywhere; changing a value here
propagates to data loading, model construction, training, post-processing,
and inference.

The defaults reproduce the WhaleVAD/WhaleVAD-BPN training recipe of
Geldenhuys et al. (DCASE 2025; arXiv:2510.21280) so that Triton baselines
remain numerically comparable to those references during development. The
naming (Triton, TIDE) is ours; the numbers are theirs.

Sections
--------
- Filesystem paths
- Dataset splits
- Audio and spectrogram parameters
- Classes and label collapsing
- Segmentation
- Negative undersampling
- Model architecture (backbone)
- Training hyperparameters
- Loss
- Post-processing
- Runtime
"""

from pathlib import Path


# ======================================================================
# Filesystem paths
# ======================================================================
# These are machine-specific. Everything else can stay at defaults.

#: Root of the BioDCASE 2026 development dataset. Expected layout:
#:     DATA_ROOT/
#:       train/
#:         annotations/{dataset_name}.csv
#:         audio/{dataset_name}/*.wav
#:       validation/
#:         annotations/{dataset_name}.csv
#:         audio/{dataset_name}/*.wav
DATA_ROOT = Path("/home/matthias-nagl/BioDCASE/task/2026_BioDCASE_development_set/")

#: Per-run output directory. Each ``train.py`` invocation creates a
#: timestamped subdirectory under here (e.g. ``runs/triton_20260516_141500/``).
OUTPUT_DIR = Path("./runs")

#: Default destination for challenge submission CSVs.
SUBMISSION_PATH = Path("./submission.csv")

#: Optional path to pretrained encoder weights (e.g. from contrastive
#: pretraining). Set this via ``train.py --pretrained ...`` rather than
#: by editing this value directly.
PRETRAINED_PATH = None


# ======================================================================
# Dataset splits
# ======================================================================

#: Training sites (labels available).
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

#: Validation sites (labels available, used for model selection and
#: post-hoc threshold tuning).
VAL_DATASETS = [
    "casey2017",
    "kerguelen2014",
    "kerguelen2015",
]

#: Evaluation sites (labels withheld; challenge submission only).
EVAL_DATASETS = [
    "kerguelen2020",
    "ddu2021",
]


# ======================================================================
# Audio and spectrogram parameters
# ======================================================================

#: Audio sample rate (Hz). Provider resamples all ATBFL hydrophone
#: recordings to 250 Hz.
SAMPLE_RATE = 250

#: Per-frame temporal stride (seconds). Model produces one prediction
#: every 20 ms.
FRAME_STRIDE_S = 0.02

#: STFT window size in samples (≈ 1 s at 250 Hz). Suits low-frequency
#: whale vocalisations whose fundamental periods are 1–10 s.
N_FFT = 256
WIN_LENGTH = 256

#: STFT hop length (samples). ``HOP_LENGTH = SAMPLE_RATE * FRAME_STRIDE_S``,
#: so 5 samples = 20 ms.
HOP_LENGTH = 5

#: Per-frequency mean subtraction. ``"demean"`` subtracts the time-mean of
#: the complex STFT (before magnitude/phase split). Removes stationary
#: channel noise; non-trivial — the official Geldenhuys checkpoint was
#: trained against complex-domain demeaning, not magnitude demeaning.
NORM_FEATURES = "demean"

#: Complex-spectrum representation. ``"trig"`` stacks
#: ``[magnitude, cos(phase), sin(phase)]`` as 3 input channels — empirically
#: more learnable than raw real/imaginary parts.
COMPLEX_REPR = "trig"


# ======================================================================
# Classes and label collapsing
# ======================================================================

#: Fine-grained call type labels (as they appear in the raw CSVs).
CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]

#: Coarse labels used by the challenge.
#:   - ``bmabz``: Antarctic blue whale Z-call and its syllabic components
#:   - ``d``:     D-call (blue whale downsweep)
#:   - ``bp``:    fin whale pulse calls
CALL_TYPES_3 = ["bmabz", "d", "bp"]

#: Fine-grained → coarse map.
COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d",     "bpd": "d",
    "bp20": "bp",   "bp20plus": "bp",
}

#: When True, models output 3 logits matching the coarse classes. When
#: False, models output 7 fine-grained logits and we collapse to 3 at
#: evaluation time via COLLAPSE_MAP. The DCASE 2025 ablation favours 3.
USE_3CLASS = True


# ======================================================================
# Segmentation
# ======================================================================

#: Minimum random "collar" prepended/appended to a training call (seconds).
#: Ensures the model learns to localise calls rather than triggering on
#: segment-length cues.
COLLAR_MIN_S = 1.0

#: Maximum collar duration (seconds).
COLLAR_MAX_S = 5.0

#: Fixed-window length for validation/inference segments (seconds).
#: Empirically 60 s outperformed 30 s in our reproduction — likely
#: because the BiLSTM benefits from more bilateral context and the
#: stitch-overlap zone shrinks as a fraction of each window. Going
#: longer than ~120 s shows diminishing returns. Changing this
#: affects: per-epoch validation F1 (and therefore LR scheduling,
#: best-checkpoint selection, and the per-class thresholds saved
#: with the final model), and inference segment length downstream.
EVAL_SEGMENT_S = 60.0

#: Overlap between consecutive validation windows (seconds). Kept
#: absolute (not a fraction of EVAL_SEGMENT_S) on purpose — the
#: stitch-zone artifact magnitude depends on the absolute overlap,
#: not the ratio, and 2 s is already wider than the boundary
#: uncertainty for the longest call type.
EVAL_OVERLAP_S = 2.0

#: Annotations shorter than this are dropped (likely errors).
MIN_CALL_DURATION_S = 0.5

#: Annotations longer than this are dropped (likely multi-call groupings
#: or end-time inheritance errors in the CSV).
MAX_CALL_DURATION_S = 30.0


# ======================================================================
# Negative undersampling
# ======================================================================

#: Negative-to-positive segment ratio per epoch. ~1:1 balance via fresh
#: stochastic resampling each epoch so that, over many epochs, the model
#: sees a diverse sample of the negative distribution.
NEG_RATIO = 1.0


# ======================================================================
# Model architecture (shared backbone for both Triton and Triton-TIDE)
# ======================================================================

#: Channels out of the learnable filterbank (first Conv2d).
FILTERBANK_OUT_CH = 64

#: Main feature-extractor channel count.
FEAT_EXTRACTOR_CH = 128

#: Bottleneck channel count inside the residual block.
BOTTLENECK_CH = 64

#: Per-frame feature dim fed into the BiLSTM.
PROJECTION_DIM = 64

#: Hidden size per BiLSTM direction (so output dim is ``2 * LSTM_HIDDEN``).
LSTM_HIDDEN = 128

#: Stacked BiLSTM layers.
LSTM_LAYERS = 2

#: Dropout between BiLSTM layers.
LSTM_DROPOUT = 0.5

#: Dropout inside the residual bottleneck.
BOTTLENECK_DROPOUT = 0.1

#: Spatial (channel-wise) dropout at the depthwise aggregation entry.
AGG_DROPOUT = 0.2


# ======================================================================
# Training
# ======================================================================

#: Hard cap on epochs. ``train.py`` early-stops well before this in
#: practice.
EPOCHS = 150

#: Mini-batch size (segments per gradient step).
BATCH_SIZE = 32

#: AdamW learning rate.
#:
#: 5e-5 is the empirically validated value for this codebase, matched
#: against the canonical seed-42 baseline run (runs/whalevad_20260502_175547)
#: which reaches F1=0.474 at epoch 30 with patience-8/25 schedule.
#:
#: The papers disagree: DCASE 2025 tech report says 1e-5, BPN paper says
#: 1e-3 (fixed). Neither matches our working reproduction. 1e-5 doesn't
#: move weights (paper's own LR doesn't converge with no schedule).
#: 1e-3 overshoots with the patience-8/25 schedule we use (a separate
#: experimental run with patience-4/12 and 1e-3 hit F1=0.465 — different
#: config, not the canonical baseline). 5e-5 is the rate at which the
#: schedule we actually use produces the F1=0.474 result that the report
#: cites. Don't change this without first running a full baseline at the
#: new value to validate.
LR = 5e-5

#: AdamW weight decay.
WEIGHT_DECAY = 0.001

#: AdamW betas.
BETA1 = 0.9
BETA2 = 0.999

#: Gradient L2-norm clip. Prevents occasional BiLSTM gradient spikes.
GRAD_CLIP = 1.0


# ======================================================================
# Loss
# ======================================================================

#: Apply per-class positive weights in BCE. Weights are ``w_c = N / P_c``
#: where ``N`` is the number of negative segments and ``P_c`` is the
#: number of positive segments for class c (see ``loss.compute_class_weights``).
USE_WEIGHTED_BCE = True

#: Apply focal modulation on top of BCE. DCASE 2025 Table 2 shows this
#: adds ~8% absolute F1 over plain weighted BCE.
USE_FOCAL_LOSS = True

#: Focal class-imbalance parameter α. Following Lin et al. 2018.
FOCAL_ALPHA = 0.25

#: Focal focusing parameter γ. Following Lin et al. 2018.
FOCAL_GAMMA = 2.0


# ======================================================================
# Post-processing
# ======================================================================

#: Median-filter width applied to per-frame probabilities (milliseconds).
SMOOTH_KERNEL_MS = 500

#: Max gap (s) between two same-class detections on the same file that
#: should be merged into one event.
MERGE_GAP_S = 0.5

#: Min post-merge event duration (s).
POST_MIN_DUR_S = 0.5

#: Max post-merge event duration (s).
POST_MAX_DUR_S = 30.0

#: Default per-class thresholds when no tuned ones are supplied. Run
#: per-epoch threshold tuning in ``train.py`` to learn better ones.
DEFAULT_THRESHOLDS = [0.5, 0.5, 0.5]


# ======================================================================
# Runtime
# ======================================================================

#: DataLoader worker processes.
NUM_WORKERS = 16

#: Master random seed. Reproducibility starts here; see
#: ``utils.seed_everything``.
SEED = 42


# ======================================================================
# Convenience helpers (depend on the 3-class toggle)
# ======================================================================

def n_classes() -> int:
    """Return the number of output classes the model should produce."""
    return 3 if USE_3CLASS else 7


def class_names() -> list[str]:
    """Return the ordered list of class label strings currently in use."""
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)


def class_to_idx() -> dict[str, int]:
    """Mapping from class name to zero-based output index."""
    return {c: i for i, c in enumerate(class_names())}
