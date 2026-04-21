# Whale-VAD Reproduction

A clean-room reproduction of the Whale-VAD paper:

> Geldenhuys, Tonitz, Niesler. *Whale-VAD: Whale Vocalisation Activity Detection.*
> DCASE 2025 (F1 = 0.440).

This repository contains a paper-faithful implementation of every component of the Whale-VAD pipeline, plus three practical extensions developed during the reproduction: training stabilization, per-class binary specialization, and optional contrastive self-supervised pretraining on unlabeled Antarctic hydrophone audio.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | Centralized hyperparameters and paths. All paper settings live here. |
| `spectrogram.py` | Phase-aware STFT front end (`[magnitude, cos θ, sin θ]`) with per-frequency mean subtraction. |
| `model.py` | Whale-VAD architecture (Table 2 of the paper): filterbank → CNN → residual bottleneck + depthwise aggregation → BiLSTM → linear classifier. Also contains the weighted BCE + focal loss and the class-weight computation. |
| `dataset.py` | ATBFL dataset loader with collar-extended training segments, stochastic negative undersampling, and 30 s / 2 s overlap validation windows. |
| `postprocess.py` | 500 ms median filter → per-class thresholding → merge close events → duration filter → CSV export, plus event-level 1D IoU evaluation. |
| `train.py` | Supervised training loop with early stopping, `ReduceLROnPlateau`, and periodic negative resampling. |
| `train_binary.py` | Trains one binary model per class (for the ensemble approach). |
| `inference.py` | Generate the challenge submission CSV from a single multi-class checkpoint. |
| `inference_binary.py` | Combine the three binary models for submission or evaluation. |
| `tune_thresholds.py` | Iterative per-class threshold search with finer grids for rare classes. |
| `eval_only.py` | Evaluate any checkpoint on the validation set and print Table-4-style per-site metrics. |
| `diagnose.py` | Standalone sanity-check script: verifies STFT shapes, class balance, model init. |
| `pretrain_contrastive.py` | Optional SimCLR pretraining on unlabeled audio (extension). |

## Quick start

### 1. Configure paths

Edit `config.py` so that `DATA_ROOT` points at your local copy of the BioDCASE development set:

```python
DATA_ROOT = Path("/path/to/2026_BioDCASE_development_set/")
UNLABELED_DATA_DIR = Path("/path/to/unlabeled/audio")   # optional, for SSL extension
```

### 2. Train the paper reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

Single-GPU training completes in a few hours on an RTX 2080 Ti. For multi-GPU, set `CUDA_VISIBLE_DEVICES=0,1,2,3`; `DataParallel` is applied automatically.

### 3. Evaluate a checkpoint

```bash
python eval_only.py --checkpoint runs/whalevad_XXXX/final_model.pt
```

### 4. Generate a challenge submission

```bash
python inference.py --checkpoint runs/whalevad_XXXX/final_model.pt \
                    --output submission.csv
```

## Paper reproduction checklist

Every element of the paper's specification is implemented:

- **Audio preprocessing**: 250 Hz sample rate, 256-point FFT, 20 ms stride (Section 5.2) ✓
- **Phase-aware features**: `[|STFT|, cos θ, sin θ]` as three channels (Section 5.4) ✓
- **Per-frequency mean subtraction** (Section 5.2) ✓
- **Architecture**: learnable filterbank + CNN + Residual(Bottleneck + DepthwiseConv) stack (Table 2) ✓
- **BiLSTM**: 2 layers, hidden size 128, inter-layer dropout 0.5 (Section 5.3) ✓
- **3-class collapsed training** (`USE_3CLASS=True`; Section 6 reports +15.2% F1 vs 7-class) ✓
- **Weighted BCE** with *w<sub>c</sub>* = N / P<sub>c</sub> (Section 5.6) ✓
- **Focal loss**, α = 0.25, γ = 2.0 (Section 5.6; disabled by default in our config — see *Notes* below) ✓
- **AdamW**: lr = 1e-5, weight decay = 1e-3, β = (0.9, 0.999) (Section 5.6) ✓
- **Stochastic negative undersampling** (Section 5.5) ✓
- **Training segments**: call + random [1, 5] s collars (Section 5.1) ✓
- **Validation segments**: 30 s windows, 2 s overlap (Section 5.1) ✓
- **Post-processing**: 500 ms median smoothing → per-class thresholds → 7→3 collapse → 500 ms merge → duration filter (Section 5.8) ✓
- **Evaluation metric**: event-level 1D IoU ≥ 0.3 with greedy matching ✓

## Extension A — Training stabilization

In our reproduction we found that strict paper settings produced noisy validation F1 curves and some overfitting in the long tail of training. `train.py` therefore adds three stabilizers that do not change the paper's method but make training more reliable:

- **Negatives resampled every 5 epochs** (instead of every epoch). Paper's every-epoch resampling is noisier and makes early-stopping decisions unreliable.
- **`ReduceLROnPlateau`** scheduler halves the learning rate after 5 stagnant epochs.
- **Early stopping** with 15-epoch patience on validation F1.

These extensions improved our reproduction F1 from ≈ 0.23 (strict paper) to ≈ 0.29 (stabilized).

## Extension B — Binary per-class specialization

The three target classes have very different acoustic properties and prior frequencies. Training one binary model per class and combining their detections is a natural alternative to the multi-label model.

```bash
# Train one model per class (can be run in parallel on different GPUs)
CUDA_VISIBLE_DEVICES=0 python train_binary.py --class bmabz
CUDA_VISIBLE_DEVICES=1 python train_binary.py --class d
CUDA_VISIBLE_DEVICES=2 python train_binary.py --class bp

# Combine the three for evaluation
python inference_binary.py \
    --bmabz_ckpt runs/binary_bmabz_XXXX/final_model.pt \
    --d_ckpt     runs/binary_d_XXXX/final_model.pt     \
    --bp_ckpt    runs/binary_bp_XXXX/final_model.pt    \
    --mode eval
```

Results from our reproduction:

| Class | Multi-class F1 | Binary F1 (tuned) | Δ |
|---|---|---|---|
| bmabz | 0.388 | **0.485** | +25 % |
| d     | 0.046 | 0.031 | −33 % |
| bp    | 0.156 | 0.161 | +3 % |

Binary specialization helps the well-represented class (bmabz) significantly but does not rescue the rare classes — d and bp are architecturally difficult at this scale regardless of the training protocol, consistent with the paper's own low per-site numbers for these classes in Table 4.

## Extension C — Optional contrastive pretraining

For experiments with self-supervised pretraining on unlabeled Antarctic audio (e.g. Kerguelen2020, DDU2021), the encoder architecture is identical to the supervised one, so pretrained weights load directly into `train.py`.

```bash
# 1. Symlink unlabeled WAV files into one directory
mkdir -p ./data_unlabeled/combined
ln -sf /path/to/kerguelen2020/*.wav ./data_unlabeled/combined/
ln -sf /path/to/ddu2021/*.wav       ./data_unlabeled/combined/

# 2. Pretrain the encoder with SimCLR (~hours)
CUDA_VISIBLE_DEVICES=0 python pretrain_contrastive.py

# 3. Fine-tune on labeled data with the pretrained encoder
CUDA_VISIBLE_DEVICES=0 python train.py \
    --pretrained runs/pretrain/contrastive_XXXX/best_pretrained.pt \
    --freeze_epochs 5
```

At fine-tuning time only the BiLSTM and classifier are randomly initialized; the encoder starts from the pretrained weights. The `--freeze_epochs 5` flag keeps the encoder frozen for the first 5 epochs to let the LSTM adapt before end-to-end training.

## Notes and implementation details

- **Lazy projection layer**: `model.feat_proj` is created on the first forward pass so its input dimension can be inferred from the actual CNN output. Every entry point (`train.py`, `inference.py`, `eval_only.py`, etc.) therefore runs a dummy forward pass before calling `load_state_dict`. If you subclass the model, preserve this pattern.

- **Class weight normalization**: raw N / P<sub>c</sub> weights can exceed 15 for rare classes. We normalize so that the minimum weight is 1.0 (instead of shifting by the mean), which preserves the relative class importance without actively down-weighting the common class.

- **Focal loss disabled by default**: the paper combines focal loss with weighted BCE. In our reproduction, enabling both simultaneously produced unstable early training (loss spikes). We kept only the weighted BCE, which matches the paper's Section 5.6 specification minus the focal modulation; re-enable via `USE_FOCAL_LOSS=True` in `config.py` to reproduce the paper's full loss.

- **Multi-GPU**: `DataParallel` is wrapped automatically when multiple GPUs are visible. Checkpoints are always saved with `.module` unwrapped so they load cleanly on any GPU count.

- **Thresholds**: `train.py` performs a simple threshold tuning pass at the end of training. For a more thorough search (finer grid for rare classes, three passes, per-site breakdown), run `tune_thresholds.py` on the `best_model.pt` afterwards.
