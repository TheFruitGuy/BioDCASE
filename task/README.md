# Whale-VAD Reproduction

Clean-slate reproduction of the Whale-VAD paper:
> Geldenhuys, Tonitz, Niesler. *Whale-VAD: Whale Vocalisation Activity Detection.* DCASE 2025 (F1 = 0.440).

Paper-faithful implementation of every component, with optional hooks for contrastive self-supervised pretraining on unlabeled audio.

## Files

| File | What it does |
|---|---|
| `config.py` | All paper hyperparameters in one place |
| `spectrogram.py` | STFT feature extractor — phase-aware `[mag, cos θ, sin θ]` with per-freq mean subtraction |
| `model.py` | Whale-VAD architecture (Table 2): filterbank → CNN → residual bottleneck + depthwise → BiLSTM → linear |
| `dataset.py` | ATBFL loader with collar-extended training segments, stochastic negative undersampling, 30s/2s overlap validation |
| `postprocess.py` | 500ms median filter → threshold → 500ms merge → duration filter → CSV, plus 1D IoU evaluation |
| `train.py` | Training: AdamW lr=1e-5, weighted BCE + focal loss |
| `inference.py` | Generate challenge submission CSV |
| `pretrain_contrastive.py` | Optional SimCLR pretraining on unlabeled audio (extension) |

## Quick start

### 1. Edit paths in `config.py`

```python
DATA_ROOT = Path("/path/to/2026_BioDCASE_development_set/")
UNLABELED_DATA_DIR = Path("/path/to/unlabeled/audio")   # only for SSL
```

### 2. Train from scratch (paper reproduction)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

Single-GPU — should hit F1 ≈ 0.44 after 60 epochs. For multi-GPU, set `CUDA_VISIBLE_DEVICES=0,1,2,3`; `DataParallel` is automatic.

### 3. Generate submission

```bash
python inference.py --checkpoint runs/whalevad_XXXX/final_model.pt
```

## Paper reproduction checklist

Everything the paper specifies is implemented:

- **Audio**: 250 Hz, 256-pt FFT, 20 ms stride (Section 5.2) ✓
- **Phase features**: `[r, cos θ, sin θ]` (Section 5.4) ✓
- **Per-freq mean subtraction** (Section 5.2) ✓
- **Filterbank + CNN + Residual(Bottleneck + DepthwiseConv)** (Table 2) ✓
- **BiLSTM**, 2 layers, hidden=128, dropout=0.5 (Section 5.3) ✓
- **3-class collapsed training** (`USE_3CLASS=True` — Section 6: "+15.2% F1") ✓
- **Weighted BCE**, w_c = N / P_c (Section 5.6) ✓
- **Focal loss**, α=0.25, γ=2 (Section 5.6) ✓
- **AdamW**, lr=1e-5, wd=1e-3, betas=(0.9, 0.999) (Section 5.6) ✓
- **Stochastic negative undersampling** (Section 5.5) ✓
- **Training segments**: call + random [1, 5]s collars ✓
- **Val segments**: 30s windows, 2s overlap (Section 5.1) ✓
- **Postproc**: 500ms median filter → per-class thresholds → collapse → 500ms merge → duration filter (Section 5.8) ✓
- **Event metric**: 1D IoU ≥ 0.3 greedy matching ✓

## Optional: Contrastive pretraining extension

If you want to try self-supervised pretraining on unlabeled data (e.g. Kerguelen2020, DDU2021), the encoder weights transfer cleanly into the supervised model.

```bash
# 1. Symlink all unlabeled data into one folder
mkdir -p ./data_unlabeled/combined
ln -sf /path/to/kerguelen2020/*.wav ./data_unlabeled/combined/
ln -sf /path/to/ddu2021/*.wav       ./data_unlabeled/combined/

# 2. Pretrain the encoder (SimCLR-style, ~hours)
CUDA_VISIBLE_DEVICES=0 python pretrain_contrastive.py

# 3. Fine-tune on labeled data, loading pretrained encoder
CUDA_VISIBLE_DEVICES=0 python train.py \
    --pretrained runs/pretrain/contrastive_XXXX/best_pretrained.pt \
    --freeze_epochs 5
```

The encoder is the same class as used in `train.py` — no architectural difference between pretraining and fine-tuning. Only the BiLSTM + classifier are randomly initialized during fine-tuning.

## Notes

- `feat_proj` is lazily initialized on the first forward pass so input frequency dimension can be inferred automatically. Calling `model(spec_extractor(dummy))` once before `load_state_dict` ensures the layer exists — both `train.py` and `inference.py` do this.
- Class weights are clamped at 15 and normalized to mean=1 for loss stability. Raw N/P_c values can be 50+ and cause NaN with bfloat16.
- For multi-GPU training, `DataParallel` is used. Model state dicts are saved with the `.module` unwrapped so checkpoints are portable.
