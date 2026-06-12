# NAVE -- Normalized, Adaptive Conformer for Whale Vocalization-Event Detection

Clean, hardcoded implementation of the locked conformer recipe for BioDCASE 2026
Task 2 (3-class Antarctic baleen-whale SED: BMABZ / D / BP). Single-model dev
macro F1 = 0.495; multi-seed probability ensemble on top.

- **N**ormalized -- fixed PCEN channel (per-bin AGC; recovers background-buried D downsweeps)
- **A**daptive -- frequency-dynamic (FDY) convolutions in the stem
- Conformer backbone (macaron FFN / RoPE MHSA / wide depthwise conv k=129)

## Files

| file | role |
|------|------|
| `nave_config.py`   | single hardcoded source of truth (recipe constants) |
| `nave_features.py` | 4-ch STFT + PCEN front end (parameter-free) |
| `nave_model.py`    | `NAVE` architecture + legacy-checkpoint loader |
| `nave_train.py`    | training entry point (EMA, RAdam, masked BCE, tuned-macro selection) |
| `nave_evaluate.py` | single-checkpoint eval + per-class threshold tuning |
| `nave_ensemble.py` | multi-seed probability-averaging ensemble |

## Recipe (all in `nave_config.py`)

STFT SR=250 / N_FFT=256 / hop=5 (129 bins, 20 ms). 4 channels [demeaned |S|,
cos phi, sin phi, PCEN]. FDY on filterbank+feat0 (basis 4). d_model 128, 4 heads,
4 layers, ffn x4, dropout 0.1, depthwise conv k=129. RAdam const LR 5e-5, wd 1e-3,
EMA 0.999, 40 epochs, batch 32, neg-ratio 1.0, per-epoch negative resampling.
Post: 500 ms median smooth -> tuned per-class thresholds -> 0.5 s merge gap ->
0.5-30 s duration filter. Exactly 2,693,499 parameters.

## Usage

```bash
# train one seed (vary --seed for the ensemble)
CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed 42 --tune-workers 20

# evaluate one checkpoint (native or legacy phase13r both work)
CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/nave_s42_*/nave_best.pt --workers 13

# multi-seed ensemble
CUDA_VISIBLE_DEVICES=0 python nave_ensemble.py \
    runs/nave_s42_*/nave_best.pt runs/nave_s2024_*/nave_best.pt \
    runs/nave_s7777_*/nave_best.pt --workers 13
```

## Checkpoint compatibility

Every existing `train_phase13r` checkpoint (the 0.495 best and all seeds) loads
into `NAVE` with **no retraining**: `NAVE.from_legacy_checkpoint(path)` remaps the
old module names (`_inner.* -> stem.*`, `_proj -> proj`, `classifier -> head`),
drops the dead WhaleVAD BiLSTM keys, and surfaces the stored tuned thresholds.
Verified: 202/202 keys, 0 missing / 0 unexpected, output byte-identical to the
existing model (max abs diff 0.0), param count exactly 2,693,499. `nave_evaluate`
and `nave_ensemble` auto-detect legacy vs native checkpoints.

Confirm against the real best checkpoint on the cluster:

```bash
python -c "from nave_model import NAVE; m,meta=NAVE.from_legacy_checkpoint('runs/phase13r_3c_s42_<ts>/phase13r_best.pt'); print('loaded NAVE', sum(p.numel() for p in m.parameters()), meta)"
```

## Experiment tracking

`nave_train.py` logs to a fresh W&B project `the_fruit_guy/nave-whale-sed`
(group `nave_final`), independent of the old phase registry, so later NAVE
experiments stay separate. Disable with `--no-wandb`.

## Consolidation note

`nave_train.py` / `nave_evaluate.py` / `nave_ensemble.py` reuse the verified data
pipeline (`dataset_final`), validation helpers (`validation_core`, `tuned_val`),
post-processing (`postprocess_final`) and the threshold tuner (`eval_conformer`).
Those modules currently read `config_final`, whose shared values are identical to
`nave_config` (only `EPOCHS` differs: 30 vs 40, and the entry points set epochs
themselves). At new-git rename time, point those helpers at `nave_config` and the
pipeline is fully self-contained. The core (`nave_config` / `nave_features` /
`nave_model`) is already self-contained and sandbox-verified.
