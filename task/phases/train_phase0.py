"""
Phase 0 Sanity Baseline
=======================

The simplest possible Whale-VAD training run that should still produce
non-trivial F1. The point is **not** to match the paper — it's to verify
the pipeline can train monotonically end-to-end on something easy. If
this stops climbing or starts degrading by epoch 5, there's a pipeline
bug we need to fix before adding any real complexity back in.

What's stripped down vs full ``train.py``
------------------------------------------
- **Single training site**: ``kerguelen2005`` only (~200 files, ~3000
  annotations). Tiny enough to iterate fast (<2 min per epoch).
- **Single validation site**: ``casey2017`` only.
- **Single class**: only the ``bmabz`` (collapsed blue-whale) class.
  Model has one output channel, target is one-hot for bmabz only,
  loss is plain BCE.
- **No class weighting, no focal loss**: just the standard BCE that
  PyTorch ships. One less moving variable.
- **No negative resampling** during training: positives + negatives are
  fixed for the whole run. Eliminates between-epoch noise.
- **Smaller LSTM**: ``hidden=32, layers=1`` instead of ``128, 2``. Fewer
  params means less to overfit; the encoder is what mostly matters.
- **No threshold tuning**: just F1 at threshold 0.5. We're not trying
  to maximise the metric, we're trying to verify it climbs.

If this works (monotonic F1 climb to ~0.30+), we know the pipeline is
sound and we can layer complexity back in one axis at a time. If it
fails, we have a tiny reproducible problem to debug instead of a giant
one.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
)


# ======================================================================
# Hyperparameters — deliberately conservative
# ======================================================================

#: Just one site for training. Smaller than casey2014 / elephantisland
#: but big enough to learn from.
PHASE0_TRAIN_SITES = ["kerguelen2005"]

#: One validation site.
PHASE0_VAL_SITES = ["casey2017"]

#: We only train against this one coarse class.
TARGET_CLASS_IDX = 0   # 0=bmabz, 1=d, 2=bp in CALL_TYPES_3 ordering
TARGET_CLASS_NAME = cfg.CALL_TYPES_3[TARGET_CLASS_IDX]

#: Learning rate. Mid-range; not too small (we saw 1e-5 doesn't move
#: weights) and not too large (we saw 1e-4 destabilises).
PHASE0_LR = 5e-5

#: Weight decay. Same as paper.
PHASE0_WEIGHT_DECAY = 0.001

#: Batch size. Small enough that BatchNorm batch stats are noisy but
#: not catastrophically so.
PHASE0_BATCH_SIZE = 32

#: Hidden size for the BiLSTM. Reduced from 128 to 32.
PHASE0_LSTM_HIDDEN = 32

#: Number of BiLSTM layers. Reduced from 2 to 1.
PHASE0_LSTM_LAYERS = 1

#: How many epochs to train. We expect to see clear convergence well
#: before this; anything past 15 is just confirming the trend.
PHASE0_EPOCHS = 20

#: Single-threshold evaluation point. We're not tuning, just checking
#: that training is healthy.
PHASE0_THRESHOLD = 0.3


# ======================================================================
# Single-class dataset wrapper
# ======================================================================

class SingleClassDataset(WhaleDataset):
    """
    Wraps WhaleDataset to return a single-class target.

    Picks the channel for ``target_class_idx`` (in the 3-class layout)
    out of the full multi-class target tensor. Output target shape is
    ``(n_frames, 1)`` instead of ``(n_frames, 3)``, so the loss and
    model output dimensions stay matched at 1.
    """

    def __init__(self, segments, target_class_idx: int):
        super().__init__(segments)
        self.target_class_idx = target_class_idx

    def __getitem__(self, idx: int):
        audio, targets, mask, meta = super().__getitem__(idx)
        # If the underlying dataset returned 3-class targets (USE_3CLASS=True),
        # just slice. If it returned 7-class targets, we'd need a collapse —
        # but we assert USE_3CLASS at startup so this branch is fine.
        single = targets[:, self.target_class_idx:self.target_class_idx + 1]
        return audio, single, mask, meta


# ======================================================================
# Build a small model with a single output channel
# ======================================================================

def build_phase0_model(device: torch.device):
    """
    Build a small WhaleVAD with one output class.

    Temporarily monkey-patches the relevant config values so the LSTM
    is built smaller. Restores them afterwards. Hacky but keeps the
    model.py code unchanged.
    """
    # Snapshot the values we'll override.
    orig_hidden = cfg.LSTM_HIDDEN
    orig_layers = cfg.LSTM_LAYERS

    cfg.LSTM_HIDDEN = PHASE0_LSTM_HIDDEN
    cfg.LSTM_LAYERS = PHASE0_LSTM_LAYERS
    try:
        model = WhaleVAD(num_classes=1).to(device)
        # Materialise lazy projection layer.
        spec = SpectrogramExtractor().to(device)
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            model(spec(dummy))
    finally:
        # Always put them back, even if the build failed.
        cfg.LSTM_HIDDEN = orig_hidden
        cfg.LSTM_LAYERS = orig_layers

    return model, spec


# ======================================================================
# Training and validation
# ======================================================================

def train_one_epoch(model, spec_extractor, loader, criterion, optimizer, device):
    """Standard training pass. Returns mean loss."""
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)

        # Trim model output to match target length (small drift from STFT
        # boundary effects; copied from train.py's align_lengths logic).
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        # Mask out padded frames.
        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / valid.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        # Mild gradient clip to catch any explosions early.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


@torch.no_grad()
def validate_one_class(model, spec_extractor, loader, criterion, device,
                      val_annotations, file_start_dts, threshold: float):
    """
    Validation pass for the single-class case.

    Builds detections at a single fixed threshold, computes event-level
    F1 against the ground-truth annotations of the target class only.
    """
    model.eval()
    losses, n = 0.0, 0

    # Collect per-window probabilities, keyed for stitching.
    all_probs = {}
    hop = spec_extractor.hop_length

    for audio, targets, mask, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / valid.sum().clamp(min=1.0)

        losses += loss.item()
        n += 1

        # Stash probabilities for event-level evaluation.
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Build single-class threshold array, run the standard postprocess.
    # postprocess_predictions expects (n_classes,)-shaped thresholds and
    # arrays — works fine with n_classes=1.
    pred_events = postprocess_predictions(all_probs, np.array([threshold]))
    # The label that postprocess_predictions emits comes from
    # cfg.class_names()[c]; with USE_3CLASS=True and only 1 channel it
    # would emit "bmabz" — but only if it loops over a 1-class list.
    # Override the label here to be safe.
    for d in pred_events:
        d.label = TARGET_CLASS_NAME

    # GT events for our target class only.
    gt_events = []
    for _, row in val_annotations.iterrows():
        if row["label_3class"] != TARGET_CLASS_NAME:
            continue
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=TARGET_CLASS_NAME,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    cls = metrics.get(TARGET_CLASS_NAME, {})
    return {
        "loss": losses / max(n, 1),
        "f1": cls.get("f1", 0.0),
        "precision": cls.get("precision", 0.0),
        "recall": cls.get("recall", 0.0),
        "tp": cls.get("tp", 0),
        "fp": cls.get("fp", 0),
        "fn": cls.get("fn", 0),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    """Run the Phase 0 sanity baseline end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0 expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
    })

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0 configuration:")
    print(f"  Train sites: {PHASE0_TRAIN_SITES}")
    print(f"  Val sites:   {PHASE0_VAL_SITES}")
    print(f"  Target class: {TARGET_CLASS_NAME} (idx {TARGET_CLASS_IDX})")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0_EPOCHS}")
    print(f"  LSTM: hidden={PHASE0_LSTM_HIDDEN}, layers={PHASE0_LSTM_LAYERS}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\nLoading data...")
    train_manifest = get_file_manifest(PHASE0_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0_TRAIN_SITES,
                                         manifest=train_manifest)
    val_manifest = get_file_manifest(PHASE0_VAL_SITES)
    val_annotations = load_annotations(PHASE0_VAL_SITES,
                                       manifest=val_manifest)

    # Build positive and negative training segments. For Phase 0 we
    # build negatives ONCE and never resample — eliminates between-epoch
    # noise that masked the underlying training trajectory.
    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    train_segments = pos_segs + neg_segs
    print(f"Training segments: {len(pos_segs)} positive + "
          f"{len(neg_segs)} negative")

    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"Validation segments: {len(val_segments)}")

    train_ds = SingleClassDataset(train_segments, TARGET_CLASS_IDX)
    val_ds = SingleClassDataset(val_segments, TARGET_CLASS_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        **wbu.seeded_dataloader_kwargs(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model + loss + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Plain BCE-with-logits, no class weighting, no focal. The 'none'
    # reduction lets us mask padded frames before averaging.
    criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)

    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop with per-epoch monotonic F1 logging
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0_EPOCHS} epochs")
    print(f"{'=' * 60}")

    for epoch in range(1, PHASE0_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_one_class(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=PHASE0_THRESHOLD,
        )
        epoch_time = time.time() - t0

        print(f"\nEpoch {epoch:2d}/{PHASE0_EPOCHS}  ({epoch_time:.0f}s)")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        print(f"  {TARGET_CLASS_NAME}: TP={val['tp']:4} FP={val['fp']:5} "
              f"FN={val['fn']:4}  P={val['precision']:.3f} "
              f"R={val['recall']:.3f} F1={val['f1']:.3f}")

        wbu.log_epoch(epoch, train_loss, val)


        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "precision": val["precision"],
            "recall": val["recall"],
        })

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
        }, run_dir / f"phase0_epoch_{epoch:02d}.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0 VERDICT")
    print(f"{'=' * 60}")
    f1s = [h["f1"] for h in history]
    print(f"F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Best F1: {max(f1s):.3f} at epoch {f1s.index(max(f1s)) + 1}")
    print(f"Final F1: {f1s[-1]:.3f}")

    # Health checks: does F1 improve in the second half of training?
    first_half_max = max(f1s[:len(f1s) // 2])
    second_half_max = max(f1s[len(f1s) // 2:])
    print(f"\nFirst-half best:  {first_half_max:.3f}")
    print(f"Second-half best: {second_half_max:.3f}")

    if second_half_max > first_half_max + 0.02:
        print("→ Pipeline is healthy. Training improves over time.")
        print("  Next: Phase 1 (add a second class, then more, "
              "until pipeline breaks).")
    elif second_half_max < first_half_max - 0.05:
        print("→ DEGRADATION DETECTED. Pipeline peaks early then gets worse.")
        print("  This is the same pattern as the full pipeline. Even at the")
        print("  simplest possible configuration, training is unstable.")
        print("  Investigate: BatchNorm dynamics, target alignment, "
              "or stitching.")
    else:
        print("→ Plateau. Training is stable but not improving much. "
              "Either the simplified setup is too weak (small LSTM, single "
              "site) or there's a subtle mid-training stall. Either way, "
              "no degradation, so the pipeline isn't actively broken.")


    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    verdict_text = (
        f"Phase 0: best F1 {max(f1s):.3f} at epoch "
        f"{f1s.index(max(f1s)) + 1}, final F1 {f1s[-1]:.3f}."
    )
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=None,
    )

if __name__ == "__main__":
    main()
