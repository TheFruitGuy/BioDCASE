"""
Phase 0e: Match Training Segment Length to Validation
=====================================================

Phase 0c showed the model can hit F1=0.443 within-site, but with wild
oscillation (0.44 → 0.004 → 0.18 → 0.39 across consecutive epochs).
Phase 0d ruled out BatchNorm running statistics as the primary cause.

Diagnostic of segment lengths revealed:

  - Training positive segments: median 9.5s, range 3-30s
  - Training negative segments: median 16.1s, range 5-29s
  - Validation segments: ALL exactly 30s

The model is being trained on short clips and evaluated on clips 3×
longer. This mismatches the BiLSTM's hidden state dynamics, the
BatchNorm spatial statistics, and effectively asks the model to
extrapolate to a context length it never saw during training.

Phase 0e tests the fix: **extend every training segment to a fixed
30 seconds** (the same length validation uses). The annotations stay
where they are within the segment, only the surrounding context grows.
A 7-second call+collar segment becomes a 30s segment with the call
sitting somewhere inside; targets are 1 only at frames where the call
is, 0 elsewhere. The expectation is the same as it always was, but
now the model sees 30s of audio with sparse positive frames during
training, exactly matching what it sees at evaluation time.

Three possible outcomes
-----------------------
1. F1 stabilises at 0.30+ across multiple consecutive epochs. Best F1
   similar to or higher than Phase 0c. → Train/val length mismatch
   was the cause. Apply this fix to the full pipeline.

2. F1 still oscillates but is smoother (max swing 0.10-0.20 instead
   of 0.40+). → Length is part of the problem; combine with BN
   momentum reduction in Phase 0f.

3. F1 oscillates as before. → Length wasn't it. Look elsewhere
   (positive/negative ratio per batch, optimizer choice, or whether
   the BiLSTM hidden state initialization is the issue).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0e.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    collate_fn,
)
from train_phase0 import (
    SingleClassDataset, build_phase0_model, validate_one_class,
    train_one_epoch,
    TARGET_CLASS_IDX, TARGET_CLASS_NAME,
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE,
    PHASE0_EPOCHS, PHASE0_THRESHOLD,
)
from train_phase0c import (
    PHASE0C_SITE, PHASE0C_VAL_FRACTION, PHASE0C_SEED,
    split_manifest_by_file,
)


# Segment-length helpers now live in segment_length_core so they can be
# reused without importing the whole phase-0 training chain. Re-exported
# here for backward compatibility with existing `from train_phase0e import ...`.
from segment_length_core import (  # noqa: E402,F401
    PHASE0E_SEGMENT_S,
    extend_segment_to_fixed_length,
    extend_all_segments,
)


def main():
    """Run Phase 0e end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0e expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0e", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
    })

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0e_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0e configuration:")
    print(f"  Site: {PHASE0C_SITE} (same-site split)")
    print(f"  Val fraction: {PHASE0C_VAL_FRACTION}")
    print(f"  Target class: {TARGET_CLASS_NAME}")
    print(f"  *** Training segments forced to {PHASE0E_SEGMENT_S}s ***")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0_EPOCHS}")

    # ------------------------------------------------------------------
    # Data — like Phase 0c but with the segment-extension step
    # ------------------------------------------------------------------
    print(f"\nLoading data...")
    full_manifest = get_file_manifest([PHASE0C_SITE])
    full_annotations = load_annotations([PHASE0C_SITE], manifest=full_manifest)

    train_manifest, val_manifest = split_manifest_by_file(
        full_manifest, PHASE0C_VAL_FRACTION, seed=PHASE0C_SEED,
    )

    val_filenames = set(val_manifest["filename"])
    train_annotations = full_annotations[
        ~full_annotations["filename"].isin(val_filenames)
    ].reset_index(drop=True)
    val_annotations = full_annotations[
        full_annotations["filename"].isin(val_filenames)
    ].reset_index(drop=True)

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )

    # Phase 0e intervention: extend every training segment to 30s.
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs

    # Quick verification that the extension actually worked.
    pos_durs = [(s.end_sample - s.start_sample) / cfg.SAMPLE_RATE
                for s in pos_segs[:200]]
    print(f"Training segment durations after extension (first 200 positives):")
    print(f"  min={min(pos_durs):.1f}s  max={max(pos_durs):.1f}s  "
          f"mean={sum(pos_durs)/len(pos_durs):.1f}s")

    val_segments = build_val_segments(val_manifest, val_annotations)

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
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0_EPOCHS} epochs (30s training segments)")
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
        }, run_dir / f"phase0e_epoch_{epoch:02d}.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0e VERDICT")
    print(f"{'=' * 60}")
    f1s = [h["f1"] for h in history]
    print(f"F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Best F1: {max(f1s):.3f} at epoch {f1s.index(max(f1s)) + 1}")
    print(f"Final F1: {f1s[-1]:.3f}")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0

    print(f"\nSecond-half stability:")
    print(f"  Mean epoch-to-epoch F1 swing: {mean_swing:.3f}")
    print(f"  Max  epoch-to-epoch F1 swing: {max_swing:.3f}")
    print(f"  Phase 0c reference: max swing 0.44, mean swing 0.20")
    print(f"  Phase 0d reference: max swing 0.37, mean swing 0.18")

    if max_swing < 0.10:
        print("→ STABILITY FIXED. Train/val length mismatch was the cause.")
        print("  Update dataset.py to extend training segments to 30s, then")
        print("  scale up to multi-site / multi-class.")
    elif max_swing < 0.20:
        print("→ Improved but still wobbling. Combine with BN momentum")
        print("  reduction in Phase 0f, or investigate batch composition.")
    else:
        print("→ Same oscillation. Length wasn't the primary cause.")
        print("  Investigate: positive-fraction-per-batch invariance, LSTM")
        print("  hidden initialization, or whether input normalization")
        print("  differs between training and validation pipelines.")


    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    verdict_text = (
        f"Phase 0e: best F1 {max(f1s):.3f} at epoch "
        f"{f1s.index(max(f1s)) + 1}, final F1 {f1s[-1]:.3f}."
    )
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=None,
    )

if __name__ == "__main__":
    main()
