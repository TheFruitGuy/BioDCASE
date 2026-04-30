"""
Phase 0d: Stabilizing BatchNorm Dynamics
========================================

Phase 0c proved the pipeline can hit F1=0.443 on within-site validation
(matching the paper's overall F1). But the F1 oscillates wildly between
epochs — 0.44 then 0.004 then 0.18 then 0.40 — because BatchNorm
running statistics shift faster than the model converges.

Diagnosis: PyTorch's default ``BatchNorm2d`` momentum is 0.1, meaning
the running mean/var update by 10% of each batch's stats per step. When
training batches are small and shuffled (different mix of sites,
positives, negatives) every epoch, the running stats track the latest
batches' distribution rather than the long-term distribution. Eval uses
those wandering stats, so the feature distribution at validation time
shifts each epoch and the threshold cuts a different part of the
probability mass. Result: F1 swings.

Fix: reduce ``BatchNorm2d.momentum`` to 0.01, so running stats track
the average over ~100 batches instead of ~10. Slower-moving stats =
more stable validation features.

This script is identical to Phase 0c except for ``apply_bn_momentum()``
which is called once after model construction. Everything else is the
same so the comparison is clean.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0d.py

Expected outcome
----------------
If BN was the cause, you should see:
  - F1 climbs more smoothly (no 0.44 → 0.004 jumps)
  - Multiple consecutive epochs near peak F1
  - First-half-best ≈ second-half-best (training plateau, not collapse)

If BN was NOT the cause:
  - Same oscillating F1 pattern as Phase 0c
  - We then look at other stabilisers (gradient clipping floor,
    different optimizer, larger batch).
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
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


#: BatchNorm running-stats momentum. PyTorch default is 0.1; we use 0.01
#: so the running statistics track ~100 batches instead of ~10. That
#: slows the rate at which BN's idea of "typical activations" shifts as
#: the model trains, which (we hypothesize) is the main cause of the
#: epoch-to-epoch F1 oscillation in Phase 0c.
BN_MOMENTUM = 0.01


def apply_bn_momentum(model: nn.Module, momentum: float) -> int:
    """
    Walk the model and set every BatchNorm layer's ``momentum`` attribute.

    Operates in place. Returns the number of layers modified, useful as
    a sanity check that the override actually hit something.

    Parameters
    ----------
    model : nn.Module
        The model whose BatchNorm layers should be reconfigured.
    momentum : float
        The new momentum value (typical range 0.001-0.1).

    Returns
    -------
    int
        How many BatchNorm layers were updated.
    """
    n_modified = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = momentum
            n_modified += 1
    return n_modified


def main():
    """Run Phase 0d end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0d expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0d_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0d configuration:")
    print(f"  Site: {PHASE0C_SITE} (same-site split, like Phase 0c)")
    print(f"  Val fraction: {PHASE0C_VAL_FRACTION}")
    print(f"  Target class: {TARGET_CLASS_NAME}")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0_EPOCHS}")
    print(f"  *** BN momentum: {BN_MOMENTUM} *** (vs PyTorch default 0.1)")

    # ------------------------------------------------------------------
    # Data — identical to Phase 0c
    # ------------------------------------------------------------------
    print(f"\nLoading data...")
    full_manifest = get_file_manifest([PHASE0C_SITE])
    full_annotations = load_annotations([PHASE0C_SITE], manifest=full_manifest)

    train_manifest, val_manifest = split_manifest_by_file(
        full_manifest, PHASE0C_VAL_FRACTION, seed=PHASE0C_SEED,
    )
    print(f"Train files: {len(train_manifest)}  "
          f"Val files: {len(val_manifest)}")

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
    train_segments = pos_segs + neg_segs
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"Training segments: {len(pos_segs)} positive + "
          f"{len(neg_segs)} negative")
    print(f"Validation segments: {len(val_segments)}")

    train_ds = SingleClassDataset(train_segments, TARGET_CLASS_IDX)
    val_ds = SingleClassDataset(val_segments, TARGET_CLASS_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
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
    # Model + the Phase 0d intervention
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0_model(device)

    # The single change vs Phase 0c: reduce BatchNorm momentum.
    # All other model code, loss, optimizer, training loop are identical.
    n_bn = apply_bn_momentum(model, BN_MOMENTUM)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"BatchNorm layers reconfigured: {n_bn} (momentum={BN_MOMENTUM})")

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
    print(f"Training {PHASE0_EPOCHS} epochs (BN momentum {BN_MOMENTUM})")
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
        }, run_dir / f"phase0d_epoch_{epoch:02d}.pt")

    # ------------------------------------------------------------------
    # Verdict — comparing oscillation behaviour to Phase 0c
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0d VERDICT")
    print(f"{'=' * 60}")
    f1s = [h["f1"] for h in history]
    print(f"F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Best F1: {max(f1s):.3f} at epoch {f1s.index(max(f1s)) + 1}")
    print(f"Final F1: {f1s[-1]:.3f}")

    # Stability metrics: how much does F1 jump epoch-to-epoch in the
    # second half (after the model has roughly converged)?
    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0

    print(f"\nSecond-half stability:")
    print(f"  Mean epoch-to-epoch F1 swing: {mean_swing:.3f}")
    print(f"  Max  epoch-to-epoch F1 swing: {max_swing:.3f}")
    print(f"  Phase 0c reference: max swing 0.44, mean swing ~0.20")
    if max_swing < 0.10:
        print("→ STABILITY FIXED. BN momentum was the issue.")
        print("  Apply BN_MOMENTUM=0.01 to the full pipeline.")
    elif max_swing < 0.20:
        print("→ Partially stabilized. BN helps but isn't the only factor.")
        print("  Try BN_MOMENTUM=0.001, or also disable BN (use GroupNorm).")
    else:
        print("→ Same oscillation as Phase 0c. BN is not the main driver.")
        print("  Investigate batch composition (positive ratio per batch),")
        print("  optimizer choice (AdamW → SGD?), or loss formulation.")


if __name__ == "__main__":
    main()
