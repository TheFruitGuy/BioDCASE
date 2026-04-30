"""
Phase 0c: Same-Site Validation Sanity Baseline
==============================================

Phase 0 (Phase 0a) used kerguelen2005 for training and casey2017 for
validation — completely different sites, different hydrophones,
different ocean environments. F1 stayed at zero across 10 epochs while
train loss dropped cleanly. Diagnosis: the model fits training data but
doesn't transfer between sites.

This script (Phase 0c) eliminates the cross-site distribution shift by
splitting *one site* into train and validation portions:

  - Take all files in kerguelen2005
  - Reserve the last 20% (deterministic split by sorted filename) for
    validation
  - Train on the remaining 80%
  - Validate on the held-out 20%

Same model, same loss, same hyperparameters as Phase 0a. The only thing
that changes is the train/val split. So if F1 climbs here while it
stayed at zero in Phase 0a, we have clean evidence that the issue is
cross-site generalization, not the pipeline itself.

Three possible outcomes:

  1. F1 climbs to 0.30+ and stays. → Pipeline is fine. The full-pipeline
     train/val gap is purely site-shift, and we should look at
     domain-adaptation tricks (matched-site training subsets, BatchNorm
     adjustments at inference, simple feature normalisation across
     sites, etc.).

  2. F1 climbs but mildly (0.10-0.20). → Within-site is partially
     learnable but the small per-site dataset is the main bottleneck.
     Same conclusion as outcome 1, slightly weaker.

  3. F1 stays at zero. → There's something fundamentally broken in the
     pipeline that has nothing to do with site shift. Investigate target
     construction, stitching offsets, or the loss/sigmoid pairing.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0c.py
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
    collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
)
from train_phase0 import (
    SingleClassDataset, build_phase0_model, validate_one_class,
    train_one_epoch,
    TARGET_CLASS_IDX, TARGET_CLASS_NAME,
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE,
    PHASE0_EPOCHS, PHASE0_THRESHOLD,
)


#: Use the same single site for both train and validation.
PHASE0C_SITE = "kerguelen2005"

#: Fraction of files held out for validation. Files are sorted
#: deterministically before splitting so the same files always end up in
#: the validation set across runs.
PHASE0C_VAL_FRACTION = 0.20

#: A fixed random seed for the within-site split. Not strictly needed
#: since we sort by filename, but useful if anyone wants to swap the
#: split logic for a random partition later.
PHASE0C_SEED = 42


def split_manifest_by_file(manifest, val_fraction: float, seed: int = 42):
    """
    Deterministic 80/20 file-level split of a manifest DataFrame.

    Files are sorted alphabetically and split by index, so the same files
    always land in the same partition. This avoids the trap where train
    and val accidentally share files (which would inflate F1 by giving
    the model identical-distribution data at evaluation time).

    Parameters
    ----------
    manifest : pandas.DataFrame
        File manifest with at least a ``filename`` column.
    val_fraction : float
        Fraction of files to put in validation, e.g. ``0.20`` for 20%.
    seed : int
        Unused for the sorted-split implementation; preserved so this
        function can be swapped for a random-split version without
        changing callers.

    Returns
    -------
    train_manifest, val_manifest : pandas.DataFrame
        The two partitions of the input manifest.
    """
    sorted_man = manifest.sort_values("filename").reset_index(drop=True)
    n = len(sorted_man)
    n_val = max(1, int(n * val_fraction))
    train_manifest = sorted_man.iloc[:-n_val].reset_index(drop=True)
    val_manifest = sorted_man.iloc[-n_val:].reset_index(drop=True)
    return train_manifest, val_manifest


def main():
    """Run Phase 0c end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0c expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0c", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
    })

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0c_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0c configuration (same-site split):")
    print(f"  Site: {PHASE0C_SITE}")
    print(f"  Val fraction: {PHASE0C_VAL_FRACTION}")
    print(f"  Target class: {TARGET_CLASS_NAME}")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0_EPOCHS}")

    # ------------------------------------------------------------------
    # Build manifest and annotations for the single site, then split
    # ------------------------------------------------------------------
    print(f"\nLoading data...")
    full_manifest = get_file_manifest([PHASE0C_SITE])
    full_annotations = load_annotations([PHASE0C_SITE], manifest=full_manifest)

    train_manifest, val_manifest = split_manifest_by_file(
        full_manifest, PHASE0C_VAL_FRACTION, seed=PHASE0C_SEED,
    )
    print(f"Train files: {len(train_manifest)}  "
          f"Val files: {len(val_manifest)}")

    # Slice annotations by file partition. This is the critical step:
    # any annotation whose filename is in the val manifest goes into the
    # val annotations; all others go into train. This guarantees zero
    # file-level leakage between the two partitions.
    val_filenames = set(val_manifest["filename"])
    train_annotations = full_annotations[
        ~full_annotations["filename"].isin(val_filenames)
    ].reset_index(drop=True)
    val_annotations = full_annotations[
        full_annotations["filename"].isin(val_filenames)
    ].reset_index(drop=True)
    print(f"Train annotations: {len(train_annotations)}  "
          f"Val annotations: {len(val_annotations)}")

    # Sanity check: zero filename overlap between the two partitions.
    assert (set(train_annotations["filename"])
            & set(val_annotations["filename"])) == set(), \
        "FATAL: train and val share files! split_manifest_by_file is broken."

    # ------------------------------------------------------------------
    # Build training segments (positives + negatives, fixed for the run)
    # ------------------------------------------------------------------
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
    # Model + loss + optimizer (identical to Phase 0a)
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

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
    print(f"Training {PHASE0_EPOCHS} epochs (same-site validation)")
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
        }, run_dir / f"phase0c_epoch_{epoch:02d}.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0c VERDICT")
    print(f"{'=' * 60}")
    f1s = [h["f1"] for h in history]
    print(f"F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Best F1: {max(f1s):.3f} at epoch {f1s.index(max(f1s)) + 1}")
    print(f"Final F1: {f1s[-1]:.3f}")

    first_half_max = max(f1s[:len(f1s) // 2])
    second_half_max = max(f1s[len(f1s) // 2:])
    print(f"\nFirst-half best:  {first_half_max:.3f}")
    print(f"Second-half best: {second_half_max:.3f}")

    if max(f1s) > 0.30:
        print("→ Within-site training works. The full-pipeline failure is")
        print("  cross-site distribution shift, not a pipeline bug.")
        print("  Next: per-site BatchNorm, site-aware sampling, or simple")
        print("  feature normalisation per recording.")
    elif max(f1s) > 0.10:
        print("→ Within-site training partially works; ceiling is low.")
        print("  Per-site dataset is small. Consider multi-site training")
        print("  on similar Antarctic sites.")
    else:
        print("→ Even within-site training failed. The pipeline has a")
        print("  deeper problem. Investigate target construction or")
        print("  stitching alignment more carefully.")


    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    verdict_text = (
        f"Phase 0c: best F1 {max(f1s):.3f} at epoch "
        f"{f1s.index(max(f1s)) + 1}, final F1 {f1s[-1]:.3f}."
    )
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=None,
    )

if __name__ == "__main__":
    main()
