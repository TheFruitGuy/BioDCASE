"""
Continue Training from the Official Checkpoint
==============================================

Loads ``WhaleVAD_ATBFL_3P-c6f6a07a.pt`` (Geldenhuys et al.) into our
training pipeline and runs a few epochs of continued training, validating
after each one.

Why this script exists
----------------------
We have repeatedly observed that training from random init under our
recipe peaks around F1=0.28 in the first 1-2 epochs and then collapses,
across many different (LR × resample × loss) combinations. This script
isolates whether the problem is in the **training pipeline** (loss,
optimizer, data, BatchNorm dynamics) versus the **initialisation**
(random init in a hard loss landscape).

Two possible outcomes:

  1. The official checkpoint, which evaluates at F1≈0.44 in our
     ``load_official_checkpoint.py`` pipeline, **stays near 0.44** for
     a few epochs of continued training. Conclusion: our training
     pipeline is mostly fine; from-scratch training simply struggles to
     find a similar minimum, likely because of init dynamics or because
     the paper trained much longer.

  2. F1 **drops sharply within 1-2 epochs**, e.g. to 0.10 or below.
     Conclusion: our training pipeline is actively destroying weights
     that work, which points to a real bug in the loss, data targets,
     or BatchNorm — and we should look there, not at hyperparameters.

Either outcome is informative. Run this script before running yet
another from-scratch sweep.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python continue_from_official.py \\
        --checkpoint WhaleVAD_ATBFL_3P-c6f6a07a.pt \\
        --epochs 3 \\
        --lr 1e-6

Notes
-----
* Forces ``num_classes=7`` regardless of the value of ``cfg.USE_3CLASS``,
  because the official checkpoint has a 7-class classifier head.
* Uses our ``validate()`` and ``train_epoch()`` directly, so the loss,
  data loader, and metrics are exactly what ``train.py`` would use —
  this is the point of the test.
* Saves a per-epoch checkpoint to ``runs/continue_from_official_*/``.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from train import validate, train_epoch, set_seed


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the official Geldenhuys checkpoint, e.g. "
                        "WhaleVAD_ATBFL_3P-c6f6a07a.pt.")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of continued-training epochs to run "
                        "(default 3).")
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Learning rate for continued training. Defaults to "
                        "1e-6 — small enough that healthy gradients should "
                        "barely move the weights, so any large F1 swing is "
                        "evidence of a pipeline problem rather than "
                        "legitimate optimization.")
    p.add_argument("--validate_only", action="store_true",
                   help="Skip training entirely; just load the checkpoint "
                        "and run validation. Useful as a baseline before "
                        "the continued-training drift kicks in.")
    return p.parse_args()


# ======================================================================
# Checkpoint key remap — same as load_official_checkpoint.py
# ======================================================================

def remap_checkpoint_keys(state_dict: dict) -> dict:
    """
    Translate official-checkpoint key names into our model's key names.

    Drops the bounding-box regressor head (``bb_proj.*``), which is not
    part of our model. See ``load_official_checkpoint.py`` for the full
    rationale of this remap.
    """
    def rename(k: str) -> str:
        if k.startswith("fbank."):
            return k.replace("fbank.", "filterbank.")
        if k.startswith("cnn_blocks.0."):
            return k.replace("cnn_blocks.0.", "feat_extractor.")
        if k.startswith("cnn_blocks.1.blocks.0."):
            return k.replace("cnn_blocks.1.blocks.0.", "residual_stack.blocks.0.")
        if k.startswith("cnn_blocks.1.blocks.1."):
            return k.replace("cnn_blocks.1.blocks.1.", "residual_stack.blocks.1.")
        return k

    return {
        rename(k): v for k, v in state_dict.items()
        if not k.startswith("bb_proj")
    }


# ======================================================================
# Main
# ======================================================================

def main():
    """Load official checkpoint, optionally train, validate after each epoch."""
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Run directory for any per-epoch checkpoints we save
    # ------------------------------------------------------------------
    run_dir = Path(cfg.OUTPUT_DIR) / f"continue_from_official_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data — exactly the same loaders train.py uses
    # ------------------------------------------------------------------
    train_ds, train_loader, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model — forced to 7 classes to match the official checkpoint
    # ------------------------------------------------------------------
    if cfg.USE_3CLASS:
        print("NOTE: cfg.USE_3CLASS=True, but the official checkpoint outputs "
              "7 classes. We build the model with num_classes=7 here so the "
              "checkpoint loads cleanly. Validation collapses 7→3 via the "
              "logic in validate() / postprocess.collapse_probs_to_3class. "
              "The training data targets, however, will be 3-class because "
              "dataset.py honours cfg.USE_3CLASS — which means training will "
              "fail with a shape mismatch.")
        print()
        print("To run continued training, set cfg.USE_3CLASS=False in "
              "config.py first. To just validate the official checkpoint "
              "without training, pass --validate_only and the shape "
              "mismatch never triggers.")
        if not args.validate_only:
            print("\nAborting. Either pass --validate_only or flip "
                  "cfg.USE_3CLASS to False.")
            return

    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=7).to(device)

    # Forward dummy to materialize the lazy projection layer before
    # load_state_dict, otherwise it would fail with a missing-key error.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    remapped = remap_checkpoint_keys(raw)
    model.load_state_dict(remapped, strict=True)
    print(f"Loaded {args.checkpoint} (strict=True)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss + optimizer — same as train.py would build
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    if pos_weight is not None:
        print(f"Class weights: {pos_weight.tolist()}")
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    print(f"Optimizer: AdamW, LR={args.lr}, weight_decay={cfg.WEIGHT_DECAY}")

    # ------------------------------------------------------------------
    # Initial validation (epoch 0) — this is the reference point. It
    # should reproduce ~F1=0.44 if our pipeline is correct.
    # ------------------------------------------------------------------
    # The thresholds passed to validate() are 3-long because validate()
    # collapses 7→3 before postprocessing. 0.5 across the board gives a
    # default operating point; the absolute value isn't important — we're
    # tracking *change* across epochs.
    thresholds = torch.tensor([0.5, 0.5, 0.5], device=device)

    print("\n" + "=" * 60)
    print("Epoch 0 — validation of the loaded official checkpoint")
    print("=" * 60)
    val0 = validate(
        model, spec_extractor, val_loader, criterion, device,
        thresholds, val_annotations, file_start_dts,
    )
    print(f"\n  Initial val loss: {val0['loss']:.4f}")
    print(f"  Initial F1:       {val0['mean_f1']:.3f}")
    clf = (model.module.classifier if isinstance(model, nn.DataParallel)
           else model.classifier)
    bias_str = ", ".join(f"{b:+.3f}" for b in clf.bias.detach().cpu().tolist())
    print(f"  Classifier bias:  [{bias_str}]")

    # Save the epoch-0 state for diff comparisons later if anyone wants them.
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "f1": val0["mean_f1"],
        "val_loss": val0["loss"],
    }, run_dir / "epoch_00_initial.pt")

    if args.validate_only:
        print("\n--validate_only set; exiting after initial validation.")
        return

    # ------------------------------------------------------------------
    # Continued training — a few short epochs with per-epoch validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Continued training: {args.epochs} epochs at LR={args.lr}")
    print("=" * 60)
    print("If F1 stays near the initial value, our training pipeline is fine")
    print("and the from-scratch difficulty is in initialisation/duration.")
    print("If F1 drops sharply, the pipeline is actively breaking the model")
    print("and we should look at loss / targets / BatchNorm.")

    # Snapshot the initial classifier weights so we can quantify drift later.
    init_clf_weight = clf.weight.detach().cpu().clone()
    init_clf_bias = clf.bias.detach().cpu().clone()

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'-' * 60}\nEpoch {epoch}/{args.epochs}\n{'-' * 60}")

        # We deliberately DO NOT call train_ds.resample_negatives() here
        # because we want a clean A/B comparison: same data each epoch,
        # only the training step changes. If you want resampling, comment
        # this out and uncomment the next two lines.
        # train_ds.resample_negatives()
        # train_loader = DataLoader(...)  # rebuild with new segments

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
        )
        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts,
        )

        # How far did the classifier drift from its loaded values?
        clf = (model.module.classifier if isinstance(model, nn.DataParallel)
               else model.classifier)
        weight_delta = (clf.weight.detach().cpu() - init_clf_weight).norm().item()
        bias_delta = (clf.bias.detach().cpu() - init_clf_bias).norm().item()
        bias_str = ", ".join(f"{b:+.3f}" for b in clf.bias.detach().cpu().tolist())

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  F1: {val['mean_f1']:.3f}  (initial was {val0['mean_f1']:.3f})")
        print(f"  Classifier drift since load — weight: {weight_delta:.4f}, "
              f"bias: {bias_delta:.4f}")
        print(f"  Classifier bias now: [{bias_str}]")

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "f1": val["mean_f1"],
            "val_loss": val["loss"],
            "weight_delta": weight_delta,
            "bias_delta": bias_delta,
        }, run_dir / f"epoch_{epoch:02d}.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    final_f1 = val["mean_f1"]
    delta = final_f1 - val0["mean_f1"]
    print(f"  Initial F1: {val0['mean_f1']:.3f}")
    print(f"  Final F1:   {final_f1:.3f}  (Δ = {delta:+.3f})")
    if delta < -0.10:
        print("  → Pipeline is actively destroying useful weights.")
        print("    Look at loss function, target construction, or BatchNorm.")
    elif delta < -0.03:
        print("  → Mild drift; pipeline is borderline. Worth investigating "
              "but not catastrophic.")
    else:
        print("  → Pipeline preserves good weights. The from-scratch problem "
              "is likely about initialisation dynamics or training duration.")


if __name__ == "__main__":
    main()
