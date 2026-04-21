"""
Supervised Training Loop
========================

Main entry point for training the Whale-VAD model on the BioDCASE 2026
development set. Implements the training recipe from Section 5.6 of the
paper, with three stability-focused extensions that proved beneficial in
our reproduction:

    1. **Stochastic undersampling at a lower frequency**: the paper resamples
       negative segments every epoch; we resample every 5 epochs. Reshuffling
       less often reduces validation-loss jitter and makes early-stopping
       decisions more reliable, at no observed cost to final F1.

    2. **ReduceLROnPlateau scheduler**: halves the learning rate if the
       validation F1 stagnates for 5 consecutive epochs. Helps the model
       settle into a minimum after the initial rapid improvement phase.

    3. **Early stopping**: aborts training if validation F1 does not improve
       for 15 epochs, preventing overfitting during the long tail when the
       model memorizes training noise.

After training completes, the best checkpoint (by validation F1) is
automatically re-evaluated with a per-class threshold tuning pass, and the
resulting ``final_model.pt`` contains both the best weights and the tuned
thresholds ready for inference.

Usage
-----
From-scratch reproduction (paper setting)::

    python train.py

With contrastive-pretrained encoder weights::

    python train.py --pretrained path/to/best_pretrained.pt \\
                    --freeze_epochs 5
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest,
    collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level,
)


# ======================================================================
# Stabilization hyperparameters
# ======================================================================
# These values were chosen empirically on the BioDCASE validation set and
# do not appear in the paper. They address stability issues that emerged
# in our reproduction rather than the paper's own training dynamics.

#: Resample negative segments every N epochs. Lower values match the paper
#: more closely but increase validation noise; higher values (e.g. 10) risk
#: overfitting to the same negative distribution.
RESAMPLE_EVERY = 5

#: Stop training if validation F1 does not improve for this many epochs.
EARLY_STOP_PATIENCE = 15

#: Number of stagnant epochs before the LR scheduler fires.
LR_PATIENCE = 5

#: Multiplicative factor applied to the LR each time the scheduler fires.
LR_FACTOR = 0.5

#: Floor for the learning rate; scheduler will not reduce below this.
MIN_LR = 1e-7


# ======================================================================
# CLI and helpers
# ======================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=None,
                   help="Optional path to a contrastive-pretrained encoder "
                        "checkpoint produced by pretrain_contrastive.py.")
    p.add_argument("--freeze_epochs", type=int, default=0,
                   help="If --pretrained is set, keep the encoder weights "
                        "frozen for this many initial epochs before "
                        "unfreezing them for end-to-end fine-tuning.")
    return p.parse_args()


def set_seed(seed: int = cfg.SEED):
    """Seed all stochastic sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_lengths(logits, targets, mask):
    """
    Reconcile small length mismatches between logits and targets.

    The CNN front end can produce one or two fewer/more frames than the
    naive ``n_samples // hop_length`` count, depending on padding and pool
    boundary effects. This helper truncates or zero-pads ``targets`` and
    ``mask`` to match ``logits`` so the loss can be computed unchanged.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, T_m, C)
    targets : torch.Tensor, shape (B, T_t, C)
    mask : torch.Tensor, shape (B, T_t), dtype=bool

    Returns
    -------
    targets, mask : torch.Tensor
        Both now with time dimension equal to ``T_m``.
    """
    T_m, T_t = logits.size(1), targets.size(1)
    if T_m < T_t:
        # Model produced fewer frames than the naive count. Truncate.
        targets = targets[:, :T_m, :]
        mask = mask[:, :T_m]
    elif T_m > T_t:
        # Model produced more frames. Pad targets with zeros and mark the
        # extra positions as invalid so the loss ignores them.
        pad_t = torch.zeros(targets.size(0), T_m - T_t, targets.size(2),
                            device=targets.device)
        targets = torch.cat([targets, pad_t], dim=1)
        pad_m = torch.zeros(mask.size(0), T_m - T_t, dtype=torch.bool,
                            device=mask.device)
        mask = torch.cat([mask, pad_m], dim=1)
    return targets, mask


# ======================================================================
# Validation
# ======================================================================

@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             thresholds, val_annotations, file_start_dts):
    """
    Run the full event-level validation pipeline.

    Performs a forward pass over the entire validation loader, then runs
    the same post-processing pipeline used at inference time (stitch,
    smooth, threshold, merge, filter) and reports per-class and overall
    event-level precision/recall/F1.

    Parameters
    ----------
    model : nn.Module
    spec_extractor : nn.Module
    loader : DataLoader
        Validation loader.
    criterion : WhaleVADLoss
        Used only to compute validation loss for monitoring; does not
        affect metrics.
    device : torch.device
    thresholds : torch.Tensor
        Per-class confidence thresholds.
    val_annotations : pd.DataFrame
        Ground-truth annotations.
    file_start_dts : dict
        Maps ``(dataset, filename)`` → file start datetime.

    Returns
    -------
    dict with keys:
        - ``loss``       : mean validation loss
        - ``mean_f1``    : overall event-level F1
        - ``per_class``  : full per-class metrics dict
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_probs: dict = {}

    for audio, targets, mask, metas in tqdm(loader, desc="Validating", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)
        targets, mask = align_lengths(logits, targets, mask)

        # Compute validation loss for logging; metric decisions below do
        # not depend on this value.
        total_loss += criterion(logits, targets, mask).item()
        n_batches += 1

        # Stash probabilities for event-level evaluation below. We keep
        # per-window raw probs (not yet stitched) so the same data can be
        # reused by threshold tuning without re-running inference.
        probs = torch.sigmoid(logits).cpu().numpy()
        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Run the standard post-processing pipeline end-to-end.
    pred_events = postprocess_predictions(all_probs, thresholds.cpu().numpy())

    # Build ground-truth Detection objects in the same format for matching.
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    # Pretty-print per-class metrics.
    print("\n  Event-level 1D IoU validation:")
    for cls, m in metrics.items():
        if cls == "overall":
            continue
        print(f"    {cls.upper():6} TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_f1": overall_f1,
        "per_class": metrics,
    }


# ======================================================================
# Training loop (one epoch)
# ======================================================================

def train_epoch(model, spec_extractor, loader, criterion, optimizer, device, epoch):
    """
    Run one training epoch.

    Parameters
    ----------
    model, spec_extractor, loader, criterion, optimizer, device
        Standard training objects.
    epoch : int
        1-indexed epoch number (used only for the progress bar).

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        spec = spec_extractor(audio)
        logits = model(spec)
        targets, mask = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets, mask)

        # Defend against rare NaN/Inf gradients. Skipping the batch is
        # preferable to poisoning the optimizer state with bad updates.
        if torch.isnan(loss) or torch.isinf(loss):
            print("*** NaN detected, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Main entry point
# ======================================================================

def main():
    """End-to-end training driver."""
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Timestamped run directory keeps artifacts from different runs isolated.
    run_dir = Path(cfg.OUTPUT_DIR) / f"whalevad_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, train_loader, val_loader = build_dataloaders()

    # We need the validation annotations and file start-times separately
    # (not just via the loader) for event-level metric computation.
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Forward a dummy batch to initialize the lazy projection layer. This
    # must happen before loading any checkpoint; otherwise load_state_dict
    # will report the projection weights as "missing".
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    # Optional: load pretrained encoder weights (from SSL pretraining).
    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
        # strict=False because the pretrained checkpoint may lack the
        # classifier weights, which we intentionally train from scratch.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)}")

    # Multi-GPU (data parallel). Wrapping in DataParallel is transparent to
    # the rest of the loop; state dicts are saved with ``.module`` unwrapped
    # below so checkpoints are GPU-count-agnostic.
    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss and optimizer
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    print(f"Scheduler: ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"Negative resampling: every {RESAMPLE_EVERY} epochs")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    no_improve_epochs = 0
    thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device)

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}\n"
              f"{'=' * 60}")

        # Periodic negative resampling. Dataset shape changes, so we must
        # rebuild the DataLoader to re-shuffle from the new segment list.
        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
            )

        # Encoder freeze schedule (only active with --pretrained).
        if args.pretrained and epoch <= args.freeze_epochs:
            # Freeze everything except BiLSTM + classifier: the encoder
            # weights are already good from SSL; only the downstream
            # temporal model and classifier need to be trained.
            for name, p in model.named_parameters():
                if "classifier" not in name and "lstm" not in name:
                    p.requires_grad = False
            print("  [frozen encoder]")
        elif args.pretrained and epoch == args.freeze_epochs + 1:
            # Unfreeze for end-to-end fine-tuning.
            for p in model.parameters():
                p.requires_grad = True
            print("  [unfroze encoder]")

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
        )

        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts,
        )

        scheduler.step(val["mean_f1"])

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}  Best F1: {best_f1:.3f}")

        # Always unwrap DataParallel before saving, so checkpoints load
        # cleanly on any GPU count.
        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs}/{EARLY_STOP_PATIENCE} epochs")

        # Always save the latest checkpoint so training can be resumed
        # from the very last state if a run is killed.
        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs")
            break

    # ------------------------------------------------------------------
    # Post-training threshold tuning
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])

    tuned = tune_thresholds_event_level(
        model_to_load, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    # Save a third checkpoint that bundles best weights + tuned thresholds;
    # this is the checkpoint to pass to inference.py.
    final_state = model_to_load.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "thresholds": torch.tensor(tuned),
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")
    print(f"Next: python tune_thresholds.py --checkpoint "
          f"{run_dir}/best_model.pt")


if __name__ == "__main__":
    main()
