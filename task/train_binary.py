"""
Binary Per-Class Whale-VAD Training
===================================

Trains a specialized single-class Whale-VAD model. Unlike the main
``train.py`` which trains one multi-label model over all three classes,
this script trains one binary detector per class. Three separate runs
(one each for ``bmabz``, ``d``, ``bp``) produce three models whose
predictions are then combined by ``inference_binary.py`` at test time.

Motivation
----------
The three target classes have very different properties:

    - **bmabz**: abundant (~24k annotations), temporally coherent, frequency-
      localized. The multi-label model already performs reasonably well.
    - **bp**:    moderately abundant but spectrally variable across
      sub-types (bpd / bp20 / bp20plus all collapse into bp).
    - **d**:     rarest (~13k annotations), highly variable downsweep
      whose acoustic signature overlaps with environmental noise.

In the multi-label model the loss is dominated by the common class, and
the model effectively learns a single shared representation that is a
compromise between all three. Binary specialization lets each model
dedicate all of its capacity to one class and use class-appropriate
thresholds and training hyperparameters.

Usage
-----
::

    python train_binary.py --class bmabz
    python train_binary.py --class d
    python train_binary.py --class bp

Each invocation creates a separate run directory under ``runs/`` and
saves ``final_model.pt`` containing both the trained weights and the
tuned threshold.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss
from dataset import (
    load_annotations, get_file_manifest, collate_fn,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, Segment,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
)


# ======================================================================
# Stabilization hyperparameters (see train.py for rationale)
# ======================================================================

RESAMPLE_EVERY = 5
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-7


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="target_class", type=str, required=True,
                   choices=cfg.CALL_TYPES_3,
                   help="Which class to train a specialized binary model for.")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                   help="Maximum training epochs (early stopping may cut short).")
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

    See ``train.align_lengths`` for details. Duplicated here to avoid a
    cross-script import dependency for this utility.
    """
    T_m, T_t = logits.size(1), targets.size(1)
    if T_m < T_t:
        targets = targets[:, :T_m, :]
        mask = mask[:, :T_m]
    elif T_m > T_t:
        pad_t = torch.zeros(targets.size(0), T_m - T_t, targets.size(2),
                            device=targets.device)
        targets = torch.cat([targets, pad_t], dim=1)
        pad_m = torch.zeros(mask.size(0), T_m - T_t, dtype=torch.bool,
                            device=mask.device)
        mask = torch.cat([mask, pad_m], dim=1)
    return targets, mask


# ======================================================================
# Binary dataset wrappers
# ======================================================================

class BinaryWhaleDataset(WhaleDataset):
    """
    Variant of ``WhaleDataset`` whose targets are restricted to one class.

    Only annotations matching ``target_class`` contribute positive frames;
    annotations for the other two classes become implicit negatives (they
    are neither explicitly positive nor excluded). This is important
    because it means the binary model learns to distinguish its target
    class from calls of other classes, not just from silence.

    Parameters
    ----------
    segments : list of Segment
        Pre-built segments.
    target_class : str
        Coarse class label this dataset targets (one of CALL_TYPES_3).
    """

    def __init__(self, segments: list[Segment], target_class: str):
        super().__init__(segments)
        self.target_class = target_class
        # Override the parent's multi-class setting.
        self.n_classes = 1

    def __getitem__(self, idx: int):
        """
        Load one segment and build a 1-dim (binary) frame-level target.

        Returns
        -------
        audio : torch.Tensor
        targets : torch.Tensor, shape (n_frames, 1)
            Single-class binary labels.
        mask : torch.Tensor, dtype=bool
        meta : dict
        """
        import soundfile as sf

        seg = self.segments[idx]
        n_samples = seg.end_sample - seg.start_sample

        audio, sr = sf.read(
            seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32"
        )
        audio = torch.from_numpy(audio)

        n_frames = n_samples // self.stride_samp
        targets = torch.zeros(n_frames, 1)  # single-class targets

        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE
        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            # Skip annotations of other classes: they should not contribute
            # positive labels to this binary model, but their mere presence
            # in the segment is fine (implicit negative context).
            if label != self.target_class:
                continue
            local_start_s = max(0.0, a["start_s"] - seg_start_s)
            local_end_s = min(n_samples / cfg.SAMPLE_RATE, a["end_s"] - seg_start_s)
            f0 = int(local_start_s / cfg.FRAME_STRIDE_S)
            f1 = int(local_end_s / cfg.FRAME_STRIDE_S)
            targets[f0:f1, 0] = 1.0

        mask = torch.ones(n_frames, dtype=torch.bool)
        meta = {
            "dataset": seg.dataset,
            "filename": seg.filename,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
        }
        return audio, targets, mask, meta


class BinaryTrainingDataset(BinaryWhaleDataset):
    """
    Binary-target training dataset with resampleable negative segments.

    Parameters
    ----------
    positive_segments : list of Segment
        Segments containing at least one target-class annotation.
    manifest : pd.DataFrame
        Full file manifest (used by the negative sampler).
    annotations : pd.DataFrame
        Annotations for *all* classes, not just the target. Passed to the
        negative sampler so that it avoids windows overlapping any call
        (not just target-class calls).
    target_class : str
    """

    def __init__(self, positive_segments, manifest, annotations, target_class):
        self.positive_segments = positive_segments
        self.manifest = manifest
        self.annotations = annotations
        self.target_class = target_class
        self.negative_segments = []
        self.resample_negatives()
        super().__init__(self.positive_segments + self.negative_segments, target_class)

    def resample_negatives(self):
        """Draw a fresh set of negative segments for this epoch."""
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.segments = self.positive_segments + self.negative_segments


# ======================================================================
# Annotation filtering
# ======================================================================

def filter_segments_for_class(
    annotations: pd.DataFrame, target_class: str,
) -> pd.DataFrame:
    """
    Return only the annotations that belong to ``target_class`` after
    7→3 label collapsing.

    Parameters
    ----------
    annotations : pd.DataFrame
        Full annotations DataFrame.
    target_class : str
        Coarse 3-class label.

    Returns
    -------
    pd.DataFrame
        Subset of the input containing only rows whose raw annotation
        label maps to ``target_class`` via ``COLLAPSE_MAP``.
    """
    orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == target_class]
    return annotations[annotations["annotation"].isin(orig_labels)].reset_index(drop=True)


# ======================================================================
# Binary validation
# ======================================================================

@torch.no_grad()
def validate_binary(model, spec_extractor, loader, criterion, device,
                    threshold, val_annotations, file_start_dts, target_class):
    """
    Single-class event-level validation.

    Mirrors ``train.validate`` but works with a 1-class model and only
    evaluates against ground-truth annotations of ``target_class``.

    Parameters
    ----------
    model : nn.Module
    spec_extractor : nn.Module
    loader : DataLoader
    criterion : WhaleVADLoss
    device : torch.device
    threshold : float
        Single confidence threshold for the binary model.
    val_annotations : pd.DataFrame
    file_start_dts : dict
    target_class : str

    Returns
    -------
    dict
        ``{"loss": mean loss, "f1": target-class F1}``.
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
        total_loss += criterion(logits, targets, mask).item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Reimplement the post-processing pipeline here because the shared
    # helper assumes a multi-class model and uses cfg.class_names() to
    # label detections; we need single-class output labelled with the
    # target class name.
    from postprocess import (
        stitch_segments, smooth_probabilities, merge_and_filter,
    )

    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        # Single-class model: only one probability channel to threshold.
        active = probs[:, 0] > threshold
        diffs = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            all_dets.append(Detection(
                dataset=ds, filename=fn, label=target_class,
                start_s=s * cfg.FRAME_STRIDE_S, end_s=e * cfg.FRAME_STRIDE_S,
                confidence=float(probs[s:e, 0].mean()),
            ))

    pred_events = merge_and_filter(all_dets)

    # Ground-truth events, filtered to target class only.
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        if label != target_class:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=target_class,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    f1 = metrics.get(target_class, {}).get("f1", 0.0)
    p = metrics.get(target_class, {}).get("precision", 0.0)
    r = metrics.get(target_class, {}).get("recall", 0.0)
    tp = metrics.get(target_class, {}).get("tp", 0)
    fp = metrics.get(target_class, {}).get("fp", 0)
    fn = metrics.get(target_class, {}).get("fn", 0)

    print(f"\n  {target_class.upper():6} TP={tp:5} FP={fp:6} FN={fn:6}  "
          f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "f1": f1,
    }


# ======================================================================
# Training loop (one epoch)
# ======================================================================

def train_epoch(model, spec_extractor, loader, criterion, optimizer, device,
                epoch, total_epochs):
    """
    Run one training epoch. Same semantics as ``train.train_epoch`` but
    works with the single-class binary targets.
    """
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        spec = spec_extractor(audio)
        logits = model(spec)
        targets, mask = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets, mask)

        # Skip bad batches instead of crashing; occasional NaN from the
        # BiLSTM is a rare but recoverable event.
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    """End-to-end binary training driver."""
    args = parse_args()
    set_seed()
    target_class = args.target_class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target class: {target_class}")

    # Class-named run directory so the three binary runs are easy to tell
    # apart on disk.
    run_dir = Path(cfg.OUTPUT_DIR) / (
        f"binary_{target_class}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------
    print("\nLoading train datasets...")
    train_annotations_all = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)

    # Filter to target-class positives only. The full annotation set is
    # still kept around for negative sampling.
    train_annotations_class = filter_segments_for_class(train_annotations_all,
                                                         target_class)
    print(f"  Total annotations: {len(train_annotations_all)}")
    print(f"  {target_class} annotations: {len(train_annotations_class)}")

    pos_segs = build_positive_segments(train_annotations_class, train_manifest)
    print(f"  Positive segments: {len(pos_segs)}")

    # Negative sampler uses the full annotations dataframe so that windows
    # overlapping any call (including non-target classes) are excluded —
    # we want negatives to be real silence, not other-class events.
    train_ds = BinaryTrainingDataset(
        pos_segs, train_manifest, train_annotations_all, target_class,
    )
    print(f"  Negative segments: {len(train_ds.negative_segments)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Validation data
    # ------------------------------------------------------------------
    print("\nLoading val datasets...")
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    # Full fixed-window tiling; the binary dataset will filter to target
    # class targets internally.
    val_segs = build_val_segments(val_manifest, val_annotations)
    val_ds = BinaryWhaleDataset(val_segs, target_class)
    print(f"  Val segments: {len(val_segs)}")

    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    # num_classes=1 → binary model with a single output channel.
    model = WhaleVAD(num_classes=1).to(device)

    # Initialize the lazy projection layer.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss and optimizer
    # ------------------------------------------------------------------
    # No positive weight is needed for the binary model: the 1:1 positive/
    # negative segment ratio (from NEG_RATIO=1.0) already balances the
    # segment-level label distribution, and the frame-level imbalance
    # within positive segments is moderate enough that plain BCE works.
    criterion = WhaleVADLoss(pos_weight=None).to(device)

    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    no_improve = 0
    threshold = 0.5  # default threshold; tuned after training

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{args.epochs}  "
              f"LR={lr:.2e}\n{'=' * 60}")

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
            )

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device,
            epoch, args.epochs,
        )

        val = validate_binary(
            model, spec_extractor, val_loader, criterion, device,
            threshold, val_annotations, file_start_dts, target_class,
        )

        scheduler.step(val["f1"])

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  F1: {val['f1']:.3f}  Best F1: {best_f1:.3f}")

        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_f1": best_f1,
            "target_class": target_class,
            "threshold": threshold,
        }

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{EARLY_STOP_PATIENCE} epochs")

        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve >= EARLY_STOP_PATIENCE:
            print("\n  Early stopping")
            break

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------
    # Per-class grid chosen empirically: bmabz scores are well-separated
    # so a coarse 0.05 grid suffices; d and bp have flatter score
    # distributions (rare positives), so we use a finer low-threshold
    # grid to find the sensitivity sweet spot.
    print(f"\n{'=' * 60}\nTuning threshold for {target_class}\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])
    model_to_load.eval()

    # Collect per-window probabilities once; re-use for all threshold trials.
    all_probs = {}
    hop = spec_extractor.hop_length
    with torch.no_grad():
        for audio, _, _, metas in tqdm(val_loader, desc="Collecting probs"):
            audio = audio.to(device)
            logits = model_to_load(spec_extractor(audio))
            probs = torch.sigmoid(logits).cpu().numpy()
            for j, meta in enumerate(metas):
                key = (meta["dataset"], meta["filename"], meta["start_sample"])
                n_samp = meta["end_sample"] - meta["start_sample"]
                n_frames = min(n_samp // hop, probs[j].shape[0])
                all_probs[key] = probs[j, :n_frames, :]

    # GT events for the target class.
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        if label != target_class:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=target_class,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # Class-specific threshold candidate grids.
    if target_class == "bmabz":
        # Common class: coarse grid is fine.
        candidates = np.arange(0.2, 0.8, 0.05)
    else:
        # Rare classes: finer grid near the low end where the optimum
        # typically lies due to reduced prior probability.
        candidates = np.concatenate([
            np.arange(0.02, 0.4, 0.02),
            np.arange(0.4, 0.8, 0.05),
        ])

    from postprocess import stitch_segments, smooth_probabilities

    best_thresh = 0.5
    best_f1_tuned = 0.0

    # Smooth once outside the threshold loop (smoothing is independent
    # of the threshold, and is expensive enough to hoist).
    file_probs = stitch_segments(all_probs)
    smoothed = {k: smooth_probabilities(v) for k, v in file_probs.items()}

    for t in candidates:
        all_dets = []
        for (ds, fn), probs in smoothed.items():
            active = probs[:, 0] > t
            diffs = np.diff(active.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                all_dets.append(Detection(
                    dataset=ds, filename=fn, label=target_class,
                    start_s=s * cfg.FRAME_STRIDE_S,
                    end_s=e * cfg.FRAME_STRIDE_S,
                    confidence=float(probs[s:e, 0].mean()),
                ))
        from postprocess import merge_and_filter
        preds = merge_and_filter(all_dets)
        m = compute_metrics(preds, gt_events, iou_threshold=0.3)
        f1 = m.get(target_class, {}).get("f1", 0.0)
        if f1 > best_f1_tuned:
            best_f1_tuned = f1
            best_thresh = t

    print(f"\nBest threshold: {best_thresh:.3f}")
    print(f"Tuned F1: {best_f1_tuned:.3f}")

    # Save the final bundle: best weights + tuned threshold + metadata.
    final_state = model_to_load.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "target_class": target_class,
        "threshold": float(best_thresh),
        "tuned_f1": float(best_f1_tuned),
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default 0.5): {best_f1:.3f}")
    print(f"Tuned F1 (thresh={best_thresh:.3f}): {best_f1_tuned:.3f}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
