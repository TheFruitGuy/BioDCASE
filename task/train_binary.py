"""
Binary Whale-VAD training — one class per model.

Usage:
    python train_binary.py --class bmabz
    python train_binary.py --class d
    python train_binary.py --class bp

Each model is trained independently with the full WhaleVAD architecture
but with num_classes=1. Only this class's annotations become positive
targets; all other annotations are ignored (neither positive nor
explicit negative). Negative segments are randomly sampled from files
that don't contain THIS class.

At inference, run all three models and concatenate detections.
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


RESAMPLE_EVERY = 5
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="target_class", type=str, required=True,
                   choices=cfg.CALL_TYPES_3,
                   help="Which class to train (bmabz / d / bp)")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    return p.parse_args()


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_lengths(logits, targets, mask):
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


# ----------------------------------------------------------------------
# Binary dataset — filter annotations to one class
# ----------------------------------------------------------------------

class BinaryWhaleDataset(WhaleDataset):
    """WhaleDataset that only targets one class (num_classes=1)."""
    def __init__(self, segments: list[Segment], target_class: str):
        super().__init__(segments)
        self.target_class = target_class
        self.n_classes = 1      # override

    def __getitem__(self, idx: int):
        import soundfile as sf

        seg = self.segments[idx]
        n_samples = seg.end_sample - seg.start_sample

        audio, sr = sf.read(
            seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32"
        )
        audio = torch.from_numpy(audio)

        n_frames = n_samples // self.stride_samp
        targets = torch.zeros(n_frames, 1)            # single-class

        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE
        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label != self.target_class:
                continue
            local_start_s = max(0.0, a["start_s"] - seg_start_s)
            local_end_s   = min(n_samples / cfg.SAMPLE_RATE, a["end_s"] - seg_start_s)
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
    """Training dataset with resampleable negatives, for binary task."""
    def __init__(self, positive_segments, manifest, annotations, target_class):
        self.positive_segments = positive_segments
        self.manifest = manifest
        self.annotations = annotations
        self.target_class = target_class
        self.negative_segments = []
        self.resample_negatives()
        super().__init__(self.positive_segments + self.negative_segments, target_class)

    def resample_negatives(self):
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.segments = self.positive_segments + self.negative_segments


# ----------------------------------------------------------------------
# Filter training set to segments that are positive for target class
# ----------------------------------------------------------------------

def filter_segments_for_class(
    annotations: pd.DataFrame, target_class: str
) -> pd.DataFrame:
    """Return only annotations for this class."""
    orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == target_class]
    return annotations[annotations["annotation"].isin(orig_labels)].reset_index(drop=True)


# ----------------------------------------------------------------------
# Validation (binary)
# ----------------------------------------------------------------------

@torch.no_grad()
def validate_binary(model, spec_extractor, loader, criterion, device,
                    threshold, val_annotations, file_start_dts, target_class):
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

    # Postprocess with single threshold
    thresholds = np.array([threshold])

    # Build detections for this class only
    from postprocess import stitch_segments, smooth_probabilities, threshold_to_detections, merge_and_filter

    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        # Single class: relabel as target_class
        for c in range(probs.shape[1]):
            active = probs[:, c] > threshold
            diffs = np.diff(active.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                all_dets.append(Detection(
                    dataset=ds, filename=fn, label=target_class,
                    start_s=s * cfg.FRAME_STRIDE_S, end_s=e * cfg.FRAME_STRIDE_S,
                    confidence=float(probs[s:e, c].mean()),
                ))

    # Apply merge + duration filter
    from postprocess import merge_and_filter
    pred_events = merge_and_filter(all_dets)

    # Ground truth for this class only
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
            end_s=(row["end_datetime"]   - fsd).total_seconds(),
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


# ----------------------------------------------------------------------
# Train epoch
# ----------------------------------------------------------------------

def train_epoch(model, spec_extractor, loader, criterion, optimizer, device, epoch, total_epochs):
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

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed()
    target_class = args.target_class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target class: {target_class}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"binary_{target_class}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ── Build data ────────────────────────────────────────────────
    print(f"\nLoading train datasets...")
    train_annotations_all = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)

    # Only positives for this class
    train_annotations_class = filter_segments_for_class(train_annotations_all, target_class)
    print(f"  Total annotations: {len(train_annotations_all)}")
    print(f"  {target_class} annotations: {len(train_annotations_class)}")

    # Positive segments for this class
    pos_segs = build_positive_segments(train_annotations_class, train_manifest)
    print(f"  Positive segments: {len(pos_segs)}")

    # Training dataset — negatives are "no calls of ANY type" so we use full annotations for overlap check
    train_ds = BinaryTrainingDataset(
        pos_segs, train_manifest, train_annotations_all, target_class,
    )
    print(f"  Negative segments: {len(train_ds.negative_segments)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # Validation (full val set, but binary targets)
    print(f"\nLoading val datasets...")
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
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

    # ── Model ─────────────────────────────────────────────────────
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=1).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Loss (no pos_weight — binary, 1:1 sampling handles it) ────
    criterion = WhaleVADLoss(pos_weight=None).to(device)

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    # ── Training loop ─────────────────────────────────────────────
    best_f1 = 0.0
    no_improve = 0
    threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}  LR={lr:.2e}\n{'='*60}")

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print(f"  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
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
            "epoch": epoch, "model_state_dict": model_state,
            "best_f1": best_f1, "target_class": target_class,
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
            print(f"\n  Early stopping")
            break

    # ── Threshold tuning ──────────────────────────────────────────
    print(f"\n{'='*60}\nTuning threshold for {target_class}\n{'='*60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])
    model_to_load.eval()

    # Collect probs once
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

    # GT events for this class
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
            end_s=(row["end_datetime"]   - fsd).total_seconds(),
        ))

    # Try thresholds (finer for rare classes)
    if target_class == "bmabz":
        candidates = np.arange(0.2, 0.8, 0.05)
    else:
        candidates = np.concatenate([np.arange(0.02, 0.4, 0.02), np.arange(0.4, 0.8, 0.05)])

    from postprocess import stitch_segments, smooth_probabilities

    best_thresh = 0.5
    best_f1_tuned = 0.0

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
                    start_s=s * cfg.FRAME_STRIDE_S, end_s=e * cfg.FRAME_STRIDE_S,
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

    # Save final
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
