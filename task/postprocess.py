"""
Postprocessing and evaluation for BioDCASE 2026 Task 2.

Postprocessing pipeline (matching Whale-VAD):
  1. Median filter smoothing (500 ms kernel)
  2. Per-class thresholding
  3. Merge overlapping detections of the same class
  4. Join detections separated by < 500 ms
  5. Discard detections < 500 ms or > 30 s
  6. Export to challenge CSV format

Evaluation:
  - 1D temporal IoU matching
  - Per-class precision, recall, F1
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    dataset: str
    filename: str
    label: str
    start_s: float        # relative to file start
    end_s: float
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Smoothing & thresholding
# ---------------------------------------------------------------------------

def smooth_probabilities(
    probs: np.ndarray, kernel_size_ms: int = 500, frame_stride_ms: int = 20
) -> np.ndarray:
    """
    Apply median filter to per-frame probabilities.

    Args:
        probs: (T, C) array of probabilities
        kernel_size_ms: kernel size in milliseconds
        frame_stride_ms: stride between frames in milliseconds
    Returns:
        smoothed: (T, C) array
    """
    kernel_frames = max(1, kernel_size_ms // frame_stride_ms)
    if kernel_frames % 2 == 0:
        kernel_frames += 1  # median filter needs odd kernel

    smoothed = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        smoothed[:, c] = median_filter(probs[:, c], size=kernel_frames)
    return smoothed


def threshold_to_detections(
    probs: np.ndarray,
    thresholds: np.ndarray,
    frame_stride_s: float,
    dataset: str,
    filename: str,
    class_names: list[str],
    file_start_sample: int = 0,
    sample_rate: int = 250,
) -> list[Detection]:
    """
    Convert per-frame probabilities to Detection objects.

    Args:
        probs: (T, C) smoothed probabilities
        thresholds: (C,) per-class thresholds
        frame_stride_s: seconds between frames
    """
    detections = []
    T, C = probs.shape

    for c in range(C):
        active = probs[:, c] > thresholds[c]
        # Find contiguous regions
        diffs = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        offset_s = file_start_sample / sample_rate

        for s, e in zip(starts, ends):
            start_s = s * frame_stride_s + offset_s
            end_s = e * frame_stride_s + offset_s
            conf = float(probs[s:e, c].mean())
            detections.append(Detection(
                dataset=dataset,
                filename=filename,
                label=class_names[c],
                start_s=start_s,
                end_s=end_s,
                confidence=conf,
            ))

    return detections


# ---------------------------------------------------------------------------
# Merging & filtering
# ---------------------------------------------------------------------------

def merge_detections(
    detections: list[Detection],
    merge_gap_s: float = 0.5,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
) -> list[Detection]:
    """
    Merge nearby detections of the same class in the same file,
    then filter by duration.
    """
    # Group by (dataset, filename, label)
    groups: dict[tuple, list[Detection]] = {}
    for d in detections:
        key = (d.dataset, d.filename, d.label)
        groups.setdefault(key, []).append(d)

    merged = []
    for key, dets in groups.items():
        dets.sort(key=lambda x: x.start_s)
        current = dets[0]
        for d in dets[1:]:
            if d.start_s - current.end_s < merge_gap_s:
                # Merge
                current = Detection(
                    dataset=current.dataset,
                    filename=current.filename,
                    label=current.label,
                    start_s=current.start_s,
                    end_s=max(current.end_s, d.end_s),
                    confidence=max(current.confidence, d.confidence),
                )
            else:
                merged.append(current)
                current = d
        merged.append(current)

    # Filter by duration
    filtered = [
        d for d in merged
        if min_duration_s <= (d.end_s - d.start_s) <= max_duration_s
    ]
    return filtered


# ---------------------------------------------------------------------------
# Full postprocessing pipeline
# ---------------------------------------------------------------------------

def postprocess_predictions(
    all_probs: dict[tuple[str, str, int], np.ndarray],
    thresholds: np.ndarray,
    class_names: list[str],
    frame_stride_s: float = 0.02,
    sample_rate: int = 250,
    smooth_kernel_ms: int = 500,
    merge_gap_s: float = 0.5,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
) -> list[Detection]:
    """
    Full pipeline from raw probabilities to final detections.

    Args:
        all_probs: dict mapping (dataset, filename, start_sample) → (T, C) probs
    """
    # First, stitch overlapping segments per file
    file_probs = _stitch_segments(all_probs, frame_stride_s, sample_rate)

    all_detections = []
    for (dataset, filename), (probs, _) in file_probs.items():
        probs = smooth_probabilities(probs, smooth_kernel_ms)
        dets = threshold_to_detections(
            probs, thresholds, frame_stride_s,
            dataset, filename, class_names,
            file_start_sample=0, sample_rate=sample_rate,
        )
        all_detections.extend(dets)

    all_detections = merge_detections(
        all_detections, merge_gap_s, min_duration_s, max_duration_s
    )
    return all_detections


def _stitch_segments(
    all_probs: dict[tuple[str, str, int], np.ndarray],
    frame_stride_s: float,
    sample_rate: int,
) -> dict[tuple[str, str], tuple[np.ndarray, int]]:
    """
    Merge overlapping segment predictions by averaging in overlap regions.
    Returns dict mapping (dataset, filename) → (full_probs, total_frames).
    """
    # Group by file
    file_segments: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for (ds, fn, start_sample), probs in all_probs.items():
        key = (ds, fn)
        file_segments.setdefault(key, []).append((start_sample, probs))

    result = {}
    for key, segments in file_segments.items():
        segments.sort(key=lambda x: x[0])

        # Determine total length
        max_end_sample = max(
            start + probs.shape[0] * int(frame_stride_s * sample_rate)
            for start, probs in segments
        )
        total_frames = max_end_sample // int(frame_stride_s * sample_rate) + 1
        n_classes = segments[0][1].shape[1]

        accumulated = np.zeros((total_frames, n_classes), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)

        stride_samples = int(frame_stride_s * sample_rate)
        for start_sample, probs in segments:
            frame_offset = start_sample // stride_samples
            T = probs.shape[0]
            end_frame = min(frame_offset + T, total_frames)
            actual_T = end_frame - frame_offset
            accumulated[frame_offset:end_frame] += probs[:actual_T]
            counts[frame_offset:end_frame] += 1

        counts = np.maximum(counts, 1)
        avg_probs = (accumulated / counts[:, None]).astype(np.float32)
        result[key] = (avg_probs, total_frames)

    return result


# ---------------------------------------------------------------------------
# Challenge CSV export
# ---------------------------------------------------------------------------

def export_challenge_csv(
    detections: list[Detection],
    output_path: str,
    sample_rate: int = 250,
):
    """
    Export detections to the BioDCASE 2026 Task 2 CSV format.

    Expected format:
        dataset,filename,annotation,start_datetime,end_datetime
    """
    rows = []
    for d in detections:
        # Parse file start datetime from filename
        try:
            base = d.filename.replace(".wav", "").split("_")[0]
            file_start = datetime.strptime(base, "%Y-%m-%dT%H-%M-%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        start_dt = file_start + timedelta(seconds=d.start_s)
        end_dt = file_start + timedelta(seconds=d.end_s)

        rows.append({
            "dataset": d.dataset,
            "filename": d.filename,
            "annotation": d.label,
            "start_datetime": start_dt.isoformat(),
            "end_datetime": end_dt.isoformat(),
        })

    # Sort by dataset, filename, start time
    rows.sort(key=lambda r: (r["dataset"], r["filename"], r["start_datetime"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "filename", "annotation", "start_datetime", "end_datetime"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} detections to {output_path}")


# ---------------------------------------------------------------------------
# Evaluation metrics (1D IoU based)
# ---------------------------------------------------------------------------

def compute_iou_1d(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute 1D temporal IoU."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def compute_metrics(
    predictions: list[Detection],
    ground_truth: list[Detection],
    iou_threshold: float = 0.3,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute per-class and overall precision, recall, F1 using 1D IoU matching.
    Implements the challenge rule: each GT can only be matched to one prediction.
    """
    if class_names is None:
        class_names = list(set(d.label for d in predictions + ground_truth))

    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls in class_names:
        cls_preds = [d for d in predictions if d.label == cls]
        cls_gts = [d for d in ground_truth if d.label == cls]

        # Group by file
        file_keys = set(
            (d.dataset, d.filename) for d in cls_preds + cls_gts
        )

        tp, fp, fn = 0, 0, 0
        for fkey in file_keys:
            f_preds = sorted(
                [d for d in cls_preds if (d.dataset, d.filename) == fkey],
                key=lambda x: x.start_s,
            )
            f_gts = sorted(
                [d for d in cls_gts if (d.dataset, d.filename) == fkey],
                key=lambda x: x.start_s,
            )

            matched_gts = set()
            matched_preds = set()

            # For each GT, find best matching prediction
            for gi, gt in enumerate(f_gts):
                best_iou = 0.0
                best_pi = -1
                for pi, pred in enumerate(f_preds):
                    if pi in matched_preds:
                        continue
                    iou = compute_iou_1d(pred.start_s, pred.end_s, gt.start_s, gt.end_s)
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi
                if best_iou >= iou_threshold and best_pi >= 0:
                    tp += 1
                    matched_gts.add(gi)
                    matched_preds.add(best_pi)
                else:
                    fn += 1

            fp += len(f_preds) - len(matched_preds)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[cls] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0 else 0.0
    )
    results["overall"] = {
        "precision": overall_prec, "recall": overall_rec, "f1": overall_f1,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
    }

    return results


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

@torch.no_grad()
def tune_thresholds(
    model: nn.Module,
    val_loader,
    device: torch.device,
    n_classes: int,
    threshold_range: tuple[float, float] = (0.1, 0.9),
    n_steps: int = 17,
) -> torch.Tensor:
    """
    Grid search for per-class thresholds that maximise frame-level F1
    on the validation set.
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_masks = []

    for audio, targets, padding_mask, _metas in val_loader:
        audio = audio.to(device)
        logits = model(audio)
        T_model = logits.size(1)
        T_target = targets.size(1)
        if T_model < T_target:
            targets = targets[:, :T_model, :]
            padding_mask = padding_mask[:, :T_model]
        elif T_model > T_target:
            pad = torch.zeros(targets.size(0), T_model - T_target, targets.size(2))
            targets = torch.cat([targets, pad], dim=1)
            mask_pad = torch.zeros(padding_mask.size(0), T_model - T_target, dtype=torch.bool)
            padding_mask = torch.cat([padding_mask, mask_pad], dim=1)

        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(targets)
        all_masks.append(padding_mask)

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    best_thresholds = []
    candidates = np.linspace(threshold_range[0], threshold_range[1], n_steps)

    for c in range(n_classes):
        mask = all_masks.view(-1).bool()
        targs = all_targets[..., c].view(-1)[mask]
        probs_c = all_probs[..., c].view(-1)[mask]

        best_f1 = 0.0
        best_t = 0.5
        for t in candidates:
            preds = (probs_c > t).float()
            tp = (preds * targs).sum().item()
            fp = (preds * (1 - targs)).sum().item()
            fn = ((1 - preds) * targs).sum().item()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds.append(best_t)
        print(f"  Class {c}: best threshold={best_t:.3f}, F1={best_f1:.3f}")

    return torch.tensor(best_thresholds)
