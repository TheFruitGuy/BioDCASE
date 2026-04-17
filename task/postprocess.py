"""
Postprocessing and evaluation (Section 5.8).

Pipeline (paper):
  1. Stitch overlapping segment predictions (average in overlaps)
  2. Median filter with 500ms kernel
  3. Per-class thresholding (thresholds tuned on val)
  4. Collapse 7→3 class
  5. Merge gaps <500ms, discard <500ms or >30s events
  6. Export to challenge CSV

Evaluation: 1D IoU ≥ 0.3 for true positive matching.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import median_filter

import config as cfg


# ----------------------------------------------------------------------

@dataclass
class Detection:
    dataset: str
    filename: str
    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0


# ----------------------------------------------------------------------
# Stitch per-segment probabilities into per-file arrays
# ----------------------------------------------------------------------

def stitch_segments(
    all_probs: dict[tuple[str, str, int], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """
    Merge overlapping segment probs by averaging.
    Input:  dict with keys (dataset, filename, start_sample) → (T_frames, C) probs
    Output: dict with keys (dataset, filename) → (T_total, C) probs
    """
    stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)
    file_segs: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for (ds, fn, start_samp), probs in all_probs.items():
        file_segs.setdefault((ds, fn), []).append((start_samp, probs))

    result = {}
    for key, segs in file_segs.items():
        segs.sort(key=lambda x: x[0])
        max_end = max(s + p.shape[0] * stride_samp for s, p in segs)
        total_frames = max_end // stride_samp + 1
        nc = segs[0][1].shape[1]
        accum  = np.zeros((total_frames, nc), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)
        for start_samp, probs in segs:
            f0 = start_samp // stride_samp
            T = min(probs.shape[0], total_frames - f0)
            accum[f0:f0 + T] += probs[:T]
            counts[f0:f0 + T] += 1
        counts = np.maximum(counts, 1)
        result[key] = (accum / counts[:, None]).astype(np.float32)
    return result


# ----------------------------------------------------------------------
# Median filter (paper: 500ms kernel)
# ----------------------------------------------------------------------

def smooth_probabilities(probs: np.ndarray, kernel_ms: int = cfg.SMOOTH_KERNEL_MS) -> np.ndarray:
    stride_ms = int(cfg.FRAME_STRIDE_S * 1000)
    k = max(1, kernel_ms // stride_ms)
    if k % 2 == 0:
        k += 1
    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = median_filter(probs[:, c], size=k)
    return out


# ----------------------------------------------------------------------
# Threshold → Detection objects
# ----------------------------------------------------------------------

def threshold_to_detections(
    probs: np.ndarray, thresholds: np.ndarray,
    dataset: str, filename: str, offset_sample: int = 0,
) -> list[Detection]:
    names = cfg.class_names()
    dets = []
    T, C = probs.shape
    offset_s = offset_sample / cfg.SAMPLE_RATE

    for c in range(C):
        active = probs[:, c] > thresholds[c]
        diffs = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            dets.append(Detection(
                dataset=dataset, filename=filename, label=names[c],
                start_s=s * cfg.FRAME_STRIDE_S + offset_s,
                end_s=e * cfg.FRAME_STRIDE_S + offset_s,
                confidence=float(probs[s:e, c].mean()),
            ))
    return dets


# ----------------------------------------------------------------------
# Merge & filter (paper: 500ms gap, 500ms-30s duration)
# ----------------------------------------------------------------------

def merge_and_filter(detections: list[Detection]) -> list[Detection]:
    """
    - Collapse 7→3 class labels
    - Per file × class: sort by start, merge with gap < MERGE_GAP_S
    - Discard events outside [POST_MIN_DUR_S, POST_MAX_DUR_S]
    """
    # Collapse labels
    collapsed = []
    for d in detections:
        new_label = cfg.COLLAPSE_MAP.get(d.label, d.label)
        collapsed.append(Detection(
            dataset=d.dataset, filename=d.filename, label=new_label,
            start_s=d.start_s, end_s=d.end_s, confidence=d.confidence,
        ))

    # Group by (file, label)
    groups: dict[tuple, list[Detection]] = {}
    for d in collapsed:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    final = []
    for _, events in groups.items():
        events.sort(key=lambda x: x.start_s)

        # Merge close events
        merged = []
        for e in events:
            if not merged:
                merged.append(e)
            else:
                last = merged[-1]
                if e.start_s - last.end_s <= cfg.MERGE_GAP_S:
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(e)

        # Duration filter
        for m in merged:
            dur = m.end_s - m.start_s
            if cfg.POST_MIN_DUR_S <= dur <= cfg.POST_MAX_DUR_S:
                final.append(m)

    return final


# ----------------------------------------------------------------------
# Full postprocessing pipeline
# ----------------------------------------------------------------------

def postprocess_predictions(
    all_probs: dict[tuple[str, str, int], np.ndarray],
    thresholds: np.ndarray,
) -> list[Detection]:
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)                  # paper: 500ms median
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter(all_dets)


# ----------------------------------------------------------------------
# CSV export (challenge format)
# ----------------------------------------------------------------------

def export_challenge_csv(detections: list[Detection], output_path: str):
    rows = []
    for d in detections:
        file_start = _parse_filename_dt(d.filename)
        if file_start is None:
            continue
        rows.append({
            "dataset": d.dataset,
            "filename": d.filename,
            "annotation": d.label,
            "start_datetime": (file_start + timedelta(seconds=d.start_s)).isoformat(),
            "end_datetime":   (file_start + timedelta(seconds=d.end_s)).isoformat(),
        })
    rows.sort(key=lambda r: (r["dataset"], r["filename"], r["start_datetime"]))

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "filename", "annotation", "start_datetime", "end_datetime",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} detections → {output_path}")


def _parse_filename_dt(filename: str):
    from pathlib import Path
    stem = Path(filename).stem.split("_")[0]
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Event-level 1D IoU evaluation
# ----------------------------------------------------------------------

def compute_iou_1d(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_metrics(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    iou_threshold: float = 0.3,
) -> dict:
    """Per-class + overall P/R/F1 with 1D IoU matching (greedy)."""
    classes = sorted({d.label for d in list(predictions) + list(ground_truth)})
    results = {}
    tp_tot = fp_tot = fn_tot = 0

    for cls in classes:
        cp = [d for d in predictions  if d.label == cls]
        cg = [d for d in ground_truth if d.label == cls]
        files = {(d.dataset, d.filename) for d in cp + cg}
        tp = fp = fn = 0
        for fk in files:
            file_preds = sorted([d for d in cp if (d.dataset, d.filename) == fk],
                                key=lambda x: x.start_s)
            file_gts   = sorted([d for d in cg if (d.dataset, d.filename) == fk],
                                key=lambda x: x.start_s)
            matched = set()
            for gt in file_gts:
                best_iou, best_i = 0.0, -1
                for i, pr in enumerate(file_preds):
                    if i in matched:
                        continue
                    iou = compute_iou_1d(pr.start_s, pr.end_s, gt.start_s, gt.end_s)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_iou >= iou_threshold and best_i >= 0:
                    tp += 1
                    matched.add(best_i)
                else:
                    fn += 1
            fp += len(file_preds) - len(matched)

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        results[cls] = {
            "precision": p, "recall": r,
            "f1": 2*p*r / (p+r + 1e-8),
            "tp": tp, "fp": fp, "fn": fn,
        }
        tp_tot += tp; fp_tot += fp; fn_tot += fn

    p = tp_tot / (tp_tot + fp_tot + 1e-8)
    r = tp_tot / (tp_tot + fn_tot + 1e-8)
    results["overall"] = {"precision": p, "recall": r, "f1": 2*p*r/(p+r+1e-8)}
    return results


# ----------------------------------------------------------------------
# Threshold tuning (paper Section 5.8)
# ----------------------------------------------------------------------

@torch.no_grad()
def tune_thresholds_event_level(
    model: torch.nn.Module,
    spec_extractor,
    val_loader,
    device,
    val_annotations,
    file_start_dts: dict[tuple[str, str], datetime],
) -> np.ndarray:
    """
    Per-class threshold tuning based on event-level F1 (1D IoU ≥ 0.3).
    """
    model.eval()
    all_probs: dict[tuple[str, str, int], np.ndarray] = {}

    for audio, _, _, metas in val_loader:
        audio = audio.to(device)
        spec = spec_extractor(audio)
        logits = model(spec)
        probs = torch.sigmoid(logits).cpu().numpy()

        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Build ground truth Detections
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
            end_s=(row["end_datetime"]   - fsd).total_seconds(),
        ))

    # Grid search per class
    candidates = np.linspace(0.1, 0.9, 17)
    n_classes = cfg.n_classes()
    best_thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        best_f1 = 0.0
        for t_try in candidates:
            thresholds = best_thresholds.copy()
            thresholds[c] = t_try
            preds = postprocess_predictions(all_probs, thresholds)
            metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
            cls_name = cfg.class_names()[c]
            f1 = metrics.get(cls_name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = t_try
        print(f"  class {cfg.class_names()[c]}: "
              f"threshold={best_thresholds[c]:.3f}  F1={best_f1:.3f}")

    return best_thresholds
