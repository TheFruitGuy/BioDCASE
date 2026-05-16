"""
Triton — Post-processing and evaluation
=======================================

Converts the model's per-frame probability outputs into discrete
detection events, then evaluates against ground-truth annotations using
event-level 1D IoU matching.

Pipeline
--------
    1. Stitch     — average overlapping-window predictions into a single
                    per-file probability stream.
    2. Smooth     — 500 ms median filter along the time axis.
    3. Threshold  — class-specific per-frame thresholds (tuned on val).
    4. Collapse   — fine→coarse label map (no-op if model already
                    outputs the 3 coarse classes).
    5. Merge      — combine same-class events separated by < MERGE_GAP_S.
    6. Filter     — drop events outside [POST_MIN_DUR_S, POST_MAX_DUR_S].
    7. Export     — challenge-format CSV.

Evaluation
----------
Greedy event-level matching at 1D IoU ≥ 0.3. Precision, recall, F1
reported per class and overall.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import median_filter

import config as cfg


# ======================================================================
# 7-class → 3-class probability collapse
# ======================================================================
# Used when the model outputs 7 fine-grained logits but evaluation is on
# the 3 coarse classes. Max-pooling within each coarse group: a frame
# is positive for the coarse class if any of its constituent fine-grained
# outputs fired strongly.

_SEVEN_TO_THREE = {
    "bmabz": [cfg.CALL_TYPES_7.index(x) for x in ("bma", "bmb", "bmz")],
    "d":     [cfg.CALL_TYPES_7.index(x) for x in ("bmd", "bpd")],
    "bp":    [cfg.CALL_TYPES_7.index(x) for x in ("bp20", "bp20plus")],
}


def collapse_probs_to_3class(all_probs: dict) -> dict:
    """
    Collapse a dict of per-window 7-class probabilities to 3-class.

    No-op when ``cfg.USE_3CLASS`` is True or when the arrays already
    have 3 channels, so this is safe to call unconditionally.
    """
    if cfg.USE_3CLASS or not all_probs:
        return all_probs

    sample = next(iter(all_probs.values()))
    if sample.shape[1] != 7:
        return all_probs

    out = {}
    for key, p7 in all_probs.items():
        p3 = np.zeros((p7.shape[0], 3), dtype=p7.dtype)
        for i, name in enumerate(cfg.CALL_TYPES_3):
            p3[:, i] = p7[:, _SEVEN_TO_THREE[name]].max(axis=1)
        out[key] = p3
    return out


# ======================================================================
# Detection dataclass
# ======================================================================

@dataclass
class Detection:
    """A single predicted or ground-truth event."""

    dataset: str
    filename: str
    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0


# ======================================================================
# Stitching
# ======================================================================

def stitch_segments(
    all_probs: dict[tuple[str, str, int], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """
    Merge overlapping-window predictions into per-file streams by
    averaging in overlap regions. Frames covered by no window stay at
    zero, but in practice every frame is covered (the 2-second overlap
    guarantees it).
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

        accum = np.zeros((total_frames, nc), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)

        for start_samp, probs in segs:
            f0 = start_samp // stride_samp
            T = min(probs.shape[0], total_frames - f0)
            accum[f0:f0 + T] += probs[:T]
            counts[f0:f0 + T] += 1

        counts = np.maximum(counts, 1)
        result[key] = (accum / counts[:, None]).astype(np.float32)

    return result


# ======================================================================
# Smoothing
# ======================================================================

def smooth_probabilities(
    probs: np.ndarray, kernel_ms: int = cfg.SMOOTH_KERNEL_MS,
) -> np.ndarray:
    """500 ms median filter along the time axis (per class)."""
    stride_ms = int(cfg.FRAME_STRIDE_S * 1000)
    k = max(1, kernel_ms // stride_ms)
    if k % 2 == 0:
        k += 1

    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = median_filter(probs[:, c], size=k)
    return out


# ======================================================================
# Thresholding
# ======================================================================

def threshold_to_detections(
    probs: np.ndarray,
    thresholds: np.ndarray,
    dataset: str,
    filename: str,
    offset_sample: int = 0,
) -> list[Detection]:
    """
    Convert per-frame probabilities into Detection objects by finding
    contiguous runs above the per-class threshold.
    """
    names = cfg.class_names()
    dets = []
    T, C = probs.shape
    offset_s = offset_sample / cfg.SAMPLE_RATE

    for c in range(C):
        active = probs[:, c] > thresholds[c]
        diffs = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for s, e in zip(starts, ends):
            dets.append(Detection(
                dataset=dataset,
                filename=filename,
                label=names[c],
                start_s=s * cfg.FRAME_STRIDE_S + offset_s,
                end_s=e * cfg.FRAME_STRIDE_S + offset_s,
                confidence=float(probs[s:e, c].mean()),
            ))

    return dets


# ======================================================================
# Merging and filtering
# ======================================================================

def merge_and_filter(detections: list[Detection]) -> list[Detection]:
    """
    Collapse labels (7→3 via COLLAPSE_MAP, no-op in 3-class mode), merge
    same-class events on the same file separated by less than
    ``MERGE_GAP_S``, then drop merged events outside the duration window.
    """
    collapsed = []
    for d in detections:
        new_label = cfg.COLLAPSE_MAP.get(d.label, d.label)
        collapsed.append(Detection(
            dataset=d.dataset, filename=d.filename, label=new_label,
            start_s=d.start_s, end_s=d.end_s, confidence=d.confidence,
        ))

    groups: dict[tuple, list[Detection]] = {}
    for d in collapsed:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    final = []
    for _, events in groups.items():
        events.sort(key=lambda x: x.start_s)

        # Greedy left-to-right merge.
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

        for m in merged:
            dur = m.end_s - m.start_s
            if cfg.POST_MIN_DUR_S <= dur <= cfg.POST_MAX_DUR_S:
                final.append(m)

    return final


# ======================================================================
# Full pipeline
# ======================================================================

def postprocess_predictions(
    all_probs: dict[tuple[str, str, int], np.ndarray],
    thresholds: np.ndarray,
) -> list[Detection]:
    """Run stitch → smooth → threshold → merge/filter end-to-end."""
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter(all_dets)


# ======================================================================
# CSV export
# ======================================================================

def export_challenge_csv(detections: list[Detection], output_path: str):
    """Write detections to the DCASE challenge format CSV."""
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
            "end_datetime": (file_start + timedelta(seconds=d.end_s)).isoformat(),
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
    """Parse ATBFL ``YYYY-MM-DDTHH-MM-SS[_msec].wav`` → UTC datetime."""
    from pathlib import Path
    stem = Path(filename).stem.split("_")[0]
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ======================================================================
# Event-level evaluation
# ======================================================================

def compute_iou_1d(ps: float, pe: float, gs: float, ge: float) -> float:
    """1D IoU between two intervals; 0 if disjoint or union empty."""
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_metrics(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    iou_threshold: float = 0.3,
) -> dict:
    """
    Per-class and overall precision/recall/F1 via greedy 1D IoU matching.

    Returns a dict with one entry per class plus an ``"overall"`` entry.
    Each entry has ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``.
    """
    classes = sorted({d.label for d in list(predictions) + list(ground_truth)})
    results = {}
    tp_tot = fp_tot = fn_tot = 0

    for cls in classes:
        cp = [d for d in predictions if d.label == cls]
        cg = [d for d in ground_truth if d.label == cls]
        files = {(d.dataset, d.filename) for d in cp + cg}
        tp = fp = fn = 0

        for fk in files:
            file_preds = sorted([d for d in cp if (d.dataset, d.filename) == fk],
                                key=lambda x: x.start_s)
            file_gts = sorted([d for d in cg if (d.dataset, d.filename) == fk],
                              key=lambda x: x.start_s)
            matched = set()

            # Greedy per-GT match. O(n×m) but n, m are tiny per file.
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
            "f1": 2 * p * r / (p + r + 1e-8),
            "tp": tp, "fp": fp, "fn": fn,
        }
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    p = tp_tot / (tp_tot + fp_tot + 1e-8)
    r = tp_tot / (tp_tot + fn_tot + 1e-8)
    results["overall"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r + 1e-8)}
    return results


# ======================================================================
# Threshold tuning (used by train.py's post-training pass)
# ======================================================================

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
    One-pass per-class threshold grid search to maximize event-level F1.

    Grid is ``np.linspace(0.1, 0.9, 17)``; per-class coordinate descent
    with the other classes held at their current best.

    Model output convention: ``model(spec)`` returns a dict with at
    minimum a ``"probs"`` key of shape ``(B, T, C)``. This works for
    both ``Triton`` (where probs = sigmoid(logits)) and ``TritonTIDE``
    (where probs are already gated).
    """
    model.eval()
    all_probs: dict[tuple[str, str, int], np.ndarray] = {}

    # Run inference once, cache probabilities for all threshold trials.
    for audio, _, _, metas in val_loader:
        audio = audio.to(device)
        spec = spec_extractor(audio)
        out = model(spec)
        # Both Triton and TritonTIDE return a dict with "probs".
        probs = out["probs"].cpu().numpy()

        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    all_probs = collapse_probs_to_3class(all_probs)

    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"],
            filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    candidates = np.linspace(0.1, 0.9, 17)
    coarse_names = cfg.CALL_TYPES_3
    n_classes = len(coarse_names)
    best_thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        best_f1 = 0.0
        for t_try in candidates:
            thresholds = best_thresholds.copy()
            thresholds[c] = t_try
            preds = postprocess_predictions(all_probs, thresholds)
            metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
            cls_name = coarse_names[c]
            f1 = metrics.get(cls_name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = t_try
        print(f"  class {coarse_names[c]}: "
              f"threshold={best_thresholds[c]:.3f}  F1={best_f1:.3f}")

    return best_thresholds
