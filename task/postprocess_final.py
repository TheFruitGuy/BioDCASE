"""
Final Pipeline - Post-Processing and Evaluation
===============================================

Converts per-frame model probabilities into discrete detection events and
scores them against ground-truth annotations with event-level 1D IoU
matching.

Pipeline
--------
    1. Stitch overlapping-window predictions into one per-file stream.
    2. Smooth with a 500 ms median filter.
    3. Threshold into binary activations (per-class thresholds).
    4. Merge same-class events separated by less than ``MERGE_GAP_S``.
    5. Filter events by duration.

For a 7-class model, :func:`collapse_probs_to_3class` reduces the probability
arrays to the 3 coarse classes (max over each coarse group) before the
pipeline runs.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import median_filter

import config_final as cfg


# ======================================================================
# 7-class -> 3-class probability collapse
# ======================================================================
# Index lists mapping each coarse class to its constituent fine-class
# channels. Built once at module load.
_SEVEN_TO_THREE = {
    "bmabz": [cfg.CALL_TYPES_7.index(x) for x in ("bma", "bmb", "bmz")],
    "d":     [cfg.CALL_TYPES_7.index(x) for x in ("bmd", "bpd")],
    "bp":    [cfg.CALL_TYPES_7.index(x) for x in ("bp20", "bp20plus")],
}


def collapse_probs_to_3class(all_probs: dict) -> dict:
    """
    Collapse per-window 7-class probabilities to 3-class via max-pooling
    within each coarse group.

    Returns the input unchanged when ``cfg.USE_3CLASS`` is True or when the
    arrays are not 7-wide, so it is safe to call unconditionally.

    Parameters
    ----------
    all_probs : dict
        Maps ``(dataset, filename, start_sample)`` to ``(n_frames, n_classes)``
        probability arrays.

    Returns
    -------
    dict
        Same keys, with ``(n_frames, 3)`` arrays ordered as
        ``cfg.CALL_TYPES_3``.
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
    """
    A single predicted or ground-truth event.

    Attributes
    ----------
    dataset, filename : str
        Origin of the detection.
    label : str
        Class name (e.g. ``"bmabz"``, ``"d"``, ``"bp"``).
    start_s, end_s : float
        File-relative start and end times in seconds.
    confidence : float, default 1.0
        Mean per-frame probability over the event span (1.0 for ground truth).
    """

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
    Merge overlapping-window predictions into one per-file probability stream
    by averaging predictions in overlap regions.

    Parameters
    ----------
    all_probs : dict
        Keys ``(dataset, filename, start_sample)``; values
        ``(n_frames, n_classes)`` probability arrays.

    Returns
    -------
    dict
        Keys ``(dataset, filename)``; values ``(n_total_frames, n_classes)``.
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
    probs: np.ndarray, kernel_ms: int = cfg.SMOOTH_KERNEL_MS
) -> np.ndarray:
    """
    Apply a temporal median filter to per-frame class probabilities.

    Parameters
    ----------
    probs : np.ndarray, shape (n_frames, n_classes)
    kernel_ms : int
        Kernel width in milliseconds.

    Returns
    -------
    np.ndarray
        Smoothed probabilities, same shape as the input.
    """
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
    Convert per-frame probabilities into Detection objects by thresholding.

    For each class, emit one Detection per contiguous run of frames above the
    class-specific threshold.

    Parameters
    ----------
    probs : np.ndarray, shape (n_frames, n_classes)
    thresholds : np.ndarray, shape (n_classes,)
    dataset, filename : str
    offset_sample : int, default 0
        Sample offset added to every detection timestamp.

    Returns
    -------
    list of Detection
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
    Collapse labels, merge nearby same-class detections, and filter by
    duration.

    Steps:
      1. Map fine-grained to coarse labels via ``COLLAPSE_MAP``.
      2. Per ``(file, class)``, merge events separated by less than
         ``MERGE_GAP_S``.
      3. Drop events whose duration is outside
         ``[POST_MIN_DUR_S, POST_MAX_DUR_S]``.

    Parameters
    ----------
    detections : list of Detection

    Returns
    -------
    list of Detection
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
    """
    Run stitch -> smooth -> threshold -> merge/filter end-to-end.

    Parameters
    ----------
    all_probs : dict
        Per-window probabilities; see :func:`stitch_segments`.
    thresholds : np.ndarray, shape (n_classes,)

    Returns
    -------
    list of Detection
    """
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter(all_dets)


# ======================================================================
# Event-level evaluation
# ======================================================================

def compute_iou_1d(ps: float, pe: float, gs: float, ge: float) -> float:
    """
    1D Intersection-over-Union between intervals ``[ps, pe)`` and ``[gs, ge)``.

    Returns 0 for disjoint intervals or an empty union.
    """
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_metrics(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    iou_threshold: float = 0.3,
) -> dict:
    """
    Per-class and overall precision / recall / F1 via greedy 1D IoU matching.

    Each ground-truth event is matched to the highest-IoU unmatched prediction
    on the same file; matches below ``iou_threshold`` count as false negatives
    and unmatched predictions as false positives.

    Parameters
    ----------
    predictions : list of Detection
    ground_truth : list of Detection
    iou_threshold : float, default 0.3

    Returns
    -------
    dict
        Per-class entries plus an ``"overall"`` (micro-averaged) entry; each
        holds ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``.
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
    results["overall"] = {"precision": p, "recall": r,
                          "f1": 2 * p * r / (p + r + 1e-8)}
    return results
