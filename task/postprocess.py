"""
Post-Processing and Evaluation
==============================

Converts the model's per-frame probability outputs into a list of discrete
detection events, and evaluates those events against ground-truth
annotations using event-level 1D IoU matching.

Post-processing pipeline (Section 5.8 of the paper)
---------------------------------------------------
    1. **Stitch**: average overlapping-window predictions back into one
       continuous per-file probability stream.
    2. **Smooth**: apply a 500 ms median filter along the time axis to
       remove short spurious spikes.
    3. **Threshold**: convert per-frame probabilities into binary activations
       using per-class thresholds (tuned on validation data).
    4. **Collapse**: map the 7 fine-grained labels to 3 coarse labels via
       ``COLLAPSE_MAP``. (Only relevant in 7-class inference mode; 3-class
       runs already output coarse labels.)
    5. **Merge**: merge neighbouring events of the same class separated by
       less than ``MERGE_GAP_S``.
    6. **Filter**: discard events with duration outside
       ``[POST_MIN_DUR_S, POST_MAX_DUR_S]``.
    7. **Export**: write detections as a challenge-format CSV.

Evaluation
----------
Event-level greedy matching with a 1D Intersection-over-Union threshold of
0.3 (the standard DCASE metric). Precision, recall, and F1 are reported
per class and overall.
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
        Mean per-frame probability over the event's span. Always 1.0 for
        ground-truth entries.
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
    Merge overlapping-window predictions into a single per-file probability
    stream by averaging predictions in overlap regions.

    Parameters
    ----------
    all_probs : dict
        Keys are ``(dataset, filename, start_sample)`` triples identifying
        each window; values are arrays of shape ``(n_frames, n_classes)``
        containing per-frame probabilities for that window.

    Returns
    -------
    dict
        Keys are ``(dataset, filename)`` pairs; values are arrays of shape
        ``(n_total_frames, n_classes)`` holding the stitched probability
        stream. Frames that no window covered remain at zero, but in
        practice every frame is covered by at least one window because of
        the 2-second overlap.
    """
    stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)

    # Group windows by file.
    file_segs: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for (ds, fn, start_samp), probs in all_probs.items():
        file_segs.setdefault((ds, fn), []).append((start_samp, probs))

    result = {}
    for key, segs in file_segs.items():
        # Sort by start sample so later loops see windows in order.
        segs.sort(key=lambda x: x[0])

        # Determine total frame count needed to cover every window.
        max_end = max(s + p.shape[0] * stride_samp for s, p in segs)
        total_frames = max_end // stride_samp + 1
        nc = segs[0][1].shape[1]

        # Accumulate sum and count for averaging. Using float64 avoids
        # precision loss when many windows contribute to the same frame.
        accum = np.zeros((total_frames, nc), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)

        for start_samp, probs in segs:
            f0 = start_samp // stride_samp
            T = min(probs.shape[0], total_frames - f0)
            accum[f0:f0 + T] += probs[:T]
            counts[f0:f0 + T] += 1

        # Avoid division by zero for frames that no window covered.
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

    A median filter is robust to isolated spurious activations while
    preserving the edges of real events, making it preferable to a
    moving average for this task.

    Parameters
    ----------
    probs : np.ndarray, shape (n_frames, n_classes)
        Per-frame probabilities for one file.
    kernel_ms : int
        Kernel width in milliseconds. Paper uses 500 ms.

    Returns
    -------
    np.ndarray, shape (n_frames, n_classes)
        Smoothed probabilities.
    """
    stride_ms = int(cfg.FRAME_STRIDE_S * 1000)
    # Convert milliseconds to frames; ensure odd kernel size as required
    # by scipy's median_filter.
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
    Convert per-frame probabilities into a list of Detection objects by
    thresholding.

    For each class, find contiguous runs where the probability exceeds the
    class-specific threshold and emit one Detection per run. The numpy
    "diff of masked integer" trick identifies run starts and ends in O(n)
    without any explicit loop over frames.

    Parameters
    ----------
    probs : np.ndarray, shape (n_frames, n_classes)
        Smoothed per-frame probabilities.
    thresholds : np.ndarray, shape (n_classes,)
        Per-class thresholds.
    dataset, filename : str
        Metadata to embed in each Detection.
    offset_sample : int, default 0
        Sample offset to add to every detection's timestamp (used when
        processing sub-windows rather than whole files).

    Returns
    -------
    list of Detection
    """
    names = cfg.class_names()
    dets = []
    T, C = probs.shape
    offset_s = offset_sample / cfg.SAMPLE_RATE

    for c in range(C):
        # Build a binary activation vector for this class, then detect
        # run starts (0→1 transitions) and ends (1→0). The prepend/append
        # of zeros ensures runs touching the start or end are terminated
        # correctly.
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
    Collapse labels, merge nearby detections, and filter by duration.

    Three steps applied in order:
      1. Map fine-grained to coarse labels via COLLAPSE_MAP (no-op if
         already in 3-class mode).
      2. For each (file, class), sort detections by start time and merge
         any pair separated by less than ``MERGE_GAP_S``.
      3. Discard merged events whose duration lies outside
         ``[POST_MIN_DUR_S, POST_MAX_DUR_S]``.

    Parameters
    ----------
    detections : list of Detection
        Raw detections from ``threshold_to_detections``.

    Returns
    -------
    list of Detection
        Final event list ready for evaluation or CSV export.
    """
    # Step 1: label collapsing (7-class → 3-class). In 3-class mode this
    # is an identity mapping for the primary classes.
    collapsed = []
    for d in detections:
        new_label = cfg.COLLAPSE_MAP.get(d.label, d.label)
        collapsed.append(Detection(
            dataset=d.dataset, filename=d.filename, label=new_label,
            start_s=d.start_s, end_s=d.end_s, confidence=d.confidence,
        ))

    # Step 2: group by (file, class) and merge close events. The group key
    # uses dataset + filename so different files never interact, and uses
    # class so different call types never merge.
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
                    # Extend the previous event to cover this one.
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(e)

        # Step 3: duration filter. Removes both spurious short activations
        # (noise) and unrealistically long events (runaway detections).
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
    Run the complete post-processing pipeline end-to-end.

    Parameters
    ----------
    all_probs : dict
        Output of model inference; see ``stitch_segments``.
    thresholds : np.ndarray, shape (n_classes,)
        Per-class probability thresholds.

    Returns
    -------
    list of Detection
        Final list of detections for all files.
    """
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)  # 500 ms median filter
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter(all_dets)


# ======================================================================
# CSV export
# ======================================================================

def export_challenge_csv(detections: list[Detection], output_path: str):
    """
    Write detections to a CSV file in the DCASE challenge format.

    The challenge expects absolute ``start_datetime`` and ``end_datetime``
    columns, which we construct by adding each detection's file-relative
    offset to the file's start datetime parsed from its filename.

    Parameters
    ----------
    detections : list of Detection
    output_path : str
        Destination CSV path.
    """
    rows = []
    for d in detections:
        file_start = _parse_filename_dt(d.filename)
        if file_start is None:
            # Skip detections on files with unparseable names (shouldn't
            # happen in practice on the ATBFL corpus).
            continue
        rows.append({
            "dataset": d.dataset,
            "filename": d.filename,
            "annotation": d.label,
            "start_datetime": (file_start + timedelta(seconds=d.start_s)).isoformat(),
            "end_datetime": (file_start + timedelta(seconds=d.end_s)).isoformat(),
        })
    # Stable sort so the output is deterministic.
    rows.sort(key=lambda r: (r["dataset"], r["filename"], r["start_datetime"]))

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "filename", "annotation", "start_datetime", "end_datetime",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} detections → {output_path}")


def _parse_filename_dt(filename: str):
    """
    Parse the UTC start datetime from an ATBFL-style filename.

    Mirrors the helper in ``dataset.py`` but is duplicated here to avoid a
    circular import when post-processing is invoked standalone.
    """
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
    """
    Compute the 1D Intersection-over-Union between two intervals.

    Parameters
    ----------
    ps, pe : float
        Predicted interval [start, end).
    gs, ge : float
        Ground-truth interval [start, end).

    Returns
    -------
    float
        Intersection / Union, in ``[0, 1]``. Returns 0 if the intervals
        are disjoint or the union is empty.
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
    Per-class and overall precision / recall / F1 using greedy 1D IoU matching.

    Each ground-truth event is matched to the highest-IoU unmatched
    prediction on the same file; if the best IoU is below ``iou_threshold``
    the ground-truth event is counted as a false negative. Unmatched
    predictions become false positives.

    Parameters
    ----------
    predictions : list of Detection
        Model outputs after post-processing.
    ground_truth : list of Detection
        Annotated events.
    iou_threshold : float, default 0.3
        Minimum IoU to accept a match. 0.3 is the standard DCASE value.

    Returns
    -------
    dict
        Nested dict with per-class and an ``"overall"`` entry. Each entry
        contains ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``.
    """
    # Evaluate each class that appears in either predictions or GT.
    classes = sorted({d.label for d in list(predictions) + list(ground_truth)})
    results = {}
    tp_tot = fp_tot = fn_tot = 0

    for cls in classes:
        cp = [d for d in predictions if d.label == cls]
        cg = [d for d in ground_truth if d.label == cls]
        # Iterate file-by-file so predictions can only match GT events
        # from the same recording.
        files = {(d.dataset, d.filename) for d in cp + cg}
        tp = fp = fn = 0

        for fk in files:
            file_preds = sorted([d for d in cp if (d.dataset, d.filename) == fk],
                                key=lambda x: x.start_s)
            file_gts = sorted([d for d in cg if (d.dataset, d.filename) == fk],
                              key=lambda x: x.start_s)
            matched = set()

            # Greedy per-GT match: for each GT, pick the best available
            # prediction. This is O(n×m) but n, m are small per file.
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

            # Any prediction that didn't match a GT counts as FP.
            fp += len(file_preds) - len(matched)

        # Use 1e-8 epsilon to avoid division by zero when all three counts
        # are zero (degenerate but possible for empty eval subsets).
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

    # Micro-average overall metric.
    p = tp_tot / (tp_tot + fp_tot + 1e-8)
    r = tp_tot / (tp_tot + fn_tot + 1e-8)
    results["overall"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r + 1e-8)}
    return results


# ======================================================================
# Threshold tuning (simple version - see tune_thresholds.py for the
# full iterative version with per-class finer grids).
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
    Grid-search per-class thresholds to maximize event-level F1.

    One-pass algorithm: for each class, try thresholds in ``{0.1, ..., 0.9}``
    with the other classes held at their current best. Single pass, no
    iteration. For a more thorough search see ``tune_thresholds.py``.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    spec_extractor : nn.Module
        Feature extractor to apply before the model.
    val_loader : DataLoader
        Validation data.
    device : torch.device
    val_annotations : pd.DataFrame
        Used to build ground-truth events.
    file_start_dts : dict
        Maps ``(dataset, filename)`` to the file's start datetime.

    Returns
    -------
    np.ndarray, shape (n_classes,)
        Best-found per-class thresholds.
    """
    model.eval()
    all_probs: dict[tuple[str, str, int], np.ndarray] = {}

    # Run inference once, cache probabilities for all threshold trials.
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

    # Build ground-truth Detections for the validation set.
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"],
            filename=row["filename"],
            label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # Grid search per class, one class at a time. Other classes stay at
    # their previous best thresholds.
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
