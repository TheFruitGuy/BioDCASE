"""
Postprocessing and evaluation for BioDCASE 2026 Task 2.

Pipeline:
  1. Stitch overlapping segment predictions (averaging in overlaps)
  2. Median filter smoothing
  3. Per-class thresholding
  4. Merge nearby detections, discard outlier durations
  5. Export to challenge CSV
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter

import config as cfg


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    dataset: str
    filename: str
    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Smoothing & thresholding
# ---------------------------------------------------------------------------

def smooth_probabilities(probs: np.ndarray, kernel_ms: int = cfg.SMOOTH_KERNEL_MS) -> np.ndarray:
    """Median filter per class.  probs: (T, C)."""
    stride_ms = int(cfg.FRAME_STRIDE_S * 1000)
    k = max(1, kernel_ms // stride_ms)
    if k % 2 == 0:
        k += 1
    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = median_filter(probs[:, c], size=k)
    return out


def threshold_to_detections(
    probs: np.ndarray, thresholds: np.ndarray,
    dataset: str, filename: str, offset_sample: int = 0,
) -> list[Detection]:
    """Convert per-frame probabilities → Detection objects."""
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
                dataset=dataset, filename=filename, label=names[c],
                start_s=s * cfg.FRAME_STRIDE_S + offset_s,
                end_s=e * cfg.FRAME_STRIDE_S + offset_s,
                confidence=float(probs[s:e, c].mean()),
            ))
    return dets


# ---------------------------------------------------------------------------
# Merging & filtering
# ---------------------------------------------------------------------------

def collapse_to_3class(detections: list[Detection]) -> list[Detection]:
    """Map 7 fine-grained predictions back to the 3 challenge evaluation classes."""
    collapsed = []
    for d in detections:
        # Map class down if a 7-class label, otherwise keep as is
        new_label = cfg.COLLAPSE_MAP.get(d.label, d.label)
        collapsed.append(Detection(
            dataset=d.dataset,
            filename=d.filename,
            label=new_label,
            start_s=d.start_s,
            end_s=d.end_s,
            confidence=d.confidence
        ))

    # Run merge_detections again!
    # If a 'bma' predicted right next to a 'bmb' they will now merge.
    return filter_and_merge_events(collapsed)


def filter_and_merge_events(events: list[Detection]) -> list[Detection]:
    """Applies class-specific minimum durations and merge gaps per file."""

    # FIX 1: Group by dataset AND filename so we don't merge the whole dataset!
    events_by_group = {}
    for e in events:
        eval_label = cfg.COLLAPSE_MAP.get(e.label, e.label)
        key = (e.dataset, e.filename, eval_label)
        events_by_group.setdefault(key, []).append(e)

    final_events = []

    for (ds, fn, label), class_events in events_by_group.items():
        # Look up class-specific minimum duration (or default to 0.5)
        min_dur = cfg.CLASS_MIN_DURATION_S.get(label, 0.5)

        # Smart Merge Gap: Check if it's a dictionary or a universal float
        if isinstance(cfg.CLASS_MERGE_GAP_S, dict):
            max_gap = cfg.CLASS_MERGE_GAP_S.get(label, 0.5)
        else:
            max_gap = float(cfg.CLASS_MERGE_GAP_S)  # Use the universal 0.5s gap!

        # 1. Sort by start time
        class_events.sort(key=lambda x: x.start_s)

        # 2. Merge close events
        merged = []
        for e in class_events:
            if not merged:
                merged.append(e)
            else:
                last = merged[-1]
                # If gap is smaller than max_gap, merge them
                if e.start_s - last.end_s <= max_gap:
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(e)

        # 3. Filter out events that are too short
        for m in merged:
            if (m.end_s - m.start_s) >= min_dur:
                final_events.append(m)

    return final_events


# ---------------------------------------------------------------------------
# Stitch overlapping segments
# ---------------------------------------------------------------------------

def _stitch_segments(
    all_probs: dict[tuple[str, str, int], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Merge overlapping segment probs by averaging.  Returns per-file arrays."""
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


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def postprocess_predictions(
    all_probs: dict[tuple[str, str, int], np.ndarray],
    thresholds: np.ndarray,
) -> list[Detection]:
    """Raw segment probs → mapped to 3-class → merged and filtered."""
    file_probs = _stitch_segments(all_probs)
    all_dets = []

    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        dets = threshold_to_detections(probs, thresholds, ds, fn)
        all_dets.extend(dets)

    mapped_dets = []
    for d in all_dets:
        new_label = cfg.COLLAPSE_MAP.get(d.label, d.label)
        mapped_dets.append(Detection(
            dataset=d.dataset, filename=d.filename, label=new_label,
            start_s=d.start_s, end_s=d.end_s, confidence=d.confidence
        ))

    return filter_and_merge_events(mapped_dets)


# ---------------------------------------------------------------------------
# Challenge CSV export
# ---------------------------------------------------------------------------

def export_challenge_csv(detections: list[Detection], output_path: str):
    """Write detections in the official BioDCASE 2026 Task 2 format."""
    rows = []
    for d in detections:
        try:
            base = d.filename.replace(".wav", "").split("_")[0]
            file_start = datetime.strptime(base, "%Y-%m-%dT%H-%M-%S").replace(
                tzinfo=timezone.utc)
        except ValueError:
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
            "dataset", "filename", "annotation", "start_datetime", "end_datetime"])
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} detections → {output_path}")


# ---------------------------------------------------------------------------
# Evaluation (1D IoU)
# ---------------------------------------------------------------------------

def compute_iou_1d(ps, pe, gs, ge) -> float:
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_metrics(
    predictions: list[Detection],
    ground_truth: list[Detection],
    iou_threshold: float = 0.3,
) -> dict:
    """Per-class + overall precision / recall / F1 with 1D IoU matching."""
    classes = list(set(d.label for d in predictions + ground_truth))
    results = {}
    tot_tp = tot_fp = tot_fn = 0

    for cls in classes:
        cp = [d for d in predictions if d.label == cls]
        cg = [d for d in ground_truth if d.label == cls]
        files = set((d.dataset, d.filename) for d in cp + cg)
        tp = fp = fn = 0
        for fk in files:
            fp_ = sorted([d for d in cp if (d.dataset, d.filename) == fk],
                         key=lambda x: x.start_s)
            fg_ = sorted([d for d in cg if (d.dataset, d.filename) == fk],
                         key=lambda x: x.start_s)
            matched = set()
            for gt in fg_:
                best_iou, best_i = 0.0, -1
                for i, pr in enumerate(fp_):
                    if i in matched:
                        continue
                    iou = compute_iou_1d(pr.start_s, pr.end_s, gt.start_s, gt.end_s)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_iou >= iou_threshold and best_i >= 0:
                    tp += 1; matched.add(best_i)
                else:
                    fn += 1
            fp += len(fp_) - len(matched)

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        results[cls] = {"precision": p, "recall": r,
                        "f1": 2*p*r / (p + r + 1e-8), "tp": tp, "fp": fp, "fn": fn}
        tot_tp += tp; tot_fp += fp; tot_fn += fn

    op = tot_tp / (tot_tp + tot_fp + 1e-8)
    or_ = tot_tp / (tot_tp + tot_fn + 1e-8)
    results["overall"] = {"precision": op, "recall": or_,
                          "f1": 2*op*or_ / (op + or_ + 1e-8)}
    return results


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

@torch.no_grad()
def tune_thresholds(model: nn.Module, val_loader, device, n_classes,
                    lo=0.1, hi=0.9, steps=17) -> torch.Tensor:
    """Grid-search per-class thresholds maximising frame-level F1."""
    model.eval()
    all_p, all_t, all_m = [], [], []
    for audio, targets, mask, _ in val_loader:
        logits = model(audio.to(device))
        Tm = logits.size(1); Tt = targets.size(1)
        if Tm < Tt:
            targets = targets[:, :Tm]; mask = mask[:, :Tm]
        elif Tm > Tt:
            targets = torch.cat([targets,
                torch.zeros(targets.size(0), Tm-Tt, targets.size(2))], 1)
            mask = torch.cat([mask,
                torch.zeros(mask.size(0), Tm-Tt, dtype=torch.bool)], 1)
        all_p.append(torch.sigmoid(logits).cpu())
        all_t.append(targets); all_m.append(mask)

    all_p = torch.cat(all_p); all_t = torch.cat(all_t); all_m = torch.cat(all_m)
    cands = np.linspace(lo, hi, steps)
    best = []
    for c in range(n_classes):
        m = all_m.reshape(-1).bool()
        tg = all_t[..., c].reshape(-1)[m]
        pr = all_p[..., c].reshape(-1)[m]
        bf1, bt = 0.0, 0.5
        for t in cands:
            pd_ = (pr > t).float()
            tp = (pd_ * tg).sum().item()
            fp = (pd_ * (1-tg)).sum().item()
            fn = ((1-pd_) * tg).sum().item()
            p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
            f1 = 2*p*r/(p+r+1e-8)
            if f1 > bf1: bf1, bt = f1, t
        best.append(bt)
        print(f"  class {c}: threshold={bt:.3f}  F1={bf1:.3f}")
    return torch.tensor(best)
