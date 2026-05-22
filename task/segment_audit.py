"""
Segment Construction Audit
==========================

Standalone diagnostic that exercises the existing ``build_positive_segments``,
``build_negative_segments``, and target-painting logic from ``dataset.py``
and reports the kinds of subtle bugs Christiaan flagged in his email
(2026-05-21):

  "a non-trivial amount of effort went into correctly building training
   segments from the raw recordings and annotations, and the segment-
   construction logic is more nuanced than it first appears."

This is read-only. It does not change anything. It just prints statistics
that make silent failure modes visible:

  1. **Dropped annotations** — how many calls fell below
     ``MIN_CALL_DURATION_S`` or above ``MAX_CALL_DURATION_S`` and were
     silently skipped, broken down by class.
  2. **Annotation-frame fidelity** — sum of annotation durations vs sum
     of painted positive frames per class. The ratio reveals quantization
     loss from the ``int()`` truncation in target painting.
  3. **Negative-positive proximity** — distribution of how close (in
     seconds) each negative segment sits to the nearest annotated call.
     If a meaningful tail is within ~50ms, those "negatives" carry call
     residue and you're effectively training the model to ignore real
     signal.
  4. **Overlap between positive segments** — for each positive segment,
     mean Jaccard overlap with its neighbours. High values indicate the
     training set is denser than the annotation count suggests.
  5. **Time-conversion sanity** — flags any segment whose
     ``start_sample`` or ``end_sample`` falls outside the underlying
     file's duration. Catches timestamp / timezone parsing bugs.
  6. **Per-segment annotation density** — histogram of how many calls
     each positive segment contains. A bimodal distribution (lots with 1,
     lots with many) is normal; lots with 0 would indicate a bug.

Run::

    python segment_audit.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

import config as cfg
from dataset import (
    get_file_manifest, load_annotations,
    build_positive_segments, build_negative_segments,
    _build_annotations_by_file,
)


def _header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def audit_dropped_annotations(annotations, manifest) -> None:
    """Count annotations that get filtered out before becoming segments."""
    _header("1. Dropped annotations (filtered by build_positive_segments)")
    ann_by_file = _build_annotations_by_file(annotations, manifest)
    file_starts = {(r["dataset"], r["filename"]): r["start_dt"]
                   for _, r in manifest.iterrows()}

    counts = {
        "total":          0,
        "no_file":        0,
        "no_start_dt":    0,
        "neg_or_zero":    0,
        "too_long":       0,
        "too_short":      0,
        "kept":           0,
    }
    dropped_short_by_class: Counter = Counter()
    dropped_long_by_class: Counter = Counter()

    for _, row in annotations.iterrows():
        counts["total"] += 1
        key = (row["dataset"], row["filename"])
        if key not in file_starts:
            counts["no_file"] += 1
            continue
        fsd = file_starts[key]
        if fsd is None:
            counts["no_start_dt"] += 1
            continue
        try:
            import pandas as pd
            if pd.isna(fsd):
                counts["no_start_dt"] += 1
                continue
        except Exception:
            pass
        cs = (row["start_datetime"] - fsd).total_seconds()
        ce = (row["end_datetime"] - fsd).total_seconds()
        if ce <= cs or ce <= 0:
            counts["neg_or_zero"] += 1
            continue
        dur = ce - cs
        label = row.get("label_3class", row.get("annotation", "?"))
        if dur > cfg.MAX_CALL_DURATION_S:
            counts["too_long"] += 1
            dropped_long_by_class[label] += 1
            continue
        if dur < cfg.MIN_CALL_DURATION_S:
            counts["too_short"] += 1
            dropped_short_by_class[label] += 1
            continue
        counts["kept"] += 1

    pct = lambda n: (100.0 * n / max(counts["total"], 1))
    for k, v in counts.items():
        print(f"  {k:14s} {v:>8,d}  ({pct(v):5.2f} %)")
    if dropped_short_by_class:
        print(f"\n  Dropped because dur < {cfg.MIN_CALL_DURATION_S}s by class:")
        for cls, n in dropped_short_by_class.most_common():
            print(f"    {cls:8s} {n:>6,d}")
    if dropped_long_by_class:
        print(f"\n  Dropped because dur > {cfg.MAX_CALL_DURATION_S}s by class:")
        for cls, n in dropped_long_by_class.most_common():
            print(f"    {cls:8s} {n:>6,d}")
    if counts["too_short"] > counts["kept"] * 0.05:
        print(f"\n  *** WARNING: {counts['too_short']:,} short calls dropped "
              f"({pct(counts['too_short']):.1f}%). If these are real "
              f"calls, lowering MIN_CALL_DURATION_S would recover them.")


def audit_frame_painting_fidelity(segments, n_classes) -> None:
    """Compare painted positive frames against true annotation durations."""
    _header("2. Frame-painting fidelity (annotation seconds → painted frames)")
    class_idx = cfg.class_to_idx()
    stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)
    painted_frames_by_class = np.zeros(n_classes, dtype=np.float64)
    annotated_seconds_by_class = np.zeros(n_classes, dtype=np.float64)

    for seg in segments:
        n_samples = seg.end_sample - seg.start_sample
        n_frames = n_samples // stride_samp
        if n_frames <= 0:
            continue
        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE
        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label not in class_idx:
                continue
            c = class_idx[label]
            local_start_s = max(0.0, a["start_s"] - seg_start_s)
            local_end_s = min(n_samples / cfg.SAMPLE_RATE,
                              a["end_s"] - seg_start_s)
            if local_end_s <= local_start_s:
                continue
            f0 = int(local_start_s / cfg.FRAME_STRIDE_S)
            f1 = int(local_end_s / cfg.FRAME_STRIDE_S)
            if f1 > f0:
                painted_frames_by_class[c] += (f1 - f0)
                annotated_seconds_by_class[c] += (local_end_s - local_start_s)

    names = cfg.class_names()
    print(f"\n  Per-class quantization summary (positive segments only):")
    print(f"  {'class':8s} {'painted frames':>15s} {'≈ painted s':>13s}"
          f" {'annot. s':>12s} {'recovery %':>12s}")
    for c, name in enumerate(names):
        painted_s = painted_frames_by_class[c] * cfg.FRAME_STRIDE_S
        recovery = (100.0 * painted_s / max(annotated_seconds_by_class[c], 1e-9))
        print(f"  {name:8s} {int(painted_frames_by_class[c]):>15,d}"
              f" {painted_s:>13,.1f}s {annotated_seconds_by_class[c]:>11,.1f}s"
              f" {recovery:>11.2f} %")
    print(f"\n  Recovery % = (painted s) / (annotation s). With int() "
          f"truncation in target painting, expect ~95-98% per class.")
    print(f"  Anything below ~90% would suggest a systematic bias in the "
          f"painting code (e.g. truncation losing whole frames on short calls).")


def audit_negative_proximity(positive_segments, negative_segments,
                             annotations, manifest) -> None:
    """How close (in seconds) is each negative to its nearest annotated call?"""
    _header("3. Negative-segment proximity to nearest call")
    ann_by_file = _build_annotations_by_file(annotations, manifest)

    distances = []
    for seg in negative_segments:
        key = (seg.dataset, seg.filename)
        calls = ann_by_file.get(key, [])
        if not calls:
            continue
        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE
        seg_end_s = seg.end_sample / cfg.SAMPLE_RATE
        # Min temporal distance to any call's interval (0 means strictly
        # adjacent or overlapping, though overlapping ones should already
        # have been rejected by the negative sampler).
        d_min = float("inf")
        for c in calls:
            cs, ce = c["start_s"], c["end_s"]
            if seg_end_s < cs:
                d = cs - seg_end_s
            elif seg_start_s > ce:
                d = seg_start_s - ce
            else:
                d = 0.0
            d_min = min(d_min, d)
        if d_min < float("inf"):
            distances.append(d_min)

    if not distances:
        print("  No negatives to score.")
        return
    arr = np.asarray(distances)
    print(f"  {len(arr):,} negatives scored")
    print(f"  Distance to nearest call (seconds):")
    for q in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        v = np.quantile(arr, q)
        print(f"    {q*100:5.1f}% quantile: {v:>10.3f}s")
    pct_under_50ms = 100.0 * (arr < 0.05).mean()
    pct_under_200ms = 100.0 * (arr < 0.2).mean()
    pct_under_1s = 100.0 * (arr < 1.0).mean()
    print(f"\n  {pct_under_50ms:5.1f}%  of negatives are within 50 ms of a call")
    print(f"  {pct_under_200ms:5.1f}%  within 200 ms")
    print(f"  {pct_under_1s:5.1f}%  within 1 s")
    if pct_under_200ms > 5.0:
        print(f"\n  *** Consider adding a guard band to "
              f"build_negative_segments — these negatives are essentially "
              f"adjacent to real calls and may contain residual signal "
              f"(e.g. call onset/decay).")


def audit_positive_overlap(segments) -> None:
    """How much do positive training segments overlap each other in audio?"""
    _header("4. Overlap between positive segments (per file)")
    # Group by file and sort by start_sample.
    by_file: dict = {}
    for seg in segments:
        if seg.is_positive:
            by_file.setdefault((seg.dataset, seg.filename), []).append(seg)

    jaccards = []
    for key, segs in by_file.items():
        segs_sorted = sorted(segs, key=lambda s: s.start_sample)
        for i in range(len(segs_sorted) - 1):
            a, b = segs_sorted[i], segs_sorted[i + 1]
            inter = max(0, min(a.end_sample, b.end_sample)
                        - max(a.start_sample, b.start_sample))
            union = (max(a.end_sample, b.end_sample)
                     - min(a.start_sample, b.start_sample))
            if union > 0:
                jaccards.append(inter / union)

    if not jaccards:
        print("  No neighbouring segments to compare.")
        return
    arr = np.asarray(jaccards)
    print(f"  {len(arr):,} adjacent positive-segment pairs (within file)")
    print(f"  Jaccard overlap of consecutive segments:")
    for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]:
        v = np.quantile(arr, q)
        print(f"    {q*100:5.1f}% quantile: {v:>6.3f}")
    print(f"  mean = {arr.mean():.3f}  ;  fraction > 0.5 = "
          f"{100.0*(arr > 0.5).mean():.1f}%")
    if (arr > 0.5).mean() > 0.1:
        print(f"\n  *** A non-trivial fraction of positive segments overlap "
              f"each other by more than half. Two close calls produce two "
              f"heavily-redundant segments; the training set is denser than "
              f"the annotation count suggests.")


def audit_bounds_sanity(segments, manifest) -> None:
    """Flag segments whose [start, end] sample range escapes the file."""
    _header("5. Time-conversion sanity (segment bounds vs file duration)")
    file_durs = {(r["dataset"], r["filename"]): r["duration_s"]
                 for _, r in manifest.iterrows()}
    bad = 0
    examples = []
    for seg in segments:
        dur_samp = int(file_durs.get((seg.dataset, seg.filename), 0)
                       * cfg.SAMPLE_RATE)
        if seg.start_sample < 0 or seg.end_sample > dur_samp + 1:
            bad += 1
            if len(examples) < 5:
                examples.append((seg.dataset, seg.filename,
                                 seg.start_sample, seg.end_sample, dur_samp))
    print(f"  {bad:,} of {len(segments):,} segments have bounds outside "
          f"their file.")
    if bad > 0:
        print(f"  Sample of offenders:")
        for ex in examples:
            print(f"    {ex[0]}/{ex[1]}  "
                  f"[{ex[2]:,}, {ex[3]:,}]  vs  file_samples={ex[4]:,}")
    else:
        print(f"  All segments lie within their file. Time-conversion "
              f"appears correct.")


def audit_annotation_density(segments) -> None:
    """Histogram of how many annotations each positive segment contains."""
    _header("6. Annotations per positive segment")
    counts = [len(s.annotations) for s in segments if s.is_positive]
    if not counts:
        print("  No positive segments.")
        return
    h = Counter(counts)
    print(f"  {len(counts):,} positive segments scored")
    print(f"  Histogram (annotations per segment):")
    for k in sorted(h.keys())[:15]:
        bar = "█" * min(40, h[k] * 40 // max(h.values()))
        print(f"    {k:>3d}  {h[k]:>8,d}  {bar}")
    if h.get(0, 0) > 0:
        print(f"\n  *** WARNING: {h[0]:,} positive segments contain ZERO "
              f"annotations after construction. They were built around a "
              f"call but the call got filtered out — bug in segment "
              f"construction or the focal annotation being inadvertently "
              f"excluded from its own segment.")


def main():
    print("Segment-construction audit")
    print("==========================")
    print("Loading training manifest and annotations...")
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    train_annotations = load_annotations(cfg.TRAIN_DATASETS,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest):,} files, "
          f"{len(train_annotations):,} annotations")

    print("\nBuilding segments (this can take 30-60s)...")
    pos = build_positive_segments(train_annotations, train_manifest)
    neg = build_negative_segments(
        train_annotations, train_manifest,
        n_segments=int(len(pos) * cfg.NEG_RATIO),
    )
    print(f"  positives: {len(pos):,}   negatives: {len(neg):,}")

    audit_dropped_annotations(train_annotations, train_manifest)
    audit_frame_painting_fidelity(pos, cfg.n_classes())
    audit_negative_proximity(pos, neg, train_annotations, train_manifest)
    audit_positive_overlap(pos)
    audit_bounds_sanity(pos + neg, train_manifest)
    audit_annotation_density(pos)

    print("\n" + "=" * 70)
    print("Audit complete. Anything above marked *** WARNING is worth")
    print("investigating before treating the canonical reproduction as final.")
    print("=" * 70)


if __name__ == "__main__":
    main()
