"""
Sweep the D-class MERGE_GAP_S on cached ensemble probabilities.
===============================================================

Motivation: the bout filter's ICI diagnostic showed that 37% of all 'd'
inter-call intervals fall in the 2–4 s bin — well above cfg.MERGE_GAP_S =
0.5 s. That's almost certainly fragmentation: one true D call producing
multiple short detections that the current merge step doesn't consolidate.

This script reuses the existing 5-model ensemble cache and sweeps D's
merge gap independently of bmabz/bp (which stay at cfg.MERGE_GAP_S). For
each candidate D gap it:

  1. Reproduces the standard pipeline up through threshold_to_detections.
  2. Applies a per-class merge: D uses the candidate gap, others use
     cfg.MERGE_GAP_S as currently configured.
  3. Applies the standard duration filter [cfg.POST_MIN_DUR_S,
     cfg.POST_MAX_DUR_S].
  4. Scores against ground truth using compute_metrics @ IoU=0.3.

Optional --retune_d: at each gap, re-grid the D threshold (other classes
fixed). With the same cost as the bout filter retune. Useful since the
optimal D threshold may shift once fragments are consolidated.

Usage
-----
    # Quick sweep at baseline thresholds (no retune):
    python eval_d_merge_gap.py \
        --checkpoints runs/whalevad_*/best_model.pt \
        --use_fp16

    # Sweep + retune D threshold at each gap (slower, more honest):
    python eval_d_merge_gap.py \
        --checkpoints runs/whalevad_*/best_model.pt \
        --use_fp16 --retune_d

    # Pick specific gap values:
    python eval_d_merge_gap.py \
        --checkpoints ... --gaps 0.5 2 4 6 8 10 15
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, stitch_segments, smooth_probabilities,
    threshold_to_detections, compute_metrics,
)
from spectrogram import SpectrogramExtractor

from ensemble_predict import average_prob_dicts, tune_thresholds_on_probs
from ensemble_predict_cached import get_or_compute_probs


# ---------------------------------------------------------------------
# Per-class merge: copy of postprocess.merge_and_filter but with the
# merge gap looked up per class from a dict.
# ---------------------------------------------------------------------

def merge_and_filter_per_class(
    detections: list[Detection],
    merge_gap_per_class: dict[str, float],
) -> list[Detection]:
    """
    Same logic as postprocess.merge_and_filter but each class can have its
    own merge gap. Classes absent from the dict fall back to cfg.MERGE_GAP_S.
    The collapse step is omitted because we're already on 3-class events.
    """
    groups: dict[tuple, list[Detection]] = {}
    for d in detections:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    final = []
    for (ds, fn, cls), events in groups.items():
        events.sort(key=lambda x: x.start_s)
        gap = float(merge_gap_per_class.get(cls, cfg.MERGE_GAP_S))

        merged = []
        for e in events:
            if not merged:
                merged.append(Detection(
                    dataset=e.dataset, filename=e.filename, label=e.label,
                    start_s=e.start_s, end_s=e.end_s, confidence=e.confidence,
                ))
            else:
                last = merged[-1]
                if e.start_s - last.end_s <= gap:
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(Detection(
                        dataset=e.dataset, filename=e.filename, label=e.label,
                        start_s=e.start_s, end_s=e.end_s, confidence=e.confidence,
                    ))

        for m in merged:
            dur = m.end_s - m.start_s
            if cfg.POST_MIN_DUR_S <= dur <= cfg.POST_MAX_DUR_S:
                final.append(m)
    return final


def predict_events_per_class_gap(
    all_probs, thresholds, merge_gap_per_class,
) -> list[Detection]:
    """Stitch → smooth → threshold → per-class merge → duration filter."""
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter_per_class(all_dets, merge_gap_per_class)


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

def class_f1_table(metrics, label):
    macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                           for c in cfg.CALL_TYPES_3]))
    out = []
    for c in cfg.CALL_TYPES_3:
        m = metrics.get(c, {})
        out.append((c.upper(), m.get("tp", 0), m.get("fp", 0),
                    m.get("fn", 0), m.get("precision", 0),
                    m.get("recall", 0), m.get("f1", 0.0)))
    overall = metrics.get("overall", {})
    return out, macro, overall.get("f1", 0.0)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--cache_dir", type=str, default="runs/prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--gaps", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0],
                   help="D-class merge gaps in seconds to sweep.")
    p.add_argument("--retune_d", action="store_true",
                   help="At each gap, re-tune D's operating threshold "
                        "(bmabz and bp fixed at baseline). Slower.")
    p.add_argument("--d_grid", nargs="+", type=float,
                   default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
                   help="D threshold grid for the optional retune.")
    p.add_argument("--base_thr", nargs=3, type=float, default=None,
                   metavar=("BMABZ", "D", "BP"),
                   help="Per-class operating thresholds for the ensemble "
                        "(bmabz d bp). If unset, --auto_tune is used; if "
                        "neither, falls back to [0.25, 0.40, 0.40] with a warning.")
    p.add_argument("--auto_tune", action="store_true",
                   help="Tune thresholds on the ensemble probs before "
                        "sweeping merge gaps. Use this on new ensembles.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Cache dir: {args.cache_dir}, current cfg.MERGE_GAP_S={cfg.MERGE_GAP_S}s")

    # --- Validation loader + ground truth ---
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  {len(val_segs)} 30s tiles")

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    gt_events = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"  {len(gt_events)} ground-truth events")

    spec_extractor = SpectrogramExtractor().to(device)

    # --- Get/compute per-checkpoint probabilities ---
    cache_dir = Path(args.cache_dir)
    all_prob_dicts = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        probs = get_or_compute_probs(
            ckpt_path, spec_extractor, val_loader, device,
            cache_dir, no_cache=args.no_cache, use_fp16=args.use_fp16)
        all_prob_dicts.append(probs)

    weights = None
    if args.weights:
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    ens_probs = (all_prob_dicts[0] if len(all_prob_dicts) == 1
                 else average_prob_dicts(all_prob_dicts, weights=weights))

    # --- Resolve baseline thresholds for this ensemble ---
    if args.base_thr is not None:
        base_thr = np.array(args.base_thr, dtype=np.float64)
        print(f"\nUsing user-specified baseline thresholds: "
              f"bmabz={base_thr[0]:.2f}  d={base_thr[1]:.2f}  bp={base_thr[2]:.2f}")
    elif args.auto_tune:
        print(f"\n{'='*78}\nAUTO-TUNING thresholds on ensemble probabilities\n{'='*78}")
        t0 = time.time()
        base_thr = tune_thresholds_on_probs(ens_probs, gt_events)
        print(f"  tuned in {time.time()-t0:.1f}s: "
              f"bmabz={base_thr[0]:.2f}  d={base_thr[1]:.2f}  bp={base_thr[2]:.2f}")
    else:
        base_thr = np.array([0.25, 0.40, 0.40], dtype=np.float64)
        print(f"\n[warn] No --base_thr or --auto_tune given. "
              f"Falling back to source-ensemble thresholds "
              f"[{base_thr[0]:.2f}, {base_thr[1]:.2f}, {base_thr[2]:.2f}]. "
              f"For HNM_PGI or any other ensemble, pass --auto_tune.")

    # --- Baseline at cfg.MERGE_GAP_S for reference ---
    print(f"\n{'='*78}\nBASELINE (D merge gap = cfg.MERGE_GAP_S = {cfg.MERGE_GAP_S}s)\n{'='*78}")
    t0 = time.time()
    base_events = predict_events_per_class_gap(
        ens_probs, base_thr, {"d": cfg.MERGE_GAP_S})
    base_metrics = compute_metrics(base_events, gt_events, iou_threshold=0.3)
    base_rows, base_macro, base_overall = class_f1_table(base_metrics, "BASELINE")
    print(f"  computed in {time.time()-t0:.1f}s")
    for cls, tp, fp, fn, p, r, f1 in base_rows:
        print(f"    {cls:6}  TP={tp:5} FP={fp:5} FN={fn:5}  "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"    overall F1={base_overall:.3f}  macro F1={base_macro:.4f}")

    # --- Sweep D merge gap ---
    print(f"\n{'='*78}\nSWEEP: D merge gap (bmabz/bp held at cfg.MERGE_GAP_S={cfg.MERGE_GAP_S}s)"
          f"\n{'='*78}")
    print(f"\n{'gap':>6}  {'D_TP':>5} {'D_FP':>5} {'D_FN':>5}  "
          f"{'D_P':>5} {'D_R':>5} {'D_F1':>5}  "
          f"{'macroF1':>7} {'Δ':>7}"
          + ("  d_thr" if args.retune_d else ""))
    print("-" * 78)

    results = []
    for gap in args.gaps:
        per_class_gap = {"d": float(gap)}

        if args.retune_d:
            # Try each D threshold; bmabz/bp use baseline.
            best_macro, best_t = -1.0, base_thr[1]
            best_events = None
            for t_d in args.d_grid:
                trial = base_thr.copy()
                trial[1] = t_d
                events = predict_events_per_class_gap(
                    ens_probs, trial, per_class_gap)
                m = compute_metrics(events, gt_events, iou_threshold=0.3)
                mac = float(np.mean([m.get(c, {}).get("f1", 0.0)
                                     for c in cfg.CALL_TYPES_3]))
                if mac > best_macro:
                    best_macro, best_t = mac, t_d
                    best_events = events
            events, t_d_used = best_events, best_t
            metrics = compute_metrics(events, gt_events, iou_threshold=0.3)
        else:
            events = predict_events_per_class_gap(
                ens_probs, base_thr, per_class_gap)
            metrics = compute_metrics(events, gt_events, iou_threshold=0.3)
            t_d_used = base_thr[1]

        rows, macro, overall = class_f1_table(metrics, f"gap={gap}")
        d_row = next(r for r in rows if r[0] == "D")
        _, tp, fp, fn, p, r, f1 = d_row
        delta = macro - base_macro
        sign = "+" if delta >= 0 else ""
        extra = f"  {t_d_used:.2f}" if args.retune_d else ""
        print(f"  {gap:>5.1f}s  {tp:5d} {fp:5d} {fn:5d}  "
              f"{p:.3f} {r:.3f} {f1:.3f}  "
              f"{macro:.4f} {sign}{delta:+.4f}{extra}")
        results.append((gap, t_d_used, macro, f1, p, r, tp, fp, fn))

    # --- Summary: pick the best ---
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    best = max(results, key=lambda r: r[2])
    g, t_d, mac, f1d, p, r, tp, fp, fn = best
    print(f"  best D merge gap : {g}s")
    if args.retune_d:
        print(f"  best D threshold : {t_d:.2f}")
    print(f"  D F1             : {f1d:.3f}  (was {base_rows[1][6]:.3f})")
    print(f"  macro F1         : {mac:.4f}  (was {base_macro:.4f}, "
          f"Δ={mac-base_macro:+.4f})")

    # Also report whether bmabz/bp moved (they shouldn't much; tiny shifts
    # come from the duration filter touching post-merge events that now
    # may slightly differ in their pre-merge boundary alignment).


if __name__ == "__main__":
    main()
