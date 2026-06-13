"""
NAVE evaluation in the OFFICIAL BioDCASE Task 2 metric (convention B).
=====================================================================
The official ranking metric (see `evaluation_submissions.py`,
`global_ranking_filtered.csv`) is **F1 of mean-precision / mean-recall** across
the three classes -- NOT the mean of per-class F1s. This script reports that as
the headline, evaluated directly at 60 s tiles (the dev-selected eval length),
for one checkpoint or a probability ensemble.

    # single checkpoint, official metric at 60s
    CUDA_VISIBLE_DEVICES=0 python nave_eval_official.py --segment-s 60 --use_fp16 \
        --checkpoints runs/phase13r_3c_s666_20260612_132012/phase13r_best.pt --val-workers 20

    # ensemble + tune thresholds FOR the official metric + cross-check with the
    # real scorer
    CUDA_VISIBLE_DEVICES=0 python nave_eval_official.py --segment-s 60 --use_fp16 \
        --tune-metric official --official-scorer \
        --checkpoints runs/nave_s*/nave_best.pt --val-workers 20

Metric conventions reported:
  OFFICIAL (B) = F1(mean P, mean R)   <-- the leaderboard number
  per-class F1 + (A) mean-of-F1       <-- diagnostics only

Threshold tuning:
  --tune-metric f1        per-class F1 coordinate descent (validated; default).
  --tune-metric official  per-class F1 warm start, then coordinate descent on B.

--official-scorer writes submission-format CSVs and runs the bundled
`evaluation_submissions.run` (±2 min window, IoU 0.3, D label 'fm') for the
authoritative number; it must sit alongside this file.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from postprocess_final import Detection, postprocess_predictions
from eval_conformer import (
    tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)
from nave_eval import get_or_compute_probs, combine_probs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--segment-s", type=float, default=60.0,
                   help="Evaluation tile length (default 60).")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--tune-metric", choices=["f1", "official"], default="f1",
                   help="'f1' = per-class F1 (validated); 'official' = refine for B.")
    p.add_argument("--official-scorer", action="store_true",
                   help="Cross-check with the bundled evaluation_submissions.run().")
    p.add_argument("--out-dir", type=str, default="runs/nave_submission",
                   help="Where submission CSVs are written for --official-scorer.")
    p.add_argument("--cache_dir", type=str, default="runs/nave_prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--val-workers", type=int, default=min((os.cpu_count() or 1), 13))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        p.error(f"--weights has {len(args.weights)} values, need {len(args.checkpoints)}.")
    return args


def tune_for_official(probs, gt, start, grid=None, rounds=3):
    """Coordinate descent on the official metric B (warm-started)."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.025), 4)
    thr = [float(t) for t in start]

    def B(t):
        return macro_f1_paper(evaluate_with_thresholds(probs, gt, np.array(t)))

    best = B(thr)
    for _ in range(rounds):
        improved = False
        for c in range(len(thr)):
            for v in grid:
                cand = thr.copy(); cand[c] = float(v)
                s = B(cand)
                if s > best + 1e-9:
                    best, thr, improved = s, cand, True
        if not improved:
            break
    return np.array(thr), best


def write_submission_csvs(detections, file_start_dts, out_dir):
    """Detections -> submission-format CSVs (one per dataset)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in detections:
        fsd = file_start_dts.get((d.dataset, d.filename))
        if fsd is None:
            continue
        rows.append(dict(dataset=d.dataset, filename=d.filename, annotation=d.label,
                         start_datetime=fsd + pd.Timedelta(seconds=d.start_s),
                         end_datetime=fsd + pd.Timedelta(seconds=d.end_s)))
    df = pd.DataFrame(rows)
    for ds, g in df.groupby("dataset"):
        g.to_csv(out_dir / f"{ds}.csv", index=False)
    return out_dir, len(df)


def write_gt_csvs(val_anns, out_dir):
    """Ground truth -> CSVs with the FINE annotation (official scorer collapses it)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "filename", "annotation", "start_datetime", "end_datetime"]
    gt = val_anns[cols].copy()
    for ds, g in gt.groupby("dataset"):
        g.to_csv(out_dir / f"{ds}.csv", index=False)
    return out_dir


def official_B_from_confmatrix(conf):
    """Leaderboard B from evaluation_submissions.run()'s confusion matrix."""
    joined = conf[conf["dataset"] == "joined"]
    P = float(joined["precision"].mean())
    R = float(joined["recall"].mean())
    return 2 * P * R / (P + R + 1e-8), P, R


def main():
    args = parse_args()
    import config_final as _cf
    _cf.USE_3CLASS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.checkpoints)
    print(f"[NAVE official-eval] {n} ckpt | {args.segment_s:.0f}s tiles | "
          f"tune={args.tune_metric} | official_scorer={args.official_scorer}")

    val_sites = list(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(val_sites)
    val_anns = load_annotations(val_sites, manifest=val_manifest)
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}
    gt = []
    for _, row in val_anns.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds()))
    print(f"  {len(gt)} ground-truth events")

    spec = NAVEFeatureExtractor().to(device)
    _loader = {"obj": None}

    def loader_factory():
        if _loader["obj"] is None:
            segs = build_val_segments(val_manifest, val_anns,
                                      segment_s=args.segment_s, overlap_s=args.overlap_s)
            _loader["obj"] = DataLoader(
                WhaleDataset(segs), batch_size=args.batch_size, shuffle=False,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
            print(f"  built loader: {len(segs)} tiles")
        return _loader["obj"]

    cache_dir = Path(args.cache_dir)
    prob_dicts = []
    for i, ckpt in enumerate(args.checkpoints):
        print(f"  [{i+1}/{n}] {Path(ckpt).parent.name}")
        prob_dicts.append(get_or_compute_probs(
            ckpt, spec, loader_factory, device, cache_dir,
            segment_s=args.segment_s, no_cache=args.no_cache, use_fp16=args.use_fp16))
    probs = combine_probs(prob_dicts, args.weights)

    # thresholds: per-class F1 (validated), optionally refined for B
    thr = tune_thresholds_per_class(probs, gt, workers=args.val_workers)
    if args.tune_metric == "official":
        thr, b_internal = tune_for_official(probs, gt, start=thr)
        print(f"  refined thresholds for official metric (internal B={b_internal:.4f})")

    metrics = evaluate_with_thresholds(probs, gt, thr)
    print(f"\n{'='*60}\nINTERNAL matcher\n{'='*60}")
    for c in CLASS_NAMES:
        m = metrics.get(c, {})
        print(f"  {c.upper():6} P={m.get('precision',0):.3f} "
              f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")
    print(f"  OFFICIAL (B, F1 of mean-P/mean-R) = {macro_f1_paper(metrics):.4f}   <-- leaderboard metric")
    print(f"  (A, mean of per-class F1)         = {macro_f1(metrics):.4f}   (diagnostic only)")
    print(f"  thresholds = {[round(float(t),3) for t in thr]}")

    if args.official_scorer:
        print(f"\n{'='*60}\nOFFICIAL scorer (evaluation_submissions.run)\n{'='*60}")
        try:
            import evaluation_submissions as offsc
        except ImportError:
            print("  evaluation_submissions.py not found next to this script; skipping.")
            return
        dets = postprocess_predictions(probs, np.asarray(thr))
        pred_dir, n_pred = write_submission_csvs(dets, file_start_dts, Path(args.out_dir) / "pred")
        gt_dir = write_gt_csvs(val_anns, Path(args.out_dir) / "gt")
        print(f"  wrote {n_pred} predictions -> {pred_dir}")
        conf = offsc.run(pred_dir, gt_dir, iou_threshold=cfg.IOU_THRESHOLD)
        B, P, R = official_B_from_confmatrix(conf)
        print(conf[conf["dataset"] == "joined"][["tp", "fp", "fn", "recall", "precision"]]
              .to_string())
        print(f"\n  OFFICIAL B (leaderboard) = {B:.4f}   (mean P={P:.3f}, mean R={R:.3f})")


if __name__ == "__main__":
    main()
