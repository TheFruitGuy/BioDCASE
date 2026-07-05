"""
nave_segment_eval.py -- evaluation-segment-length robustness check.
==============================================================================
Runs a set of (already-trained) checkpoints at several EVALUATION segment/tile
lengths and reports, per length, the pooled LOSO F1 (macro + per class) and the
number of detected events per class -- aggregated across the checkpoints as
mean (sample std) over seeds. This is a robustness check on the 60s eval choice,
NOT a context-length study: the model is trained at TRAIN_SEGMENT_S and only the
eval tiling changes here, and the per-segment mean-subtraction statistic shifts
with length, so read a flat result as "insensitive to eval segment length" and a
non-flat one as mixed context/normalisation, not a clean context effect.

Pure inference (no retraining). The prob cache keys on segment length, so each
length caches separately.

    python nave_segment_eval.py runs/nave_s*_pcen1_fdy1_k153_*/nave_best.pt \
        --segments 30 60 90 120 --workers 20

    python nave_segment_eval.py <ckpt1> <ckpt2> ... --segments 45 60 90
"""

from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np
import torch

import nave_config as cfg
from nave_train import tune_thresholds_per_class, postprocess_predictions, CLASS_NAMES
from nave_evaluate import evaluate_with_thresholds
from nave_loso import get_probs, macro_f1, split_by_site


def loso_perclass(probs, gt, workers):
    """Pooled leave-one-site-out per-class F1 + macro F1."""
    pooled = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    for h in cfg.VAL_DATASETS:
        tr_p, tr_g = split_by_site(probs, gt, h, keep=False)
        te_p, te_g = split_by_site(probs, gt, h, keep=True)
        thr = tune_thresholds_per_class(tr_p, tr_g, workers=workers)
        m = evaluate_with_thresholds(te_p, te_g, thr)
        for c in CLASS_NAMES:
            for k in ("tp", "fp", "fn"):
                pooled[c][k] += int(m.get(c, {}).get(k, 0))
    pr, f1 = {}, {}
    for c in CLASS_NAMES:
        tp, fp, fn = pooled[c]["tp"], pooled[c]["fp"], pooled[c]["fn"]
        P = tp / (tp + fp + 1e-8)
        R = tp / (tp + fn + 1e-8)
        pr[c] = {"precision": P, "recall": R}
        f1[c] = 2 * P * R / (P + R + 1e-8)
    return f1, macro_f1(pr)


def event_counts(probs, gt, workers):
    """Detected-event count per class (in-sample tuned thresholds) + in-sample macro."""
    thr = tune_thresholds_per_class(probs, gt, workers=workers)
    dets = postprocess_predictions(probs, np.asarray(thr, dtype=np.float64))
    counts = {c: 0 for c in CLASS_NAMES}
    for d in dets:
        if d.label in counts:
            counts[d.label] += 1
    ismacro = macro_f1(evaluate_with_thresholds(probs, gt, thr))
    return counts, ismacro


def gt_counts(gt):
    counts = {c: 0 for c in CLASS_NAMES}
    for g in gt:
        if g.label in counts:
            counts[g.label] += 1
    return counts


def _ms(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="Checkpoint path(s); globs ok.")
    p.add_argument("--segments", type=float, nargs="+", default=[30, 60, 90, 120],
                   help="Evaluation segment/tile lengths in seconds.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default=None, help="CSV path.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # expand globs -> unique checkpoint list
    ckpts = []
    for pat in args.checkpoints:
        ckpts.extend(sorted(glob.glob(pat)) or ([pat] if Path(pat).exists() else []))
    ckpts = sorted(dict.fromkeys(ckpts))
    if not ckpts:
        raise SystemExit("No checkpoints matched.")
    segments = sorted(set(args.segments))

    print(f"Segment-length robustness | {len(ckpts)} checkpoint(s) x "
          f"{len(segments)} lengths {segments}")
    for c in ckpts:
        print(f"  {Path(c).parent.name}")
    print(f"(train segment = {cfg.TRAIN_SEGMENT_S:.0f}s; only eval tiling changes)\n")

    # data[seg] = {"loso_macro":[...], "f1": {c:[...]}, "counts": {c:[...]}, "gt": {c:int}}
    data = {s: {"loso_macro": [], "is_macro": [],
                "f1": {c: [] for c in CLASS_NAMES},
                "counts": {c: [] for c in CLASS_NAMES}, "gt": None}
            for s in segments}

    t0 = time.time()
    total = len(segments) * len(ckpts)
    step = 0
    for seg in segments:
        for ckpt in ckpts:
            step += 1
            print(f"[{step}/{total}] seg={int(seg)}s  {Path(ckpt).parent.name}")
            _tag, probs, gt = get_probs(ckpt, device, seg, args.fp16, args.no_cache)
            lf1, lmacro = loso_perclass(probs, gt, args.workers)
            counts, ismacro = event_counts(probs, gt, args.workers)
            d = data[seg]
            d["loso_macro"].append(lmacro)
            d["is_macro"].append(ismacro)
            for c in CLASS_NAMES:
                d["f1"][c].append(lf1[c])
                d["counts"][c].append(counts[c])
            if d["gt"] is None:
                d["gt"] = gt_counts(gt)
            print(f"    LOSO macro={lmacro:.4f} | events "
                  + " ".join(f"{c.upper()}={counts[c]}" for c in CLASS_NAMES) + "\n")

    # aggregate + CSV
    rows = []
    for seg in segments:
        d = data[seg]
        lm, lms = _ms(d["loso_macro"])
        im, ims = _ms(d["is_macro"])
        row = {"segment_s": int(seg), "n": len(d["loso_macro"]),
               "loso_macro": round(lm, 4), "loso_macro_std": round(lms, 4),
               "insample_macro": round(im, 4)}
        for c in CLASS_NAMES:
            fm, fs = _ms(d["f1"][c])
            cm, _cs = _ms(d["counts"][c])
            row[f"{c}_f1"] = round(fm, 4)
            row[f"{c}_f1_std"] = round(fs, 4)
            row[f"{c}_events"] = round(cm, 1)
            row[f"{c}_gt"] = d["gt"][c]
        rows.append(row)

    out = Path(args.out) if args.out else Path("runs/segment_eval.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- print tables ----
    print("=" * 104)
    print(f"EVAL SEGMENT-LENGTH ROBUSTNESS -- LOSO F1 mean (sample std) over seeds   "
          f"[total {time.time() - t0:.0f}s]")
    print("=" * 104)
    print(f"{'seg':>5} {'n':>2}  {'LOSO macro':>16}  "
          + "  ".join(f"{c.upper()+' F1':>14}" for c in CLASS_NAMES)
          + f"  {'in-sample':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['segment_s']:>4}s {r['n']:>2}  {r['loso_macro']:.3f} ({r['loso_macro_std']:.3f})  "
              + "  ".join(f"{r[c+'_f1']:.3f} ({r[c+'_f1_std']:.3f})" for c in CLASS_NAMES)
              + f"  {r['insample_macro']:>10.3f}")
    print("-" * 104)
    print("\nDetected events per class (mean over seeds)   vs ground-truth count:")
    print(f"{'seg':>5}  " + "  ".join(f"{c.upper():>18}" for c in CLASS_NAMES))
    for r in rows:
        print(f"{r['segment_s']:>4}s  "
              + "  ".join(f"{r[c+'_events']:>8.0f} / gt {r[c+'_gt']:<6}" for c in CLASS_NAMES))
    print("-" * 104)
    print(f"wrote {len(rows)} rows -> {out}")
    print("flat LOSO across lengths => 60s eval is fine (robustness footnote); watch the "
          "event counts for boundary-fragmentation effects, esp. on the longer D-calls.")


if __name__ == "__main__":
    main()
