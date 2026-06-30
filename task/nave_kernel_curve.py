"""
nave_kernel_curve.py -- per-class and macro F1 across the conformer-kernel sweep,
written to a CSV for plotting F1 vs kernel size (or receptive field).
==============================================================================
Discovers every full-NAVE kernel run for a seed (tag pcen1_fdy1_k*, newest per
kernel), evaluates each at the chosen tile length, and writes one row per kernel:

    kernel, rf_s, bmabz_f1, d_f1, bp_f1, macro_f1   [, loso_* with --loso]

rf_s = receptive field in seconds = kernel * 0.02 (20 ms/frame). F1 numbers are
in-sample (thresholds tuned on the full dev set); add --loso to also emit the
pooled leave-one-site-out F1 per class. Reuses the LOSO prob cache, so kernels
already evaluated elsewhere are instant.

    python nave_kernel_curve.py --seed 666 --segment-s 60 --workers 20
    python nave_kernel_curve.py --seed 666 --segment-s 60 --loso \
        --out runs/kernel_curve_s666.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import time
from pathlib import Path

import torch

import nave_config as cfg
from nave_train import tune_thresholds_per_class, CLASS_NAMES
from nave_evaluate import evaluate_with_thresholds
from nave_loso import get_probs, macro_f1, split_by_site

FRAME_S = 0.02  # 20 ms per frame -> receptive field = kernel * FRAME_S


def discover_kernels(seed):
    """kernel -> newest checkpoint, for tag pcen1_fdy1_k<kernel> at this seed."""
    pat = re.compile(rf"nave_s{seed}_pcen1_fdy1_k(\d+)_")
    found = {}
    for pt in glob.glob(f"runs/nave_s{seed}_pcen1_fdy1_k*_*/nave_best.pt"):
        m = pat.search(Path(pt).parent.name)
        if not m:
            continue
        k = int(m.group(1))
        if (k not in found or
                Path(pt).parent.stat().st_mtime > Path(found[k]).parent.stat().st_mtime):
            found[k] = pt
    return dict(sorted(found.items()))


def per_class_f1(metrics):
    return {c: float(metrics.get(c, {}).get("f1", 0.0)) for c in CLASS_NAMES}


def insample_f1(probs, gt, workers):
    thr = tune_thresholds_per_class(probs, gt, workers=workers)
    m = evaluate_with_thresholds(probs, gt, thr)
    return per_class_f1(m), macro_f1(m)


def loso_f1(probs, gt, workers):
    pooled = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    for h in cfg.VAL_DATASETS:
        tr_p, tr_g = split_by_site(probs, gt, h, keep=False)
        te_p, te_g = split_by_site(probs, gt, h, keep=True)
        thr = tune_thresholds_per_class(tr_p, tr_g, workers=workers)
        m = evaluate_with_thresholds(te_p, te_g, thr)
        for c in CLASS_NAMES:
            for k in ("tp", "fp", "fn"):
                pooled[c][k] += int(m.get(c, {}).get(k, 0))
    pr = {}
    for c in CLASS_NAMES:
        tp, fp, fn = pooled[c]["tp"], pooled[c]["fp"], pooled[c]["fn"]
        P = tp / (tp + fp + 1e-8)
        R = tp / (tp + fn + 1e-8)
        pr[c] = {"precision": P, "recall": R, "f1": 2 * P * R / (P + R + 1e-8)}
    return {c: pr[c]["f1"] for c in CLASS_NAMES}, macro_f1(pr)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--loso", action="store_true",
                   help="Also compute pooled leave-one-site-out F1 per kernel.")
    p.add_argument("--out", default=None, help="CSV path (default runs/kernel_curve_*.csv).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kernels = discover_kernels(args.seed)
    if not kernels:
        raise SystemExit(f"No pcen1_fdy1_k* runs found for seed {args.seed}.")

    print(f"Found {len(kernels)} kernels for seed {args.seed} "
          f"(tag pcen1_fdy1_k*, newest per kernel):")
    for k, ckpt in kernels.items():
        print(f"  k={k:>3}  RF={k * FRAME_S:5.2f}s   {Path(ckpt).parent.name}")
    print(f"\nEvaluating at {args.segment_s:.0f}s | metric: in-sample"
          + (" + LOSO" if args.loso else "") + "\n")

    rows = []
    t_start = time.time()
    for i, (k, ckpt) in enumerate(kernels.items(), 1):
        print(f"[{i}/{len(kernels)}] k={k}  ({Path(ckpt).parent.name})")
        _tag, probs, gt = get_probs(ckpt, device, args.segment_s, args.fp16, args.no_cache)

        isf1, ismacro = insample_f1(probs, gt, args.workers)
        row = {"kernel": k, "rf_s": round(k * FRAME_S, 3),
               "bmabz_f1": round(isf1["bmabz"], 4),
               "d_f1": round(isf1["d"], 4),
               "bp_f1": round(isf1["bp"], 4),
               "macro_f1": round(ismacro, 4)}
        msg = (f"    in-sample  BMABZ={isf1['bmabz']:.3f} D={isf1['d']:.3f} "
               f"BP={isf1['bp']:.3f} | macro={ismacro:.3f}")

        if args.loso:
            lf1, lmacro = loso_f1(probs, gt, args.workers)
            row.update(loso_bmabz_f1=round(lf1["bmabz"], 4),
                       loso_d_f1=round(lf1["d"], 4),
                       loso_bp_f1=round(lf1["bp"], 4),
                       loso_macro_f1=round(lmacro, 4))
            msg += (f"\n    LOSO       BMABZ={lf1['bmabz']:.3f} D={lf1['d']:.3f} "
                    f"BP={lf1['bp']:.3f} | macro={lmacro:.3f}")
        print(msg + "\n")
        rows.append(row)

    # ---- write CSV ----
    out = Path(args.out) if args.out else Path(
        f"runs/kernel_curve_s{args.seed}_seg{int(args.segment_s)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- print a compact table ----
    print("=" * 72)
    print(f"KERNEL SWEEP (seed {args.seed}, {args.segment_s:.0f}s)   "
          f"[total {time.time() - t_start:.0f}s]")
    print("=" * 72)
    hdr = f"{'k':>4} {'RF(s)':>6} {'BMABZ':>7} {'D':>7} {'BP':>7} {'macro':>7}"
    if args.loso:
        hdr += f"   {'LOSO-macro':>10}"
    print(hdr)
    print("-" * 72)
    for r in rows:
        line = (f"{r['kernel']:>4} {r['rf_s']:>6.2f} {r['bmabz_f1']:>7.3f} "
                f"{r['d_f1']:>7.3f} {r['bp_f1']:>7.3f} {r['macro_f1']:>7.3f}")
        if args.loso:
            line += f"   {r['loso_macro_f1']:>10.3f}"
        print(line)
    print("-" * 72)
    print(f"wrote {len(rows)} rows -> {out}")
    print("plot: x=kernel (or rf_s), y=bmabz_f1/d_f1/bp_f1 (three lines) + macro_f1.")


if __name__ == "__main__":
    main()
