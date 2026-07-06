"""
nave_kernel_curve.py -- per-class and macro F1 across the conformer-kernel sweep.
==============================================================================
Two modes:

SINGLE SEED (curve): every full-NAVE kernel run for one seed (tag pcen1_fdy1_k*,
newest per kernel), evaluated at the chosen tile length, written to a CSV:
    kernel, rf_s, bmabz_f1, d_f1, bp_f1, macro_f1   [, loso_* with --loso]

    python nave_kernel_curve.py --seed 666 --segment-s 60 --workers 20 --loso

MULTI SEED (error bars): given several seeds, aggregate each kernel across them
and report mean (sample std). This is the fluke-vs-real test -- e.g. does k153's
single-seed lead survive over seeds, compared with k129?

    python nave_kernel_curve.py --seeds 0 1 666 --kernels 129 153 \
        --segment-s 60 --workers 20

rf_s = receptive field in seconds = kernel * 0.02 (20 ms/frame). Reuses the LOSO
prob cache, so kernels already evaluated elsewhere are instant.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import time
from pathlib import Path

import numpy as np
import torch

import nave_config as cfg
from nave_train import tune_thresholds_per_class, CLASS_NAMES
from nave_evaluate import evaluate_with_thresholds
from nave_loso import get_probs, macro_f1, split_by_site

FRAME_S = 0.02  # 20 ms per frame -> receptive field = kernel * FRAME_S


def newest_ckpt(seed, k):
    """Newest checkpoint for a specific seed + kernel (full-NAVE tag), or None."""
    hits = glob.glob(f"runs/nave_s{seed}_pcen1_fdy1_k{k}_*/nave_best.pt")
    return max(hits, key=lambda p: Path(p).parent.stat().st_mtime) if hits else None


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


def _ms(a):
    a = np.asarray(a, dtype=float)
    sd = a.std(ddof=1) if a.size > 1 else 0.0
    return a.mean(), sd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=None, help="Single-seed curve mode.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multi-seed mode: aggregate each kernel across these seeds "
                        "(mean +/- sample std).")
    p.add_argument("--kernels", type=int, nargs="+", default=None,
                   help="Restrict to these kernels (multi-seed mode). Default: union "
                        "of kernels present across the given seeds.")
    p.add_argument("--min-seeds", type=int, default=1,
                   help="(multi-seed) only include kernels present at >= this many "
                        "seeds, so error bars are meaningful (e.g. 3 for the full sweep).")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--loso", action="store_true",
                   help="(single-seed) also emit pooled LOSO F1 per kernel.")
    p.add_argument("--out", default=None, help="CSV path.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  single-seed curve                                                          #
# --------------------------------------------------------------------------- #
def run_single(args, device):
    kernels = discover_kernels(args.seed)
    if not kernels:
        raise SystemExit(f"No pcen1_fdy1_k* runs found for seed {args.seed}.")
    print(f"Found {len(kernels)} kernels for seed {args.seed} (newest per kernel):")
    for k, ckpt in kernels.items():
        print(f"  k={k:>3}  RF={k * FRAME_S:5.2f}s   {Path(ckpt).parent.name}")
    print(f"\nEvaluating at {args.segment_s:.0f}s | in-sample"
          + (" + LOSO" if args.loso else "") + "\n")

    rows = []
    t0 = time.time()
    for i, (k, ckpt) in enumerate(kernels.items(), 1):
        print(f"[{i}/{len(kernels)}] k={k}  ({Path(ckpt).parent.name})")
        _tag, probs, gt = get_probs(ckpt, device, args.segment_s, args.fp16, args.no_cache)
        isf1, ismacro = insample_f1(probs, gt, args.workers)
        row = {"kernel": k, "rf_s": round(k * FRAME_S, 3),
               "bmabz_f1": round(isf1["bmabz"], 4), "d_f1": round(isf1["d"], 4),
               "bp_f1": round(isf1["bp"], 4), "macro_f1": round(ismacro, 4)}
        msg = (f"    in-sample  BMABZ={isf1['bmabz']:.3f} D={isf1['d']:.3f} "
               f"BP={isf1['bp']:.3f} | macro={ismacro:.3f}")
        if args.loso:
            lf1, lmacro = loso_f1(probs, gt, args.workers)
            row.update(loso_bmabz_f1=round(lf1["bmabz"], 4), loso_d_f1=round(lf1["d"], 4),
                       loso_bp_f1=round(lf1["bp"], 4), loso_macro_f1=round(lmacro, 4))
            msg += (f"\n    LOSO       BMABZ={lf1['bmabz']:.3f} D={lf1['d']:.3f} "
                    f"BP={lf1['bp']:.3f} | macro={lmacro:.3f}")
        print(msg + "\n")
        rows.append(row)

    out = Path(args.out) if args.out else Path(
        f"runs/kernel_curve_s{args.seed}_seg{int(args.segment_s)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("=" * 72)
    print(f"KERNEL SWEEP (seed {args.seed}, {args.segment_s:.0f}s)   [total {time.time() - t0:.0f}s]")
    print("=" * 72)
    hdr = f"{'k':>4} {'RF(s)':>6} {'BMABZ':>7} {'D':>7} {'BP':>7} {'macro':>7}"
    if args.loso:
        hdr += f"   {'LOSO-macro':>10}"
    print(hdr); print("-" * 72)
    for r in rows:
        line = (f"{r['kernel']:>4} {r['rf_s']:>6.2f} {r['bmabz_f1']:>7.3f} "
                f"{r['d_f1']:>7.3f} {r['bp_f1']:>7.3f} {r['macro_f1']:>7.3f}")
        if args.loso:
            line += f"   {r['loso_macro_f1']:>10.3f}"
        print(line)
    print("-" * 72)
    print(f"wrote {len(rows)} rows -> {out}")


# --------------------------------------------------------------------------- #
#  multi-seed aggregation (error bars per kernel)                             #
# --------------------------------------------------------------------------- #
def run_multi(args, device):
    seeds = args.seeds
    if args.kernels:
        kernels = sorted(set(args.kernels))
    else:
        ks = set()
        for s in seeds:
            ks |= set(discover_kernels(s).keys())
        kernels = sorted(ks)
    if not kernels:
        raise SystemExit("No kernels found for the given seeds.")

    # resolve grid + drop kernels without enough seeds for a real error bar
    grid = {(k, s): newest_ckpt(s, k) for k in kernels for s in seeds}
    avail = {k: sum(grid[(k, s)] is not None for s in seeds) for k in kernels}
    dropped = [k for k in kernels if avail[k] < args.min_seeds]
    kernels = [k for k in kernels if avail[k] >= args.min_seeds]
    if dropped:
        print("Skipping kernels with < %d seeds: " % args.min_seeds
              + ", ".join("k%d(n=%d)" % (k, avail[k]) for k in dropped))
    if not kernels:
        raise SystemExit("No kernels have >= %d seeds." % args.min_seeds)
    print(f"Multi-seed kernel comparison | seeds={seeds} | {len(kernels)} kernels "
          f"| min_seeds={args.min_seeds} | segment={args.segment_s:.0f}s\n")
    print("Found checkpoints (per kernel x seed):")
    hdr = f"  {'k':>4} {'RF(s)':>6}  " + "  ".join(f"s{s:>5}" for s in seeds)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in kernels:
        cells = "  ".join(f"{'OK   ' if grid[(k, s)] else 'MISS ':>6}" for s in seeds)
        print(f"  {k:>4} {k * FRAME_S:>6.2f}  {cells}")
    total = sum(v is not None for v in grid.values())
    print(f"\n{total} run(s) present; evaluating ...\n")

    # data[k] = {"loso_macro":[...], "is_macro":[...], "loso_bmabz":[...], ...}
    data = {k: {"seeds": [], "loso_macro": [], "is_macro": [],
                "loso_bmabz": [], "loso_d": [], "loso_bp": []} for k in kernels}
    t0 = time.time()
    step = 0
    for k in kernels:
        for s in seeds:
            ckpt = grid[(k, s)]
            if ckpt is None:
                continue
            step += 1
            print(f"[{step}/{total}] k={k} s{s}  ({Path(ckpt).parent.name})")
            _tag, probs, gt = get_probs(ckpt, device, args.segment_s, args.fp16, args.no_cache)
            isf1, ismacro = insample_f1(probs, gt, args.workers)
            lf1, lmacro = loso_f1(probs, gt, args.workers)
            print(f"    LOSO macro={lmacro:.4f} (BMABZ={lf1['bmabz']:.3f} "
                  f"D={lf1['d']:.3f} BP={lf1['bp']:.3f}) | in-sample macro={ismacro:.4f}\n")
            d = data[k]
            d["seeds"].append(s); d["loso_macro"].append(lmacro); d["is_macro"].append(ismacro)
            d["loso_bmabz"].append(lf1["bmabz"]); d["loso_d"].append(lf1["d"]); d["loso_bp"].append(lf1["bp"])

    # aggregate + write CSV
    rows = []
    for k in kernels:
        d = data[k]
        if not d["loso_macro"]:
            continue
        lm, lms = _ms(d["loso_macro"])
        im, ims = _ms(d["is_macro"])
        bm, bms = _ms(d["loso_bmabz"])
        dm, dms = _ms(d["loso_d"])
        pm, pms = _ms(d["loso_bp"])
        rows.append({"kernel": k, "rf_s": round(k * FRAME_S, 3), "n": len(d["seeds"]),
                     "loso_macro": round(lm, 4), "loso_macro_std": round(lms, 4),
                     "loso_bmabz_f1": round(bm, 4), "loso_bmabz_std": round(bms, 4),
                     "loso_d_f1": round(dm, 4), "loso_d_std": round(dms, 4),
                     "loso_bp_f1": round(pm, 4), "loso_bp_std": round(pms, 4),
                     "insample_macro": round(im, 4), "insample_macro_std": round(ims, 4),
                     "per_seed_loso": ";".join(f"s{s}={v:.3f}"
                                               for s, v in zip(d["seeds"], d["loso_macro"]))})

    out = Path(args.out) if args.out else Path(
        f"runs/kernel_multiseed_seg{int(args.segment_s)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("=" * 100)
    print(f"MULTI-SEED KERNEL COMPARISON -- LOSO F1, mean (sample std) over seeds   "
          f"[total {time.time() - t0:.0f}s]")
    print("=" * 100)
    print(f"{'k':>4} {'RF(s)':>6} {'n':>2}  {'LOSO macro':>16}  {'BMABZ':>14}  "
          f"{'D':>14}  {'BP':>14}   per-seed LOSO")
    print("-" * 100)
    for r in rows:
        print(f"{r['kernel']:>4} {r['rf_s']:>6.2f} {r['n']:>2}  "
              f"{r['loso_macro']:.3f} ({r['loso_macro_std']:.3f})  "
              f"{r['loso_bmabz_f1']:.3f} ({r['loso_bmabz_std']:.3f})  "
              f"{r['loso_d_f1']:.3f} ({r['loso_d_std']:.3f})  "
              f"{r['loso_bp_f1']:.3f} ({r['loso_bp_std']:.3f})   {r['per_seed_loso']}")
    print("-" * 100)
    print(f"wrote {len(rows)} rows -> {out}")
    print("read overlap, not just means: if two kernels' (mean +/- std) intervals "
          "overlap, the difference is within seed noise.")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seeds:
        run_multi(args, device)
    elif args.seed is not None:
        run_single(args, device)
    else:
        raise SystemExit("Pass --seed SEED (curve) or --seeds S1 S2 ... (error bars).")


if __name__ == "__main__":
    main()
