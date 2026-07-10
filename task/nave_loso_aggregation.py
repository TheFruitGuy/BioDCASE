"""
nave_loso_aggregation.py
========================
Compare fold-aggregation conventions under the NAVE leave-one-site-out protocol.

The LOSO loop is identical in all cases: the trained model is fixed, the
per-class thresholds are tuned on two validation sites and applied to the
held-out third, rotating over all three sites. What differs is how the three
held-out folds are combined into one number:

  POOLED    Sum TP/FP/FN over the three folds, then P_c, R_c per class, then
            F1 = 2*Pbar*Rbar/(Pbar+Rbar) over the class means.
            -> what nave_loso.py / nave_kernel_curve.py report, and what the
               paper's Experimental Setup describes.

  FOLD-AVG  Per fold compute P_c, R_c. Average each over folds. Then
            F1 = 2*Pbar*Rbar/(Pbar+Rbar) over the class means.
            -> WhaleVAD-BPN, Sec. V-C3 ("precision and recall metrics are
               computed for each test fold ... recomputed from the averaged
               precision/recall scores"). Also what loso_eval.py::aggregate does.

  MEAN-F1   Compute the macro F1 of each fold, then average the three F1s.
            -> the naive reading of "just average the results".

POOLED weights every *event* equally. FOLD-AVG and MEAN-F1 weight every *site*
equally, which matters here because the validation sites are very unequal in
event count (casey2017 ~3.3k, kerguelen2015 ~5.5k, kerguelen2014 ~8.8k).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python nave_loso_aggregation.py \
        runs/nave_s*_pcen1_fdy1_k153_*/nave_best.pt \
        --segment-s 60 --workers 20

Prints per-seed and multi-seed mean (sample std, ddof=1) for each convention.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------
# Segment length must be set before dataset_final / eval_conformer read it.
# ----------------------------------------------------------------------

def _set_segment_length(segment_s: float, overlap_s: float) -> None:
    import config_final
    import nave_config
    config_final.EVAL_SEGMENT_S = float(segment_s)
    config_final.EVAL_OVERLAP_S = float(overlap_s)
    nave_config.EVAL_SEGMENT_S = float(segment_s)
    nave_config.EVAL_OVERLAP_S = float(overlap_s)


# ----------------------------------------------------------------------
# Aggregation maths. `folds` is a list (one per held-out site) of dicts
# {class_name: {"tp","fp","fn","precision","recall"}}.
# ----------------------------------------------------------------------

EPS = 1e-8


def _f1(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r + EPS)


def agg_pooled(folds, classes):
    """Sum counts across folds, then per-class P/R, then F1 of the class means."""
    per_class = {}
    for c in classes:
        tp = sum(f[c]["tp"] for f in folds)
        fp = sum(f[c]["fp"] for f in folds)
        fn = sum(f[c]["fn"] for f in folds)
        p = tp / (tp + fp + EPS)
        r = tp / (tp + fn + EPS)
        per_class[c] = {"precision": p, "recall": r, "f1": _f1(p, r)}
    pbar = float(np.mean([per_class[c]["precision"] for c in classes]))
    rbar = float(np.mean([per_class[c]["recall"] for c in classes]))
    return _f1(pbar, rbar), per_class


def agg_foldavg(folds, classes):
    """Average P and R across folds, then F1 of the class means (BPN Sec. V-C3)."""
    per_class = {}
    for c in classes:
        p = float(np.mean([f[c]["precision"] for f in folds]))
        r = float(np.mean([f[c]["recall"] for f in folds]))
        per_class[c] = {"precision": p, "recall": r, "f1": _f1(p, r)}
    pbar = float(np.mean([per_class[c]["precision"] for c in classes]))
    rbar = float(np.mean([per_class[c]["recall"] for c in classes]))
    return _f1(pbar, rbar), per_class


def agg_mean_fold_f1(folds, classes):
    """Macro F1 per fold, then the mean of the three."""
    per_fold = []
    for f in folds:
        pbar = float(np.mean([f[c]["precision"] for c in classes]))
        rbar = float(np.mean([f[c]["recall"] for c in classes]))
        per_fold.append(_f1(pbar, rbar))
    return float(np.mean(per_fold)), per_fold


# ----------------------------------------------------------------------
# Site slicing. probs keys are (dataset, filename, start_sample); gt entries
# are either Detection dataclasses or plain tuples (dataset, filename, ...).
# ----------------------------------------------------------------------

def _gt_site(g):
    return g.dataset if hasattr(g, "dataset") else g[0]


def subset_sites(probs, gt, sites):
    sset = set(sites)
    p = {k: v for k, v in probs.items() if k[0] in sset}
    g = [d for d in gt if _gt_site(d) in sset]
    return p, g


# ----------------------------------------------------------------------
# Probability cache, keyed like the rest of the stack (run dir + segment len).
# ----------------------------------------------------------------------

def _load_probs(ckpt: Path, segment_s: float, workers: int, fp16: bool,
                cache_dir: Path, use_cache: bool):
    from nave_evaluate import build_nave_from_ckpt
    from nave_features import NAVEFeatureExtractor
    from eval_conformer import collect_val_probs

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ckpt.parent.name}_seg{int(segment_s)}_agg.pkl"

    if use_cache and cache_file.exists():
        with cache_file.open("rb") as fh:
            probs, gt = pickle.load(fh)
        print(f"    cache HIT  ({cache_file.name})")
        return probs, gt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_nave_from_ckpt(ckpt, device)
    spec = NAVEFeatureExtractor().to(device)
    probs, gt = collect_val_probs(model, spec, device, fp16)

    if use_cache:
        with cache_file.open("wb") as fh:
            pickle.dump((probs, gt), fh, protocol=4)
        print(f"    cache WRITE ({cache_file.name})")
    return probs, gt


# ----------------------------------------------------------------------
# One seed: run the LOSO loop, return the three folds' per-class blocks.
# ----------------------------------------------------------------------

def loso_folds(probs, gt, sites, classes, workers):
    from eval_conformer import tune_thresholds_per_class, evaluate_with_thresholds

    folds, thresholds = [], []
    for held in sites:
        dev = [s for s in sites if s != held]
        dev_probs, dev_gt = subset_sites(probs, gt, dev)
        test_probs, test_gt = subset_sites(probs, gt, [held])

        thr = tune_thresholds_per_class(dev_probs, dev_gt, workers=workers)
        metrics = evaluate_with_thresholds(test_probs, test_gt, thr)

        block = {}
        for c in classes:
            m = metrics.get(c)
            if m is None:
                print(f"    WARNING: class '{c}' absent from fold '{held}'; "
                      f"counted as zeros (biases FOLD-AVG downward).")
                m = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0}
            block[c] = {k: m[k] for k in ("tp", "fp", "fn", "precision", "recall")}
        folds.append(block)
        thresholds.append([round(float(t), 3) for t in thr])

        pbar = float(np.mean([block[c]["precision"] for c in classes]))
        rbar = float(np.mean([block[c]["recall"] for c in classes]))
        print(f"    fold held-out={held:<15} thr={thresholds[-1]}  "
              f"P={pbar:.3f} R={rbar:.3f} F1={_f1(pbar, rbar):.4f}")
    return folds


# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", type=Path, nargs="+",
                   help="One NAVE checkpoint per seed (shell glob is fine).")
    p.add_argument("--segment-s", type=float, default=60.0,
                   help="Evaluation tile length in seconds (paper uses 60).")
    p.add_argument("--overlap-s", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=8, help="Threshold-tuner workers.")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("runs/_agg_cache"))
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute probabilities (required if the tile length changed).")
    return p.parse_args()


def main():
    args = parse_args()
    _set_segment_length(args.segment_s, args.overlap_s)

    import nave_config as ncfg
    from eval_conformer import CLASS_NAMES

    sites = list(ncfg.VAL_DATASETS)
    classes = list(CLASS_NAMES)

    print(f"LOSO aggregation comparison | {len(args.ckpts)} seed(s) | "
          f"segment={args.segment_s:g}s | sites={', '.join(sites)}\n")

    rows = {"pooled": [], "foldavg": [], "meanf1": []}
    percls = {"pooled": [], "foldavg": []}
    t0 = time.time()

    for i, ckpt in enumerate(args.ckpts, 1):
        if not ckpt.exists():
            sys.exit(f"missing checkpoint: {ckpt}")
        print(f"[{i}/{len(args.ckpts)}] {ckpt.parent.name}")
        probs, gt = _load_probs(ckpt, args.segment_s, args.workers, args.fp16,
                                args.cache_dir, not args.no_cache)
        folds = loso_folds(probs, gt, sites, classes, args.workers)

        f_pool, pc_pool = agg_pooled(folds, classes)
        f_favg, pc_favg = agg_foldavg(folds, classes)
        f_mean, _ = agg_mean_fold_f1(folds, classes)

        rows["pooled"].append(f_pool)
        rows["foldavg"].append(f_favg)
        rows["meanf1"].append(f_mean)
        percls["pooled"].append([pc_pool[c]["f1"] for c in classes])
        percls["foldavg"].append([pc_favg[c]["f1"] for c in classes])

        print(f"    -> POOLED   {f_pool:.4f}   FOLD-AVG {f_favg:.4f}   "
              f"MEAN-F1 {f_mean:.4f}\n")

    def ms(v):
        v = np.asarray(v, dtype=float)
        s = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        return float(v.mean()), s

    print("=" * 78)
    print(f"MULTI-SEED SUMMARY  (n={len(args.ckpts)}, mean (sample std))"
          f"   [{time.time() - t0:.0f}s]")
    print("=" * 78)
    hdr = f"{'convention':<12}" + "".join(f"{c.upper():>16}" for c in classes) + f"{'macro F1':>18}"
    print(hdr)
    print("-" * 78)
    for key, label in (("pooled", "POOLED"), ("foldavg", "FOLD-AVG")):
        pc = np.asarray(percls[key], dtype=float)          # (seeds, classes)
        cells = ""
        for j in range(pc.shape[1]):
            m, s = ms(pc[:, j])
            cells += f"{m:>10.3f} ({s:.3f})"
        m, s = ms(rows[key])
        print(f"{label:<12}{cells}{m:>11.4f} ({s:.3f})")
    m, s = ms(rows["meanf1"])
    print(f"{'MEAN-F1':<12}{'':>48}{m:>11.4f} ({s:.3f})")
    print("-" * 78)

    mp, _ = ms(rows["pooled"])
    mf, _ = ms(rows["foldavg"])
    d = mp - mf
    print(f"\nPOOLED - FOLD-AVG = {d:+.4f}")
    print("FOLD-AVG is the WhaleVAD-BPN convention (their Sec. V-C3), so this is")
    print("the number directly comparable to their reported 0.475.")
    if abs(d) < 0.005:
        print("The two conventions agree to within 0.005; the matched-protocol")
        print("claim in Sec. 4.2 holds under either aggregation.")
    else:
        print("The conventions differ by more than 0.005. Report the FOLD-AVG")
        print("number against BPN, or state the aggregation difference explicitly.")


if __name__ == "__main__":
    main()
