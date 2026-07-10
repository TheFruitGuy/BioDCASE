"""
nave_loso_aggregation.py -- fold-aggregation conventions under the NAVE LOSO protocol.
=====================================================================================
The LOSO loop is identical in all cases and is the one from ``nave_loso.py``: the
trained model is fixed, per-class thresholds are tuned on two validation sites and
applied to the held-out third, rotating over all three sites. What differs is only
how the three held-out folds are combined into one number.

  POOLED    Sum tp/fp/fn over the three folds, then per-class P/R, then
            F1 = 2*Pbar*Rbar/(Pbar+Rbar) over the class means.
            -> what nave_loso.py reports, what the paper's Eq. 1 describes, and
               what the official scorer does (evaluation_submissions.py:153).

  FOLD-AVG  Per fold compute per-class P/R. Average each over folds. Then
            F1 = 2*Pbar*Rbar/(Pbar+Rbar) over the class means.
            -> WhaleVAD-BPN, Sec. V-C3. Also what loso_eval.py::aggregate does.

  MEAN-F1   Macro F1 of each fold, then the mean of the three.
            -> the naive reading of "just average the results".

POOLED weights every event equally. FOLD-AVG and MEAN-F1 weight every site
equally, which matters because the validation sites are unequal in event count
(casey2017 ~3.3k, kerguelen2015 ~5.5k, kerguelen2014 ~8.8k).

Reuses nave_loso.get_probs, so it hits the existing runs/nave_loso_cache and does
no inference if the seg60 caches are already there.

    python nave_loso_aggregation.py runs/nave_s*_pcen1_fdy1_k153_*/nave_best.pt \
        --segment-s 60 --workers 40
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import nave_config as cfg
from nave_train import tune_thresholds_per_class, CLASS_NAMES
from nave_evaluate import evaluate_with_thresholds
from nave_loso import get_probs, split_by_site

EPS = 1e-8


def _f1(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r + EPS)


def _macro_from_per_class(per_class) -> float:
    """F1 of the per-class mean precision and mean recall (paper Eq. 1)."""
    p = float(np.mean([per_class[c]["precision"] for c in CLASS_NAMES]))
    r = float(np.mean([per_class[c]["recall"] for c in CLASS_NAMES]))
    return _f1(p, r)


# ----------------------------------------------------------------------
# The three aggregations. `folds` is a list of per-class metric dicts, one per
# held-out site, each {class: {tp, fp, fn, precision, recall}}.
# ----------------------------------------------------------------------

def agg_pooled(folds):
    per_class = {}
    for c in CLASS_NAMES:
        tp = sum(int(f[c]["tp"]) for f in folds)
        fp = sum(int(f[c]["fp"]) for f in folds)
        fn = sum(int(f[c]["fn"]) for f in folds)
        p = tp / (tp + fp + EPS)
        r = tp / (tp + fn + EPS)
        per_class[c] = {"precision": p, "recall": r, "f1": _f1(p, r)}
    return _macro_from_per_class(per_class), per_class


def agg_foldavg(folds):
    per_class = {}
    for c in CLASS_NAMES:
        p = float(np.mean([f[c]["precision"] for f in folds]))
        r = float(np.mean([f[c]["recall"] for f in folds]))
        per_class[c] = {"precision": p, "recall": r, "f1": _f1(p, r)}
    return _macro_from_per_class(per_class), per_class


def agg_mean_fold_f1(folds):
    per_fold = []
    for f in folds:
        p = float(np.mean([f[c]["precision"] for c in CLASS_NAMES]))
        r = float(np.mean([f[c]["recall"] for c in CLASS_NAMES]))
        per_fold.append(_f1(p, r))
    return float(np.mean(per_fold)), per_fold


# ----------------------------------------------------------------------

def loso_folds(probs, gt, workers, verbose=True):
    """Run the LOSO loop, returning one per-class metric block per held-out site."""
    folds = []
    for held in cfg.VAL_DATASETS:
        t0 = time.time()
        tr_p, tr_g = split_by_site(probs, gt, held, keep=False)
        te_p, te_g = split_by_site(probs, gt, held, keep=True)
        thr = tune_thresholds_per_class(tr_p, tr_g, workers=workers)
        m = evaluate_with_thresholds(te_p, te_g, thr)

        block = {}
        for c in CLASS_NAMES:
            mc = m.get(c)
            if mc is None:
                print(f"      WARNING: class '{c}' absent from fold '{held}'. "
                      f"Zeros substituted, which biases FOLD-AVG downward.")
                mc = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0}
            block[c] = {k: mc.get(k, 0) for k in
                        ("tp", "fp", "fn", "precision", "recall")}
        folds.append(block)

        if verbose:
            p = float(np.mean([block[c]["precision"] for c in CLASS_NAMES]))
            r = float(np.mean([block[c]["recall"] for c in CLASS_NAMES]))
            print(f"      fold held-out {held:16} P={p:.3f} R={r:.3f} "
                  f"F1={_f1(p, r):.4f}  thr={[round(float(t), 3) for t in thr]}  "
                  f"({time.time() - t0:.0f}s)")
    return folds


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="One checkpoint per seed; globs ok.")
    p.add_argument("--segment-s", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def _ms(a):
    a = np.asarray(a, dtype=float)
    sd = a.std(ddof=1) if a.size > 1 else 0.0
    return float(a.mean()), float(sd)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [Path(c) for c in args.checkpoints]

    print(f"LOSO aggregation comparison | {len(ckpts)} seed(s) | "
          f"segment={args.segment_s:.0f}s | sites={', '.join(cfg.VAL_DATASETS)}\n")

    macro = {"pooled": [], "foldavg": [], "meanf1": []}
    percls = {"pooled": [], "foldavg": []}
    t_start = time.time()

    for i, ckpt in enumerate(ckpts, 1):
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        print(f"[{i}/{len(ckpts)}] {ckpt.parent.name}")
        _tag, probs, gt = get_probs(str(ckpt), device, args.segment_s,
                                    args.fp16, args.no_cache)
        folds = loso_folds(probs, gt, args.workers)

        f_pool, pc_pool = agg_pooled(folds)
        f_favg, pc_favg = agg_foldavg(folds)
        f_mean, _ = agg_mean_fold_f1(folds)

        macro["pooled"].append(f_pool)
        macro["foldavg"].append(f_favg)
        macro["meanf1"].append(f_mean)
        percls["pooled"].append([pc_pool[c]["f1"] for c in CLASS_NAMES])
        percls["foldavg"].append([pc_favg[c]["f1"] for c in CLASS_NAMES])

        print(f"    => POOLED {f_pool:.4f} | FOLD-AVG {f_favg:.4f} | "
              f"MEAN-F1 {f_mean:.4f}\n")

    print("=" * 92)
    print(f"MULTI-SEED SUMMARY  (n={len(ckpts)}, mean (sample std, ddof=1))"
          f"   [total {time.time() - t_start:.0f}s]")
    print("=" * 92)
    cls_hdr = "  ".join(f"{c.upper() + ' F1':>15}" for c in CLASS_NAMES)
    print(f"{'convention':12} {cls_hdr}  {'macro F1':>15}")
    print("-" * 92)
    for key, label in (("pooled", "POOLED"), ("foldavg", "FOLD-AVG")):
        pc = np.asarray(percls[key], dtype=float)
        cells = "  ".join(
            f"{_ms(pc[:, j])[0]:.3f} ({_ms(pc[:, j])[1]:.3f})".rjust(15)
            for j in range(pc.shape[1]))
        m, s = _ms(macro[key])
        print(f"{label:12} {cells}  {f'{m:.4f} ({s:.3f})':>15}")
    m, s = _ms(macro["meanf1"])
    print(f"{'MEAN-F1':12} {'':>51}  {f'{m:.4f} ({s:.3f})':>15}")
    print("-" * 92)

    mp, _ = _ms(macro["pooled"])
    mf, _ = _ms(macro["foldavg"])
    d = mp - mf
    print(f"\nPOOLED - FOLD-AVG = {d:+.4f}")
    print("POOLED is the official metric (evaluation_submissions.py) and the paper's Eq. 1.")
    print("FOLD-AVG is the WhaleVAD-BPN convention (their Sec. V-C3), so that column is")
    print("the one directly comparable to their reported 0.475.")
    if abs(d) < 0.005:
        print("\nThe two agree to within 0.005, so the comparison in Sec. 4.2 holds")
        print("under either aggregation.")
    else:
        print("\nThe two differ by more than 0.005. Quote the FOLD-AVG number when")
        print("comparing against BPN, and state the aggregation difference explicitly.")


if __name__ == "__main__":
    main()
