"""
nave_loso.py -- leave-one-site-out (LOSO) evaluation for NAVE checkpoints, with
per-class breakdown, error bars across seeds, a prob cache, and progress output.
==============================================================================
For each checkpoint: collect validation probabilities once (cached), then hold
out each of the three dev sites, tune per-class thresholds on the other two, and
score the held-out site. Reports per-fold and pooled per-class P/R/F1, the macro
F1 (official: F1 of the per-class mean precision and recall), the in-sample F1,
and the gap. Pass several seeds to --ladder for per-rung mean +/- std over seeds.

    python nave_loso.py --ladder 0 1 1337 666 --segment-s 60 --workers 20
    python nave_loso.py --ladder 666 --segment-s 60 --per-site
    python nave_loso.py runs/nave_s666_pcen1_fdy1_k129_*/nave_best.pt --segment-s 60
"""

from __future__ import annotations

import argparse
import glob
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from nave_train import tune_thresholds_per_class, CLASS_NAMES
from nave_evaluate import (
    build_nave_from_ckpt, collect_val_probs, evaluate_with_thresholds,
)

LADDER = [
    ("base",  "pcen0_fdy0_k31"),
    ("+fdy",  "pcen0_fdy1_k31"),
    ("+pcen", "pcen1_fdy1_k31"),
    ("+k65",  "pcen1_fdy1_k65"),
    ("full",  "pcen1_fdy1_k129"),
]
CACHE_DIR = Path("runs/nave_loso_cache")


def macro_f1(per_class) -> float:
    """Official macro F1: F1 of the per-class mean precision and mean recall."""
    p = float(np.mean([per_class.get(c, {}).get("precision", 0.0) for c in CLASS_NAMES]))
    r = float(np.mean([per_class.get(c, {}).get("recall", 0.0) for c in CLASS_NAMES]))
    return 2 * p * r / (p + r + 1e-8)


def split_by_site(probs, gt, site, keep):
    pf = {k: v for k, v in probs.items() if (k[0] == site) == keep}
    gf = [g for g in gt if (g.dataset == site) == keep]
    return pf, gf


def _config_tag(ckpt):
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    conf = meta.get("config", {}) if isinstance(meta, dict) else {}
    cfg.apply_ablation(use_pcen=conf.get("use_pcen", True),
                       use_fdy=conf.get("use_fdy", True),
                       conv_kernel=conf.get("conv_kernel", cfg.CONV_KERNEL))
    return cfg.ablation_tag()


def get_probs(ckpt, device, seg, fp16, no_cache):
    cp = CACHE_DIR / f"{Path(ckpt).parent.name}_seg{int(seg)}.pkl"
    if not no_cache and cp.exists():
        tag = _config_tag(ckpt)
        print(f"    cache HIT  ({cp.name})")
        with open(cp, "rb") as f:
            probs, gt = pickle.load(f)
        return tag, probs, gt

    print(f"    cache MISS -> loading model + running inference at {int(seg)}s ...")
    t0 = time.time()
    model, _ = build_nave_from_ckpt(ckpt, device)
    tag = cfg.ablation_tag()
    spec = NAVEFeatureExtractor().to(device)
    probs, gt = collect_val_probs(model, spec, device, fp16, segment_s=seg, progress=True)
    print(f"    inference done in {time.time() - t0:.0f}s  ({len(probs)} tiles, {len(gt)} gt events)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not no_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump((probs, gt), f)
        print(f"    cached -> {cp}")
    return tag, probs, gt


def loso_for_probs(probs, gt, workers, per_site):
    pooled = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    folds = []
    for h in cfg.VAL_DATASETS:
        t0 = time.time()
        tr_p, tr_g = split_by_site(probs, gt, h, keep=False)
        te_p, te_g = split_by_site(probs, gt, h, keep=True)
        thr = tune_thresholds_per_class(tr_p, tr_g, workers=workers)
        m = evaluate_with_thresholds(te_p, te_g, thr)
        folds.append((h, macro_f1(m), [round(float(t), 3) for t in thr], m))
        print(f"      fold held-out {h:16} F1={macro_f1(m):.4f}  ({time.time() - t0:.0f}s)")
        if per_site:
            for c in CLASS_NAMES:
                mc = m.get(c, {})
                print(f"          {c.upper():6} P={mc.get('precision', 0):.3f} "
                      f"R={mc.get('recall', 0):.3f} F1={mc.get('f1', 0):.3f}  "
                      f"tp={mc.get('tp', 0)} fp={mc.get('fp', 0)} fn={mc.get('fn', 0)}")
        for c in CLASS_NAMES:
            for k in ("tp", "fp", "fn"):
                pooled[c][k] += int(m.get(c, {}).get(k, 0))

    # pooled per-class P/R/F1 (from summed tp/fp/fn over folds)
    pooled_pr = {}
    for c in CLASS_NAMES:
        tp, fp, fn = pooled[c]["tp"], pooled[c]["fp"], pooled[c]["fn"]
        P = tp / (tp + fp + 1e-8)
        R = tp / (tp + fn + 1e-8)
        pooled_pr[c] = {"precision": P, "recall": R,
                        "f1": 2 * P * R / (P + R + 1e-8),
                        "tp": tp, "fp": fp, "fn": fn}
    pooled_macro = macro_f1(pooled_pr)

    print("      pooled LOSO per-class:")
    for c in CLASS_NAMES:
        pc = pooled_pr[c]
        print(f"        {c.upper():6} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f}  tp={pc['tp']} fp={pc['fp']} fn={pc['fn']}")

    t0 = time.time()
    thr_all = tune_thresholds_per_class(probs, gt, workers=workers)
    insample_metrics = evaluate_with_thresholds(probs, gt, thr_all)
    insample_macro = macro_f1(insample_metrics)
    print(f"      in-sample (tune-all) macro F1={insample_macro:.4f}  ({time.time() - t0:.0f}s)")
    return folds, pooled_pr, pooled_macro, insample_macro


def resolve_ladder_map(seed):
    out = {}
    for label, tag in LADDER:
        hits = glob.glob(f"runs/nave_s{seed}_{tag}_*/nave_best.pt")
        out[label] = (max(hits, key=lambda p: Path(p).parent.stat().st_mtime)
                      if hits else None)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", help="Checkpoint path(s); globs ok.")
    p.add_argument("--ladder", type=int, nargs="+", default=None,
                   help="One or more seeds; resolves the 5 rungs for each and reports "
                        "per-rung mean +/- std over seeds.")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--per-site", action="store_true",
                   help="Also print per-FOLD per-class detail.")
    p.add_argument("--no-cache", action="store_true", help="Ignore/skip the prob cache.")
    return p.parse_args()


def _ms(a):
    """mean (sample std) over seeds; sample std (ddof=1) for n>1, else 0."""
    a = np.asarray(a, dtype=float)
    sd = a.std(ddof=1) if a.size > 1 else 0.0
    return f"{a.mean():.3f} ({sd:.3f})"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.ladder or []

    items = []   # (group_label, seed_or_None, ckpt)
    if seeds:
        print("Resolving ablation ladder checkpoints (newest per seed+rung):")
        total_missing = 0
        for seed in seeds:
            fmap = resolve_ladder_map(seed)
            n = sum(v is not None for v in fmap.values())
            total_missing += (len(LADDER) - n)
            cells = "  ".join(f"{lab}={'OK ' if fmap[lab] else 'MISS'}" for lab, _ in LADDER)
            print(f"  seed {seed:>5}:  {cells}   ({n}/{len(LADDER)})")
            for lab, _ in LADDER:
                if fmap[lab]:
                    items.append((lab, seed, fmap[lab]))
        if total_missing:
            print(f"  NOTE: {total_missing} rung(s) missing across seeds -- skipped.")
        print()
    for c in args.checkpoints:
        items.append((Path(c).parent.name, None, c))
    if not items:
        raise SystemExit("Pass checkpoint path(s) or --ladder SEED [SEED ...].")

    print(f"LOSO over {list(cfg.VAL_DATASETS)} | segment={args.segment_s:.0f}s "
          f"| {len(items)} run(s)" + (" | cache OFF" if args.no_cache else "") + "\n")

    # group_label -> list of (seed, macro_f1, insample_f1, {class: f1})
    agg, order = {}, []
    t_start = time.time()
    for i, (label, seed, ckpt) in enumerate(items, 1):
        sd = f"s{seed} " if seed is not None else ""
        print(f"[{i}/{len(items)}] {label} {sd}({Path(ckpt).parent.name})")
        tag, probs, gt = get_probs(ckpt, device, args.segment_s, args.fp16, args.no_cache)
        folds, pooled_pr, pooled_macro, insample_macro = loso_for_probs(
            probs, gt, args.workers, args.per_site)
        gap = insample_macro - pooled_macro
        print(f"    => LOSO macro F1 = {pooled_macro:.4f} | in-sample = {insample_macro:.4f} "
              f"| gap = {gap:+.4f}\n")
        if label not in agg:
            agg[label] = []
            order.append((label, tag))
        agg[label].append((seed, pooled_macro, insample_macro,
                           {c: pooled_pr[c]["f1"] for c in CLASS_NAMES}))

    # ---- aggregate table: per-rung mean +/- std over seeds, with per-class F1 ----
    print("=" * 110)
    print(f"ABLATION SUMMARY -- LOSO F1 (mean +/- std over seeds)   [total {time.time() - t_start:.0f}s]")
    print("=" * 110)
    cls_hdr = "  ".join(f"{c.upper()+' F1':>15}" for c in CLASS_NAMES)
    print(f"{'rung':8} {'n':>2}  {'macro F1':>15}  {cls_hdr}  {'in-sample':>15}")
    print("-" * 110)
    for label, tag in order:
        rows = agg[label]
        macro = [r[1] for r in rows]
        insamp = [r[2] for r in rows]
        cls = {c: [r[3][c] for r in rows] for c in CLASS_NAMES}
        cls_cells = "  ".join(f"{_ms(cls[c]):>15}" for c in CLASS_NAMES)
        print(f"{label:8} {len(rows):>2}  {_ms(macro):>15}  {cls_cells}  {_ms(insamp):>15}")
    print("-" * 110)
    print("values: mean (sample std, ddof=1) over seeds | macro F1 = official "
          "metric (F1 of mean P, mean R) | per-class F1 from pooled tp/fp/fn.")


if __name__ == "__main__":
    main()
