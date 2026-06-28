"""
nave_loso.py -- leave-one-site-out (LOSO) evaluation for NAVE checkpoints, with
optional error bars across seeds.
==============================================================================
For each checkpoint: collect the validation probabilities once, then for each of
the three development sites hold it out, tune per-class thresholds on the OTHER
two, and score the held-out site. Reports per-fold B, the pooled LOSO B (sum
tp/fp/fn over folds -> one F1; the headline generalisation number), the in-sample
B, and the gap.

Official metric B = F1 of the per-class mean precision and mean recall.

Pass several seeds to --ladder to get, per ablation rung, the LOSO mean +/- std
across seeds (the error bar for the paper table).

    # one seed, full ladder:
    python nave_loso.py --ladder 666 --segment-s 60 --workers 20

    # error bars across seeds (per rung: mean +/- std over seeds):
    python nave_loso.py --ladder 0 1 1337 666 --segment-s 60 --workers 20

    # explicit checkpoint(s) (globs ok):
    python nave_loso.py runs/nave_s666_pcen1_fdy1_k129_*/nave_best.pt --segment-s 60
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from nave_train import tune_thresholds_per_class, CLASS_NAMES
from nave_evaluate import (
    build_nave_from_ckpt, collect_val_probs, evaluate_with_thresholds,
)

# canonical ablation ladder: label -> run-dir tag (must match cfg.ablation_tag())
LADDER = [
    ("base",  "pcen0_fdy0_k31"),
    ("+fdy",  "pcen0_fdy1_k31"),
    ("+pcen", "pcen1_fdy1_k31"),
    ("+k65",  "pcen1_fdy1_k65"),
    ("full",  "pcen1_fdy1_k129"),
]


def macro_b(metrics) -> float:
    """Official B: F1 of the mean precision and mean recall over the 3 classes."""
    p = float(np.mean([metrics.get(c, {}).get("precision", 0.0) for c in CLASS_NAMES]))
    r = float(np.mean([metrics.get(c, {}).get("recall", 0.0) for c in CLASS_NAMES]))
    return 2 * p * r / (p + r + 1e-8)


def split_by_site(probs, gt, site, keep):
    """keep=True -> only `site`; keep=False -> all sites except `site`."""
    pf = {k: v for k, v in probs.items() if (k[0] == site) == keep}
    gf = [g for g in gt if (g.dataset == site) == keep]
    return pf, gf


def loso_for_ckpt(ckpt, device, segment_s, workers, fp16):
    model, _ = build_nave_from_ckpt(ckpt, device)        # sets cfg ablation switches
    tag = cfg.ablation_tag()
    spec = NAVEFeatureExtractor().to(device)
    probs, gt = collect_val_probs(model, spec, device, fp16, segment_s=segment_s)

    pooled = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    folds = []
    for h in cfg.VAL_DATASETS:
        tr_p, tr_g = split_by_site(probs, gt, h, keep=False)
        te_p, te_g = split_by_site(probs, gt, h, keep=True)
        thr = tune_thresholds_per_class(tr_p, tr_g, workers=workers)
        m = evaluate_with_thresholds(te_p, te_g, thr)
        folds.append((h, macro_b(m), [round(float(t), 3) for t in thr], m))
        for c in CLASS_NAMES:
            for k in ("tp", "fp", "fn"):
                pooled[c][k] += int(m.get(c, {}).get(k, 0))

    pooled_pr = {}
    for c in CLASS_NAMES:
        tp, fp, fn = pooled[c]["tp"], pooled[c]["fp"], pooled[c]["fn"]
        pooled_pr[c] = {"precision": tp / (tp + fp + 1e-8),
                        "recall": tp / (tp + fn + 1e-8)}
    pooled_b = macro_b(pooled_pr)

    thr_all = tune_thresholds_per_class(probs, gt, workers=workers)
    insample_b = macro_b(evaluate_with_thresholds(probs, gt, thr_all))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tag, folds, pooled_b, insample_b


def resolve_ladder(seed):
    """(label, seed, tag, ckpt) for each rung that exists for this seed."""
    out = []
    for label, tag in LADDER:
        hits = glob.glob(f"runs/nave_s{seed}_{tag}_*/nave_best.pt")
        if not hits:
            print(f"  [ladder] missing rung '{label}' (tag {tag}) for seed {seed}")
            continue
        newest = max(hits, key=lambda p: Path(p).parent.stat().st_mtime)
        out.append((label, seed, tag, newest))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", help="Checkpoint path(s); globs are fine.")
    p.add_argument("--ladder", type=int, nargs="+", default=None,
                   help="One or more seeds; resolves the 5 ablation rungs for each "
                        "and reports per-rung mean +/- std across seeds.")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Eval tile length in seconds (use 60 to match the paper).")
    p.add_argument("--workers", type=int, default=8, help="Threshold-tuner workers.")
    p.add_argument("--fp16", action="store_true", help="fp16 inference.")
    p.add_argument("--per-site", action="store_true",
                   help="Print per-fold per-class P/R/F1 + counts.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the work list: (group_label, seed_or_None, tag_or_None, ckpt)
    items = []
    if args.ladder is not None:
        for seed in args.ladder:
            items += resolve_ladder(seed)
    items += [(Path(c).parent.name, None, None, c) for c in args.checkpoints]
    if not items:
        raise SystemExit("Pass checkpoint path(s) or --ladder SEED [SEED ...].")

    seeds = args.ladder or []
    print(f"LOSO over {list(cfg.VAL_DATASETS)} | segment={args.segment_s:.0f}s "
          f"| {len(items)} run(s)" + (f" | seeds={seeds}" if seeds else "") + "\n")

    # group_label -> list of (seed, pooled_b, insample_b)
    agg: dict = {}
    order: list = []
    for label, seed, tag, ckpt in items:
        rtag, folds, pooled_b, insample_b = loso_for_ckpt(
            ckpt, device, args.segment_s, args.workers, args.fp16)
        gap = insample_b - pooled_b
        sd = f"s{seed} " if seed is not None else ""
        print(f"=== {label}  [{rtag}]  {sd}({Path(ckpt).parent.name}) ===")
        for h, b, thr, m in folds:
            print(f"    held-out {h:16} B={b:.4f}  thr={thr}")
            if args.per_site:
                for c in CLASS_NAMES:
                    mc = m.get(c, {})
                    print(f"        {c.upper():6} P={mc.get('precision', 0):.3f} "
                          f"R={mc.get('recall', 0):.3f} F1={mc.get('f1', 0):.3f}  "
                          f"tp={mc.get('tp', 0)} fp={mc.get('fp', 0)} fn={mc.get('fn', 0)}")
        print(f"  pooled LOSO B = {pooled_b:.4f} | in-sample = {insample_b:.4f} "
              f"| gap = {gap:+.4f}\n")
        if label not in agg:
            agg[label] = []
            order.append((label, rtag))
        agg[label].append((seed, pooled_b, insample_b))

    # ---- aggregate table (mean +/- std across seeds, per rung) ----
    print("=" * 92)
    print("ABLATION SUMMARY -- pooled LOSO B (mean +/- std over seeds)")
    print("=" * 92)
    print(f"{'rung':8} {'tag':18} {'n':>2}  {'pooled LOSO':>18}  {'in-sample':>18}  per-seed LOSO")
    print("-" * 92)
    for label, tag in order:
        rows = agg[label]
        pooled = np.array([r[1] for r in rows], dtype=float)
        insamp = np.array([r[2] for r in rows], dtype=float)
        per_seed = "  ".join(
            f"{('s'+str(s)) if s is not None else '?'}={b:.3f}" for s, b, _ in rows)
        print(f"{label:8} {tag:18} {len(rows):>2}  "
              f"{pooled.mean():.4f} +/- {pooled.std(ddof=0):.4f}  "
              f"{insamp.mean():.4f} +/- {insamp.std(ddof=0):.4f}  {per_seed}")
    print("-" * 92)
    print("std is population std over seeds (ddof=0); report as the table error bar.")


if __name__ == "__main__":
    main()
