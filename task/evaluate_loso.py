"""
evaluate_loso.py -- leave-one-site-out generalization for the conformer line.
=============================================================================
Threshold-LOSO: the model is trained once; for each held-out dev site we tune the
per-class thresholds on the OTHER sites and score the held-out site with them.
Measures how well the operating point generalizes across sites. Metric = official
B = F1(mean-P, mean-R). Now also prints a PER-SITE breakdown (per-class precision /
recall / F1 for each held-out site).

Multiple checkpoints are reported two honest ways (no cherry-picking the best):
  - PER-SEED mean +/- std : each seed's LOSO scored independently, then averaged.
  - ENSEMBLE              : the seeds' probabilities are averaged into one model.

    CUDA_VISIBLE_DEVICES=0 python evaluate_loso.py --segment-s 60 --use_fp16 --val-workers 20 \
        --checkpoints runs/phase13r_3c_s666_*/phase13r_best.pt \
                      runs/phase13r_3c_s1337_*/phase13r_best.pt \
                      runs/phase13r_3c_s42_*/phase13r_best.pt

Each fold aggregates B two ways: mean-of-folds and POOLED (sum tp/fp/fn over folds,
one F1). Pooled is the headline LOSO number. Per-site detail is printed for the
ensemble (multi-seed) or the single model; add --per-site-seeds for every seed.
Probabilities cache per (run, segment_s), shared with nave_eval / evaluate_phase13.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from postprocess_final import Detection
from eval_conformer import (
    load_conformer, tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)
from nave_eval import (
    cache_path_for, load_cached_probs, save_probs_to_cache, build_nave_from_ckpt,
    combine_probs,
)
from nave_eval_official import tune_for_official
from evaluate_phase13 import predict_probs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--segment-s", type=float, default=60.0)
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint ENSEMBLE weights (default uniform).")
    p.add_argument("--tune-metric", choices=["f1", "official"], default="f1")
    p.add_argument("--no-ensemble", action="store_true",
                   help="Skip the ensemble; only per-seed mean +/- std.")
    p.add_argument("--per-site-seeds", action="store_true",
                   help="Print the per-site breakdown for every seed (not just the ensemble).")
    p.add_argument("--cache_dir", type=str, default="runs/nave_prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--val-workers", type=int, default=min((os.cpu_count() or 1), 13))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        p.error(f"--weights has {len(args.weights)} values, need {len(args.checkpoints)}.")
    return args


def load_model_spec(ckpt, device):
    """Native NAVE -> NAVE + NAVEFeatureExtractor; any legacy phase variant -> load_conformer."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    native = (isinstance(obj, dict) and obj.get("model") == "NAVE") \
        or any(k.startswith("stem.") for k in sd)
    if native:
        return build_nave_from_ckpt(ckpt, device), NAVEFeatureExtractor().to(device)
    model, spec, _ = load_conformer(ckpt, device)
    return model.to(device).eval(), spec.to(device)


def get_or_compute(ckpt, loader_factory, device, cache_dir, seg, no_cache, use_fp16):
    cp = cache_path_for(Path(ckpt), cache_dir, seg)
    if not no_cache and cp.exists():
        print(f"    cache hit ({cp.name})")
        return load_cached_probs(cp)
    print("    cache miss, loading architecture + running inference...")
    model, spec = load_model_spec(ckpt, device)
    probs = predict_probs(model, spec, loader_factory(), device, use_fp16)
    if not no_cache:
        save_probs_to_cache(probs, cp)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def _tune(probs, gt, workers, metric):
    thr = tune_thresholds_per_class(probs, gt, workers=workers)
    if metric == "official":
        thr, _ = tune_for_official(probs, gt, start=thr)
    return thr


def _filter(probs, gt, keep_sites):
    p = {k: v for k, v in probs.items() if k[0] in keep_sites}
    g = [d for d in gt if d.dataset in keep_sites]
    return p, g


def run_loso(probs, gt, sites, workers, tune_metric):
    """One model's LOSO. Returns per-fold metrics, pooled B, mean-of-folds, in-sample."""
    folds, pooled = [], {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    for held in sites:
        p_tr, g_tr = _filter(probs, gt, [s for s in sites if s != held])
        thr = _tune(p_tr, g_tr, workers, tune_metric)
        p_h, g_h = _filter(probs, gt, [held])
        m = evaluate_with_thresholds(p_h, g_h, thr)
        folds.append({"site": held, "B": macro_f1_paper(m), "metrics": m, "thr": thr})
        for c in CLASS_NAMES:
            for k in ("tp", "fp", "fn"):
                pooled[c][k] += int(m.get(c, {}).get(k, 0))
    fold_Bs = np.array([f["B"] for f in folds])
    pP = [pooled[c]["tp"] / (pooled[c]["tp"] + pooled[c]["fp"] + 1e-8) for c in CLASS_NAMES]
    pR = [pooled[c]["tp"] / (pooled[c]["tp"] + pooled[c]["fn"] + 1e-8) for c in CLASS_NAMES]
    mP, mR = float(np.mean(pP)), float(np.mean(pR))
    pooled_B = 2 * mP * mR / (mP + mR + 1e-8)
    thr_all = _tune(probs, gt, workers, tune_metric)
    insample_B = macro_f1_paper(evaluate_with_thresholds(probs, gt, thr_all))
    return dict(folds=folds, pooled=pooled, pooled_B=pooled_B, pooled_P=mP, pooled_R=mR,
                mean_fold=float(fold_Bs.mean()), std_fold=float(fold_Bs.std()),
                insample_B=insample_B)


def print_per_site(res, label):
    """Per held-out site: per-class P / R / F1 (+ counts) and the fold B."""
    print(f"\n  PER-SITE LOSO breakdown -- {label}")
    print("  (thresholds tuned on the OTHER sites, then applied to the held-out site)")
    for f in res["folds"]:
        print(f"    held-out {f['site']:16} fold B={f['B']:.4f}  "
              f"thr={[round(float(t),2) for t in f['thr']]}")
        print(f"      {'class':6} {'P':>6} {'R':>6} {'F1':>6}   "
              f"{'tp':>5} {'fp':>5} {'fn':>5}")
        for c in CLASS_NAMES:
            m = f["metrics"].get(c, {})
            print(f"      {c.upper():6} {m.get('precision',0):6.3f} {m.get('recall',0):6.3f} "
                  f"{m.get('f1',0):6.3f}   {int(m.get('tp',0)):5d} {int(m.get('fp',0)):5d} "
                  f"{int(m.get('fn',0)):5d}")
    # pooled-over-sites per-class
    print(f"    POOLED over sites: meanP={res['pooled_P']:.3f} meanR={res['pooled_R']:.3f} "
          f"-> B={res['pooled_B']:.4f}")
    print(f"      {'class':6} {'P':>6} {'R':>6}   {'tp':>5} {'fp':>5} {'fn':>5}")
    for c in CLASS_NAMES:
        pc = res["pooled"][c]
        P = pc["tp"] / (pc["tp"] + pc["fp"] + 1e-8)
        R = pc["tp"] / (pc["tp"] + pc["fn"] + 1e-8)
        print(f"      {c.upper():6} {P:6.3f} {R:6.3f}   "
              f"{pc['tp']:5d} {pc['fp']:5d} {pc['fn']:5d}")


def main():
    args = parse_args()
    import config_final as _cf
    _cf.USE_3CLASS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.checkpoints)
    print(f"[LOSO] {n} ckpt | {args.segment_s:.0f}s tiles | tune={args.tune_metric}")

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
    prob_dicts, names = [], []
    for i, ckpt in enumerate(args.checkpoints):
        names.append(Path(ckpt).parent.name)
        print(f"  [{i+1}/{n}] {names[-1]}")
        _cf.USE_3CLASS = True
        prob_dicts.append(get_or_compute(ckpt, loader_factory, device, cache_dir,
                                         args.segment_s, args.no_cache, args.use_fp16))

    sites = sorted({d.dataset for d in gt})
    print(f"\nLOSO over {len(sites)} sites: {sites}")

    # ---- per-seed LOSO ----
    per_seed = []
    for name, probs in zip(names, prob_dicts):
        res = run_loso(probs, gt, sites, args.val_workers, args.tune_metric)
        per_seed.append((name, res))
        print(f"\n--- seed: {name} ---")
        print(f"  pooled LOSO B={res['pooled_B']:.4f} | mean-of-folds={res['mean_fold']:.4f}"
              f"+/-{res['std_fold']:.4f} | in-sample={res['insample_B']:.4f} | "
              f"gap={res['insample_B']-res['pooled_B']:+.4f}")
        if n == 1 or args.per_site_seeds:
            print_per_site(res, f"seed {name}")

    print(f"\n{'='*68}\nSUMMARY (official B, {args.segment_s:.0f}s, tune={args.tune_metric})\n{'='*68}")
    print(f"{'seed':32} {'pooledLOSO':>11} {'in-sample':>10} {'gap':>7}")
    print("-" * 62)
    for name, res in per_seed:
        print(f"{name[:32]:32} {res['pooled_B']:11.4f} {res['insample_B']:10.4f} "
              f"{res['insample_B']-res['pooled_B']:+7.4f}")

    if n > 1:
        pooled = np.array([r["pooled_B"] for _, r in per_seed])
        ins = np.array([r["insample_B"] for _, r in per_seed])
        print("-" * 62)
        print(f"  SINGLE-MODEL mean over {n} seeds (report this, not the best seed):")
        print(f"    pooled LOSO B = {pooled.mean():.4f} +/- {pooled.std():.4f}")
        print(f"    in-sample  B = {ins.mean():.4f} +/- {ins.std():.4f}")
        print(f"    best seed pooled LOSO = {pooled.max():.4f} (shown for reference only)")

        if not args.no_ensemble:
            print(f"\n  ENSEMBLE (probabilities averaged over {n} seeds -- a separate system):")
            ens = combine_probs(prob_dicts, args.weights)
            re = run_loso(ens, gt, sites, args.val_workers, args.tune_metric)
            print(f"    pooled LOSO B = {re['pooled_B']:.4f} | in-sample B = {re['insample_B']:.4f}"
                  f" | gap = {re['insample_B']-re['pooled_B']:+.4f}")
            print_per_site(re, f"ENSEMBLE of {n} seeds")


if __name__ == "__main__":
    main()
