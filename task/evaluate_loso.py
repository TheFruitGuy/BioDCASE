"""
evaluate_loso.py -- leave-one-site-out generalization for the conformer line.
=============================================================================
Threshold-LOSO: the model is trained once; for each held-out dev site we tune the
per-class thresholds on the OTHER sites and score the held-out site with them.
This measures how well the operating point generalizes across sites (the honest
number for the baselines comparison, which is LOSO) without retraining.

Reports the official BioDCASE metric B = F1(mean-P, mean-R) three ways:
  - per fold (held-out site)
  - mean-of-folds B (+/- std)        -- average generalization across sites
  - pooled B                         -- sum tp/fp/fn over held-out folds, one F1
                                        (each event scored under thresholds NOT
                                         tuned on its own site)
plus the in-sample B (thresholds tuned on all sites) and the gap = in-sample - LOSO.

    # single checkpoint (legacy phase13 or native NAVE both work), 60 s
    CUDA_VISIBLE_DEVICES=0 python evaluate_loso.py --segment-s 60 --use_fp16 --val-workers 20 \
        --checkpoints runs/phase13r_3c_s666_20260612_132012/phase13r_best.pt

    # ensemble (probabilities averaged first, then LOSO on the ensemble)
    CUDA_VISIBLE_DEVICES=0 python evaluate_loso.py --segment-s 60 --use_fp16 \
        --checkpoints runs/nave_s*/nave_best.pt --val-workers 20

Probabilities are cached per (run, segment_s), shared with nave_eval /
evaluate_phase13, so anything scored at 60 s already is reused.
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
                   help="Per-checkpoint ensemble weights (default uniform).")
    p.add_argument("--tune-metric", choices=["f1", "official"], default="f1")
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
    """Native NAVE checkpoints -> NAVE + NAVEFeatureExtractor; any legacy phase
    variant -> load_conformer (plain / FDY / PCEN / CRNN / LEAF / ...)."""
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
    prob_dicts = []
    for i, ckpt in enumerate(args.checkpoints):
        print(f"  [{i+1}/{n}] {Path(ckpt).parent.name}")
        _cf.USE_3CLASS = True
        prob_dicts.append(get_or_compute(ckpt, loader_factory, device, cache_dir,
                                         args.segment_s, args.no_cache, args.use_fp16))
    probs = combine_probs(prob_dicts, args.weights)

    sites = sorted({d.dataset for d in gt})
    print(f"\nLOSO over {len(sites)} sites: {sites}")

    # ---- per-fold: tune on others, score held-out ----
    folds = []
    pooled = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    for held in sites:
        p_tr, g_tr = _filter(probs, gt, [s for s in sites if s != held])
        thr = _tune(p_tr, g_tr, args.val_workers, args.tune_metric)
        p_h, g_h = _filter(probs, gt, [held])
        m = evaluate_with_thresholds(p_h, g_h, thr)
        B = macro_f1_paper(m)
        folds.append((held, B, m, thr))
        for c in CLASS_NAMES:
            for k in ("tp", "fp", "fn"):
                pooled[c][k] += int(m.get(c, {}).get(k, 0))
        print(f"\n  held-out {held}:  B={B:.4f}  thr={[round(float(t),2) for t in thr]}")
        for c in CLASS_NAMES:
            mc = m.get(c, {})
            print(f"    {c.upper():6} P={mc.get('precision',0):.3f} "
                  f"R={mc.get('recall',0):.3f} F1={mc.get('f1',0):.3f} "
                  f"(tp={mc.get('tp',0)} fp={mc.get('fp',0)} fn={mc.get('fn',0)})")

    # ---- aggregates ----
    fold_Bs = np.array([B for _, B, _, _ in folds])
    mean_fold_B = float(fold_Bs.mean())
    std_fold_B = float(fold_Bs.std())

    pooledP, pooledR = [], []
    for c in CLASS_NAMES:
        tp, fp, fn = pooled[c]["tp"], pooled[c]["fp"], pooled[c]["fn"]
        pooledP.append(tp / (tp + fp + 1e-8))
        pooledR.append(tp / (tp + fn + 1e-8))
    mP, mR = float(np.mean(pooledP)), float(np.mean(pooledR))
    pooled_B = 2 * mP * mR / (mP + mR + 1e-8)

    # ---- in-sample (tune on all, score all) ----
    thr_all = _tune(probs, gt, args.val_workers, args.tune_metric)
    m_all = evaluate_with_thresholds(probs, gt, thr_all)
    insample_B = macro_f1_paper(m_all)

    print(f"\n{'='*64}\nLOSO SUMMARY (official B, {args.segment_s:.0f}s, tune={args.tune_metric})\n{'='*64}")
    for held, B, _, _ in folds:
        print(f"  {held:28} B={B:.4f}")
    print("-" * 40)
    print(f"  mean-of-folds B = {mean_fold_B:.4f} +/- {std_fold_B:.4f}")
    print(f"  pooled       B = {pooled_B:.4f}   (meanP={mP:.3f}, meanR={mR:.3f})")
    print(f"  in-sample    B = {insample_B:.4f}   (thresholds tuned on all sites)")
    print(f"  generalization gap (in-sample - pooled LOSO) = {insample_B - pooled_B:+.4f}")
    print("\n  Report the pooled LOSO B as the cross-site generalization number;")
    print("  per-fold values are noisy when a site is thin in a class.")


if __name__ == "__main__":
    main()
