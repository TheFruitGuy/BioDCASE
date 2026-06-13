"""
NAVE evaluation -- single or multiple checkpoints, 30s / 60s tiles, cached.
===========================================================================
One script for both single-checkpoint scoring and multi-seed probability
ensembling, with a switchable evaluation tile length and per-checkpoint
probability caching (the same pattern as ``ensemble_predict_cached``).

    # single checkpoint, default 30s tiles
    CUDA_VISIBLE_DEVICES=0 python nave_eval.py --checkpoints runs/nave_s42_*/nave_best.pt --val-workers 13

    # multi-seed ensemble at 60s tiles, per-member breakdown, fp16
    CUDA_VISIBLE_DEVICES=0 python nave_eval.py --per_model_eval --use_fp16 --segment-s 60 \
        --checkpoints runs/nave_s42_*/nave_best.pt runs/nave_s7_*/nave_best.pt \
                      runs/phase13r_3c_s2024_*/phase13r_best.pt --val-workers 13

Legacy ``train_phase13r`` checkpoints and native NAVE checkpoints can be mixed
freely (identical architecture, identical segment keys). The recipe tile length
is 30s; 60s is available for comparison (the conformer handles any T, so the
only change is how files are tiled).

Caching
-------
Per-checkpoint probabilities are cached under ``--cache_dir`` keyed by
``(run directory name, segment_s)``: 30s keeps the bare name, other lengths get
a ``_seg<N>`` suffix, so 30s and 60s never collide. Re-running an ensemble that
reuses a checkpoint skips its inference entirely. ``--no_cache`` recomputes.

Metrics
-------
Both conventions are printed: ``MACRO F1`` (mean of per-class F1s -- the official
BioDCASE Task 2 metric, and what thresholds are tuned on) and ``PAPER MACRO``
(F1 of mean precision / mean recall -- the Geldenhuys/WhaleVAD-BPN convention,
report-only). Thresholds are tuned per class on the official macro.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from nave_model import NAVE

from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn, get_file_manifest,
    load_annotations,
)
from postprocess_final import collapse_probs_to_3class
from eval_conformer import (
    tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more NAVE checkpoints (native or legacy).")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Evaluation tile length in seconds (e.g. 30 or 60). "
                        "Caches are keyed by this.")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S,
                   help="Overlap between consecutive tiles in seconds.")
    p.add_argument("--per_model_eval", action="store_true",
                   help="Also tune + report each checkpoint individually.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint ensemble weights (default: uniform).")
    p.add_argument("--cache_dir", type=str, default="runs/nave_prob_cache")
    p.add_argument("--no_cache", action="store_true", help="Disable prob caching.")
    p.add_argument("--use_fp16", action="store_true", help="fp16 autocast inference.")
    p.add_argument("--val-workers", type=int, default=min((os.cpu_count() or 1), 13),
                   help="Worker processes for the per-class threshold sweep.")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        p.error(f"--weights has {len(args.weights)} values, need {len(args.checkpoints)}.")
    return args


# ---------------------------------------------------------------- caching
def _seg_tag(segment_s: float) -> str:
    return "" if abs(segment_s - 30.0) < 1e-6 else f"_seg{int(segment_s)}"


def cache_path_for(ckpt_path: Path, cache_dir: Path, segment_s: float) -> Path:
    return cache_dir / f"{ckpt_path.parent.name}{_seg_tag(segment_s)}_nave_probs.pkl"


def load_cached_probs(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {k: v.astype(np.float32) for k, v in d.items()}


def save_probs_to_cache(probs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: v.astype(np.float16) for k, v in probs.items()}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------- model
def build_nave_from_ckpt(path, device):
    """Auto-detect legacy vs native NAVE checkpoint -> eval-mode model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_inner.") for k in sd):
        model, _ = NAVE.from_legacy_checkpoint(path, map_location=device)
    else:
        model = NAVE()
        model.load_checkpoint(path, map_location=device)
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model, spec, loader, device, use_fp16):
    """Per-segment sigmoid probabilities keyed by (dataset, filename, start_sample)."""
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_fp16 and device.type == "cuda") else nullcontext())
    hop = spec.hop_length
    raw = {}
    for audio, _t, _m, metas in loader:
        audio = audio.to(device)
        with autocast:
            logits = model(spec(audio))
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            raw[key] = probs[j, :n_frames, :]
    return collapse_probs_to_3class(raw)


def get_or_compute_probs(ckpt_path, spec, loader_factory, device, cache_dir,
                         segment_s, no_cache, use_fp16):
    cp = cache_path_for(Path(ckpt_path), cache_dir, segment_s)
    if not no_cache and cp.exists():
        print(f"  cache hit: {cp.name}")
        return load_cached_probs(cp)
    print("  cache miss, running inference...")
    t0 = time.time()
    model = build_nave_from_ckpt(ckpt_path, device)
    probs = predict_probs(model, spec, loader_factory(), device, use_fp16)
    print(f"    {len(probs)} segment arrays in {time.time()-t0:.0f}s")
    if not no_cache:
        save_probs_to_cache(probs, cp)
        print(f"    cached -> {cp.name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


# ---------------------------------------------------------------- combine
def combine_probs(prob_dicts, weights=None):
    """Weighted average over the common segment keys (frame-truncated to min)."""
    n = len(prob_dicts)
    w = np.ones(n) / n if weights is None else np.asarray(weights, float) / np.sum(weights)
    keys = set(prob_dicts[0])
    for d in prob_dicts[1:]:
        keys &= set(d)
    out = {}
    for k in keys:
        t = min(d[k].shape[0] for d in prob_dicts)
        out[k] = sum(w[i] * prob_dicts[i][k][:t] for i in range(n))
    return out


def report(metrics, thr, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for c in CLASS_NAMES:
        m = metrics.get(c, {})
        print(f"  {c.upper():6} P={m.get('precision',0):.3f} "
              f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")
    print(f"  MACRO F1     = {macro_f1(metrics):.4f}   (official, mean of per-class F1)")
    print(f"  PAPER MACRO  = {macro_f1_paper(metrics):.4f}   (F1 of mean-P/mean-R)")
    print(f"  thresholds   = {[round(float(t),3) for t in thr]}")


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    import config_final as _cf
    _cf.USE_3CLASS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.checkpoints)
    print(f"[NAVE eval] {n} checkpoint(s) | {args.segment_s:.0f}s tiles "
          f"({args.overlap_s:.1f}s overlap) | fp16={args.use_fp16} | "
          f"tune workers={args.val_workers} | cache={'off' if args.no_cache else args.cache_dir}")

    # ground truth (tuples), built once
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
        gt.append((row["dataset"], row["filename"], row["label_3class"],
                   (row["start_datetime"] - fsd).total_seconds(),
                   (row["end_datetime"] - fsd).total_seconds()))
    print(f"  {len(gt)} ground-truth events")

    spec = NAVEFeatureExtractor().to(device)

    # lazy val loader at the requested tile length (built only on a cache miss)
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
    prob_dicts, member_macros = [], []
    for i, ckpt in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{n}] {ckpt}")
        probs = get_or_compute_probs(ckpt, spec, loader_factory, device, cache_dir,
                                     args.segment_s, args.no_cache, args.use_fp16)
        prob_dicts.append(probs)
        if args.per_model_eval or n == 1:
            thr = tune_thresholds_per_class(probs, gt, workers=args.val_workers)
            metrics = evaluate_with_thresholds(probs, gt, thr)
            member_macros.append(macro_f1(metrics))
            report(metrics, thr, label=f"MEMBER {Path(ckpt).parent.name}")

    if n == 1:
        return

    ens = combine_probs(prob_dicts, args.weights)
    ens_thr = tune_thresholds_per_class(ens, gt, workers=args.val_workers)
    metrics = evaluate_with_thresholds(ens, gt, ens_thr)
    report(metrics, ens_thr, label=f"ENSEMBLE ({n} members)")
    if member_macros:
        best = max(member_macros)
        print(f"  lift over best member ({best:.4f}): {macro_f1(metrics)-best:+.4f}")


if __name__ == "__main__":
    main()
