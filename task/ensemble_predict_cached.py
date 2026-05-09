"""
Cached ensemble predict
=======================

Wraps ensemble_predict.py's pipeline with per-checkpoint probability
caching. First eval of a checkpoint runs full inference (~3 min) and
caches probs to disk. Subsequent ensemble evals that include the same
checkpoint skip inference entirely (~5 s for the whole pipeline).

Useful when the same checkpoints appear in multiple ensemble combinations
(A, B, C, D, leave-one-out ablations, etc.).

Optional FP16 autocast for ~1.5–2× speedup on the inference pass when
GPUs are free.
"""

from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import Detection, collapse_probs_to_3class
from spectrogram import SpectrogramExtractor

from ensemble_predict import (
    average_prob_dicts, build_model_for_ckpt, predict_probabilities,
    tune_thresholds_on_probs, evaluate_with_thresholds,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--per_model_eval", action="store_true")
    p.add_argument("--cache_dir", type=str, default="runs/prob_cache",
                   help="Directory for per-checkpoint cached probs.")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable caching (always recompute).")
    p.add_argument("--use_fp16", action="store_true",
                   help="FP16 autocast during inference. ~1.5–2× speedup, "
                        "negligible accuracy hit (<0.001 F1 typically).")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


def cache_path_for(ckpt_path: Path, cache_dir: Path) -> Path:
    """One pickle per source checkpoint, keyed by parent dir name."""
    return cache_dir / f"{ckpt_path.parent.name}_probs.pkl"


def load_cached_probs(path: Path):
    """Restore float32 from float16-on-disk cache."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {k: v.astype(np.float32) for k, v in d.items()}


def save_probs_to_cache(probs, path: Path):
    """Cast to float16 for ~50% disk savings (lossless at eval precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: v.astype(np.float16) for k, v in probs.items()}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


@torch.no_grad()
def get_or_compute_probs(ckpt_path, spec_extractor, val_loader, device,
                         cache_dir, no_cache=False, use_fp16=False):
    """Cache hit → load pickle. Cache miss → run inference, save, return."""
    cp = cache_path_for(Path(ckpt_path), cache_dir)
    if not no_cache and cp.exists():
        print(f"  cache hit: {cp.name}")
        return load_cached_probs(cp)

    print(f"  cache miss, running inference...")
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"    type: {model_type}")

    # Materialize lazy projection layer before load_state_dict.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            probs = predict_probabilities(
                model, model_type, spec_extractor, val_loader, device)
    else:
        probs = predict_probabilities(
            model, model_type, spec_extractor, val_loader, device)
    probs = collapse_probs_to_3class(probs)

    print(f"    computed in {time.time()-t0:.0f}s, "
          f"{len(probs)} segment prob arrays")
    if not no_cache:
        save_probs_to_cache(probs, cp)
        print(f"    cached → {cp}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def per_class_print(metrics, thresholds, label="ENSEMBLE"):
    """Pretty-print the per-class table matching ensemble_predict's format."""
    print(f"\n  {label} RESULT:")
    for c in cfg.CALL_TYPES_3:
        m = metrics.get(c, {})
        idx = cfg.CALL_TYPES_3.index(c)
        print(f"    {c.upper():6} t={thresholds[idx]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):5} "
              f"FN={m.get('fn', 0):5}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    overall = metrics.get("overall", {})
    macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                           for c in cfg.CALL_TYPES_3]))
    print(f"    OVERALL F1={overall.get('f1', 0):.3f}  "
          f"MACRO F1={macro:.3f}")
    print(f"    Tuned thresholds: "
          f"{['%.2f' % t for t in thresholds]}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir} (no_cache={args.no_cache}, "
          f"use_fp16={args.use_fp16})")

    # Validation loader is shared across all checkpoints.
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  {len(val_segs)} 30s tiles")

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    gt_events = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"  {len(gt_events)} ground-truth events")

    spec_extractor = SpectrogramExtractor().to(device)

    # Get probs per checkpoint (cache hit or compute).
    all_prob_dicts = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        probs = get_or_compute_probs(
            ckpt_path, spec_extractor, val_loader, device,
            cache_dir, no_cache=args.no_cache, use_fp16=args.use_fp16)
        all_prob_dicts.append(probs)

        if args.per_model_eval:
            thr = tune_thresholds_on_probs(probs, gt_events)
            metrics = evaluate_with_thresholds(probs, gt_events, thr)
            macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                                   for c in cfg.CALL_TYPES_3]))
            print(f"  individual: overall F1="
                  f"{metrics.get('overall', {}).get('f1', 0):.3f}, "
                  f"macro={macro:.3f}, thr={['%.2f' % t for t in thr]}")
            for c in cfg.CALL_TYPES_3:
                m = metrics.get(c, {})
                print(f"    {c.upper():6} F1={m.get('f1', 0):.3f} "
                      f"P={m.get('precision', 0):.3f} "
                      f"R={m.get('recall', 0):.3f}")

    # Average + threshold tune + final metrics.
    print(f"\n{'='*64}\nENSEMBLE\n{'='*64}")
    weights = None
    if args.weights:
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
        print(f"Per-model weights (normalized): {weights}")

    ens_probs = (all_prob_dicts[0] if len(all_prob_dicts) == 1
                 else average_prob_dicts(all_prob_dicts, weights=weights))

    print("Tuning thresholds on ensemble probabilities...")
    ens_thr = tune_thresholds_on_probs(ens_probs, gt_events)
    metrics = evaluate_with_thresholds(ens_probs, gt_events, ens_thr)
    per_class_print(metrics, ens_thr, label="ENSEMBLE")


if __name__ == "__main__":
    main()