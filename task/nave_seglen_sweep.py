"""
NAVE evaluation tile-length sweep.
==================================
Evaluates one or more NAVE checkpoints across a range of tile lengths and
prints a macro-F1-vs-length table, so you can find the best evaluation segment
length (the model is trained on 30s segments, but RoPE attention generalises to
longer sequences and longer tiles reduce boundary splits during stitching).

    # one checkpoint, default sweep
    CUDA_VISIBLE_DEVICES=0 python nave_seglen_sweep.py --use_fp16 --val-workers 20 \
        --checkpoints runs/phase13r_3c_s666_20260612_132012/phase13r_best.pt

    # custom lengths + an ensemble
    CUDA_VISIBLE_DEVICES=0 python nave_seglen_sweep.py --use_fp16 --val-workers 20 \
        --segment-lengths 30 45 60 90 120 150 \
        --checkpoints runs/nave_s7_*/nave_best.pt runs/phase13r_3c_s666_*/phase13r_best.pt

Probabilities are cached per (checkpoint, length) under --cache_dir (reusing
nave_eval's cache), so a given length is inferred once; re-running the sweep, or
later evaluating one length with nave_eval, hits the cache. Thresholds are tuned
per class at every length on the official macro (mean of per-class F1).

Ground truth is event-based and identical across lengths, so it is built once.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from postprocess_final import Detection
from eval_conformer import (
    tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)
# reuse nave_eval's verified prob collection + caching + combination
from nave_eval import get_or_compute_probs, combine_probs

DEFAULT_LENGTHS = [20.0, 30.0, 45.0, 60.0, 90.0, 120.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more NAVE checkpoints (native or legacy).")
    p.add_argument("--segment-lengths", nargs="+", type=float, default=DEFAULT_LENGTHS,
                   help=f"Tile lengths in seconds to sweep (default {DEFAULT_LENGTHS}).")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S,
                   help="Tile overlap in seconds (fixed across lengths).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint ensemble weights (default: uniform).")
    p.add_argument("--cache_dir", type=str, default="runs/nave_prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--val-workers", type=int, default=min((os.cpu_count() or 1), 13))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        p.error(f"--weights has {len(args.weights)} values, need {len(args.checkpoints)}.")
    return args


def main():
    args = parse_args()
    import config_final as _cf
    _cf.USE_3CLASS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.checkpoints)
    lengths = sorted(set(args.segment_lengths))
    print(f"[NAVE seglen sweep] {n} checkpoint(s) | lengths {lengths} | "
          f"overlap {args.overlap_s:.1f}s | fp16={args.use_fp16} | workers={args.val_workers}")

    # ---- ground truth (event-based, length-independent): build once ----
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
    print(f"  {len(gt)} ground-truth events\n")

    spec = NAVEFeatureExtractor().to(device)
    cache_dir = Path(args.cache_dir)
    rows = []

    for seg in lengths:
        print(f"{'='*60}\nTILE LENGTH {seg:.0f}s\n{'='*60}")
        _loader = {"obj": None, "n": 0}

        def loader_factory(_seg=seg):
            if _loader["obj"] is None:
                segs = build_val_segments(val_manifest, val_anns,
                                          segment_s=_seg, overlap_s=args.overlap_s)
                _loader["obj"] = DataLoader(
                    WhaleDataset(segs), batch_size=args.batch_size, shuffle=False,
                    num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
                _loader["n"] = len(segs)
                print(f"  built loader: {len(segs)} tiles")
            return _loader["obj"]

        prob_dicts = []
        for i, ckpt in enumerate(args.checkpoints):
            print(f"  [{i+1}/{n}] {Path(ckpt).parent.name}")
            prob_dicts.append(get_or_compute_probs(
                ckpt, spec, loader_factory, device, cache_dir,
                segment_s=seg, no_cache=args.no_cache, use_fp16=args.use_fp16))

        probs = combine_probs(prob_dicts, args.weights)
        thr = tune_thresholds_per_class(probs, gt, workers=args.val_workers)
        metrics = evaluate_with_thresholds(probs, gt, thr)
        macro, paper = macro_f1(metrics), macro_f1_paper(metrics)
        per = {c: metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES}
        rows.append((seg, macro, paper, per, thr, _loader["n"]))
        print(f"  -> MACRO {macro:.4f} | PAPER {paper:.4f} | "
              f"{' '.join(f'{c.upper()}={per[c]:.3f}' for c in CLASS_NAMES)}\n")

    # ---- summary table ----
    best = max(rows, key=lambda r: r[1])
    print(f"\n{'='*72}\nSWEEP SUMMARY ({n} ckpt, "
          f"{'uniform' if args.weights is None else 'weighted'} ensemble)\n{'='*72}")
    hdr = f"{'len(s)':>7} {'MACRO':>7} {'PAPER':>7} " \
          + " ".join(f"{c.upper():>6}" for c in CLASS_NAMES) + f" {'tiles':>7}  thresholds"
    print(hdr)
    print("-" * len(hdr))
    for seg, macro, paper, per, thr, ntiles in rows:
        star = "  <-- best" if (seg, macro) == (best[0], best[1]) else ""
        print(f"{seg:7.0f} {macro:7.4f} {paper:7.4f} "
              + " ".join(f"{per[c]:6.3f}" for c in CLASS_NAMES)
              + f" {ntiles:7d}  {[round(float(t),2) for t in thr]}{star}")
    print(f"\nBest tile length: {best[0]:.0f}s -> MACRO F1 {best[1]:.4f}")


if __name__ == "__main__":
    main()
