"""
Hard-Negative Mining for D-Class False Positives (multi-checkpoint)
====================================================================

Run one or more trained checkpoints on the labeled training data, average
their per-frame probabilities, identify event-level false positives of
the target class, and persist their locations to disk for fine-tuning.

Single-checkpoint mode mines from one model. Multi-checkpoint mode mines
from the ensemble's averaged probabilities — useful when you want
"consensus FPs" that all models in the ensemble fail on.

Usage
-----
::

    # single model (per-model mining; recommended for ensemble retraining)
    python mine_hard_negatives.py \\
        --checkpoints runs/whalevad_20260507_191223/best_model.pt \\
        --target d --threshold 0.2 --top_k 1500 \\
        --output runs/hardnegs/d_seed3.json

    # full ensemble (consensus FP signal)
    python mine_hard_negatives.py \\
        --checkpoints runs/whalevad_20260504_152450/best_model.pt \\
                      runs/phase5_20260506_204358/best_model.pt \\
                      runs/whalevad_20260507_191223/best_model.pt \\
                      runs/phase5_20260507_211504/best_model.pt \\
        --target d --threshold 0.2 --top_k 1500 \\
        --output runs/hardnegs/d_ensemble.json
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, collapse_probs_to_3class, compute_iou_1d, merge_and_filter,
    smooth_probabilities, stitch_segments, threshold_to_detections,
)
from spectrogram import SpectrogramExtractor

# Reuse the ensemble's model/inference plumbing — the file already
# handles baseline vs BPN auto-detection and BPN's dict return value.
from ensemble_predict import (
    average_prob_dicts, build_model_for_ckpt, predict_probabilities,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint paths. Multiple → "
                        "averaged probabilities (ensemble mining).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint weights (must match "
                        "len(checkpoints)). Default: equal.")
    p.add_argument("--target", type=str, default="d", choices=cfg.CALL_TYPES_3)
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Frame-level threshold for proposal generation. "
                        "Lower than tuned operating point on purpose.")
    p.add_argument("--max_iou_for_fp", type=float, default=0.1)
    p.add_argument("--top_k", type=int, default=1500)
    p.add_argument("--datasets", type=str, nargs="+", default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


def build_target_class_thresholds(target_idx: int, threshold: float) -> np.ndarray:
    """Threshold array: `threshold` for target class, 0.999 for others."""
    thr = np.full(cfg.n_classes(), 0.999, dtype=np.float64)
    thr[target_idx] = threshold
    return thr


def build_gt_events_for_class(annotations, file_start_dts, target_class: str):
    gt = []
    for _, row in annotations.iterrows():
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        if label != target_class:
            continue
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=target_class,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt


def predictions_to_fps(predictions, gt, target_class: str, max_iou_for_fp: float):
    gt_by_file = {}
    for g in gt:
        gt_by_file.setdefault((g.dataset, g.filename), []).append(g)
    fps = []
    for p in predictions:
        if p.label != target_class:
            continue
        candidates = gt_by_file.get((p.dataset, p.filename), [])
        max_iou = max(
            (compute_iou_1d(p.start_s, p.end_s, g.start_s, g.end_s)
             for g in candidates),
            default=0.0,
        )
        if max_iou < max_iou_for_fp:
            fps.append(p)
    return fps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_idx = cfg.CALL_TYPES_3.index(args.target)
    print(f"Target class: '{args.target}' (idx {target_idx})")
    print(f"Proposal threshold: {args.threshold}")
    print(f"Mining from {len(args.checkpoints)} checkpoint(s)")

    # ------------------------------------------------------------------
    # Weights handling — same convention as ensemble_predict.py
    # ------------------------------------------------------------------
    if args.weights is not None:
        assert len(args.weights) == len(args.checkpoints)
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
        print(f"Per-checkpoint weights (normalized): {weights}")
    else:
        weights = None

    # ------------------------------------------------------------------
    # Build inference loader — same fixed 30s tiles as validation
    # ------------------------------------------------------------------
    datasets = args.datasets or cfg.TRAIN_DATASETS
    print(f"Mining on: {datasets}")

    manifest = get_file_manifest(datasets)
    annotations = load_annotations(datasets, manifest=manifest)
    print(f"  {len(manifest)} files, {len(annotations)} annotations")

    segments = build_val_segments(manifest, annotations)
    print(f"  {len(segments)} 30s tiles to score")

    loader = DataLoader(
        WhaleDataset(segments),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    spec_extractor = SpectrogramExtractor().to(device)

    # ------------------------------------------------------------------
    # Run inference with each checkpoint, collecting prob dicts
    # ------------------------------------------------------------------
    all_prob_dicts = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_type = build_model_for_ckpt(ckpt, device)
        print(f"  type: {model_type}")

        # Materialize lazy projection layer before load_state_dict.
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            spec = spec_extractor(dummy)
            _ = model(spec)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        t0 = time.time()
        probs = predict_probabilities(
            model, model_type, spec_extractor, loader, device)
        # Collapse to 3-class if the model is 7-class (no-op for 3-class).
        probs = collapse_probs_to_3class(probs)
        all_prob_dicts.append(probs)
        print(f"  inference {time.time()-t0:.0f}s, {len(probs)} prob arrays")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Average probabilities (single checkpoint → identity)
    # ------------------------------------------------------------------
    if len(all_prob_dicts) == 1:
        all_probs = all_prob_dicts[0]
        print(f"\nSingle-checkpoint mode: skipping averaging.")
    else:
        all_probs = average_prob_dicts(all_prob_dicts, weights=weights)
        print(f"\nAveraged probs across {len(all_prob_dicts)} models, "
              f"{len(all_probs)} segments")

    # ------------------------------------------------------------------
    # Stitch + smooth + threshold + merge
    # ------------------------------------------------------------------
    file_probs = stitch_segments(all_probs)
    thr = build_target_class_thresholds(target_idx, args.threshold)
    raw_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        raw_dets.extend(threshold_to_detections(probs, thr, ds, fn))
    pred_events = merge_and_filter(raw_dets)
    print(f"  {len(pred_events)} '{args.target}' proposals "
          f"at threshold {args.threshold}")

    # ------------------------------------------------------------------
    # Match to GT, identify FPs, top-k by confidence
    # ------------------------------------------------------------------
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }
    gt = build_gt_events_for_class(annotations, file_start_dts, args.target)
    print(f"  {len(gt)} GT '{args.target}' events on training sites")

    fps = predictions_to_fps(pred_events, gt, args.target, args.max_iou_for_fp)
    print(f"  {len(fps)} hard FPs (max IoU < {args.max_iou_for_fp})")

    fps.sort(key=lambda d: d.confidence, reverse=True)
    if args.top_k > 0 and len(fps) > args.top_k:
        fps = fps[:args.top_k]
        print(f"  truncated to top {args.top_k} by confidence")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoints": [str(c) for c in args.checkpoints],
        "weights": weights,
        "target_class": args.target,
        "threshold": args.threshold,
        "max_iou_for_fp": args.max_iou_for_fp,
        "datasets": list(datasets),
        "n_fps": len(fps),
        "fps": [
            {"dataset": d.dataset, "filename": d.filename,
             "start_s": float(d.start_s), "end_s": float(d.end_s),
             "confidence": float(d.confidence),
             "target_class": args.target}
            for d in fps
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {len(fps)} hard negatives → {out_path}")
    if fps:
        confs = np.array([d.confidence for d in fps])
        print(f"  confidence range: {confs.min():.3f}–{confs.max():.3f} "
              f"(median {np.median(confs):.3f})")


if __name__ == "__main__":
    main()