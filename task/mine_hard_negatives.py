"""
Hard-Negative Mining for D-Class False Positives
=================================================

Run a trained Whale-VAD checkpoint on the labeled training data, identify
event-level false positives (model fires confidently on a region with no
GT call of the target class), and persist their locations to disk so a
fine-tuning script can resample them as explicit negatives.

Why bother
----------
Random negative segments are drawn uniformly from non-call regions of
the corpus. The probability of hitting a ship-noise transient or an
ice-crack — the patterns the model actually mistakes for D-calls — is
low. Hard-negative mining replaces this uniform draw with a targeted one:
every mined segment is *guaranteed* to contain a pattern the current
model gets wrong. Repeated exposure with explicit ``D=0`` targets gives
the model the negative gradient signal it never got from random
sampling alone.

Usage
-----
::

    python mine_hard_negatives.py \\
        --checkpoint runs/baseline_seed42/best_model.pt \\
        --target d --threshold 0.2 --top_k 1500 \\
        --output runs/hardnegs/d_top1500.json
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
from model import WhaleVAD
from postprocess import (
    Detection, compute_iou_1d, merge_and_filter,
    smooth_probabilities, stitch_segments, threshold_to_detections,
)
from spectrogram import SpectrogramExtractor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--target", type=str, default="d", choices=cfg.CALL_TYPES_3,
                   help="Class to mine FPs for. D has the worst FP rate.")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Frame-level threshold for proposal generation. "
                        "Lower than the tuned operating point on purpose.")
    p.add_argument("--max_iou_for_fp", type=float, default=0.1,
                   help="A proposal is a FP iff max IoU with any same-class "
                        "GT event is below this. Keeps near-misses out.")
    p.add_argument("--top_k", type=int, default=1500)
    p.add_argument("--datasets", type=str, nargs="+", default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, spec_extractor, loader, device):
    """Mirror eval_only.py / train.validate inference block exactly."""
    all_probs = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Mining inference"):
        audio = audio.to(device, non_blocking=True)
        logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]
    return all_probs


def build_target_class_thresholds(target_idx: int, threshold: float) -> np.ndarray:
    """Threshold array that's `threshold` for target class, 0.999 for others.
    Disabling other classes prevents cross-class detections we'd just filter."""
    thr = np.full(cfg.n_classes(), 0.999, dtype=np.float64)
    thr[target_idx] = threshold
    return thr


def build_gt_events_for_class(annotations, file_start_dts, target_class: str):
    """Same-class GT Detections for IoU matching."""
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
    """Filter to predictions whose max same-class IoU is below threshold."""
    gt_by_file: dict = {}
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

    datasets = args.datasets or cfg.TRAIN_DATASETS
    print(f"Mining on: {datasets}")

    manifest = get_file_manifest(datasets)
    annotations = load_annotations(datasets, manifest=manifest)
    print(f"  {len(manifest)} files, {len(annotations)} annotations")

    # Same fixed 30s tiles as validation — mining must operate on the
    # input distribution the training-time inference path produces.
    segments = build_val_segments(manifest, annotations)
    print(f"  {len(segments)} 30s tiles to score")

    loader = DataLoader(
        WhaleDataset(segments),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t0 = time.time()
    all_probs = run_inference(model, spec_extractor, loader, device)
    print(f"  inference done in {time.time() - t0:.0f}s")

    file_probs = stitch_segments(all_probs)
    thr = build_target_class_thresholds(target_idx, args.threshold)
    raw_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        raw_dets.extend(threshold_to_detections(probs, thr, ds, fn))
    pred_events = merge_and_filter(raw_dets)
    print(f"  {len(pred_events)} candidate '{args.target}' proposals "
          f"at threshold {args.threshold}")

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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
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