"""
Diagnose Phase 0 IoU-Matching Failure
=====================================

Phase 0 produced FPs but zero TPs across 10 epochs. That's not "model
didn't learn" — that means predictions and GT events are in disjoint
spaces (different labels, different filenames, different time ranges).

This script loads a Phase 0 checkpoint, runs inference on ONE validation
file, dumps:

  1. The prediction Detection objects (label, dataset, filename, time range)
  2. The GT Detection objects for that same file
  3. Whether their (label, dataset, filename) tuples actually align

If the tuples don't match, we've found the bug. If they DO match but
times are wildly off, that's a separate (smaller) issue.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python diagnose_phase0.py \\
        --checkpoint runs/phase0_*/phase0_epoch_10.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess import postprocess_predictions, Detection, compute_iou_1d
from train_phase0 import (
    SingleClassDataset, build_phase0_model, TARGET_CLASS_NAME,
    TARGET_CLASS_IDX, PHASE0_VAL_SITES, PHASE0_THRESHOLD,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--n_files", type=int, default=3,
                        help="Number of files to inspect (default 3).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------
    # Load model (Phase 0 small variant)
    # --------------------------------------------------------------
    model, spec_extractor = build_phase0_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"reported F1 {ckpt['f1']:.3f})")

    # --------------------------------------------------------------
    # Build val data
    # --------------------------------------------------------------
    val_man = get_file_manifest(PHASE0_VAL_SITES)
    val_ann = load_annotations(PHASE0_VAL_SITES, manifest=val_man)
    val_segs = build_val_segments(val_man, val_ann)
    val_ds = SingleClassDataset(val_segs, TARGET_CLASS_IDX)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_man.iterrows()
    }

    # --------------------------------------------------------------
    # Run inference, collect probabilities
    # --------------------------------------------------------------
    print("\nRunning inference...")
    all_probs = {}
    hop = spec_extractor.hop_length
    with torch.no_grad():
        for audio, _, _, metas in val_loader:
            audio = audio.to(device)
            logits = model(spec_extractor(audio))
            probs = torch.sigmoid(logits).cpu().numpy()
            for j, meta in enumerate(metas):
                key = (meta["dataset"], meta["filename"], meta["start_sample"])
                n_samp = meta["end_sample"] - meta["start_sample"]
                n_frames = min(n_samp // hop, probs[j].shape[0])
                all_probs[key] = probs[j, :n_frames, :]

    # --------------------------------------------------------------
    # Generate prediction Detections
    # --------------------------------------------------------------
    pred_events = postprocess_predictions(all_probs,
                                           np.array([PHASE0_THRESHOLD]))
    print(f"Total predictions: {len(pred_events)}")

    # --------------------------------------------------------------
    # Build GT Detections
    # --------------------------------------------------------------
    gt_events = []
    for _, row in val_ann.iterrows():
        if row["label_3class"] != TARGET_CLASS_NAME:
            continue
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=TARGET_CLASS_NAME,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"Total GT events: {len(gt_events)}")

    # --------------------------------------------------------------
    # Compare label / dataset / filename tuples
    # --------------------------------------------------------------
    pred_keys = {(d.label, d.dataset, d.filename) for d in pred_events}
    gt_keys = {(d.label, d.dataset, d.filename) for d in gt_events}
    print(f"\nUnique (label, dataset, filename) tuples:")
    print(f"  Predictions: {len(pred_keys)}")
    print(f"  GT:          {len(gt_keys)}")
    print(f"  Intersection: {len(pred_keys & gt_keys)}")

    if not (pred_keys & gt_keys):
        print("\n*** ZERO OVERLAP — predictions and GT live in different "
              "(label, dataset, filename) spaces! ***")
        print(f"\nFirst 3 prediction keys:")
        for k in list(pred_keys)[:3]:
            print(f"  {k!r}")
        print(f"\nFirst 3 GT keys:")
        for k in list(gt_keys)[:3]:
            print(f"  {k!r}")
        return

    # --------------------------------------------------------------
    # Inspect per-file details for a few files where both sides exist
    # --------------------------------------------------------------
    common_files = sorted(
        {(d.dataset, d.filename) for d in pred_events}
        & {(d.dataset, d.filename) for d in gt_events}
    )[:args.n_files]

    print(f"\nDetailed comparison on {len(common_files)} common files:")

    for ds, fn in common_files:
        print(f"\n{'-' * 70}")
        print(f"File: {ds}/{fn}")
        file_preds = sorted([d for d in pred_events
                             if d.dataset == ds and d.filename == fn],
                            key=lambda x: x.start_s)
        file_gts = sorted([d for d in gt_events
                           if d.dataset == ds and d.filename == fn],
                          key=lambda x: x.start_s)
        print(f"  Predictions ({len(file_preds)}):")
        for p in file_preds[:5]:
            print(f"    {p.label}  {p.start_s:8.2f} → {p.end_s:8.2f}s "
                  f"(conf={p.confidence:.3f})")
        if len(file_preds) > 5:
            print(f"    ... ({len(file_preds) - 5} more)")
        print(f"  GT events ({len(file_gts)}):")
        for g in file_gts[:5]:
            print(f"    {g.label}  {g.start_s:8.2f} → {g.end_s:8.2f}s")
        if len(file_gts) > 5:
            print(f"    ... ({len(file_gts) - 5} more)")

        # IoU sanity check on the first GT event
        if file_gts and file_preds:
            g0 = file_gts[0]
            best = max(((p, compute_iou_1d(p.start_s, p.end_s,
                                           g0.start_s, g0.end_s))
                       for p in file_preds),
                      key=lambda x: x[1])
            print(f"  Best IoU for first GT [{g0.start_s:.1f}-{g0.end_s:.1f}]: "
                  f"{best[1]:.3f} "
                  f"(matched pred [{best[0].start_s:.1f}-{best[0].end_s:.1f}])")


if __name__ == "__main__":
    main()
