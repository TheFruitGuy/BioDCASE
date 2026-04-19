"""
Better threshold tuning for Whale-VAD.

Finer grid for rare classes, reports per-dataset F1 for diagnostics.

Usage:
    python tune_thresholds.py --checkpoint runs/whalevad_XXXX/best_model.pt
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import build_dataloaders, load_annotations, get_file_manifest
from postprocess import postprocess_predictions, compute_metrics, Detection


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to save tuned thresholds checkpoint")
    return p.parse_args()


@torch.no_grad()
def collect_probs(model, spec_extractor, loader, device):
    """Run validation once, collect all probabilities."""
    model.eval()
    all_probs = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Collecting probs"):
        audio = audio.to(device)
        logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]
    return all_probs


def build_gt_events(val_annotations, file_start_dts):
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"]   - fsd).total_seconds(),
        ))
    return gt_events


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Init lazy layer
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build val data
    _, _, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    # Collect probabilities once (slow) → reuse for all threshold evaluations
    print("Computing predictions on validation set...")
    all_probs = collect_probs(model, spec_extractor, val_loader, device)
    gt_events = build_gt_events(val_annotations, file_start_dts)

    # Finer grid for rare classes
    candidates_per_class = [
        np.concatenate([np.arange(0.2, 0.9, 0.05)]),                    # bmabz
        np.concatenate([np.arange(0.05, 0.5, 0.02), np.arange(0.5, 0.9, 0.05)]),  # d
        np.concatenate([np.arange(0.05, 0.5, 0.02), np.arange(0.5, 0.9, 0.05)]),  # bp
    ]

    class_names = cfg.class_names()
    best_thresholds = np.array([0.5, 0.5, 0.5])

    print("\n=== Iterative threshold tuning ===")
    # Iterate 3 passes so thresholds settle together
    for iteration in range(3):
        print(f"\nPass {iteration + 1}/3")
        for c, cands in enumerate(candidates_per_class):
            best_f1 = -1.0
            best_t = best_thresholds[c]
            for t in cands:
                trial = best_thresholds.copy()
                trial[c] = t
                preds = postprocess_predictions(all_probs, trial)
                m = compute_metrics(preds, gt_events, iou_threshold=0.3)
                f1 = m.get(class_names[c], {}).get("f1", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds[c] = best_t
            print(f"  {class_names[c]:6} threshold={best_t:.3f}  F1={best_f1:.3f}")

    print("\n=== Final tuned thresholds ===")
    for c, name in enumerate(class_names):
        print(f"  {name}: {best_thresholds[c]:.3f}")

    # Final overall eval
    print("\n=== Final evaluation with tuned thresholds ===")
    preds = postprocess_predictions(all_probs, best_thresholds)
    metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)

    for cls in class_names:
        if cls in metrics:
            m = metrics[cls]
            print(f"  {cls:6} TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"  OVERALL: F1={metrics['overall']['f1']:.3f}")

    # Per-dataset
    print("\n=== Per-dataset breakdown ===")
    for ds_name in cfg.VAL_DATASETS:
        ds_preds = [d for d in preds    if d.dataset == ds_name]
        ds_gts   = [d for d in gt_events if d.dataset == ds_name]
        m = compute_metrics(ds_preds, ds_gts, iou_threshold=0.3)
        print(f"\n  {ds_name}:")
        for cls in class_names:
            if cls in m:
                r = m[cls]
                print(f"    {cls:6} TP={r['tp']:5} FP={r['fp']:6} FN={r['fn']:6}  "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
        print(f"    OVERALL F1={m['overall']['f1']:.3f}")

    # Save if requested
    if args.output:
        torch.save({
            "model_state_dict": ckpt["model_state_dict"],
            "thresholds": torch.tensor(best_thresholds, dtype=torch.float32),
        }, args.output)
        print(f"\nSaved tuned checkpoint: {args.output}")


if __name__ == "__main__":
    main()
