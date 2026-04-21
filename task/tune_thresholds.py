"""
Per-Class Threshold Tuning
==========================

Iteratively searches for the per-class confidence thresholds that
maximize event-level F1 on the validation set. Compared to the simple
one-pass tuner baked into ``postprocess.py``, this script offers:

    - **Finer grid for rare classes** (``d`` and ``bp``) where the
      F1-optimal threshold is typically well below 0.5 due to the low
      base rate.
    - **Iterative refinement** (3 passes): each class's threshold is
      re-optimized with the *current* best thresholds of the other
      classes held fixed. Because the merge-and-filter step operates
      independently per class in our pipeline the thresholds are in
      principle independent, but in practice a few passes ensure the
      grid has stabilized.
    - **Per-dataset breakdown** at the end so the tuned model can be
      compared against Table 4 of the paper site-by-site.

Usage
-----
::

    python tune_thresholds.py --checkpoint runs/whalevad_XXXX/best_model.pt

Optionally save the tuned thresholds back into a new checkpoint::

    python tune_thresholds.py --checkpoint runs/whalevad_XXXX/best_model.pt \\
                              --output runs/whalevad_XXXX/tuned_model.pt
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
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint to tune.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to save a new checkpoint bundling "
                        "the original weights with the tuned thresholds. "
                        "If omitted, the tuned thresholds are only printed.")
    return p.parse_args()


# ======================================================================
# Inference pass: cache per-window probabilities
# ======================================================================

@torch.no_grad()
def collect_probs(model, spec_extractor, loader, device):
    """
    Run inference once and return raw per-window probabilities.

    The full post-processing pipeline is threshold-independent up to the
    thresholding step; by caching probabilities here, every threshold
    trial in the search loop becomes cheap (no repeated GPU inference).

    Parameters
    ----------
    model, spec_extractor, loader, device
        Standard inference objects.

    Returns
    -------
    dict
        Maps ``(dataset, filename, start_sample)`` → ``(n_frames, n_classes)``
        probability array.
    """
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
    """
    Construct a list of ground-truth Detection objects from the validation
    annotations DataFrame.

    Parameters
    ----------
    val_annotations : pd.DataFrame
    file_start_dts : dict
        Maps ``(dataset, filename)`` to that file's start datetime.

    Returns
    -------
    list of Detection
    """
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
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt_events


# ======================================================================
# Main
# ======================================================================

def main():
    """Run iterative per-class threshold search and report results."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Initialize the lazy projection layer before load_state_dict.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ------------------------------------------------------------------
    # Validation data
    # ------------------------------------------------------------------
    _, _, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Run inference once, cache for all threshold trials
    # ------------------------------------------------------------------
    print("Computing predictions on validation set...")
    all_probs = collect_probs(model, spec_extractor, val_loader, device)
    gt_events = build_gt_events(val_annotations, file_start_dts)

    # ------------------------------------------------------------------
    # Per-class candidate grids
    # ------------------------------------------------------------------
    # Coarse grid for bmabz (common, well-separated scores) and finer
    # grid at the low end for d and bp (rare, score distribution shifted
    # toward zero).
    candidates_per_class = [
        np.arange(0.2, 0.9, 0.05),                                          # bmabz
        np.concatenate([np.arange(0.05, 0.5, 0.02), np.arange(0.5, 0.9, 0.05)]),  # d
        np.concatenate([np.arange(0.05, 0.5, 0.02), np.arange(0.5, 0.9, 0.05)]),  # bp
    ]

    class_names = cfg.class_names()
    # Start from neutral thresholds; any of these could be a local optimum.
    best_thresholds = np.array([0.5, 0.5, 0.5])

    # ------------------------------------------------------------------
    # Iterative search
    # ------------------------------------------------------------------
    # Three passes is more than enough in practice: thresholds usually
    # settle after the first pass and the second only confirms.
    print("\n=== Iterative threshold tuning ===")
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

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== Final tuned thresholds ===")
    for c, name in enumerate(class_names):
        print(f"  {name}: {best_thresholds[c]:.3f}")

    # Overall metrics with tuned thresholds.
    print("\n=== Final evaluation with tuned thresholds ===")
    preds = postprocess_predictions(all_probs, best_thresholds)
    metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)

    for cls in class_names:
        if cls in metrics:
            m = metrics[cls]
            print(f"  {cls:6} TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"  OVERALL: F1={metrics['overall']['f1']:.3f}")

    # Per-dataset breakdown for Table-4-style comparison.
    print("\n=== Per-dataset breakdown ===")
    for ds_name in cfg.VAL_DATASETS:
        ds_preds = [d for d in preds if d.dataset == ds_name]
        ds_gts = [d for d in gt_events if d.dataset == ds_name]
        m = compute_metrics(ds_preds, ds_gts, iou_threshold=0.3)
        print(f"\n  {ds_name}:")
        for cls in class_names:
            if cls in m:
                r = m[cls]
                print(f"    {cls:6} TP={r['tp']:5} FP={r['fp']:6} FN={r['fn']:6}  "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
        print(f"    OVERALL F1={m['overall']['f1']:.3f}")

    # Optional: save the tuned thresholds alongside the original weights.
    if args.output:
        torch.save({
            "model_state_dict": ckpt["model_state_dict"],
            "thresholds": torch.tensor(best_thresholds, dtype=torch.float32),
        }, args.output)
        print(f"\nSaved tuned checkpoint: {args.output}")


if __name__ == "__main__":
    main()
