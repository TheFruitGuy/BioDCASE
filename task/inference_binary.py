"""
Combined Binary Inference
=========================

Loads three class-specialized binary Whale-VAD models (one each for
``bmabz``, ``d``, ``bp``), runs inference with each on the target split,
and merges their detections into a single output. Supports two modes:

    - ``--mode eval``       : evaluate against the labeled validation set
                              and print per-class and per-dataset metrics.
                              Useful for comparing the binary ensemble to
                              the monolithic 3-class baseline.
    - ``--mode submission`` : run on the unlabeled evaluation sets and
                              export a challenge submission CSV.

Because each binary model has its own tuned threshold (baked into its
``final_model.pt`` during training), there is no threshold argument here:
each model applies its own.

Usage
-----
::

    python inference_binary.py \\
        --bmabz_ckpt runs/binary_bmabz_XXXX/final_model.pt \\
        --d_ckpt     runs/binary_d_XXXX/final_model.pt     \\
        --bp_ckpt    runs/binary_bp_XXXX/final_model.pt    \\
        --mode eval
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest,
    build_val_segments, collate_fn,
)
from postprocess import (
    Detection, stitch_segments, smooth_probabilities, merge_and_filter,
    compute_metrics, export_challenge_csv,
)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--bmabz_ckpt", type=str, required=True,
                   help="Path to the bmabz binary model checkpoint.")
    p.add_argument("--d_ckpt", type=str, required=True,
                   help="Path to the d binary model checkpoint.")
    p.add_argument("--bp_ckpt", type=str, required=True,
                   help="Path to the bp binary model checkpoint.")
    p.add_argument("--mode", choices=["eval", "submission"], default="eval",
                   help="'eval' runs on VAL_DATASETS and prints metrics; "
                        "'submission' runs on EVAL_DATASETS and exports CSV.")
    p.add_argument("--output", type=str, default="submission.csv",
                   help="Output CSV path (submission mode only).")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


# ======================================================================
# Single-model inference helper
# ======================================================================

@torch.no_grad()
def run_model_on_segments(ckpt_path, segments, class_name, device, batch_size):
    """
    Run one binary model on the provided segments, returning its detections.

    Parameters
    ----------
    ckpt_path : str
        Path to a checkpoint written by ``train_binary.py``. Expected to
        contain ``model_state_dict`` and ``threshold`` keys.
    segments : list of Segment
        Segments to score (same for all three models).
    class_name : str
        Coarse class label. Used to tag the emitted detections.
    device : torch.device
    batch_size : int

    Returns
    -------
    list of Detection
        Per-class detections after stitching, smoothing, thresholding,
        merging, and duration filtering.
    """
    from dataset import WhaleDataset

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    threshold = ckpt.get("threshold", 0.5)

    spec_ext = SpectrogramExtractor().to(device)
    # Binary model: single output channel.
    model = WhaleVAD(num_classes=1).to(device)

    # Initialize the lazy projection layer before loading weights.
    dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
    model(spec_ext(dummy))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [{class_name}] threshold={threshold:.3f}, "
          f"loaded from {Path(ckpt_path).name}")

    loader = DataLoader(
        WhaleDataset(segments), batch_size=batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # Collect per-window probabilities for stitching.
    all_probs = {}
    hop = spec_ext.hop_length
    for audio, _, _, metas in tqdm(loader, desc=f"Inference [{class_name}]"):
        audio = audio.to(device)
        logits = model(spec_ext(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Stitch → smooth → threshold. We don't use the multi-class
    # postprocess_predictions because this is a single-channel model.
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        active = probs[:, 0] > threshold
        diffs = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            all_dets.append(Detection(
                dataset=ds, filename=fn, label=class_name,
                start_s=s * cfg.FRAME_STRIDE_S,
                end_s=e * cfg.FRAME_STRIDE_S,
                confidence=float(probs[s:e, 0].mean()),
            ))

    # Apply the merge + duration filter per-class. Since each binary model
    # produces only its own class, this won't mix detections across classes.
    return merge_and_filter(all_dets)


# ======================================================================
# Main
# ======================================================================

def main():
    """Entry point: load three binary models, combine their detections."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Build segments for the target split
    # ------------------------------------------------------------------
    if args.mode == "eval":
        print(f"Mode: EVAL on validation sets {cfg.VAL_DATASETS}")
        manifest = get_file_manifest(cfg.VAL_DATASETS)
        annotations = load_annotations(cfg.VAL_DATASETS)
    else:
        print(f"Mode: SUBMISSION on eval sets {cfg.EVAL_DATASETS}")
        manifest = get_file_manifest(cfg.EVAL_DATASETS)
        # No labels available for eval sites; empty DataFrame keeps the
        # rest of the pipeline happy.
        annotations = pd.DataFrame(columns=[
            "dataset", "filename", "start_datetime", "end_datetime",
            "annotation", "label_3class",
        ])

    print(f"  {len(manifest)} audio files, {len(annotations)} annotations")
    segments = build_val_segments(manifest, annotations)
    print(f"  {len(segments)} segments")

    # ------------------------------------------------------------------
    # Run each binary model and concatenate detections
    # ------------------------------------------------------------------
    all_detections = []
    for class_name, ckpt_path in [
        ("bmabz", args.bmabz_ckpt),
        ("d", args.d_ckpt),
        ("bp", args.bp_ckpt),
    ]:
        print(f"\n--- {class_name} ---")
        dets = run_model_on_segments(
            ckpt_path, segments, class_name, device, args.batch_size,
        )
        print(f"  {len(dets)} detections")
        all_detections.extend(dets)

    print(f"\nTotal detections: {len(all_detections)}")

    # ------------------------------------------------------------------
    # Eval or export
    # ------------------------------------------------------------------
    if args.mode == "eval":
        # Build ground-truth events and compute metrics.
        file_start_dts = {
            (r.dataset, r.filename): r.start_dt
            for _, r in manifest.iterrows()
        }
        gt_events = []
        for _, row in annotations.iterrows():
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

        # Overall metrics (micro-averaged across all validation sites).
        metrics = compute_metrics(all_detections, gt_events, iou_threshold=0.3)
        print("\n=== Overall Evaluation ===")
        for cls in cfg.class_names():
            if cls in metrics:
                m = metrics[cls]
                print(f"  {cls:6} TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
                      f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        print(f"  OVERALL: F1={metrics['overall']['f1']:.3f}")

        # Per-dataset breakdown (matches Table 4 layout in the paper).
        print("\n=== Per-dataset ===")
        for ds_name in cfg.VAL_DATASETS:
            ds_preds = [d for d in all_detections if d.dataset == ds_name]
            ds_gts = [d for d in gt_events if d.dataset == ds_name]
            m = compute_metrics(ds_preds, ds_gts, iou_threshold=0.3)
            print(f"\n  {ds_name}:")
            for cls in cfg.class_names():
                if cls in m:
                    r = m[cls]
                    print(f"    {cls:6} P={r['precision']:.3f} "
                          f"R={r['recall']:.3f} F1={r['f1']:.3f}")
            print(f"    OVERALL F1={m['overall']['f1']:.3f}")
    else:
        export_challenge_csv(all_detections, args.output)


if __name__ == "__main__":
    main()
