"""
Challenge Submission Inference
==============================

Generates the final challenge submission CSV from a trained Whale-VAD
checkpoint. Runs the model over the unlabeled evaluation sites
(Kerguelen2020 and DDU2021), applies the full post-processing pipeline
with tuned thresholds, and writes detections in the DCASE competition
format.

Usage
-----
::

    python inference.py --checkpoint runs/whalevad_XXXX/final_model.pt \\
                        --output submission.csv

The ``final_model.pt`` bundle produced by ``train.py`` already contains
the tuned thresholds, so no additional threshold argument is needed.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    WhaleDataset, build_val_segments, get_file_manifest, collate_fn,
)
from postprocess import postprocess_predictions, export_challenge_csv


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint produced by train.py. Must "
                        "contain both 'model_state_dict' and 'thresholds'.")
    p.add_argument("--output", type=str, default=str(cfg.SUBMISSION_PATH),
                   help="Destination path for the submission CSV.")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE,
                   help="Batch size for inference. Larger values are "
                        "faster but use more GPU memory.")
    return p.parse_args()


@torch.no_grad()
def main():
    """Load model, run inference on evaluation sites, export CSV."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load checkpoint and thresholds
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Thresholds may be stored as a torch.Tensor (newer checkpoints) or a
    # Python list (older checkpoints) — handle both.
    thresholds = (ckpt["thresholds"].cpu().numpy()
                  if torch.is_tensor(ckpt["thresholds"])
                  else np.array(ckpt["thresholds"]))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Initialize the lazy projection layer before loading weights.
    dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
    model(spec_extractor(dummy))

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------
    # Evaluation data
    # ------------------------------------------------------------------
    # Eval sites have no public annotations; build_val_segments accepts an
    # empty annotations DataFrame and just tiles each file with 30s/2s
    # overlap windows.
    manifest = get_file_manifest(cfg.EVAL_DATASETS)
    print(f"{len(manifest)} eval files")

    import pandas as pd
    empty_annots = pd.DataFrame(columns=[
        "dataset", "filename", "start_datetime", "end_datetime",
        "annotation", "label_3class",
    ])
    segments = build_val_segments(manifest, empty_annots)
    print(f"{len(segments)} segments (30s × 2s overlap)")

    loader = DataLoader(
        WhaleDataset(segments),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Inference — accumulate per-window probabilities for stitching
    # ------------------------------------------------------------------
    all_probs = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Inference"):
        audio = audio.to(device)
        logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            # Defend against the 1-2 frame mismatch between the raw-sample
            # frame count and the model's actual output length.
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # ------------------------------------------------------------------
    # Post-processing and export
    # ------------------------------------------------------------------
    print("Postprocessing…")
    detections = postprocess_predictions(all_probs, thresholds)
    print(f"{len(detections)} detections")

    export_challenge_csv(detections, args.output)


if __name__ == "__main__":
    main()
