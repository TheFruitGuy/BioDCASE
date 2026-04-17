"""
Inference — generate challenge submission CSV from a trained model.

    python inference.py --checkpoint ./runs/whalevad_XXXX/final_model.pt

Uses fixed 30s windows with 2s overlap (Section 5.1).
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
    load_annotations,
)
from postprocess import postprocess_predictions, export_challenge_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=str(cfg.SUBMISSION_PATH))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    thresholds = ckpt["thresholds"].cpu().numpy() if torch.is_tensor(ckpt["thresholds"]) \
        else np.array(ckpt["thresholds"])

    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Initialize lazy projection layer first
    dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
    model(spec_extractor(dummy))

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Eval data (no annotations needed — just generate segments)
    manifest = get_file_manifest(cfg.EVAL_DATASETS)
    print(f"{len(manifest)} eval files")

    # For inference, we don't have annotations for the eval set, so
    # build_val_segments with an empty annotations frame is fine
    import pandas as pd
    segments = build_val_segments(manifest, pd.DataFrame(columns=[
        "dataset", "filename", "start_datetime", "end_datetime",
        "annotation", "label_3class",
    ]))
    print(f"{len(segments)} segments (30s × 2s overlap)")

    loader = DataLoader(
        WhaleDataset(segments),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    all_probs = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Inference"):
        audio = audio.to(device)
        logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    print("Postprocessing…")
    detections = postprocess_predictions(all_probs, thresholds)
    print(f"{len(detections)} detections")

    export_challenge_csv(detections, args.output)


if __name__ == "__main__":
    main()
