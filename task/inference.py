"""
Inference script for BioDCASE 2026 Task 2 submission.

Usage:
    python inference.py \
        --checkpoint ./runs/conformer_XXXX/final_model.pt \
        --data_root ./data \
        --eval_datasets kerguelen2020 ddu2021 \
        --output submission.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import WhaleConformer
from dataset import (
    DataConfig, WhaleCallDataset, build_eval_segments,
    get_file_manifest, collate_fn, CALL_TYPES_3, CALL_TYPES_7,
)
from postprocess import postprocess_predictions, export_challenge_csv


def parse_args():
    p = argparse.ArgumentParser(description="Whale-Conformer Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--eval_datasets", nargs="+", default=["kerguelen2020", "ddu2021"])
    p.add_argument("--output", type=str, default="submission.csv")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def run_inference(
    model: WhaleConformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[tuple[str, str, int], np.ndarray]:
    """
    Run model inference on all segments.
    Returns dict: (dataset, filename, start_sample) → probs array (T, C).
    """
    model.eval()
    all_probs = {}

    for batch_idx, (audio, _targets, _mask, metas) in enumerate(loader):
        audio = audio.to(device)
        logits = model(audio)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            # Trim to actual length (remove padding)
            n_samples = meta["end_sample"] - meta["start_sample"]
            n_frames = n_samples // (model.hop_length)
            all_probs[key] = probs[i, :n_frames, :]

        if (batch_idx + 1) % 100 == 0:
            print(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    return all_probs


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    thresholds = ckpt["thresholds"].numpy()

    n_classes = 3 if config.get("use_3class", True) else 7
    class_names = CALL_TYPES_3 if n_classes == 3 else CALL_TYPES_7

    # Build model
    model = WhaleConformer(
        n_classes=n_classes,
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 1024),
        n_layers=config.get("n_layers", 4),
        conv_kernel_size=config.get("conv_kernel", 15),
        dropout=0.0,  # No dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from {args.checkpoint}")

    # Build eval data
    data_config = DataConfig(
        data_root=args.data_root,
        use_3class=(n_classes == 3),
    )
    manifest = get_file_manifest(args.data_root, args.eval_datasets)
    print(f"Found {len(manifest)} files across {args.eval_datasets}")

    segments = build_eval_segments(manifest, data_config)
    print(f"Created {len(segments)} evaluation segments")

    eval_ds = WhaleCallDataset(segments, data_config, is_train=False)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Run inference
    print("Running inference...")
    all_probs = run_inference(model, eval_loader, device)
    print(f"Got predictions for {len(all_probs)} segments")

    # Postprocess
    print("Postprocessing...")
    detections = postprocess_predictions(
        all_probs, thresholds, class_names,
        frame_stride_s=0.02, sample_rate=250,
    )
    print(f"Generated {len(detections)} detections")

    # Export
    export_challenge_csv(detections, args.output)
    print(f"Done! Submission saved to {args.output}")


if __name__ == "__main__":
    main()
