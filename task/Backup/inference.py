"""
Inference — generate challenge submission CSV from a trained model.

    python inference.py --checkpoint ./runs/conformer_XXXX/final_model.pt

Reads eval dataset paths and all other settings from config.py.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from model import WhaleConformer
from dataset import WhaleCallDataset, build_eval_segments, get_file_manifest, collate_fn
from postprocess import postprocess_predictions, export_challenge_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=str(cfg.SUBMISSION_PATH))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs = {}
    for i, (audio, _, _, metas) in enumerate(loader):
        logits = model(audio.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = n_samp // model.hop_length
            all_probs[key] = probs[j, :n_frames, :]
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(loader)} batches")
    return all_probs


def main():
    args = parse_args()
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    thresholds = ckpt["thresholds"].numpy()

    model = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=0.0,  # no dropout at inference
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.WIN_LENGTH,
        sample_rate=cfg.SAMPLE_RATE,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    manifest = get_file_manifest(cfg.EVAL_DATASETS)
    print(f"{len(manifest)} files across {cfg.EVAL_DATASETS}")

    segments = build_eval_segments(manifest)
    print(f"{len(segments)} evaluation segments")

    loader = DataLoader(
        WhaleCallDataset(segments, is_train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    print("Running inference...")
    all_probs = run_inference(model, loader, device)

    print("Postprocessing...")
    detections = postprocess_predictions(all_probs, thresholds)
    print(f"{len(detections)} detections")

    export_challenge_csv(detections, args.output)


if __name__ == "__main__":
    main()
