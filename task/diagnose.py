"""
Diagnose why Whale-VAD reproduction is at F1=0.29 instead of 0.44.

Checks:
  1. Frame rate and dimensions match paper
  2. Annotation coverage in training segments
  3. Class weight distribution
  4. Model output scale sanity
  5. Positive/negative ratio per batch
"""

import torch
import numpy as np
from pathlib import Path

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, compute_class_weights
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments,
    WhaleDataset, TrainingDatasetWithResample,
)


def check_1_stft_shapes():
    """Verify STFT produces correct shapes."""
    print("=" * 60)
    print("1. STFT shape check")
    print("=" * 60)

    spec_ext = SpectrogramExtractor()
    audio = torch.randn(1, cfg.SAMPLE_RATE * 30)  # 30s segment

    feat = spec_ext(audio)
    B, C, F, T = feat.shape

    expected_frames = (cfg.SAMPLE_RATE * 30) // cfg.HOP_LENGTH  # = 1500
    expected_freq = cfg.N_FFT // 2 + 1                          # = 129

    print(f"Input: 30s × {cfg.SAMPLE_RATE}Hz = {cfg.SAMPLE_RATE * 30} samples")
    print(f"Spec: {feat.shape}")
    print(f"  Expected frames (T): ~{expected_frames}, got {T}")
    print(f"  Expected freq (F):   {expected_freq}, got {F}")
    print(f"  Channels (C):        {C} (should be 3: mag, cos, sin)")
    print(f"  Frame rate: {T / 30:.1f} frames/sec (paper: 50)")

    if abs(T / 30 - 50) > 2:
        print("  ⚠️  FRAME RATE MISMATCH — paper is 50 fps (20ms stride)")
    else:
        print("  ✅ Frame rate matches paper")


def check_2_annotation_coverage():
    """Check if training segments actually contain calls."""
    print("\n" + "=" * 60)
    print("2. Training segment annotation coverage")
    print("=" * 60)

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(annotations, manifest)

    # Sample some segments and check
    n_check = 10
    sample_segs = pos_segs[:n_check]
    ds = WhaleDataset(sample_segs)

    print(f"Total positive segments: {len(pos_segs)}")
    print(f"\nChecking first {n_check}:")
    for i in range(n_check):
        audio, targets, mask, meta = ds[i]
        pos_frames = (targets.sum(dim=-1) > 0).sum().item()
        total_frames = targets.size(0)
        ratio = 100 * pos_frames / total_frames
        dur = audio.size(0) / cfg.SAMPLE_RATE
        print(f"  Seg {i}: dur={dur:.1f}s, frames={total_frames}, "
              f"pos_frames={pos_frames} ({ratio:.1f}%), "
              f"annots={len(sample_segs[i].annotations)}")

    # Average
    total_pos = 0
    total_frames = 0
    n_to_check = min(100, len(ds))
    for i in range(n_to_check):
        _, t, _, _ = ds[i]
        total_pos += (t.sum(dim=-1) > 0).sum().item()
        total_frames += t.size(0)
    print(f"\nAverage positive-frame ratio across 100 segments: "
          f"{100 * total_pos / total_frames:.1f}%")
    print("(Paper expects ~5% prevalence overall)")


def check_3_class_weights():
    """Print class weights being used."""
    print("\n" + "=" * 60)
    print("3. Class weights")
    print("=" * 60)
    w = compute_class_weights()
    print(f"Ratios (vs bmabz):")
    for name, weight in zip(cfg.class_names(), w.tolist()):
        print(f"  {name}: {weight:.3f}  (ratio {weight/w[0]:.2f}x)")


def check_4_model_output_scale():
    """Check model output is reasonable at init."""
    print("\n" + "=" * 60)
    print("4. Model output at init")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Init lazy layer
    dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
    with torch.no_grad():
        out = model(spec(dummy))
    probs = torch.sigmoid(out)
    print(f"Output shape: {out.shape}")
    print(f"Logit range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"Prob range:  [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Mean prob:   {probs.mean():.3f}")
    if probs.mean() > 0.3 or probs.mean() < 0.05:
        print("  ⚠️  Initial prob far from 5% (target class prevalence)")


def check_5_batch_pos_neg_ratio():
    """Check that we're actually getting 1:1 pos:neg at batch level."""
    print("\n" + "=" * 60)
    print("5. Positive/negative ratio")
    print("=" * 60)

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(annotations, manifest)
    train_ds = TrainingDatasetWithResample(pos_segs, manifest, annotations)

    n_pos = len(train_ds.positive_segments)
    n_neg = len(train_ds.negative_segments)
    print(f"Positive segments: {n_pos}")
    print(f"Negative segments: {n_neg}")
    print(f"Ratio: {n_pos / max(n_neg, 1):.2f}:1 (paper: 1:1)")

    if abs(n_pos / max(n_neg, 1) - 1.0) > 0.2:
        print("  ⚠️  RATIO MISMATCH — should be ~1:1")
    else:
        print("  ✅ Ratio matches paper")


def main():
    check_1_stft_shapes()
    check_2_annotation_coverage()
    check_3_class_weights()
    check_4_model_output_scale()
    check_5_batch_pos_neg_ratio()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Flag any ⚠️ entries above as likely culprits for the F1 gap.")


if __name__ == "__main__":
    main()
