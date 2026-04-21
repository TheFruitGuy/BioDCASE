"""
Pipeline Diagnostics
====================

Standalone sanity-check script that verifies several invariants of the
training pipeline against the paper's specification. Intended to be run
once after any refactoring or configuration change to confirm that data
shapes, class balance, and model initialization all still line up with
the paper's setup.

Each check prints its own header and flags any anomalies with a ⚠️
prefix; a summary at the end points users to any issues found.

Checks performed
----------------
1. STFT output shape (frame rate, freq bins, channel count)
2. Annotation coverage in training segments (positive-frame ratio)
3. Class weights (computed from file-level statistics)
4. Model output at initialization (probability range should reflect ~5%
   prior probability of a call being present)
5. Positive/negative segment ratio (should be ~1:1 per the paper)

Usage
-----
::

    python diagnose.py
"""

import torch

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, compute_class_weights
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments,
    WhaleDataset, TrainingDatasetWithResample,
)


# ======================================================================
# Check 1: STFT output shape
# ======================================================================

def check_1_stft_shapes():
    """Verify that the STFT extractor produces the expected output shape."""
    print("=" * 60)
    print("1. STFT shape check")
    print("=" * 60)

    spec_ext = SpectrogramExtractor()
    # 30-second test segment at the configured sample rate.
    audio = torch.randn(1, cfg.SAMPLE_RATE * 30)

    feat = spec_ext(audio)
    B, C, F, T = feat.shape

    expected_frames = (cfg.SAMPLE_RATE * 30) // cfg.HOP_LENGTH  # ~1500 @ 250 Hz
    expected_freq = cfg.N_FFT // 2 + 1                          # 129

    print(f"Input: 30s × {cfg.SAMPLE_RATE}Hz = {cfg.SAMPLE_RATE * 30} samples")
    print(f"Spec: {feat.shape}")
    print(f"  Expected frames (T): ~{expected_frames}, got {T}")
    print(f"  Expected freq (F):   {expected_freq}, got {F}")
    print(f"  Channels (C):        {C} (should be 3: mag, cos, sin)")
    print(f"  Frame rate: {T / 30:.1f} frames/sec (paper: 50)")

    # The paper's 20 ms stride → 50 frames/sec. Small offset from the
    # center=False STFT boundary behavior is expected; large deviation
    # indicates a misconfigured hop length.
    if abs(T / 30 - 50) > 2:
        print("  ⚠️  FRAME RATE MISMATCH — paper is 50 fps (20ms stride)")
    else:
        print("  ✅ Frame rate matches paper")


# ======================================================================
# Check 2: Annotation coverage in training segments
# ======================================================================

def check_2_annotation_coverage():
    """
    Verify that training segments actually contain annotated calls.

    Computes the fraction of frames that are labeled positive in each
    segment. Too-low values indicate that segments aren't being populated
    with their annotations correctly (e.g. after a timestamp-parsing bug);
    too-high values suggest the collar mechanism isn't working.
    """
    print("\n" + "=" * 60)
    print("2. Training segment annotation coverage")
    print("=" * 60)

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(annotations, manifest)

    # Print per-segment details for the first 10 segments.
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

    # Average over 100 segments to smooth out per-segment variation.
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


# ======================================================================
# Check 3: Class weights
# ======================================================================

def check_3_class_weights():
    """Print the class weights currently being used."""
    print("\n" + "=" * 60)
    print("3. Class weights")
    print("=" * 60)
    w = compute_class_weights()
    print("Ratios (vs bmabz):")
    # Relative ratios make it easier to spot whether one class is being
    # weighted unreasonably more than the others.
    for name, weight in zip(cfg.class_names(), w.tolist()):
        print(f"  {name}: {weight:.3f}  (ratio {weight / w[0]:.2f}x)")


# ======================================================================
# Check 4: Model output scale at initialization
# ======================================================================

def check_4_model_output_scale():
    """
    Verify that the untrained model outputs reasonable initial probabilities.

    With the -3.0 classifier bias initialization, sigmoid outputs should
    start around 0.05, matching the ~5% prior probability of a call
    being present at any given frame. A large deviation here would mean
    the model starts with a badly biased output and will waste many
    epochs learning to regress to the base rate before it can start
    learning useful features.
    """
    print("\n" + "=" * 60)
    print("4. Model output at init")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Trigger lazy projection layer initialization with a dummy input.
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


# ======================================================================
# Check 5: Positive/negative segment ratio
# ======================================================================

def check_5_batch_pos_neg_ratio():
    """
    Verify the 1:1 positive/negative segment ratio expected by the paper.
    """
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


# ======================================================================
# Entry point
# ======================================================================

def main():
    """Run all checks in sequence and print a summary."""
    check_1_stft_shapes()
    check_2_annotation_coverage()
    check_3_class_weights()
    check_4_model_output_scale()
    check_5_batch_pos_neg_ratio()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Flag any ⚠️ entries above as likely culprits for any F1 gap.")


if __name__ == "__main__":
    main()
