"""
Load a checkpoint into the Triton model and run validation.

Tells you whether the rebrand's inference path is correct by loading the
known-good canonical weights and checking they reproduce F1≈0.474 on val.

Usage:
    python validate_checkpoint.py runs/whalevad_20260502_175547/best_model.pt
    python validate_checkpoint.py CKPT --eval-segment-s 60  # try a different eval window

What it does:
    1. Instantiate the Triton model (random init)
    2. Run a dummy forward to initialise the lazy feat_proj layer
    3. Load the canonical state_dict into it
    4. Build the validation loader
    5. Run a full validation pass at the saved thresholds (no in-training tuning)
    6. Report micro F1, macro F1, per-class F1

If the F1 lands at ~0.474 micro, the inference path is correct and the
issue is purely in *training*. If it lands much lower, even the
inference path of the rebrand has a bug.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

# Make local triton/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
import utils
from dataset import build_dataloaders, load_annotations, get_file_manifest
from spectrogram import SpectrogramExtractor
from model import Triton
from loss import TritonLoss, compute_class_weights
from train import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path,
                    help="path to canonical best_model.pt")
    ap.add_argument("--eval-segment-s", type=float, default=None,
                    help="optionally override cfg.EVAL_SEGMENT_S for this check")
    ap.add_argument("--use-saved-thresholds", action="store_true",
                    help="use thresholds saved in the .pt (no re-tune)")
    args = ap.parse_args()

    if args.eval_segment_s is not None:
        print(f"Overriding cfg.EVAL_SEGMENT_S: {cfg.EVAL_SEGMENT_S} → {args.eval_segment_s}")
        cfg.EVAL_SEGMENT_S = args.eval_segment_s

    utils.seed_everything(cfg.SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build val pipeline (also constructs train_ds we don't use)
    print("Building dataloaders...")
    _, _, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    # Build model with random init
    spec_extractor = SpectrogramExtractor().to(device)
    model = Triton(num_classes=cfg.n_classes()).to(device)

    # Dummy forward to trigger lazy feat_proj.weight init (otherwise
    # load_state_dict will complain feat_proj.weight is missing)
    print("Triggering lazy projection init...")
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    # Load checkpoint
    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print(f"  Loaded {len(sd)} tensors")

    # Read saved thresholds if available
    saved_thresh = ckpt.get("thresholds")
    if saved_thresh is not None:
        saved_thresh = torch.as_tensor(saved_thresh, dtype=torch.float32).to(device)
        print(f"  Saved thresholds: {saved_thresh.tolist()}")
    else:
        saved_thresh = torch.tensor([0.5, 0.5, 0.5], device=device)
        print("  No saved thresholds; defaulting to [0.5, 0.5, 0.5]")

    # Build criterion just to get val loss alongside (won't tune anything)
    pos_weight = compute_class_weights().to(device)
    criterion = TritonLoss(pos_weight=pos_weight).to(device)

    # Run validation. If --use-saved-thresholds, skip in-training tuning;
    # otherwise let validate() do its per-class sweep so we see the best
    # achievable F1 at any threshold.
    print(f"\n{'=' * 60}\nValidating at EVAL_SEGMENT_S={cfg.EVAL_SEGMENT_S}\n{'=' * 60}")
    result = validate(
        model, spec_extractor, val_loader, criterion, device,
        saved_thresh, val_annotations, file_start_dts,
        tune_thresholds=not args.use_saved_thresholds,
    )

    print(f"\n{'=' * 60}\nResult summary\n{'=' * 60}")
    print(f"Macro F1 (selection metric):  {result['macro_f1']:.4f}")
    print(f"Micro F1 (overall):           {result['overall_f1']:.4f}")
    print(f"Val loss:                      {result['loss']:.4f}")
    print(f"Tuned thresholds:              {result['thresholds']}")
    print(f"\nCanonical expected (epoch 30):")
    print(f"  best_f1 from .pt:           {ckpt.get('best_f1', 'n/a')}")
    print(f"  thresholds from .pt:        {saved_thresh.tolist()}")
    print()
    if abs(result['overall_f1'] - 0.474) < 0.02:
        print("✓ Micro F1 reproduces the canonical 0.474 baseline within 0.02.")
        print("  Inference path is healthy — the regression is in training-time code.")
    elif result['overall_f1'] < 0.40:
        print("✗ Micro F1 is far below 0.474 even with canonical weights.")
        print("  The rebrand's inference path itself has a bug.")
    else:
        print("△ Micro F1 is plausible but off by more than 0.02.")
        print("  Investigate EVAL_SEGMENT_S, postprocess, or threshold handling.")


if __name__ == "__main__":
    main()
