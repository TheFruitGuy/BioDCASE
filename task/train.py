"""
Whale-VAD training — paper reproduction.

Usage:
    # Train from scratch (paper default)
    python train.py

    # Load contrastive-pretrained encoder (optional)
    python train.py --pretrained /path/to/encoder.pt --freeze_epochs 5

Matches paper (Geldenhuys et al., DCASE 2025):
  - AdamW lr=1e-5, weight_decay=1e-3, betas=(0.9, 0.999)
  - Weighted BCE + Focal (alpha=0.25, gamma=2)
  - Stochastic negative mini-batch undersampling
  - 500ms median filter smoothing on validation probs
  - Event-level 1D IoU evaluation
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest,
    collate_fn, _parse_file_start_dt,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level,
)


# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=None,
                   help="Optional path to contrastive-pretrained encoder")
    p.add_argument("--freeze_epochs", type=int, default=0,
                   help="Epochs to freeze encoder if pretrained is given")
    return p.parse_args()


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_lengths(logits, targets, mask):
    """Align model output length with target/mask length."""
    T_m, T_t = logits.size(1), targets.size(1)
    if T_m < T_t:
        targets = targets[:, :T_m, :]
        mask = mask[:, :T_m]
    elif T_m > T_t:
        pad_t = torch.zeros(targets.size(0), T_m - T_t, targets.size(2),
                            device=targets.device)
        targets = torch.cat([targets, pad_t], dim=1)
        pad_m = torch.zeros(mask.size(0), T_m - T_t, dtype=torch.bool,
                            device=mask.device)
        mask = torch.cat([mask, pad_m], dim=1)
    return targets, mask


# ----------------------------------------------------------------------
# Validation: event-level 1D IoU F1
# ----------------------------------------------------------------------

@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             thresholds, val_annotations, file_start_dts):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_probs: dict = {}

    for audio, targets, mask, metas in tqdm(loader, desc="Validating", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)
        targets, mask = align_lengths(logits, targets, mask)
        total_loss += criterion(logits, targets, mask).item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Event-level metrics
    pred_events = postprocess_predictions(all_probs, thresholds.cpu().numpy())
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"]   - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    print("\n  Event-level 1D IoU validation:")
    for cls, m in metrics.items():
        if cls == "overall":
            continue
        print(f"    {cls.upper():6} TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_f1": overall_f1,
        "per_class": metrics,
    }


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_epoch(model, spec_extractor, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        spec = spec_extractor(audio)
        logits = model(spec)
        targets, mask = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"*** NaN detected, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"whalevad_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ── data ──────────────────────────────────────────────────────
    train_ds, train_loader, val_loader = build_dataloaders()

    # Cache for validation
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest    = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    # ── model ─────────────────────────────────────────────────────
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Initialize lazy projection layer with a dry run
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    # Optional pretrained encoder loading (for your SSL extension)
    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing (expected — classifier/LSTM): {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── loss ──────────────────────────────────────────────────────
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    # ── optimizer (paper: AdamW, lr=1e-5, wd=1e-3) ────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ── training loop ─────────────────────────────────────────────
    best_f1 = 0.0
    thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device)

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{cfg.EPOCHS}\n{'='*60}")

        # Re-sample negatives (Section 5.5)
        train_ds.resample_negatives()
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        # Freeze encoder for first freeze_epochs if pretrained
        if args.pretrained and epoch <= args.freeze_epochs:
            for name, p in model.named_parameters():
                if "classifier" not in name and "lstm" not in name:
                    p.requires_grad = False
            print("  [frozen encoder — training LSTM + classifier only]")
        elif args.pretrained and epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            print("  [unfroze encoder]")

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device, epoch
        )

        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts,
        )

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Mean F1:    {val['mean_f1']:.3f}")

        # Save
        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")

        torch.save(ckpt, run_dir / "latest_model.pt")

    # ── final threshold tuning ────────────────────────────────────
    print(f"\n{'='*60}\nTuning per-class thresholds on validation\n{'='*60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])

    tuned = tune_thresholds_event_level(
        model_to_load, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    # After tune_thresholds_event_level:
    print("\n=== Final evaluation with tuned thresholds ===")
    val = validate(
        model_to_load, spec_extractor, val_loader, criterion, device,
        torch.tensor(tuned, device=device), val_annotations, file_start_dts,
    )
    print(f"TUNED overall F1: {val['mean_f1']:.3f}")

    # Save final with tuned thresholds
    final_state = model_to_load.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "thresholds": torch.tensor(tuned),
    }, run_dir / "final_model.pt")

    print(f"\nDone — best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
