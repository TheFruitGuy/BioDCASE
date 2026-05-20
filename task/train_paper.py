"""
Faithful Baseline Training Script (train_paper.py)
==================================================
Strict reproduction of the DCASE 2025 Whale-VAD baseline.
Features:
  - Dynamically overwrites config.py values (no manual edits needed).
  - No early stopping (trains unconditionally for defined epochs).
  - Cosine Annealing Learning Rate Scheduler.
  - Dynamically collapses 7-class predictions to 3-class during validation.
  - Saves best_model.pt strictly based on the 3-Class Macro F1 score.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====================================================================
# 1. RUNTIME CONFIG OVERRIDES (MUST HAPPEN BEFORE PROJECT IMPORTS)
# ====================================================================
import config as cfg

print("\n" + "=" * 50)
print(">>> FORCING PAPER-FAITHFUL CONFIG OVERRIDES <<<")
cfg.USE_3CLASS = False
cfg.LR = 1e-5
print(f"  USE_3CLASS set to: {cfg.USE_3CLASS}")
print(f"  LR set to:         {cfg.LR}")
print("=" * 50 + "\n")

# ====================================================================
# 2. PROJECT IMPORTS
# ====================================================================
from dataset import get_dataloaders, load_annotations
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from spectrogram import SpectrogramExtractor
from postprocess import (
    collapse_probs_to_3class,
    postprocess_predictions,
    compute_metrics,
    Detection,
    _parse_filename_dt
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SOTA Baseline")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run folder")
    parser.add_argument("--epochs", type=int, default=80, help="Total epochs to train")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.OUTPUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting {args.run_name} on {device}")
    print(f"Targeting {args.epochs} epochs with LR={cfg.LR} + Cosine Scheduler")

    # 2. Data & Ground Truth Setup
    train_loader, val_loader, _ = get_dataloaders(cfg.BATCH_SIZE)
    val_annotations = load_annotations(cfg.VAL_DATASETS)

    # Pre-build validation ground-truth events
    file_start_dts = {}
    for ds, fn in val_annotations[["dataset", "filename"]].drop_duplicates().values:
        dt = _parse_filename_dt(fn)
        if dt: file_start_dts[(ds, fn)] = dt

    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if not fsd: continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],  # ALWAYS evaluate against 3-class labels
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # 3. Model, Optimizer, Loss, Scheduler
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    pw = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pw).to(device)

    best_macro_f1 = 0.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch}/{args.epochs}] LR: {current_lr:.2e}")

        for audio, targets, mask, metas in train_loader:
            audio, targets, mask = audio.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()
            spec = spec_extractor(audio)
            logits = model(spec)
            loss = criterion(logits, targets, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- STEP SCHEDULER ---
        scheduler.step()

        # --- VALIDATION ---
        model.eval()
        all_probs = {}
        val_loss = 0.0

        with torch.no_grad():
            for audio, targets, mask, metas in val_loader:
                audio, targets, mask = audio.to(device), targets.to(device), mask.to(device)

                spec = spec_extractor(audio)
                logits = model(spec)
                val_loss += criterion(logits, targets, mask).item()

                probs = torch.sigmoid(logits).cpu().numpy()
                hop = spec_extractor.hop_length

                for j, meta in enumerate(metas):
                    key = (meta["dataset"], meta["filename"], meta["start_sample"])
                    n_samp = meta["end_sample"] - meta["start_sample"]
                    n_frames = min(n_samp // hop, probs[j].shape[0])
                    all_probs[key] = probs[j, :n_frames, :]

        val_loss /= len(val_loader)

        # --- 3-CLASS COLLAPSE & EVALUATION ---
        # 1. Map 7 outputs to 3 (because USE_3CLASS is False)
        all_probs_3c = collapse_probs_to_3class(all_probs)

        # 2. Fast Grid Search for best Macro F1
        candidates = np.linspace(0.1, 0.9, 9)
        best_thresholds = np.full(3, 0.5)
        macro_f1_components = []

        for c, cls_name in enumerate(cfg.CALL_TYPES_3):
            best_cls_f1 = 0.0
            for t_try in candidates:
                threshs = best_thresholds.copy()
                threshs[c] = t_try
                preds = postprocess_predictions(all_probs_3c, threshs)
                metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
                f1 = metrics.get(cls_name, {}).get("f1", 0.0)

                if f1 > best_cls_f1:
                    best_cls_f1 = f1
                    best_thresholds[c] = t_try

            macro_f1_components.append(best_cls_f1)

        epoch_macro_f1 = np.mean(macro_f1_components)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Collapsed Macro F1: {epoch_macro_f1:.4f} (BMABZ: {macro_f1_components[0]:.3f}, D: {macro_f1_components[1]:.3f}, BP: {macro_f1_components[2]:.3f})")

        # --- SAVING LOGIC ---
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_f1": best_macro_f1,
            "thresholds": torch.tensor(best_thresholds)
        }

        torch.save(state, out_dir / "latest_model.pt")

        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            state["best_f1"] = best_macro_f1
            torch.save(state, out_dir / "best_model.pt")
            print(f"  *** New Best Collapsed Macro F1 saved! ***")


if __name__ == "__main__":
    main()