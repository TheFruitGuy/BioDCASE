"""
Training script for Whale-Conformer.

    python train.py

All hyperparameters live in config.py — edit there, then just run this.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config as cfg
from model import WhaleConformer, WeightedBCEWithFocal
from dataset import (
    build_dataloaders, load_annotations, collate_fn,
)
from postprocess import tune_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights() -> torch.Tensor:
    """N / P_c per class, normalised so mean = 1."""
    annotations = load_annotations(cfg.TRAIN_DATASETS)
    label_col = "label_3class" if cfg.USE_3CLASS else "annotation"
    counts = annotations[label_col].value_counts()
    total_files = len(annotations["filename"].unique())

    weights = []
    for c in cfg.class_names():
        n_pos = counts.get(c, 1)
        weights.append(max(total_files / n_pos, 1.0))
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.mean()


def _align_lengths(logits, targets, padding_mask, device):
    """Trim or pad targets/mask to match model output length."""
    T_m = logits.size(1)
    T_t = targets.size(1)
    if T_m < T_t:
        targets = targets[:, :T_m, :]
        padding_mask = padding_mask[:, :T_m]
    elif T_m > T_t:
        pad_t = torch.zeros(targets.size(0), T_m - T_t, targets.size(2), device=device)
        targets = torch.cat([targets, pad_t], dim=1)
        pad_m = torch.zeros(padding_mask.size(0), T_m - T_t, dtype=torch.bool, device=device)
        padding_mask = torch.cat([padding_mask, pad_m], dim=1)
    return targets, padding_mask


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, scaler):
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for i, (audio, targets, mask, _) in enumerate(pbar):
        # 1. non_blocking=True speeds up the CPU -> GPU transfer
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 2. Wrap the forward pass in autocast for 16-bit speed
        # To this (leveraging your Ampere hardware!):
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(audio)
            targets, mask = _align_lengths(logits, targets, mask, device)
            loss = criterion(logits, targets, mask)

        # 3. Use the scaler for the backward pass
        scaler.scale(loss).backward()

        if cfg.GRAD_CLIP > 0:
            # Unscale before clipping gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, thresholds):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_targ, all_mask = [], [], []

    for audio, targets, mask, _ in loader:
        audio, targets, mask = audio.to(device), targets.to(device), mask.to(device)
        logits = model(audio)
        targets, mask = _align_lengths(logits, targets, mask, device)
        total_loss += criterion(logits, targets, mask).item()
        n += 1
        all_probs.append(torch.sigmoid(logits).cpu())
        all_targ.append(targets.cpu())
        all_mask.append(mask.cpu())

    all_probs = torch.cat(all_probs)
    all_targ = torch.cat(all_targ)
    all_mask = torch.cat(all_mask)

    class_f1s = []
    for c in range(cfg.n_classes()):
        m = all_mask.view(-1).bool()
        preds = (all_probs[..., c].reshape(-1)[m] > thresholds[c].cpu()).float()
        targs = all_targ[..., c].reshape(-1)[m]
        tp = (preds * targs).sum().item()
        fp = (preds * (1 - targs)).sum().item()
        fn = ((1 - preds) * targs).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        class_f1s.append(2 * prec * rec / (prec + rec + 1e-8))

    return {"loss": total_loss / max(n, 1), "class_f1": class_f1s,
            "mean_f1": float(np.mean(class_f1s))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"conformer_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_ds, train_loader, val_loader = build_dataloaders()

    # --- model ---
    model = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=cfg.DROPOUT, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, sample_rate=cfg.SAMPLE_RATE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # --- loss ---
    class_weights = compute_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    criterion = WeightedBCEWithFocal(
        class_weights=class_weights, focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA, focal_weight=cfg.FOCAL_WEIGHT,
    ).to(device)

    # --- optimiser + scheduler ---
    optimizer = AdamW(model.parameters(), lr=cfg.LR,
                      weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.WARMUP_EPOCHS)
    cosine = CosineAnnealingWarmRestarts(optimizer,
                                         T_0=cfg.EPOCHS - cfg.WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[cfg.WARMUP_EPOCHS])

    # Initialize the AMP GradScaler
    scaler = GradScaler('cuda')

    # --- training loop ---
    best_f1 = 0.0
    thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device)

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}\n{'=' * 60}")

        train_ds.resample_negatives()
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        # Pass the scaler here!
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler
        )
        val = validate(model, val_loader, criterion, device, thresholds)

        print(f"\nTrain loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        for i, name in enumerate(cfg.class_names()):
            print(f"  {name}: F1={val['class_f1'][i]:.3f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1, "thresholds": thresholds.cpu()}

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")

        torch.save(ckpt, run_dir / "latest_model.pt")

    # --- final threshold tuning ---
    print(f"\n{'='*60}\nTuning thresholds on validation set\n{'='*60}")
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", map_location=device)["model_state_dict"]
    )
    best_thresholds = tune_thresholds(model, val_loader, device, cfg.n_classes())
    print(f"Tuned thresholds: {best_thresholds.tolist()}")

    torch.save({"model_state_dict": model.state_dict(),
                "thresholds": best_thresholds}, run_dir / "final_model.pt")
    print(f"\nDone — best F1: {best_f1:.3f}  →  {run_dir}")


if __name__ == "__main__":
    main()
