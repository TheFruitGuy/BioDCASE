"""
Fine-tune a contrastive-pretrained Conformer on labeled ATBFL data.

    CUDA_VISIBLE_DEVICES=8 python finetune.py --pretrained ./runs/pretrain/contrastive_XXXX/best_pretrained.pt

Compared to train.py (from scratch), this:
  1. Loads pretrained encoder weights
  2. Optionally freezes the encoder for a few epochs (train head only)
  3. Then unfreezes everything for full fine-tuning
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config as cfg
from model import WhaleConformer, WeightedBCEWithFocal
from dataset import build_dataloaders, load_annotations, collate_fn
from postprocess import tune_thresholds
from train import (
    set_seed, compute_pos_weight, _align_lengths,
    _print_prediction_stats, validate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, required=True,
                   help="Path to best_pretrained.pt from contrastive pretraining")
    p.add_argument("--freeze_epochs", type=int, default=5,
                   help="Epochs to freeze encoder, training only the classification head")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"finetune_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_ds, train_loader, val_loader = build_dataloaders()

    # --- model with pretrained weights ---
    model = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=cfg.DROPOUT, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, sample_rate=cfg.SAMPLE_RATE,
    ).to(device)

    # Load pretrained encoder
    ckpt = torch.load(args.pretrained, map_location=device)
    missing, unexpected = model.load_state_dict(
        ckpt["encoder_state_dict"], strict=False
    )
    print(f"Loaded pretrained encoder from: {args.pretrained}")
    if torch.cuda.device_count() > 1:
        print(f"Training across {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    print(f"  Pretrain loss was: {ckpt.get('loss', '?')}")
    print(f"  Missing keys (expected — classifier head): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # --- loss ---
    pos_weight = compute_pos_weight().to(device)
    # Cap pos_weight to avoid NaN explosion
    pos_weight = pos_weight.clamp(max=15.0)
    print(f"pos_weight (capped): {pos_weight}")

    criterion = WeightedBCEWithFocal(
        pos_weight=pos_weight,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        focal_weight=cfg.FOCAL_WEIGHT,
    ).to(device)

    # --- Phase 1: freeze encoder, train classifier head only ---
    if args.freeze_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1: Frozen encoder, training head only ({args.freeze_epochs} epochs)")
        print(f"{'='*60}")

        # Freeze everything except classifier
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (head only): {trainable:,}")

        head_optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=0.001,
        )
        scaler = GradScaler("cuda")
        thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device)

        for epoch in range(1, args.freeze_epochs + 1):
            print(f"\n--- Head-only epoch {epoch}/{args.freeze_epochs} ---")
            model.train()
            total_loss, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Head {epoch}", leave=False)

            for audio, targets, mask, _ in pbar:
                audio = audio.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                head_optimizer.zero_grad()
                logits = model(audio)
                targets_aligned, mask_aligned = _align_lengths(logits, targets, mask, device)
                loss = criterion(logits, targets_aligned, mask_aligned)

                if torch.isnan(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                head_optimizer.step()

                total_loss += loss.item()
                n += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val = validate(model, val_loader, criterion, device, thresholds, epoch)
            print(f"  Train loss: {total_loss/max(n,1):.4f}  Val loss: {val['loss']:.4f}"
                  f"  Mean F1: {val['mean_f1']:.3f}")

        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

    # --- Phase 2: full fine-tuning ---
    print(f"\n{'='*60}")
    print(f"Phase 2: Full fine-tuning ({cfg.EPOCHS} epochs)")
    print(f"{'='*60}")

    optimizer = AdamW(model.parameters(), lr=cfg.LR,
                      weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[cfg.WARMUP_EPOCHS])

    scaler = GradScaler("cuda")
    best_f1 = 0.0
    thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device)

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n{'='*60}\nFine-tune Epoch {epoch}/{cfg.EPOCHS}\n{'='*60}")

        train_ds.resample_negatives()
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

        for audio, targets, mask, _ in pbar:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(audio)
            targets_aligned, mask_aligned = _align_lengths(logits, targets, mask, device)
            loss = criterion(logits, targets_aligned, mask_aligned)

            if torch.isnan(loss):
                continue

            # Standard backward and step (Removed scaler)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()  # (or head_optimizer.step() in Phase 1)

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val = validate(model, val_loader, criterion, device, thresholds, epoch)

        print(f"\nTrain loss: {total_loss/max(n,1):.4f}   Val loss: {val['loss']:.4f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}")

        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

        ckpt = {"epoch": epoch, "model_state_dict": model_state,
                "best_f1": best_f1, "thresholds": thresholds.cpu(),
                "pretrained_from": args.pretrained}

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")

        torch.save(ckpt, run_dir / "latest_model.pt")

    # --- threshold tuning ---
    print(f"\n{'='*60}\nTuning thresholds\n{'='*60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_ckpt["model_state_dict"])
    else:
        model.load_state_dict(best_ckpt["model_state_dict"])
    best_thresholds = tune_thresholds(model, val_loader, device, cfg.n_classes())
    print(f"Tuned thresholds: {best_thresholds.tolist()}")

    final_state = (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

    torch.save({"model_state_dict": final_state,
                "thresholds": best_thresholds,
                "pretrained_from": args.pretrained},
               run_dir / "final_model.pt")
    print(f"\nDone — best F1: {best_f1:.3f} → {run_dir}")


if __name__ == "__main__":
    main()
