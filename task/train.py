"""
Training script for Whale-Conformer.

Usage:
    python train.py --data_root ./data --epochs 60 --batch_size 16 --gpus 1

Key features:
  - Epoch-level stochastic negative undersampling
  - Focal loss + weighted BCE
  - AdamW with cosine LR schedule + warmup
  - Per-class threshold tuning on validation set
  - Checkpointing based on best validation F1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from model import WhaleConformer, WeightedBCEWithFocal
from dataset import (
    DataConfig, build_dataloaders, CALL_TYPES_3, CALL_TYPES_7,
    collate_fn, WhaleCallDataset, build_eval_segments, get_file_manifest,
    load_annotations,
)
from postprocess import postprocess_predictions, compute_metrics, tune_thresholds


def parse_args():
    p = argparse.ArgumentParser(description="Train Whale-Conformer")
    # Data
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--use_3class", action="store_true", default=True)
    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--conv_kernel", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_weight", type=float, default=1.0)
    p.add_argument("--neg_ratio", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Infra
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="./runs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(data_config: DataConfig) -> torch.Tensor:
    """Compute N/P_c weights per class from annotation counts."""
    annotations = load_annotations(data_config.data_root, data_config.train_datasets)
    label_col = "label_3class" if data_config.use_3class else "annotation"
    classes = CALL_TYPES_3 if data_config.use_3class else CALL_TYPES_7

    counts = annotations[label_col].value_counts()
    total_files = len(annotations["filename"].unique())

    weights = []
    for c in classes:
        n_pos = counts.get(c, 1)
        # Weight = total_negative_segments / positive_segments for this class
        w = max(total_files / n_pos, 1.0)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    # Normalise so mean weight = 1
    weights = weights / weights.mean()
    return weights


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (audio, targets, padding_mask, _metas) in enumerate(loader):
        audio = audio.to(device)
        targets = targets.to(device)
        padding_mask = padding_mask.to(device)

        logits = model(audio)  # (B, T_model, C)

        # Align target length with model output length
        T_model = logits.size(1)
        T_target = targets.size(1)
        if T_model < T_target:
            targets = targets[:, :T_model, :]
            padding_mask = padding_mask[:, :T_model]
        elif T_model > T_target:
            # Pad targets
            pad = torch.zeros(
                targets.size(0), T_model - T_target, targets.size(2),
                device=device
            )
            targets = torch.cat([targets, pad], dim=1)
            mask_pad = torch.zeros(
                padding_mask.size(0), T_model - T_target,
                dtype=torch.bool, device=device
            )
            padding_mask = torch.cat([padding_mask, mask_pad], dim=1)

        loss = criterion(logits, targets, padding_mask)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")

    scheduler.step()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    thresholds: torch.Tensor,
) -> dict:
    """Run validation, return loss and per-frame metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []
    all_masks = []

    for audio, targets, padding_mask, _metas in loader:
        audio = audio.to(device)
        targets = targets.to(device)
        padding_mask = padding_mask.to(device)

        logits = model(audio)
        T_model = logits.size(1)
        T_target = targets.size(1)
        if T_model < T_target:
            targets = targets[:, :T_model, :]
            padding_mask = padding_mask[:, :T_model]
        elif T_model > T_target:
            pad = torch.zeros(
                targets.size(0), T_model - T_target, targets.size(2), device=device
            )
            targets = torch.cat([targets, pad], dim=1)
            mask_pad = torch.zeros(
                padding_mask.size(0), T_model - T_target, dtype=torch.bool, device=device
            )
            padding_mask = torch.cat([padding_mask, mask_pad], dim=1)

        loss = criterion(logits, targets, padding_mask)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_targets.append(targets.cpu())
        all_masks.append(padding_mask.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # Compute frame-level F1 per class
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    n_classes = all_probs.size(-1)
    class_f1s = []
    for c in range(n_classes):
        mask = all_masks.view(-1).bool()
        preds_c = (all_probs[..., c].view(-1)[mask] > thresholds[c].cpu()).float()
        targs_c = all_targets[..., c].view(-1)[mask]
        tp = (preds_c * targs_c).sum().item()
        fp = (preds_c * (1 - targs_c)).sum().item()
        fn = ((1 - preds_c) * targs_c).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        class_f1s.append(f1)

    return {
        "loss": avg_loss,
        "class_f1": class_f1s,
        "mean_f1": np.mean(class_f1s),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    run_dir = Path(args.output_dir) / f"conformer_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    data_config = DataConfig(
        data_root=args.data_root,
        use_3class=args.use_3class,
        neg_ratio=args.neg_ratio,
    )
    n_classes = 3 if args.use_3class else 7
    class_names = CALL_TYPES_3 if args.use_3class else CALL_TYPES_7

    train_ds, train_loader, val_loader = build_dataloaders(
        data_config, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Model
    model = WhaleConformer(
        n_classes=n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        conv_kernel_size=args.conv_kernel,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    class_weights = compute_class_weights(data_config).to(device)
    print(f"Class weights: {class_weights}")

    criterion = WeightedBCEWithFocal(
        class_weights=class_weights,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        focal_weight=args.focal_weight,
    ).to(device)

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs - args.warmup_epochs, T_mult=1,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    # Training loop
    best_f1 = 0.0
    thresholds = torch.tensor([0.5] * n_classes, device=device)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Resample negatives each epoch
        train_ds.resample_negatives()
        # Recreate loader after resampling
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, args.grad_clip, epoch,
        )

        val_result = validate(model, val_loader, criterion, device, thresholds)

        print(f"\nTrain loss: {train_loss:.4f}")
        print(f"Val loss:   {val_result['loss']:.4f}")
        for i, name in enumerate(class_names):
            print(f"  {name}: F1={val_result['class_f1'][i]:.3f}")
        print(f"  Mean F1: {val_result['mean_f1']:.3f}")

        # Save checkpoint
        if val_result["mean_f1"] > best_f1:
            best_f1 = val_result["mean_f1"]
            ckpt_path = run_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "thresholds": thresholds.cpu(),
                "config": vars(args),
            }, ckpt_path)
            print(f"  *** New best F1: {best_f1:.3f} — saved to {ckpt_path}")

        # Also save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
            "config": vars(args),
        }, run_dir / "latest_model.pt")

    # Final threshold tuning on validation set
    print("\n" + "="*60)
    print("Tuning per-class thresholds on validation set...")
    print("="*60)

    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", map_location=device)["model_state_dict"]
    )
    best_thresholds = tune_thresholds(model, val_loader, device, n_classes)
    print(f"Optimised thresholds: {best_thresholds}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "thresholds": best_thresholds,
        "config": vars(args),
    }, run_dir / "final_model.pt")

    print(f"\nTraining complete. Best F1: {best_f1:.3f}")
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
