"""
Training script for Whale-Conformer.

    python train.py

FIXES from previous version:
  1. pos_weight computed from actual frame-level positive rate (was [1,1,1])
  2. Validation uses annotated segments (was all-zeros → F1 always 0)
  3. Diagnostic prints to catch problems early
  4. focal_alpha=0.75 upweights positives (was 0.25 = downweighting them)
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


def compute_pos_weight() -> torch.Tensor:
    """
    Compute pos_weight per class from frame-level positive rates in training data.

    pos_weight_c = (# negative frames for class c) / (# positive frames for class c)

    With ~5% overall positive rate this gives pos_weight ≈ 19.
    The old version used total_files / n_annotations which gave ~1.0.
    """
    annotations = load_annotations(cfg.TRAIN_DATASETS)
    label_col = "label_3class" if cfg.USE_3CLASS else "annotation"
    class_names = cfg.class_names()

    # Estimate total annotated duration per class (in seconds)
    # and total recording duration
    from dataset import get_file_manifest
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    total_duration_s = manifest["duration_s"].sum()
    total_frames = total_duration_s / cfg.FRAME_STRIDE_S

    pos_weights = []
    for c_name in class_names:
        # Sum up annotation durations for this class
        if cfg.USE_3CLASS:
            # Find all 7-class labels that map to this 3-class label
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]

        if len(class_annots) == 0:
            pos_weights.append(20.0)  # safe default for missing class
            continue

        # Duration of positive frames
        durations = (class_annots["end_datetime"] - class_annots["start_datetime"])
        pos_duration_s = durations.dt.total_seconds().sum()
        pos_frames = pos_duration_s / cfg.FRAME_STRIDE_S
        neg_frames = total_frames - pos_frames

        pw = neg_frames / max(pos_frames, 1.0)
        # Cap at reasonable range to avoid instability
        pw = min(pw, 50.0)
        pos_weights.append(pw)

    result = torch.tensor(pos_weights, dtype=torch.float32)
    return result


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


def _print_prediction_stats(probs, targets, mask, epoch, split="val"):
    """Print diagnostic stats to verify the model is actually predicting positives."""
    m = mask.view(-1).bool()
    for c in range(probs.size(-1)):
        p = probs[..., c].reshape(-1)[m]
        t = targets[..., c].reshape(-1)[m]
        n_pos_targ = t.sum().item()
        n_pos_pred_03 = (p > 0.3).sum().item()
        n_pos_pred_05 = (p > 0.5).sum().item()
        mean_p = p.mean().item()
        mean_p_on_pos = p[t > 0].mean().item() if n_pos_targ > 0 else 0.0
        mean_p_on_neg = p[t == 0].mean().item()
        print(f"  [{split}] class {c}: "
              f"target_pos={int(n_pos_targ):,} | "
              f"pred>0.3={int(n_pos_pred_03):,} pred>0.5={int(n_pos_pred_05):,} | "
              f"mean_prob={mean_p:.4f} on_pos={mean_p_on_pos:.4f} on_neg={mean_p_on_neg:.4f}")


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, scaler):
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for i, (audio, targets, mask, _) in enumerate(pbar):
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(audio)
            targets, mask = _align_lengths(logits, targets, mask, device)
            loss = criterion(logits, targets, mask)

        scaler.scale(loss).backward()

        if cfg.GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n += 1

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  *** NaN/Inf detected at batch {i}, restoring best checkpoint ***")
            best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            optimizer = AdamW(model.parameters(), lr=cfg.LR * 0.5,
                              weight_decay=cfg.WEIGHT_DECAY)
            break

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, thresholds, epoch=0):
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

    # Diagnostic: print prediction statistics
    _print_prediction_stats(all_probs, all_targ, all_mask, epoch)

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
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        class_f1s.append(f1)
        print(f"  class {c} ({cfg.class_names()[c]}): "
              f"TP={int(tp)} FP={int(fp)} FN={int(fn)} "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

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

    if torch.cuda.device_count() > 1:
        print(f"Training across {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # --- pos_weight (CRITICAL FIX) ---
    if cfg.POS_WEIGHT is None:
        pos_weight = compute_pos_weight().to(device)
    else:
        pos_weight = torch.tensor(
            [cfg.POS_WEIGHT] * cfg.n_classes(), dtype=torch.float32, device=device
        )
    print(f"pos_weight per class: {pos_weight}")
    # Expect values around 15-30, NOT 1.0

    # --- loss ---
    criterion = WeightedBCEWithFocal(
        pos_weight=pos_weight,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        focal_weight=cfg.FOCAL_WEIGHT,
    ).to(device)

    # --- optimiser + scheduler ---
    optimizer = AdamW(model.parameters(), lr=cfg.LR,
                      weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.WARMUP_EPOCHS)
    cosine = CosineAnnealingWarmRestarts(optimizer,
                                         T_0=cfg.EPOCHS - cfg.WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[cfg.WARMUP_EPOCHS])

    scaler = GradScaler("cuda")

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

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler
        )
        val = validate(model, val_loader, criterion, device, thresholds, epoch)

        print(f"\nTrain loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}")

        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())

        ckpt = {"epoch": epoch, "model_state_dict": model_state,
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
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_ckpt["model_state_dict"])
    else:
        model.load_state_dict(best_ckpt["model_state_dict"])
    best_thresholds = tune_thresholds(model, val_loader, device, cfg.n_classes())
    print(f"Tuned thresholds: {best_thresholds.tolist()}")

    final_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                   else model.state_dict())
    torch.save({"model_state_dict": final_state,
                "thresholds": best_thresholds}, run_dir / "final_model.pt")
    print(f"\nDone — best F1: {best_f1:.3f}  →  {run_dir}")


if __name__ == "__main__":
    main()
