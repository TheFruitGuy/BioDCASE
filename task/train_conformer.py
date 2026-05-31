"""
Conformer-SED Training Driver
=============================

Trains ``WhaleVAD_Conformer`` using the *exact* recipe, data pipeline,
validation, threshold-tuning and checkpoint-selection logic as the
CNN-BiLSTM baseline (``train.py``), so the two are a fair head-to-head.

What is reused verbatim from ``train.py`` (i.e. identical measurement):
  - ``validate``        — event-level val pipeline + per-epoch threshold sweep
  - ``align_lengths``   — logits/targets frame reconciliation
  - ``set_seed``        — seeding semantics
  - the stabilisation constants (RESAMPLE_EVERY, early-stop, LR schedule)
  - ``build_dataloaders`` / loss / optimiser / scheduler settings

What differs (and only this):
  1. The model is ``WhaleVAD_Conformer`` instead of ``WhaleVAD``.
  2. The training step passes a per-frame ``key_padding_mask`` to the
     model so attention ignores zero-padded frames in variable-length
     batches (the BiLSTM ignored them only in the loss; for global
     attention masking them in the model matters). Validation is left
     unmasked, exactly matching how the baseline is measured.
  3. ``--select-by {micro,macro}``. ``train.py`` selects the best
     checkpoint on the *micro*-averaged F1 (``metrics["overall"]``,
     see postprocess.py line ~566), even though the official BioDCASE
     metric is macro F1. ``micro`` (default) reproduces the baseline's
     selection; ``macro`` selects on the metric you actually report.
     Macro F1 is logged every epoch regardless.

Usage
-----
    CUDA_VISIBLE_DEVICES=5 python train_conformer.py
    CUDA_VISIBLE_DEVICES=5 python train_conformer.py --select-by macro
    CUDA_VISIBLE_DEVICES=5 python train_conformer.py \
        --d-model 144 --layers 4 --conv-kernel 31 --no-wandb
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVADLoss, compute_class_weights
from model_conformer import WhaleVAD_Conformer
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from postprocess import tune_thresholds_event_level

# Reuse the baseline's training/validation machinery verbatim.
from train import (
    validate, align_lengths, set_seed,
    RESAMPLE_EVERY, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, MIN_LR,
)


def parse_args():
    p = argparse.ArgumentParser()
    # Conformer hyperparameters
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--conv-kernel", type=int, default=31)
    p.add_argument("--dropout", type=float, default=0.1)
    # Selection / bookkeeping
    p.add_argument("--select-by", choices=["micro", "macro"], default="micro",
                   help="Checkpoint-selection metric. 'micro' matches train.py; "
                        "'macro' selects on the official BioDCASE metric.")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def macro_f1_from(val: dict) -> float:
    """Mean of per-class event-level F1 over the 3 coarse classes."""
    pc = val["per_class"]
    return float(np.mean([pc.get(c, {}).get("f1", 0.0) for c in cfg.CALL_TYPES_3]))


def train_epoch_masked(model, spec_extractor, loader, criterion, optimizer, device, epoch):
    """One training epoch — identical to train.train_epoch but passes a
    per-frame key_padding_mask to the model so attention skips padding."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)          # True = valid frame

        optimizer.zero_grad()

        spec = spec_extractor(audio)
        # key_padding_mask: True = padded. The model reconciles any
        # off-by-one between this length and its internal frame count.
        logits = model(spec, key_padding_mask=~mask)
        targets_a, mask_a = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets_a, mask_a)

        if torch.isnan(loss) or torch.isinf(loss):
            print("*** NaN detected, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb (optional)
    # ------------------------------------------------------------------
    run = None
    if not args.no_wandb:
        run = wbu.init_phase(
            "conformer",
            extra_tags=["from_scratch", "conformer", f"select_{args.select_by}"],
            config={
                "arch": "conformer", "d_model": args.d_model, "nhead": args.nhead,
                "layers": args.layers, "ffn_mult": args.ffn_mult,
                "conv_kernel": args.conv_kernel, "dropout": args.dropout,
                "select_by": args.select_by, "lr": cfg.LR,
                "weight_decay": cfg.WEIGHT_DECAY, "batch_size": cfg.BATCH_SIZE,
                "epochs": cfg.EPOCHS, "seed": args.seed,
                "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
                "use_focal_loss": getattr(cfg, "USE_FOCAL_LOSS", False),
                "resample_every": RESAMPLE_EVERY,
                "early_stop_patience": EARLY_STOP_PATIENCE,
            },
        )

    run_dir = Path(cfg.OUTPUT_DIR) / f"conformer_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data (identical to baseline)
    # ------------------------------------------------------------------
    train_ds, train_loader, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD_Conformer(
        num_classes=cfg.n_classes(), d_model=args.d_model, nhead=args.nhead,
        num_layers=args.layers, ffn_mult=args.ffn_mult,
        conv_kernel=args.conv_kernel, dropout=args.dropout,
    ).to(device)

    # Dummy forward to materialise the lazy projection before the optimizer
    # captures parameters (same requirement as the baseline).
    with torch.no_grad():
        model(spec_extractor(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss + optimiser + scheduler (identical to baseline)
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                      betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_FACTOR,
                                  patience=LR_PATIENCE, min_lr=MIN_LR)
    print(f"Selecting checkpoints by: {args.select_by} F1")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_metric = 0.0
    no_improve_epochs = 0
    thresholds = torch.tensor([0.5, 0.5, 0.5], device=device)

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}\n{'=' * 60}")

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
                **wbu.seeded_dataloader_kwargs(args.seed + epoch),
            )

        train_loss = train_epoch_masked(
            model, spec_extractor, train_loader, criterion, optimizer, device, epoch,
        )

        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts, tune_thresholds=True,
        )
        thresholds = torch.tensor(val["thresholds"], device=device, dtype=torch.float32)

        micro_f1 = val["mean_f1"]
        macro_f1 = macro_f1_from(val)
        select_metric = micro_f1 if args.select_by == "micro" else macro_f1
        scheduler.step(select_metric)

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  micro F1: {micro_f1:.3f}   macro F1: {macro_f1:.3f}   "
              f"(selecting by {args.select_by}, best={best_metric:.3f})")
        print(f"  Tuned thresholds: {['%.2f' % t for t in val['thresholds']]}")

        if run is not None:
            import wandb
            payload = {
                "epoch": epoch, "lr": current_lr, "train/loss": train_loss,
                "val/loss": val["loss"], "val/f1_micro": micro_f1,
                "val/f1_macro": macro_f1,
            }
            for ci, cname in enumerate(cfg.CALL_TYPES_3):
                pc = val["per_class"].get(cname, {})
                payload[f"val/f1/{cname}"] = pc.get("f1", 0.0)
                payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
                payload[f"val/recall/{cname}"] = pc.get("recall", 0.0)
                payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
            wandb.log(payload, step=epoch)

        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch, "model_state_dict": model_state,
            "best_micro_f1": micro_f1, "best_macro_f1": macro_f1,
            "select_by": args.select_by, "thresholds": thresholds.cpu(),
            "arch_kwargs": {
                "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.layers,
                "ffn_mult": args.ffn_mult, "conv_kernel": args.conv_kernel,
                "dropout": args.dropout, "num_classes": cfg.n_classes(),
            },
        }

        if select_metric > best_metric:
            best_metric = select_metric
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best ({args.select_by}) F1: {best_metric:.3f}  "
                  f"[micro={micro_f1:.3f} macro={macro_f1:.3f}]")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs}/{EARLY_STOP_PATIENCE} epochs")

        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping after {EARLY_STOP_PATIENCE} stagnant epochs")
            break

    # ------------------------------------------------------------------
    # Post-training threshold tuning on the best checkpoint
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])

    tuned = tune_thresholds_event_level(
        model_to_load, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    torch.save({
        "model_state_dict": model_to_load.state_dict(),
        "thresholds": torch.tensor(tuned),
        "arch_kwargs": best_ckpt["arch_kwargs"],
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best {args.select_by} F1: {best_metric:.3f}  "
          f"[best ckpt: micro={best_ckpt['best_micro_f1']:.3f} "
          f"macro={best_ckpt['best_macro_f1']:.3f}]")
    print(f"Run dir: {run_dir}")

    if run is not None:
        import wandb
        wandb.summary["best_micro_f1"] = float(best_ckpt["best_micro_f1"])
        wandb.summary["best_macro_f1"] = float(best_ckpt["best_macro_f1"])
        wandb.summary["select_by"] = args.select_by
        wandb.summary["final_thresholds"] = list(map(float, tuned))
        wandb.finish()


if __name__ == "__main__":
    main()
