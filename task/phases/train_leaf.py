"""
train_leaf.py — Whale-VAD training with the LEAF learnable frontend.

Mirror of ``train.py`` with the fixed phase-aware STFT frontend replaced
by ``LeafFrontend`` (SpeechBrain ``Leaf``, linear Gabor init by default,
log compression by default).

Usage
-----
::

    # Single GPU (the cluster's Blackwell GPUs aren't supported by
    # this PyTorch build, so always scope CUDA_VISIBLE_DEVICES).
    CUDA_VISIBLE_DEVICES=0 python train_leaf.py

    # Compare with PCEN compression
    CUDA_VISIBLE_DEVICES=0 python train_leaf.py --compression pcen

    # Mel-init ablation
    CUDA_VISIBLE_DEVICES=0 python train_leaf.py --init mel

    # Diagnostic only: no compression (known degenerate on whale data)
    CUDA_VISIBLE_DEVICES=0 python train_leaf.py --compression none
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from leaf_frontend import LeafFrontend
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from postprocess import tune_thresholds_event_level

from train import (
    align_lengths, validate, set_seed,
    RESAMPLE_EVERY, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, MIN_LR,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--freeze_epochs", type=int, default=0)
    p.add_argument("--n_filters", type=int, default=128)
    p.add_argument("--init", choices=["linear", "mel"], default="linear")
    p.add_argument(
        "--compression", choices=["log", "pcen", "none"], default="log",
        help="LEAF output compression. 'log' (default) is comparable to "
             "log-mel and works on whale data; 'pcen' uses learnable PCEN; "
             "'none' is broken on whale data, diagnostic only."
    )
    # Backward-compat alias.
    p.add_argument("--use_pcen", action="store_true",
                   help="(legacy) equivalent to --compression pcen")
    p.add_argument("--leaf_lr_scale", type=float, default=0.1)
    return p.parse_args()


# ======================================================================
# Training epoch
# ======================================================================

def train_epoch_leaf(model, spec_extractor, loader, criterion,
                     optimizer, device, epoch):
    model.train()
    spec_extractor.train()
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
            print("*** NaN detected, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(spec_extractor.parameters()),
            cfg.GRAD_CLIP,
        )
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Diagnostics
# ======================================================================

def gabor_centers_hz(spec_extractor: LeafFrontend) -> torch.Tensor | None:
    """Current Gabor center frequencies in Hz (angular-radian -> Hz)."""
    try:
        param = spec_extractor._gabor_param()
    except RuntimeError:
        return None
    sr = spec_extractor.sample_rate
    return (param[:, 0].detach() / math.pi * (sr / 2.0)).cpu()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    compression = "pcen" if args.use_pcen else args.compression

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"Seeing {torch.cuda.device_count()} GPU(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")

    extra_tags = [
        "leaf_frontend",
        f"leaf_init_{args.init}",
        f"leaf_filters_{args.n_filters}",
        f"leaf_compression_{compression}",
    ]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if args.pretrained:
        extra_tags.append("pretrained")

    run = wbu.init_phase(
        "baseline",
        extra_tags=extra_tags,
        config={
            "lr":              cfg.LR,
            "weight_decay":    cfg.WEIGHT_DECAY,
            "batch_size":      cfg.BATCH_SIZE,
            "epochs":          cfg.EPOCHS,
            "seed":            cfg.SEED,
            "neg_ratio":       cfg.NEG_RATIO,
            "use_3class":      cfg.USE_3CLASS,
            "n_classes":       cfg.n_classes(),
            "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
            "use_focal_loss":  getattr(cfg, "USE_FOCAL_LOSS", False),
            "lstm_hidden":     cfg.LSTM_HIDDEN,
            "lstm_layers":     cfg.LSTM_LAYERS,
            "grad_clip":       cfg.GRAD_CLIP,
            "resample_every":  RESAMPLE_EVERY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr_patience":     LR_PATIENCE,
            "lr_factor":       LR_FACTOR,
            "min_lr":          MIN_LR,
            "pretrained":      args.pretrained,
            "freeze_epochs":   args.freeze_epochs,
            "frontend":        "leaf",
            "leaf_n_filters":  args.n_filters,
            "leaf_init":       args.init,
            "leaf_compression": compression,
            "leaf_lr_scale":   args.leaf_lr_scale,
        },
    )

    run_dir = Path(cfg.OUTPUT_DIR) / f"leaf_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    train_ds, train_loader, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    spec_extractor = LeafFrontend(
        n_filters=args.n_filters,
        init=args.init,
        compression=compression,
    ).to(device)
    model = WhaleVAD(num_classes=cfg.n_classes(), feat_channels=1).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device,
                          weights_only=False)
        state = ckpt.get("encoder_state_dict",
                         ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)}")
        print("  NOTE: pretrained encoder weights were trained for "
              "feat_channels=3 (STFT). filterbank weights will be dropped.")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_leaf  = sum(p.numel() for p in spec_extractor.parameters() if p.requires_grad)
    print(f"Model params: {n_model:,}")
    print(f"LEAF params:  {n_leaf:,}")

    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        spec_extractor.gabor_param_groups(cfg.LR, args.leaf_lr_scale)
        + [{"params": model.parameters(), "lr": cfg.LR, "name": "model"}],
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    print("Optimizer parameter groups:")
    for g in optimizer.param_groups:
        gp_n = sum(p.numel() for p in g["params"])
        print(f"  {g.get('name', '?'):8s} lr={g['lr']:.2e} params={gp_n:,}")

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    init_centers = gabor_centers_hz(spec_extractor)
    if init_centers is not None:
        print(f"Initial Gabor centers (Hz): "
              f"min={init_centers.min():.2f} "
              f"med={init_centers.median():.2f} "
              f"max={init_centers.max():.2f}")

    best_f1 = 0.0
    no_improve_epochs = 0
    thresholds = torch.tensor(
        cfg.DEFAULT_THRESHOLDS[:3] if len(cfg.DEFAULT_THRESHOLDS) >= 3
        else [0.5, 0.5, 0.5],
        device=device,
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        lr_model = optimizer.param_groups[-1]["lr"]
        lr_leaf  = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{cfg.EPOCHS}  "
              f"LR(model)={lr_model:.2e}  LR(leaf)={lr_leaf:.2e}")
        print('=' * 60)

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **wbu.seeded_dataloader_kwargs(cfg.SEED + epoch),
            )

        if args.pretrained and epoch <= args.freeze_epochs:
            for name, p in model.named_parameters():
                if "classifier" not in name and "lstm" not in name:
                    p.requires_grad = False
            print("  [frozen encoder]")
        elif args.pretrained and epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            print("  [unfroze encoder]")

        train_loss = train_epoch_leaf(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
        )

        spec_extractor.eval()
        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts,
            tune_thresholds=True,
        )
        thresholds = torch.tensor(val["thresholds"], device=device,
                                  dtype=torch.float32)
        scheduler.step(val["mean_f1"])

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}  Best F1: {best_f1:.3f}")

        curr_centers = gabor_centers_hz(spec_extractor)
        drift_str = ""
        if curr_centers is not None and init_centers is not None:
            drift = (curr_centers - init_centers).abs()
            print(f"  Gabor centers Hz: "
                  f"min={curr_centers.min():.2f} "
                  f"med={curr_centers.median():.2f} "
                  f"max={curr_centers.max():.2f}  "
                  f"mean_drift={drift.mean():.3f} "
                  f"max_drift={drift.max():.3f}")
            drift_str = f" drift~{drift.mean():.2f}Hz"

        payload = {
            "epoch":         epoch,
            "lr/model":      lr_model,
            "lr/leaf":       lr_leaf,
            "train/loss":    train_loss,
            "val/loss":      val["loss"],
            "val/f1_macro":  val["mean_f1"],
        }
        for ci, cname in enumerate(cfg.CALL_TYPES_3):
            pc = val["per_class"].get(cname, {})
            payload[f"val/f1/{cname}"]        = pc.get("f1", 0.0)
            payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
            payload[f"val/recall/{cname}"]    = pc.get("recall", 0.0)
            payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
        if curr_centers is not None:
            payload["leaf/center_min_hz"] = float(curr_centers.min())
            payload["leaf/center_max_hz"] = float(curr_centers.max())
            payload["leaf/center_std_hz"] = float(curr_centers.std())
            if init_centers is not None:
                drift = (curr_centers - init_centers).abs()
                payload["leaf/drift_mean_hz"] = float(drift.mean())
                payload["leaf/drift_max_hz"]  = float(drift.max())
        wandb.log(payload, step=epoch)

        model_state = (model.module.state_dict()
                       if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model_state,
            "leaf_state_dict":  spec_extractor.state_dict(),
            "best_f1":          best_f1,
            "thresholds":       thresholds.cpu(),
            "leaf_args": {
                "n_filters":   args.n_filters,
                "init":        args.init,
                "compression": compression,
                "min_freq":    spec_extractor.min_freq,
                "max_freq":    spec_extractor.max_freq,
            },
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}{drift_str}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improvement for "
                  f"{no_improve_epochs}/{EARLY_STOP_PATIENCE} epochs")
        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs")
            break

    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = (model.module if isinstance(model, nn.DataParallel)
                     else model)
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])
    spec_extractor.load_state_dict(best_ckpt["leaf_state_dict"])
    spec_extractor.eval()

    tuned = tune_thresholds_event_level(
        model_to_load, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    final_state = model_to_load.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "leaf_state_dict":  spec_extractor.state_dict(),
        "thresholds":       torch.tensor(tuned),
        "leaf_args": {
            "n_filters":   args.n_filters,
            "init":        args.init,
            "compression": compression,
            "min_freq":    spec_extractor.min_freq,
            "max_freq":    spec_extractor.max_freq,
        },
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1: {best_f1:.3f}")
    print(f"Run dir: {run_dir}")

    wandb.summary["best_f1"]             = float(best_f1)
    wandb.summary["best_f1_post_tuning"] = float(best_ckpt.get("best_f1", best_f1))
    wandb.summary["final_thresholds"]    = list(map(float, tuned))
    wandb.summary["epochs_run"]          = epoch
    wandb.summary["early_stopped"]       = no_improve_epochs >= EARLY_STOP_PATIENCE
    wandb.summary["frontend"]            = "leaf"
    wandb.summary["leaf_init"]           = args.init
    wandb.summary["leaf_n_filters"]      = args.n_filters
    wandb.summary["leaf_compression"]    = compression
    final_centers = gabor_centers_hz(spec_extractor)
    if final_centers is not None and init_centers is not None:
        drift = (final_centers - init_centers).abs()
        wandb.summary["leaf/final_drift_mean_hz"] = float(drift.mean())
        wandb.summary["leaf/final_drift_max_hz"]  = float(drift.max())
    wandb.summary["verdict"] = (
        f"LEAF frontend ({args.init} init, {args.n_filters} filters, "
        f"compression={compression}) finished at best F1 {best_f1:.3f}."
    )

    art = wandb.Artifact(
        f"model-leaf-{run.name}", type="model",
        metadata={
            "best_f1":          float(best_f1),
            "best_epoch":       int(best_ckpt.get("epoch", 0)),
            "epochs_run":       int(epoch),
            "frontend":         "leaf",
            "leaf_init":        args.init,
            "leaf_n_filters":   args.n_filters,
            "leaf_compression": compression,
        },
    )
    art.add_file(str(run_dir / "best_model.pt"))
    art.add_file(str(run_dir / "final_model.pt"))
    run.log_artifact(art, aliases=["best", "leaf"])
    wandb.finish()


if __name__ == "__main__":
    main()
