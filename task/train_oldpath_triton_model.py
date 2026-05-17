"""
Diagnostic: OLD pipeline + Triton model class
=============================================

Literally the canonical ``train.py`` (the one that produced
``runs/whalevad_20260502_175547/best_model.pt`` with F1=0.474), with
**one change**: the model class is ``Triton`` (from
``triton/model.py``) instead of ``WhaleVAD`` (from ``model.py``).

Every other moving part is the proven canonical code path:
    - dataset.py            (OLD, same 30-60s negatives, same collation)
    - spectrogram.py        (OLD)
    - postprocess.py        (OLD)
    - wandb_utils.py        (OLD)
    - model.WhaleVADLoss    (OLD)
    - compute_class_weights (OLD)
    - This file's train/validate/main are OLD train.py verbatim

The thin ``_TritonAsLogits`` adapter is the only Triton-specific shim:
``Triton.forward()`` returns ``{"logits", "probs"}``; the OLD loss
and OLD validate expect raw logits. The adapter unwraps the dict so
no other code needs to change.

Why this diagnostic?
--------------------
If this script reproduces F1≈0.474, the Triton **model class** is
clean and the regression lives somewhere in ``triton/train.py``,
``triton/dataset.py``, or ``triton/wandb_utils.py`` — a smaller search
space we can attack next.

If this script also lands at F1≈0.245 like the new pipeline, the
Triton model class itself has a subtle training-time difference
(e.g. parameter init order, dropout site, BN momentum) that doesn't
show up at inference time but does affect training dynamics.

Either way, one short run gives a decisive answer.

Usage
-----
Place this file at ``task/train_oldpath_triton_model.py`` (i.e. next
to the canonical ``train.py``, in the project root). Then::

    CUDA_VISIBLE_DEVICES=4 python train_oldpath_triton_model.py
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# === OLD imports (canonical code path) ================================
import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
# Keep the OLD loss + class weights — same as canonical.
from model import WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest,
    collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level, collapse_probs_to_3class,
)

# === THE ONE SWAPPED IMPORT ===========================================
# Triton lives at task/triton/model.py; make it importable without
# moving anything.
sys.path.insert(0, str(Path(__file__).parent / "triton"))
from triton.model import Triton as _TritonRaw  # noqa: E402


# === Thin adapter so the OLD train loop sees raw logits ===============

class WhaleVAD(_TritonRaw):
    """
    ``Triton`` with old-style ``forward()`` returning a raw logits
    tensor instead of a ``{"logits", "probs"}`` dict.

    Subclassing preserves the saved state_dict key layout
    (``filterbank.weight``, ``feat_extractor.0.weight``, ...) exactly,
    so canonical checkpoints load and save without remapping.
    """

    def forward(self, spec):
        return super().forward(spec)["logits"]


# ======================================================================
# Stabilization hyperparameters — VERBATIM from old train.py
# ======================================================================

RESAMPLE_EVERY = 5
EARLY_STOP_PATIENCE = 25
LR_PATIENCE = 8
LR_FACTOR = 0.5
MIN_LR = 1e-7


# ======================================================================
# CLI and helpers — VERBATIM from old train.py
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--freeze_epochs", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int = cfg.SEED):
    wbu.seed_everything(seed, deterministic=False)


def align_lengths(logits, targets, mask):
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


# ======================================================================
# Validation — VERBATIM from old train.py
# ======================================================================

@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             thresholds, val_annotations, file_start_dts,
             tune_thresholds: bool = True):
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

    all_probs = collapse_probs_to_3class(all_probs)

    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    used_thresholds = np.asarray(thresholds.cpu().numpy(), dtype=np.float64).copy()
    if tune_thresholds:
        grids = [
            np.arange(0.20, 0.85, 0.05),
            np.concatenate([np.arange(0.05, 0.5, 0.05),
                            np.arange(0.5, 0.85, 0.10)]),
            np.concatenate([np.arange(0.05, 0.5, 0.05),
                            np.arange(0.5, 0.85, 0.10)]),
        ]
        for c, name in enumerate(cfg.CALL_TYPES_3):
            best_f1, best_t = -1.0, used_thresholds[c]
            for t in grids[c]:
                trial = used_thresholds.copy()
                trial[c] = t
                preds = postprocess_predictions(all_probs, trial)
                m = compute_metrics(preds, gt_events, iou_threshold=0.3)
                f1 = m.get(name, {}).get("f1", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            used_thresholds[c] = best_t

    pred_events = postprocess_predictions(all_probs, used_thresholds)
    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    label = "(tuned thresholds)" if tune_thresholds else "(fixed thresholds)"
    print(f"\n  Event-level 1D IoU validation {label}:")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name)
        if m is None:
            continue
        print(f"    {name.upper():6} t={used_thresholds[c]:.2f}  "
              f"TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_f1": overall_f1,
        "per_class": metrics,
        "thresholds": used_thresholds.tolist(),
    }


# ======================================================================
# Training loop — VERBATIM from old train.py
# ======================================================================

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
            print("*** NaN detected, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Main — VERBATIM from old train.py with one swap
# ======================================================================

def main():
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model class: {WhaleVAD.__module__}.{WhaleVAD.__name__} "
          f"(adapter over triton.model.Triton)")

    extra_tags = ["pretrained" if args.pretrained else "from_scratch",
                  "triton_model_oldpath_diagnostic"]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")

    pretrain_phase = None
    phase_id = "baseline"
    if args.pretrained:
        import re
        m = re.search(r"(3[a-z])", str(args.pretrained))
        pretrain_phase = m.group(1) if m else "unknown"
        candidate_phase = f"4{pretrain_phase[-1]}"
        if candidate_phase in wbu.PHASE_REGISTRY:
            phase_id = candidate_phase
        else:
            extra_tags.append(f"pretrained_{pretrain_phase}")

    run = wbu.init_phase(
        phase_id,
        extra_tags=extra_tags,
        config={
            "lr":               cfg.LR,
            "weight_decay":     cfg.WEIGHT_DECAY,
            "batch_size":       cfg.BATCH_SIZE,
            "epochs":           cfg.EPOCHS,
            "seed":             cfg.SEED,
            "neg_ratio":        cfg.NEG_RATIO,
            "use_3class":       cfg.USE_3CLASS,
            "n_classes":        cfg.n_classes(),
            "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
            "use_focal_loss":   getattr(cfg, "USE_FOCAL_LOSS", False),
            "focal_alpha":      getattr(cfg, "FOCAL_ALPHA", None),
            "focal_gamma":      getattr(cfg, "FOCAL_GAMMA", None),
            "lstm_hidden":      cfg.LSTM_HIDDEN,
            "lstm_layers":      cfg.LSTM_LAYERS,
            "train_sites":      list(cfg.TRAIN_DATASETS),
            "val_sites":        list(cfg.VAL_DATASETS),
            "grad_clip":        cfg.GRAD_CLIP,
            "resample_every":   RESAMPLE_EVERY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr_patience":      LR_PATIENCE,
            "lr_factor":        LR_FACTOR,
            "min_lr":           MIN_LR,
            "pretrained":       args.pretrained,
            "pretrain_phase":   pretrain_phase,
            "freeze_epochs":    args.freeze_epochs,
            "diagnostic":       "oldpath_triton_model",
        },
    )

    # Use a distinct run dir prefix so this diagnostic doesn't get
    # confused with canonical (whalevad_*) or new pipeline (triton__*).
    run_dir = Path(cfg.OUTPUT_DIR) / f"oldpath_triton_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    train_ds, train_loader, val_loader = build_dataloaders()

    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    spec_extractor = SpectrogramExtractor().to(device)
    # === THE ONE LINE THAT DIFFERS FROM CANONICAL ====================
    # Canonical: model = WhaleVAD(num_classes=cfg.n_classes()).to(device)
    # We instantiate the same name, but it's the Triton-backed subclass.
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)}")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    print(f"DEBUG class weights: {pos_weight.tolist()}")
    print(f"Scheduler: ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"Negative resampling: every {RESAMPLE_EVERY} epochs")

    best_f1 = 0.0
    no_improve_epochs = 0
    thresholds = torch.tensor(
        cfg.DEFAULT_THRESHOLDS[:3] if len(cfg.DEFAULT_THRESHOLDS) >= 3
        else [0.5, 0.5, 0.5],
        device=device,
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}\n"
              f"{'=' * 60}")

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

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
        )

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
        print(f"  Tuned thresholds: "
              f"{['%.2f' % t for t in val['thresholds']]}")

        import wandb
        wandb_payload = {
            "epoch":         epoch,
            "lr":            current_lr,
            "train/loss":    train_loss,
            "val/loss":      val["loss"],
            "val/f1_macro":  val["mean_f1"],
        }
        for ci, cname in enumerate(cfg.CALL_TYPES_3):
            pc = val["per_class"].get(cname, {})
            wandb_payload[f"val/f1/{cname}"]        = pc.get("f1", 0.0)
            wandb_payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
            wandb_payload[f"val/recall/{cname}"]    = pc.get("recall", 0.0)
            wandb_payload[f"val/tp/{cname}"]        = pc.get("tp", 0)
            wandb_payload[f"val/fp/{cname}"]        = pc.get("fp", 0)
            wandb_payload[f"val/fn/{cname}"]        = pc.get("fn", 0)
            wandb_payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
        wandb.log(wandb_payload, step=epoch)

        clf_module = (model.module.classifier if isinstance(model, nn.DataParallel)
                      else model.classifier)
        bias_str = ", ".join(f"{b:+.2f}" for b in clf_module.bias.detach().cpu().tolist())
        print(f"  Classifier bias: [{bias_str}]")

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
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs}/{EARLY_STOP_PATIENCE} epochs")

        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs")
            break

    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])

    tuned = tune_thresholds_event_level(
        model_to_load, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    final_state = model_to_load.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "thresholds": torch.tensor(tuned),
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")

    import wandb
    wandb.summary["best_f1"]              = float(best_f1)
    wandb.summary["best_f1_post_tuning"]  = float(best_ckpt.get("best_f1", best_f1))
    wandb.summary["final_thresholds"]     = list(map(float, tuned))
    wandb.summary["epochs_run"]           = epoch
    wandb.summary["early_stopped"]        = no_improve_epochs >= EARLY_STOP_PATIENCE
    wandb.summary["verdict"] = (
        f"oldpath+triton-model diagnostic finished at best F1 {best_f1:.3f} "
        f"(epoch {best_ckpt.get('epoch', '?')} of {epoch} run)."
    )
    wandb.finish()


if __name__ == "__main__":
    main()
