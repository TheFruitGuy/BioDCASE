"""
Triton — Unified training entry point
=====================================

Trains either ``Triton`` (baseline) or ``TritonTIDE`` (with the
Tap-Integrated Detection Enhancer) on the BioDCASE 2026 development
set. Single training loop handles both because both models return a
dict containing at least ``"logits"`` and ``"probs"``; the loss class
is chosen at runtime based on ``--model``.

Pipeline
--------
- Stochastic negative undersampling, resampled every N epochs to dampen
  validation-loss jitter (matches the WhaleVAD production recipe; not
  in the original paper but stabilises early stopping).
- ReduceLROnPlateau on validation F1.
- Early stopping after N epochs of no improvement.
- Per-epoch class-specific threshold tuning so best-checkpoint
  selection picks the model with the best *achievable* F1, not the
  best F1 at the (arbitrary) 0.5 threshold.
- Post-training threshold sweep on the best checkpoint (finer grid)
  saved to ``final_model.pt`` alongside the weights.
- Optional pretrained encoder + freeze schedule for SSL fine-tuning.

CLI
---
::

    # Triton baseline
    python train.py --model triton

    # Triton-TIDE (default TIDE config: 3 taps, dim=64, kernel=5,
    # near-one init, no aux loss)
    python train.py --model tide

    # With pretrained encoder + 5-epoch freeze
    python train.py --model triton --pretrained path/to/ckpt.pt --freeze_epochs 5

    # TIDE with auxiliary gate supervision
    python train.py --model tide --tide-aux-loss --tide-aux-weight 0.2

    # TIDE ablation: no gate, just the model output dict structure
    python train.py --model tide --tide-disabled
"""

from __future__ import annotations

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
import utils
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import Triton
from model_tide import TritonTIDE, TIDEConfig
from loss import TritonLoss, TritonTIDELoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level, collapse_probs_to_3class,
)


# ======================================================================
# Stabilisation hyperparameters
# ======================================================================
# Not in any paper; chosen empirically on the BioDCASE val split.

#: Resample negative segments every N epochs. Lower values match the
#: paper more closely but increase val-loss jitter; higher values risk
#: overfitting to the same negative distribution.
RESAMPLE_EVERY = 5

#: Stop training if validation F1 does not improve for this many
#: epochs. Generous because rare-class learning is slow.
EARLY_STOP_PATIENCE = 25

#: Stagnant epochs before the LR scheduler fires. Smaller than
#: ``EARLY_STOP_PATIENCE`` so the LR drops at least once before
#: stopping.
LR_PATIENCE = 8

#: Multiplicative factor applied to the LR each time the scheduler fires.
LR_FACTOR = 0.5

#: Floor for the learning rate.
MIN_LR = 1e-7


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    p.add_argument(
        "--model", choices=["triton", "tide"], default="triton",
        help="Which model to train. "
             "'triton' = baseline CNN-BiLSTM, "
             "'tide' = baseline + TIDE gate.",
    )
    p.add_argument(
        "--name", type=str, default="",
        help="Free-form tag appended to the run name and recorded in "
             "config.json. Use it to label experiments meaningfully, "
             "e.g. --name pretrained_aug_v2.",
    )
    p.add_argument(
        "--pretrained", type=str, default=None,
        help="Optional path to pretrained encoder weights.",
    )
    p.add_argument(
        "--freeze_epochs", type=int, default=0,
        help="If --pretrained is set, keep the encoder frozen for this "
             "many initial epochs before unfreezing for end-to-end "
             "fine-tuning.",
    )

    # TIDE-specific flags (only consulted when --model=tide).
    tide_grp = p.add_argument_group("TIDE configuration (used with --model=tide)")
    tide_grp.add_argument("--tide-disabled", action="store_true",
                          help="Ablation: build TritonTIDE without the gate.")
    tide_grp.add_argument("--tide-taps", type=int, default=3, choices=[1, 2, 3])
    tide_grp.add_argument("--tide-dim", type=int, default=64)
    tide_grp.add_argument("--tide-dilated-backbone", action="store_true",
                          help="Use the BPN paper's dilated (2,4,8) "
                               "depthwise stack with per-layer "
                               "residuals instead of the plain stack. "
                               "Lets you ablate the backbone change "
                               "separately from the gate.")
    tide_grp.add_argument("--tide-temporal",
                          choices=["conv1d", "bilstm", "none"],
                          default="conv1d",
                          help="Temporal model inside the gate. "
                               "'conv1d' = lightweight (~100ms RF); "
                               "'bilstm' = unbounded RF (matches BPN); "
                               "'none' = identity.")
    tide_grp.add_argument("--tide-kernel", type=int, default=5,
                          help="Conv1d kernel when --tide-temporal=conv1d "
                               "(must be odd).")
    tide_grp.add_argument("--tide-temporal-lstm-hidden", type=int, default=64,
                          help="BiLSTM hidden size when "
                               "--tide-temporal=bilstm.")
    tide_grp.add_argument("--tide-init", choices=["near_one", "random"],
                          default="near_one")
    tide_grp.add_argument("--tide-aux-loss", action="store_true",
                          help="Add auxiliary BCE on the gate against "
                               "per-frame 'any class active' targets.")
    tide_grp.add_argument("--tide-aux-weight", type=float, default=0.1)

    return p.parse_args()


# ======================================================================
# Model + loss construction
# ======================================================================

def build_model_and_loss(
    args: argparse.Namespace, device: torch.device,
) -> tuple[nn.Module, nn.Module, dict]:
    """
    Build the model and matching loss for the chosen ``--model`` mode.

    Returns
    -------
    model : nn.Module
    criterion : nn.Module
    config_dump : dict
        Hyperparameter dump for wandb logging.
    """
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None

    if args.model == "triton":
        model = Triton(num_classes=cfg.n_classes()).to(device)
        criterion = TritonLoss(pos_weight=pos_weight).to(device)
        model_cfg_dump = {"model": "triton"}

    elif args.model == "tide":
        tide_cfg = TIDEConfig(
            enabled=not args.tide_disabled,
            n_taps=args.tide_taps,
            dim=args.tide_dim,
            use_dilated_backbone=args.tide_dilated_backbone,
            temporal=args.tide_temporal,
            temporal_kernel=args.tide_kernel,
            temporal_lstm_hidden=args.tide_temporal_lstm_hidden,
            init_mode=args.tide_init,
            aux_loss=args.tide_aux_loss,
            aux_weight=args.tide_aux_weight,
        )
        model = TritonTIDE(
            num_classes=cfg.n_classes(), tide_cfg=tide_cfg,
        ).to(device)
        criterion = TritonTIDELoss(
            pos_weight=pos_weight,
            aux_loss=tide_cfg.aux_loss,
            aux_weight=tide_cfg.aux_weight,
        ).to(device)
        model_cfg_dump = {"model": "tide", "tide_cfg": tide_cfg.to_dict()}

    else:
        raise ValueError(f"Unknown --model: {args.model!r}")

    return model, criterion, model_cfg_dump


# ======================================================================
# Validation
# ======================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    spec_extractor: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: torch.Tensor,
    val_annotations,
    file_start_dts,
    tune_thresholds: bool = True,
) -> dict:
    """
    Run the full event-level validation pipeline.

    Forward-pass the whole validation loader, then run the same
    post-processing chain used at inference (stitch, smooth, threshold,
    merge, filter) and report per-class + overall event-level F1.

    With ``tune_thresholds=True`` (default) the per-class confidence
    thresholds are tuned on the cached probabilities before reporting
    F1. This matches the DCASE evaluation protocol and gives best-
    checkpoint selection a fair signal — without it, rare-class F1
    would often be reported as zero at a fixed 0.5 threshold even
    when the model has genuine predictive signal at a lower one.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_probs: dict = {}

    for audio, targets, mask, metas in tqdm(loader, desc="Validating", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        out = model(spec)
        logits = out["logits"]

        targets, mask = utils.align_lengths(logits, targets, mask)
        # Also align "probs" and "gate" if they're present; for Triton
        # probs == sigmoid(logits) so it tracks logits; for TIDE the
        # in-model alignment already ran, but doing it again is a no-op.
        if out["probs"].size(1) != logits.size(1):
            T_m = logits.size(1)
            out["probs"] = out["probs"][:, :T_m]
            if "gate" in out:
                out["gate"] = out["gate"][:, :T_m]

        total_loss += criterion(out, targets, mask).item()
        n_batches += 1

        # Cache per-window probabilities for event-level evaluation.
        probs = out["probs"].cpu().numpy()
        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Collapse 7-class → 3-class so postprocessing always sees 3 classes.
    all_probs = collapse_probs_to_3class(all_probs)

    # Build GT Detections.
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

    # Per-class threshold sweep. Coordinate descent with per-class
    # custom grids: broader for bmabz, finer near the low end for the
    # rare classes (d, bp) whose optima often live well below 0.5.
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

    # Final pass with the (tuned or fixed) thresholds.
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
# Training loop (one epoch)
# ======================================================================

def train_epoch(
    model: nn.Module,
    spec_extractor: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        spec = spec_extractor(audio)
        out = model(spec)
        logits = out["logits"]
        targets, mask = utils.align_lengths(logits, targets, mask)
        # Align the rest of the dict so the loss sees consistent shapes.
        if out["probs"].size(1) != logits.size(1):
            T_m = logits.size(1)
            out["probs"] = out["probs"][:, :T_m]
            if "gate" in out:
                out["gate"] = out["gate"][:, :T_m]

        loss = criterion(out, targets, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            print("*** NaN/Inf loss, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Encoder freeze schedule (for --pretrained runs)
# ======================================================================

def apply_freeze_schedule(model: nn.Module, epoch: int, args: argparse.Namespace) -> None:
    """
    Freeze/unfreeze encoder weights based on the freeze schedule.

    When ``--pretrained`` is set, the encoder (filterbank + feature
    extractor + bottleneck + depthwise) is kept frozen for
    ``--freeze_epochs`` initial epochs, then unfrozen for end-to-end
    fine-tuning. The criterion for "encoder" vs "head" is name-based:
    only ``classifier`` and ``lstm`` (and, for TIDE, ``tide``) get
    gradients during the freeze window.
    """
    inner = utils.unwrap(model)
    if not args.pretrained:
        return

    if epoch <= args.freeze_epochs:
        for name, p in inner.named_parameters():
            head = (
                "classifier" in name
                or "lstm" in name
                or "tide" in name
            )
            p.requires_grad = head
        print("  [frozen encoder]")
    elif epoch == args.freeze_epochs + 1:
        for p in inner.parameters():
            p.requires_grad = True
        print("  [unfroze encoder]")


# ======================================================================
# Run-directory helpers
# ======================================================================

def build_run_name(args: argparse.Namespace) -> str:
    """
    Construct an informative, filesystem-safe run name.

    Format:
        ``{model}[__{tag1}_{tag2}_...]__seed{N}__{YYYYMMDD-HHMMSS}``

    The middle tag chunk encodes the small set of interventions you
    actually care about: TIDE knobs that differ from the defaults,
    pretrained + freeze schedule, and the user-supplied ``--name``.
    Defaults that match the canonical config are *not* tagged, so a
    vanilla run reads as ``triton__seed42__20260516-141500`` rather
    than carrying noise.

    Examples
    --------
        triton__seed42__20260516-141500
        triton__pre5__seed42__20260516-141500
        tide__t3_aux__seed42__20260516-141500
        tide__t3_dilated_bilstm_aux__seed42__20260516-141500
        tide__ablate__pretrained_aug_v2__seed42__20260516-141500
    """
    parts: list[str] = [args.model]
    tags: list[str] = []

    if args.model == "tide":
        if args.tide_disabled:
            tags.append("ablate")
        else:
            if args.tide_taps != 3:
                tags.append(f"t{args.tide_taps}")
            if args.tide_dim != 64:
                tags.append(f"d{args.tide_dim}")
            if args.tide_dilated_backbone:
                tags.append("dilated")
            if args.tide_temporal != "conv1d":
                tags.append(args.tide_temporal)
            if args.tide_aux_loss:
                tags.append("aux")

    if args.pretrained:
        tags.append(f"pre{args.freeze_epochs}" if args.freeze_epochs > 0 else "pre")

    if args.name:
        # Sanitise: keep only filename-safe chars.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.name)
        if safe:
            tags.append(safe)

    if tags:
        parts.append("_".join(tags))

    parts.append(f"seed{cfg.SEED}")
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "__".join(parts)


def save_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    model_cfg_dump: dict,
    training_config: dict,
) -> None:
    """
    Dump the run's full configuration to ``config.json``.

    Includes everything needed to reproduce the run: CLI args, derived
    model config (e.g. resolved TIDEConfig), training hyperparameters,
    a snapshot of relevant ``cfg`` module values, and a start timestamp.
    """
    import json
    from datetime import datetime

    cfg_snapshot = {
        "DATA_ROOT": str(cfg.DATA_ROOT),
        "SAMPLE_RATE": cfg.SAMPLE_RATE,
        "FRAME_STRIDE_S": cfg.FRAME_STRIDE_S,
        "N_FFT": cfg.N_FFT,
        "HOP_LENGTH": cfg.HOP_LENGTH,
        "USE_3CLASS": cfg.USE_3CLASS,
        "USE_WEIGHTED_BCE": cfg.USE_WEIGHTED_BCE,
        "USE_FOCAL_LOSS": cfg.USE_FOCAL_LOSS,
        "FOCAL_ALPHA": cfg.FOCAL_ALPHA,
        "FOCAL_GAMMA": cfg.FOCAL_GAMMA,
        "LR": cfg.LR,
        "WEIGHT_DECAY": cfg.WEIGHT_DECAY,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "EPOCHS": cfg.EPOCHS,
        "SEED": cfg.SEED,
        "TRAIN_DATASETS": list(cfg.TRAIN_DATASETS),
        "VAL_DATASETS": list(cfg.VAL_DATASETS),
        "EVAL_DATASETS": list(cfg.EVAL_DATASETS),
    }

    payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "model": args.model,
        "args": vars(args),
        "model_config": model_cfg_dump,
        "training_config": training_config,
        "cfg_snapshot": cfg_snapshot,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_run_info(
    run_dir: Path,
    args: argparse.Namespace,
    best_f1: float,
    best_epoch: int,
    epochs_run: int,
    tuned_thresholds,
    early_stopped: bool,
    final_per_class: dict | None = None,
) -> None:
    """
    Write a short human-readable summary to ``run_info.txt`` at end of
    training. Complements ``config.json`` (which is machine-readable):
    this file is what you read when scanning ``runs/`` to remember what
    happened.
    """
    lines: list[str] = []
    lines.append("Triton run summary")
    lines.append("=" * 60)
    lines.append(f"Run dir:         {run_dir.name}")
    lines.append(f"Model:           {args.model}")
    lines.append(f"Best F1 (val):   {best_f1:.4f}")
    lines.append(f"Best epoch:      {best_epoch}")
    lines.append(f"Epochs run:      {epochs_run}")
    lines.append(f"Early stopped:   {early_stopped}")
    thr_str = "[" + ", ".join(f"{float(t):.3f}" for t in tuned_thresholds) + "]"
    lines.append(f"Tuned thresholds (bmabz, d, bp): {thr_str}")
    lines.append("")
    lines.append("Interventions:")
    lines.append(f"  pretrained:    {args.pretrained or '(none)'}")
    lines.append(f"  freeze_epochs: {args.freeze_epochs}")
    lines.append(f"  name tag:      {args.name or '(none)'}")
    if args.model == "tide":
        lines.append(f"  tide_disabled: {args.tide_disabled}")
        lines.append(f"  tide_taps:     {args.tide_taps}")
        lines.append(f"  tide_dim:      {args.tide_dim}")
        lines.append(f"  tide_dilated:  {args.tide_dilated_backbone}")
        lines.append(f"  tide_temporal: {args.tide_temporal}")
        if args.tide_temporal == "conv1d":
            lines.append(f"  tide_kernel:   {args.tide_kernel}")
        if args.tide_temporal == "bilstm":
            lines.append(f"  tide_lstm_h:   {args.tide_temporal_lstm_hidden}")
        lines.append(f"  tide_aux:      {args.tide_aux_loss}"
                     + (f" (w={args.tide_aux_weight})" if args.tide_aux_loss else ""))
    if final_per_class:
        lines.append("")
        lines.append("Final per-class F1 (post-training tuned):")
        for cname in cfg.CALL_TYPES_3:
            entry = final_per_class.get(cname, {})
            if entry:
                lines.append(
                    f"  {cname:6} P={entry.get('precision', 0.0):.3f} "
                    f"R={entry.get('recall', 0.0):.3f} "
                    f"F1={entry.get('f1', 0.0):.3f} "
                    f"(TP={entry.get('tp', 0)}, "
                    f"FP={entry.get('fp', 0)}, "
                    f"FN={entry.get('fn', 0)})"
                )
    lines.append("")

    with open(run_dir / "run_info.txt", "w") as f:
        f.write("\n".join(lines))


def update_latest_symlink(run_dir: Path) -> None:
    """
    Maintain ``runs/latest`` → most recent run. Best-effort: silently
    skips on platforms or filesystems where symlinks aren't supported.
    """
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # Relative target keeps the symlink portable if the runs/ dir
        # is moved around.
        latest.symlink_to(run_dir.name)
    except (OSError, NotImplementedError):
        pass  # Symlinks not available; just skip.


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    utils.seed_everything(cfg.SEED, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup
    # ------------------------------------------------------------------
    interventions: list[str] = ["from_scratch" if not args.pretrained else "pretrained"]
    if cfg.USE_WEIGHTED_BCE:
        interventions.append("weighted_bce")
    if cfg.USE_FOCAL_LOSS:
        interventions.append("focal_loss")
    if args.model == "tide":
        interventions.append("tide_gate" if not args.tide_disabled else "tide_ablation")
        if args.tide_aux_loss:
            interventions.append("tide_aux_loss")

    wandb_config = {
        "lr": cfg.LR,
        "weight_decay": cfg.WEIGHT_DECAY,
        "batch_size": cfg.BATCH_SIZE,
        "epochs": cfg.EPOCHS,
        "seed": cfg.SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "use_3class": cfg.USE_3CLASS,
        "n_classes": cfg.n_classes(),
        "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
        "use_focal_loss": cfg.USE_FOCAL_LOSS,
        "focal_alpha": cfg.FOCAL_ALPHA if cfg.USE_FOCAL_LOSS else None,
        "focal_gamma": cfg.FOCAL_GAMMA if cfg.USE_FOCAL_LOSS else None,
        "lstm_hidden": cfg.LSTM_HIDDEN,
        "lstm_layers": cfg.LSTM_LAYERS,
        "train_sites": list(cfg.TRAIN_DATASETS),
        "val_sites": list(cfg.VAL_DATASETS),
        "grad_clip": cfg.GRAD_CLIP,
        "resample_every": RESAMPLE_EVERY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "min_lr": MIN_LR,
        "pretrained": args.pretrained,
        "freeze_epochs": args.freeze_epochs,
    }

    # Use the model name (not a double-prefixed "triton_triton") as the
    # wandb run label; the structured run name carries the rest.
    run_label = args.model  # "triton" or "tide"
    run = wbu.init_run(
        name=run_label,
        config=wandb_config,
        interventions=interventions,
        notes=(
            f"{args.model} training"
            + (f" — {args.name}" if args.name else "")
        ),
    )

    run_name = build_run_name(args)
    run_dir = Path(cfg.OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(run_dir)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, train_loader, val_loader = build_dataloaders()

    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model + loss
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model, criterion, model_cfg_dump = build_model_and_loss(args, device)

    # Update wandb config with model-specific knobs.
    import wandb
    wandb.config.update(model_cfg_dump, allow_val_change=True)

    # Snapshot the full run config to disk so the run dir is
    # self-contained without needing wandb access.
    save_run_config(run_dir, args, model_cfg_dump, wandb_config)

    # Initialise the lazy projection layer before any checkpoint load.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    # Optional pretrained encoder.
    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get("encoder_state_dict",
                         ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)}")

    # Multi-GPU is transparent; checkpoints save the unwrapped state.
    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    utils.log_param_count(model)

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    print(f"Scheduler: ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"Negative resampling: every {RESAMPLE_EVERY} epochs")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    epoch = 0  # so finalize can read it if we never enter the loop

    # Thresholds are always 3-long, regardless of whether the model
    # itself outputs 3 or 7 classes. validate() collapses 7→3 before
    # postprocessing.
    thresholds = torch.tensor(
        cfg.DEFAULT_THRESHOLDS[:3] if len(cfg.DEFAULT_THRESHOLDS) >= 3
        else [0.5, 0.5, 0.5],
        device=device,
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}\n"
              f"{'=' * 60}")

        # Periodic negative resampling.
        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **utils.seeded_dataloader_kwargs(cfg.SEED + epoch),
            )

        apply_freeze_schedule(model, epoch, args)

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
        )

        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts,
            tune_thresholds=True,
        )

        # Carry the tuned thresholds forward as next epoch's starting
        # point — coordinate descent stays stable across epochs because
        # the optimum drifts slowly.
        thresholds = torch.tensor(val["thresholds"], device=device,
                                  dtype=torch.float32)

        scheduler.step(val["mean_f1"])

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}  Best F1: {best_f1:.3f}")
        print(f"  Tuned thresholds: {['%.2f' % t for t in val['thresholds']]}")

        # Wandb log
        wandb_payload = {
            "lr": current_lr,
            "train/loss": train_loss,
            "val/loss": val["loss"],
            "val/f1_macro": val["mean_f1"],
        }
        for ci, cname in enumerate(cfg.CALL_TYPES_3):
            pc = val["per_class"].get(cname, {})
            wandb_payload[f"val/f1/{cname}"] = pc.get("f1", 0.0)
            wandb_payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
            wandb_payload[f"val/recall/{cname}"] = pc.get("recall", 0.0)
            wandb_payload[f"val/tp/{cname}"] = pc.get("tp", 0)
            wandb_payload[f"val/fp/{cname}"] = pc.get("fp", 0)
            wandb_payload[f"val/fn/{cname}"] = pc.get("fn", 0)
            wandb_payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
        wbu.log_epoch(epoch, wandb_payload)

        # Classifier-bias diagnostic. Stuck biases across many epochs
        # usually mean the optimizer isn't actually updating weights
        # (e.g. learning rate too small, or a stale checkpoint loaded).
        clf = utils.unwrap(model).classifier
        bias_str = ", ".join(f"{b:+.2f}" for b in clf.bias.detach().cpu().tolist())
        print(f"  Classifier bias: [{bias_str}]")

        # Save checkpoint (always unwrap DataParallel first).
        ckpt = {
            "epoch": epoch,
            "model_state_dict": utils.unwrap(model).state_dict(),
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
            "model_type": args.model,
            "tide_cfg": (
                None if args.model == "triton"
                else utils.unwrap(model).tide_cfg.to_dict()
            ),
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            best_epoch = epoch
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

    # ------------------------------------------------------------------
    # Post-training threshold tuning
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    inner = utils.unwrap(model)
    inner.load_state_dict(best_ckpt["model_state_dict"])

    tuned = tune_thresholds_event_level(
        inner, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    final_state = inner.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "thresholds": torch.tensor(tuned),
        "model_type": args.model,
        "tide_cfg": (
            None if args.model == "triton"
            else inner.tide_cfg.to_dict()
        ),
    }, run_dir / "final_model.pt")

    # One final validation pass at the post-training-tuned thresholds
    # so the run summary records the actual final per-class metrics
    # (not the in-training tuned ones, which may differ slightly).
    print(f"\n{'=' * 60}\nFinal evaluation at tuned thresholds\n{'=' * 60}")
    final_val = validate(
        inner, spec_extractor, val_loader, criterion, device,
        torch.tensor(tuned, device=device, dtype=torch.float32),
        val_annotations, file_start_dts,
        tune_thresholds=False,
    )
    final_per_class = final_val.get("per_class", {})

    # Human-readable run summary.
    early_stopped = no_improve_epochs >= EARLY_STOP_PATIENCE
    write_run_info(
        run_dir, args,
        best_f1=best_f1,
        best_epoch=best_epoch,
        epochs_run=epoch,
        tuned_thresholds=tuned,
        early_stopped=early_stopped,
        final_per_class=final_per_class,
    )

    print(f"\nDone. Best F1 (in-training tuned): {best_f1:.3f}")
    print(f"     Final F1 (post-tune): {final_val.get('mean_f1', 0.0):.3f}")
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Wandb finalisation
    # ------------------------------------------------------------------
    verdict = (
        f"{run_label} finished at best F1 {best_f1:.3f} "
        f"(epoch {best_epoch} of {epoch} run; "
        f"final tuned thresholds {[round(float(t), 2) for t in tuned]})."
    )
    wbu.finalize_run(
        best_f1=best_f1,
        best_epoch=best_epoch,
        epochs_run=epoch,
        verdict=verdict,
        extra_summary={
            "final_thresholds": list(map(float, tuned)),
            "early_stopped": no_improve_epochs >= EARLY_STOP_PATIENCE,
        },
        artifact_paths=[run_dir / "best_model.pt", run_dir / "final_model.pt"],
        artifact_aliases=["best", run_label],
        artifact_metadata={
            "model": args.model,
            "best_f1": float(best_f1),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epoch),
            "tuned_thresholds": list(map(float, tuned)),
            "pretrained": args.pretrained,
        },
    )


if __name__ == "__main__":
    main()
