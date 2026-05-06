"""
Supervised Training Loop — WhaleVAD-BPN
=======================================

BPN-aware mirror of ``train.py``. Trains the WhaleVAD-BPN model from
``model_bpn.py``, with all architecture knobs exposed as CLI flags so
the design unknowns from the paper can be ablated without code changes.

Differences from ``train.py``:
  - Imports ``WhaleVADBPN`` and ``WhaleVADBPNLoss`` from ``model_bpn``.
  - Builds a ``BPNConfig`` from CLI flags before constructing the model.
  - Validation reads ``probs`` directly from the model's dict output
    (the BPN gate has already applied sigmoid internally), instead of
    sigmoid-on-logits.
  - Loss takes the model output dict whole, not just logits.
  - Adds optional gate-mean diagnostic to validation print.
  - Wandb hooks are kept structurally identical for now; user will
    register a phase name in ``wandb_utils.PHASE_REGISTRY`` later. If
    you want to disable wandb for this run, set ``WANDB_MODE=disabled``
    in the environment.

Usage
-----
Default reproduction with sensible BPN defaults (n_taps=3, R=4,
softmax pool, near-one init, BiLSTM temporal):

    python train_bpn.py

With a specific ablation:

    python train_bpn.py --bpn-rois 1 --bpn-pool sigmoid

Architecture-only run (dilated depthwise + spatial dropout, no BPN):

    python train_bpn.py --no-bpn

With auxiliary gate loss enabled:

    python train_bpn.py --bpn-aux-loss --bpn-aux-weight 0.2

With 4-class D-split (composes with the existing patch):

    python train_bpn.py --4class-d-split
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
from model_bpn import (
    WhaleVADBPN, WhaleVADBPNLoss, BPNConfig, compute_class_weights,
)
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level, collapse_probs_to_3class,
)


# ======================================================================
# Stabilization hyperparameters — copied verbatim from train.py
# ======================================================================

RESAMPLE_EVERY = 5
EARLY_STOP_PATIENCE = 25
LR_PATIENCE = 8
LR_FACTOR = 0.5
MIN_LR = 1e-7


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train WhaleVAD-BPN with parameterised BPN config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inherited from train.py — same flags so the CLI experience matches.
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to a contrastive-pretrained encoder.")
    p.add_argument("--freeze_epochs", type=int, default=0,
                   help="Freeze encoder for this many epochs (with --pretrained).")
    p.add_argument("--4class-d-split", dest="four_class_d_split",
                   action="store_true",
                   help="Train with 4 output classes (bmabz, bmd, bpd, bp), "
                        "collapse bmd+bpd back to d at validation.")

    # BPN-specific tunables. All have defensible defaults per the paper.
    p.add_argument("--no-bpn", dest="bpn_enabled", action="store_false",
                   default=True,
                   help="Disable the BPN gate. Keeps the dilated-depthwise "
                        "architecture changes; useful for isolating BPN "
                        "contribution in ablations.")
    p.add_argument("--bpn-taps", type=int, default=3,
                   help="Number of intermediate feature maps tapped from "
                        "the dilated depthwise block.")
    p.add_argument("--bpn-rois", type=int, default=4,
                   help="ROIs per head (R from the paper).")
    p.add_argument("--bpn-dim", type=int, default=64,
                   help="Internal channel dimension (C_bpn from the paper).")
    p.add_argument("--bpn-pool", choices=["softmax", "sigmoid", "mean"],
                   default="softmax",
                   help="How per-ROI gate scores are combined.")
    p.add_argument("--bpn-pool-scope", choices=["framewise", "global"],
                   default="framewise",
                   help="Whether the learned head weights vary per-frame "
                        "or are global (single vector for the whole sequence).")
    p.add_argument("--bpn-init", choices=["near_one", "random"],
                   default="near_one",
                   help="Gate initialisation. 'near_one' starts with the "
                        "gate at ~1 everywhere so training begins as if "
                        "BPN is absent.")
    p.add_argument("--bpn-temporal", choices=["bilstm", "lr", "none"],
                   default="bilstm",
                   help="Temporal model applied to ROI vectors.")
    p.add_argument("--bpn-lstm-hidden", type=int, default=64,
                   help="BPN BiLSTM hidden size per direction.")
    p.add_argument("--bpn-lstm-layers", type=int, default=1,
                   help="BPN BiLSTM number of layers.")
    p.add_argument("--bpn-spatial-dropout", type=float, default=0.2,
                   help="Spatial dropout in the dilated depthwise block.")
    p.add_argument("--bpn-dropout", type=float, default=0.2,
                   help="Spatial dropout inside the proposal network.")
    p.add_argument("--bpn-aux-loss", action="store_true",
                   help="Add auxiliary BCE loss on the gate against per-frame "
                        "any-class-active targets.")
    p.add_argument("--bpn-aux-weight", type=float, default=0.1,
                   help="Weight of the auxiliary gate loss (if enabled).")
    p.add_argument("--bpn-dilations", type=str, default="2,4,8",
                   help="Comma-separated time-axis dilations per layer.")

    return p.parse_args()


def build_bpn_cfg(args: argparse.Namespace) -> BPNConfig:
    """Translate argparse namespace into a BPNConfig dataclass."""
    dilations = tuple(int(d) for d in args.bpn_dilations.split(","))
    return BPNConfig(
        enabled=args.bpn_enabled,
        n_taps=args.bpn_taps,
        n_rois=args.bpn_rois,
        dim=args.bpn_dim,
        pool_mode=args.bpn_pool,
        pool_scope=args.bpn_pool_scope,
        init_mode=args.bpn_init,
        temporal=args.bpn_temporal,
        lstm_hidden=args.bpn_lstm_hidden,
        lstm_layers=args.bpn_lstm_layers,
        spatial_dropout_p=args.bpn_spatial_dropout,
        bpn_dropout_p=args.bpn_dropout,
        aux_loss=args.bpn_aux_loss,
        aux_weight=args.bpn_aux_weight,
        dilations=dilations,
    )


def set_seed(seed: int = cfg.SEED):
    wbu.seed_everything(seed, deterministic=False)


# ======================================================================
# Helpers (verbatim from train.py)
# ======================================================================

def align_lengths(probs, targets, mask):
    """Reconcile small length mismatches between model output and targets."""
    T_m, T_t = probs.size(1), targets.size(1)
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


def align_outputs(outputs: dict, targets, mask):
    """Apply align_lengths consistently to all tensors in the BPN output dict."""
    targets, mask = align_lengths(outputs["probs"], targets, mask)
    return outputs, targets, mask


# ======================================================================
# Validation — mirrors train.py.validate, adapted for BPN dict output
# ======================================================================

@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             thresholds, val_annotations, file_start_dts,
             tune_thresholds: bool = True):
    """Event-level validation. Returns dict with loss, mean_f1, per_class, thresholds, mask_mean."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_probs: dict = {}
    mask_mean_accum = 0.0
    mask_n = 0

    for audio, targets, mask, metas in tqdm(loader, desc="Validating", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        outputs = model(spec)
        outputs, targets, mask = align_outputs(outputs, targets, mask)

        total_loss += criterion(outputs, targets, mask).item()
        n_batches += 1

        # Diagnostic: track average gate value over valid frames.
        if "mask" in outputs:
            m_valid = outputs["mask"][mask]
            if m_valid.numel() > 0:
                mask_mean_accum += m_valid.mean().item()
                mask_n += 1

        # The probs tensor is already gated (sigmoid + multiplicative gate).
        # Stash for event-level evaluation.
        probs = outputs["probs"].cpu().numpy()
        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # 4-class collapse to 3-class for evaluation, when applicable.
    all_probs = collapse_probs_to_3class(all_probs)

    # Build ground-truth Detection objects (always 3-class).
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

    # Per-class threshold sweep — coordinate descent, identical to train.py.
    used_thresholds = np.asarray(thresholds.cpu().numpy(),
                                 dtype=np.float64).copy()
    if tune_thresholds:
        grids = [
            np.arange(0.20, 0.85, 0.05),                                # bmabz
            np.concatenate([np.arange(0.05, 0.5, 0.05),
                            np.arange(0.5, 0.85, 0.10)]),               # d
            np.concatenate([np.arange(0.05, 0.5, 0.05),
                            np.arange(0.5, 0.85, 0.10)]),               # bp
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
            print(f"    {name.upper():<6}                                       "
                  f"P=0.000 R=0.000 F1=0.000")
            continue
        print(f"    {name.upper():<6} t={used_thresholds[c]:.2f}  "
              f"TP={m['tp']:5d} FP={m['fp']:5d} FN={m['fn']:5d}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}")

    avg_mask = mask_mean_accum / max(mask_n, 1)
    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_f1": overall_f1,
        "per_class": metrics,
        "thresholds": used_thresholds.tolist(),
        "mask_mean": avg_mask,
    }


# ======================================================================
# Training loop step (one epoch)
# ======================================================================

def train_epoch(model, spec_extractor, loader, criterion,
                optimizer, device, epoch):
    """One training epoch over the supplied loader."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        spec = spec_extractor(audio)
        outputs = model(spec)
        outputs, targets, mask = align_outputs(outputs, targets, mask)
        loss = criterion(outputs, targets, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            print("*** NaN/Inf loss — skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    bpn_cfg = build_bpn_cfg(args)

    # Apply --4class-d-split before any class-count-dependent code runs.
    if getattr(args, "four_class_d_split", False):
        cfg.USE_4CLASS_D_SPLIT = True
        print("[train_bpn] USE_4CLASS_D_SPLIT enabled — 4 output classes "
              "(bmabz, bmd, bpd, bp), 3-class eval after bmd+bpd collapse")

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"BPN config: {bpn_cfg.to_dict()}")

    # ------------------------------------------------------------------
    # Single phase id "5" — this whole effort is a search over the BPN
    # configuration space, not a sequence of decided ablations. Axes
    # that vary across runs (gate on/off, R, pool mode, init, taps,
    # dilations, aux loss) are stamped as tags + config fields so the
    # wandb runs table becomes the search log: filter and sort to find
    # which combination wins.
    # ------------------------------------------------------------------
    phase_id = "5"

    # Tags. The per-axis discriminators that change across the search
    # run table — these are how the prof navigates the search log.
    extra_tags: list[str] = []

    # Architectural axis (paper Section V-B): with vs without gate.
    extra_tags.append("with_gate" if bpn_cfg.enabled else "no_gate")

    # Loss flags.
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")

    # Class taxonomy.
    if getattr(cfg, "USE_4CLASS_D_SPLIT", False):
        extra_tags.append("4class_d_split")

    # BPN-axis tags — only stamped when the gate is on, so 'no_gate'
    # runs aren't polluted with irrelevant config tags.
    if bpn_cfg.enabled:
        extra_tags.extend([
            f"R{bpn_cfg.n_rois}",
            f"taps{bpn_cfg.n_taps}",
            f"pool_{bpn_cfg.pool_mode}",
            f"poolscope_{bpn_cfg.pool_scope}",
            f"init_{bpn_cfg.init_mode}",
            f"temporal_{bpn_cfg.temporal}",
        ])
        if bpn_cfg.aux_loss:
            extra_tags.append("aux_gate_loss")

    # Pretrained encoder tagging — same logic as train.py.
    pretrain_phase = None
    if args.pretrained:
        import re
        m = re.search(r"(3[a-z])", str(args.pretrained))
        pretrain_phase = m.group(1) if m else "unknown"
        extra_tags.append("pretrained")
        extra_tags.append(f"pretrained_{pretrain_phase}")
    else:
        extra_tags.append("from_scratch")

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
            "use_4class_d_split": getattr(cfg, "USE_4CLASS_D_SPLIT", False),
            "n_classes":        cfg.n_classes(),
            "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
            "use_focal_loss":   getattr(cfg, "USE_FOCAL_LOSS", False),
            "focal_alpha":      getattr(cfg, "FOCAL_ALPHA", None),
            "focal_gamma":      getattr(cfg, "FOCAL_GAMMA", None),
            "lstm_hidden":      cfg.LSTM_HIDDEN,
            "lstm_layers":      cfg.LSTM_LAYERS,
            "train_sites":      list(cfg.TRAIN_DATASETS),
            "val_sites":        list(cfg.VAL_DATASETS),
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr_patience":      LR_PATIENCE,
            "lr_factor":        LR_FACTOR,
            "min_lr":           MIN_LR,
            "pretrained":       args.pretrained,
            "pretrain_phase":   pretrain_phase,
            "freeze_epochs":    args.freeze_epochs,
            # BPN config — flattened *and* nested so the wandb runs
            # table can sort/filter by any individual axis (e.g. group
            # by ``config.bpn_n_rois`` to see how F1 responds to R).
            "bpn":              bpn_cfg.to_dict(),
            "bpn_enabled":      bpn_cfg.enabled,
            "bpn_n_rois":       bpn_cfg.n_rois,
            "bpn_n_taps":       bpn_cfg.n_taps,
            "bpn_pool_mode":    bpn_cfg.pool_mode,
            "bpn_pool_scope":   bpn_cfg.pool_scope,
            "bpn_init_mode":    bpn_cfg.init_mode,
            "bpn_temporal":     bpn_cfg.temporal,
            "bpn_aux_loss":     bpn_cfg.aux_loss,
            "bpn_aux_weight":   bpn_cfg.aux_weight,
            "bpn_dilations":    list(bpn_cfg.dilations),
            "bpn_dim":          bpn_cfg.dim,
            "bpn_lstm_hidden":  bpn_cfg.lstm_hidden,
            "bpn_lstm_layers":  bpn_cfg.lstm_layers,
            "bpn_spatial_dropout": bpn_cfg.spatial_dropout_p,
            "bpn_dropout":      bpn_cfg.bpn_dropout_p,
        },
    )

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase{phase_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data — same dataloaders as the baseline; the BPN doesn't need its own.
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
    model = WhaleVADBPN(
        num_classes=cfg.n_classes(), bpn_cfg=bpn_cfg,
    ).to(device)

    # Initialise lazy projection.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    # Optional pretrained encoder. The state_dict for the BPN model is
    # NOT identical to the baseline because the depthwise block has
    # different parameter names (``dilated_depthwise.layers.N.*``).
    # Loading a baseline-trained encoder with strict=False simply skips
    # those keys, which is fine — the dilated block trains from scratch.
    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device,
                          weights_only=False)
        state = ckpt.get("encoder_state_dict",
                         ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)} keys (expected — BPN/dilated layers "
              f"start fresh)")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)}")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss + optimizer
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADBPNLoss(
        pos_weight=pos_weight, bpn_cfg=bpn_cfg,
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    print(f"DEBUG class weights: "
          f"{pos_weight.tolist() if pos_weight is not None else 'None'}")
    print(f"Scheduler: ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"Negative resampling: every {RESAMPLE_EVERY} epochs")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
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

        # Negative resampling.
        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **wbu.seeded_dataloader_kwargs(cfg.SEED + epoch),
            )

        # Encoder freeze schedule (if pretrained).
        if args.pretrained and epoch <= args.freeze_epochs:
            for name, p in model.named_parameters():
                # Train only the LSTM, classifier, and the BPN itself —
                # the BPN doesn't receive pretrained weights anyway.
                if not any(k in name for k in
                           ["classifier", "lstm", "bpn"]):
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
        print(f"  Gate mean: {val['mask_mean']:.3f}  "
              f"(1.0 = no gating; lower means BPN is suppressing more)")

        # Wandb log.
        import wandb
        wandb_payload = {
            "epoch":         epoch,
            "lr":            current_lr,
            "train/loss":    train_loss,
            "val/loss":      val["loss"],
            "val/f1_macro":  val["mean_f1"],
            "val/gate_mean": val["mask_mean"],
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

        # Classifier bias diagnostic.
        clf_module = (model.module.classifier
                      if isinstance(model, nn.DataParallel)
                      else model.classifier)
        bias_str = ", ".join(f"{b:+.2f}"
                             for b in clf_module.bias.detach().cpu().tolist())
        print(f"  Classifier bias: [{bias_str}]")

        model_state = (model.module.state_dict()
                       if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
            "bpn_cfg": bpn_cfg.to_dict(),
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")
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

    # ------------------------------------------------------------------
    # Post-training threshold tuning on best checkpoint.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = (model.module if isinstance(model, nn.DataParallel)
                     else model)
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
        "bpn_cfg": bpn_cfg.to_dict(),
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Wandb final summary + artifact.
    # ------------------------------------------------------------------
    import wandb
    wandb.summary["best_f1"] = float(best_f1)
    wandb.summary["best_f1_post_tuning"] = float(best_ckpt.get("best_f1", best_f1))
    wandb.summary["final_thresholds"] = list(map(float, tuned))
    wandb.summary["epochs_run"] = epoch
    wandb.summary["early_stopped"] = no_improve_epochs >= EARLY_STOP_PATIENCE
    wandb.summary["bpn_enabled"] = bpn_cfg.enabled
    wandb.summary["verdict"] = (
        f"Phase {phase_id} run finished at best F1 {best_f1:.3f} "
        f"(epoch {best_ckpt.get('epoch', '?')} of {epoch}; "
        f"tuned thresholds {[round(float(t),2) for t in tuned]}; "
        f"BPN config: enabled={bpn_cfg.enabled}, taps={bpn_cfg.n_taps}, "
        f"R={bpn_cfg.n_rois}, pool={bpn_cfg.pool_mode}, "
        f"init={bpn_cfg.init_mode}, aux_loss={bpn_cfg.aux_loss}). "
        f"Reference baseline F1=0.474."
    )

    art = wandb.Artifact(
        f"model-{run.name}", type="model",
        metadata={
            "best_f1": float(best_f1),
            "best_epoch": int(best_ckpt.get("epoch", 0)),
            "epochs_run": int(epoch),
            "tuned_thresholds": list(map(float, tuned)),
            "bpn_cfg": bpn_cfg.to_dict(),
            "phase": phase_id,
            "pretrained": args.pretrained,
        },
    )
    art.add_file(str(run_dir / "best_model.pt"))
    art.add_file(str(run_dir / "final_model.pt"))
    run.log_artifact(art, aliases=["best", phase_id])

    wandb.finish()


if __name__ == "__main__":
    main()
