"""
Phase 8: Per-Class Focal-γ Fine-Tune
=====================================

Continue training a converged WhaleVAD / WhaleVAD-BPN checkpoint with a
focal loss whose focusing parameter γ is class-specific. The default
recipe pushes D's γ from 2.0 → 4.0 while leaving bmabz and bp at 2.0.
Mechanism: the focal modulation ``(1 - p_t)^γ`` becomes more aggressive
for D specifically, applying stronger gradient pressure on hard D
positives -- exactly the regime where the current ensemble is leaking
recall (current D recall ≈ 0.19, vs bmabz ≈ 0.58 and bp ≈ 0.47).

Why this is justified
---------------------
Standard focal loss uses a single scalar γ across classes. When one class
is dominated by easy negatives at a far higher ratio than the others,
the modulation ``(1 - p_t)^γ`` can be too gentle on the rare positives
that the model has not yet learned. The BioDCASE community paper notes
that across all submitted models, recall stays below 50% on average and
identifies class imbalance and low-SNR vocalisations as the leading
causes. Per-class γ is the cheapest direct intervention for that:
no architectural change, no extra data, just a sharper loss for D.

What this script does
---------------------
Auto-detects baseline (WhaleVAD) vs BPN (WhaleVADBPN) from the source
checkpoint, mirroring train_phase_hnm.py. Uses the same dataset shape
(positives + resampled randoms, with hard-negs optional). Replaces the
focal loss with a per-class version local to this script -- the project's
WhaleVADLoss and WhaleVADBPNLoss are left untouched so other phases are
unaffected.

Source-checkpoint choice
------------------------
You have two reasonable choices, both supported:

1. **Pre-HNM source checkpoints** (e.g. ``runs/phase5_*/best_model.pt`` or
   ``runs/whalevad_*/best_model.pt``). Cleanest experiment: per-class γ
   *replaces* HNM rather than stacking with it. Lets you compare:
       per-class γ alone   vs   HNM alone   vs   both
   Direct ablation, three Phase 8 runs from the same source = three new
   ensemble members.

2. **HNM_D checkpoints** (e.g. ``runs/hnm_D_*/best_model.pt``). Stacks
   per-class γ on top of HNM. Higher ceiling if both interventions are
   orthogonal; risk: HNM has already pushed precision, more focal
   aggression on D could overshoot. Use ``--lr 5e-6`` for safety.

Optional --hard_negatives lets you also include hard negatives in this
run (same as Phase 6), giving you a single-shot "HNM + per-class γ"
training.

Usage
-----
::

    # Default: γ_d=4.0, fine-tune from one HNM_D model:
    python train_phase8.py \\
        --checkpoint runs/hnm_D_phase5_20260507_211504/best_model.pt \\
        --lr 5e-6

    # Sweep γ_d to find the best value:
    for GD in 2.5 3.0 4.0 5.0; do
        python train_phase8.py \\
            --checkpoint runs/hnm_D_phase5_20260507_211504/best_model.pt \\
            --gamma_d $GD --lr 5e-6 \\
            --run_name phase8_gd${GD}_hnm
    done

    # Compare per-class γ vs HNM by starting from the pre-HNM source:
    python train_phase8.py \\
        --checkpoint runs/phase5_20260507_211504/best_model.pt \\
        --gamma_d 4.0 --lr 1e-5

Wandb registry note
-------------------
Add this entry to ``PHASE_REGISTRY`` in ``wandb_utils.py``::

    "8": dict(
        parent="baseline",
        hypothesis=("Per-class focal γ fine-tune. The standard focal "
                    "loss uses a scalar γ=2 across all classes; this "
                    "phase replaces it with a per-class γ tensor that "
                    "boosts D specifically (γ_d=4.0 by default) to "
                    "apply sharper gradient pressure on hard D positives. "
                    "Direct attack on the recall bottleneck identified "
                    "in the BioDCASE community paper."),
        interventions=["per_class_focal_gamma"],
    ),

Without that entry, ``wbu.init_phase("8", ...)`` will raise.
"""

from __future__ import annotations
import argparse
import re
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
from dataset import (
    WhaleDataset, build_negative_segments, build_positive_segments,
    build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from model import compute_class_weights
from postprocess import Detection, collapse_probs_to_3class
from spectrogram import SpectrogramExtractor
from train_phase0e import extend_segment_to_fixed_length, PHASE0E_SEGMENT_S
from ensemble_predict import (
    build_model_for_ckpt, predict_probabilities,
    tune_thresholds_on_probs, evaluate_with_thresholds,
)

# Optional HNM integration. Imported lazily so Phase 8 is usable on a
# system where the HNM files were not staged.
from train_phase_hnm import (
    load_hard_negatives_json, build_hard_negative_segments,
    build_hard_neg_class_map, build_class_mask, HnmTrainingDataset,
)


# ======================================================================
# Constants
# ======================================================================

PHASE8_DEFAULT_LR = 1e-5
PHASE8_DEFAULT_EPOCHS = 15
PHASE8_DEFAULT_OVERSAMPLE = 5
PHASE8_RESAMPLE_EVERY = 5
PHASE8_EARLY_STOP = 8

# Default focal hyperparameters. γ_d > γ_other applies the per-class
# treatment to D only. α stays at the standard 0.25.
PHASE8_DEFAULT_GAMMA = {"bmabz": 2.0, "d": 4.0, "bp": 2.0}
PHASE8_DEFAULT_ALPHA = 0.25


# ======================================================================
# Per-class focal loss
# ======================================================================

def build_gamma_tensor(g_bmabz: float, g_d: float, g_bp: float,
                       device: torch.device) -> torch.Tensor:
    """Length-3 γ tensor aligned to ``cfg.CALL_TYPES_3``."""
    # Order has to match cfg.CALL_TYPES_3 exactly so broadcasting is correct.
    name_to_val = {"bmabz": g_bmabz, "d": g_d, "bp": g_bp}
    vals = [name_to_val[c] for c in cfg.CALL_TYPES_3]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def build_alpha_tensor(a_bmabz: float, a_d: float, a_bp: float,
                       device: torch.device) -> torch.Tensor:
    """Length-3 α tensor aligned to ``cfg.CALL_TYPES_3``."""
    name_to_val = {"bmabz": a_bmabz, "d": a_d, "bp": a_bp}
    vals = [name_to_val[c] for c in cfg.CALL_TYPES_3]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def per_class_focal_bce_on_probs(
    probs: torch.Tensor,            # (B, T, C) in [0, 1]
    targets: torch.Tensor,          # (B, T, C)
    mask: torch.Tensor,             # (B, T) bool
    class_mask: torch.Tensor,       # (B, C) float, for PGI
    gamma: torch.Tensor,            # (C,) float
    alpha: torch.Tensor,            # (C,) float
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Element-wise focal BCE on probabilities with per-class γ and α.

    Mirrors ``train_phase_hnm.probs_bce_focal_loss`` exactly except γ
    and α broadcast over the class axis. Used for BPN models, whose
    ``model(spec)["probs"]`` are already sigmoid'd and gated.
    """
    eps = 1e-7
    p = probs.clamp(eps, 1.0 - eps)

    # γ and α broadcast as (1, 1, C); p_t and α_t are (B, T, C).
    g = gamma.view(1, 1, -1)
    a = alpha.view(1, 1, -1)

    p_t = targets * p + (1.0 - targets) * (1.0 - p)
    a_t = targets * a + (1.0 - targets) * (1.0 - a)
    focal_w = (1.0 - p_t).pow(g)
    weight = focal_w * a_t

    pos_term = targets * torch.log(p)
    neg_term = (1.0 - targets) * torch.log(1.0 - p)
    if pos_weight is not None:
        pos_term = pos_term * pos_weight.view(1, 1, -1)
    pos_term = pos_term * weight
    neg_term = neg_term * weight

    per_elem = -(pos_term + neg_term)

    time_valid = mask.unsqueeze(-1).float()      # (B, T, 1)
    class_valid = class_mask.unsqueeze(1)        # (B, 1, C)
    valid = time_valid * class_valid             # (B, T, C)
    per_elem = per_elem * valid
    return per_elem.sum() / valid.sum().clamp(min=1.0)


def per_class_focal_bce_on_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    class_mask: torch.Tensor, gamma: torch.Tensor, alpha: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Logit-space wrapper. Used for baseline (non-BPN) models."""
    p = torch.sigmoid(logits)
    return per_class_focal_bce_on_probs(
        p, targets, mask, class_mask, gamma, alpha, pos_weight=pos_weight,
    )


# ======================================================================
# Forward + loss dispatch (baseline vs BPN, both with per-class γ)
# ======================================================================

def forward_and_loss(
    model: nn.Module, model_type: str,
    spec_extractor: SpectrogramExtractor,
    audio: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    class_mask: torch.Tensor,
    gamma: torch.Tensor, alpha: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    spec = spec_extractor(audio)
    out = model(spec)
    if model_type == "bpn":
        probs = out["probs"]
        T = min(probs.size(1), targets.size(1))
        return per_class_focal_bce_on_probs(
            probs[:, :T], targets[:, :T], mask[:, :T], class_mask,
            gamma, alpha, pos_weight=pos_weight,
        )
    logits = out
    T = min(logits.size(1), targets.size(1))
    return per_class_focal_bce_on_logits(
        logits[:, :T], targets[:, :T], mask[:, :T], class_mask,
        gamma, alpha, pos_weight=pos_weight,
    )


# ======================================================================
# Training and validation loops
# ======================================================================

def train_epoch(
    model, model_type, spec_extractor, loader,
    gamma, alpha, pos_weight, optimizer, device,
    hard_neg_class_map, n_classes, use_class_mask,
):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        class_mask = (build_class_mask(metas, hard_neg_class_map, n_classes,
                                       device)
                      if use_class_mask
                      else torch.ones(audio.size(0), n_classes, device=device))
        optimizer.zero_grad()
        loss = forward_and_loss(
            model, model_type, spec_extractor, audio, targets, mask,
            class_mask, gamma, alpha, pos_weight,
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_phase8(
    model, model_type, spec_extractor, val_loader, device, gt_events,
    gamma, alpha, pos_weight, n_classes, tune_thresholds=True,
):
    model.eval()
    total_loss, n = 0.0, 0
    for audio, targets, mask, _ in val_loader:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        class_mask = torch.ones(audio.size(0), n_classes, device=device)
        loss = forward_and_loss(
            model, model_type, spec_extractor, audio, targets, mask,
            class_mask, gamma, alpha, pos_weight,
        )
        total_loss += loss.item()
        n += 1
    val_loss = total_loss / max(n, 1)

    all_probs = predict_probabilities(model, model_type, spec_extractor,
                                      val_loader, device)
    all_probs = collapse_probs_to_3class(all_probs)

    if tune_thresholds:
        thresholds = tune_thresholds_on_probs(all_probs, gt_events)
    else:
        thresholds = np.array(cfg.DEFAULT_THRESHOLDS, dtype=np.float64)

    metrics = evaluate_with_thresholds(all_probs, gt_events, thresholds)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
    macro_f1 = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                              for c in cfg.CALL_TYPES_3]))

    print(f"  Val (loss={val_loss:.4f}):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):6} "
              f"FN={m.get('fn', 0):6}  P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} F1={m.get('f1', 0):.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}  MACRO F1={macro_f1:.3f}")

    per_class_only = {k: v for k, v in metrics.items()
                      if k in cfg.CALL_TYPES_3}
    return {"loss": val_loss, "overall_f1": overall_f1, "macro_f1": macro_f1,
            "per_class": per_class_only, "thresholds": thresholds.tolist()}


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Source checkpoint to continue training from. "
                        "Architecture is auto-detected (baseline or BPN). "
                        "Either an HNM_D checkpoint (stacks per-class γ on "
                        "top of HNM) or a pre-HNM source checkpoint (clean "
                        "per-class γ alone).")

    # Per-class focal hyperparameters.
    p.add_argument("--gamma_bmabz", type=float,
                   default=PHASE8_DEFAULT_GAMMA["bmabz"],
                   help=f"Focal γ for the bmabz class. "
                        f"Default: {PHASE8_DEFAULT_GAMMA['bmabz']}.")
    p.add_argument("--gamma_d", type=float,
                   default=PHASE8_DEFAULT_GAMMA["d"],
                   help=f"Focal γ for the D class. "
                        f"Default: {PHASE8_DEFAULT_GAMMA['d']}. "
                        f"This is the main lever for the phase: higher "
                        f"means sharper focal modulation on hard D positives.")
    p.add_argument("--gamma_bp", type=float,
                   default=PHASE8_DEFAULT_GAMMA["bp"],
                   help=f"Focal γ for the bp class. "
                        f"Default: {PHASE8_DEFAULT_GAMMA['bp']}.")

    p.add_argument("--alpha", type=float, default=PHASE8_DEFAULT_ALPHA,
                   help="Scalar α applied to all classes (default 0.25). "
                        "Per-class α not exposed in v1; if you want it, "
                        "pass --alpha_bmabz / --alpha_d / --alpha_bp.")
    p.add_argument("--alpha_bmabz", type=float, default=None)
    p.add_argument("--alpha_d", type=float, default=None)
    p.add_argument("--alpha_bp", type=float, default=None)

    # Training schedule.
    p.add_argument("--epochs", type=int, default=PHASE8_DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=PHASE8_DEFAULT_LR,
                   help="Fine-tune LR. Default 1e-5 from pre-HNM source; "
                        "drop to 5e-6 when continuing from an HNM_D "
                        "checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--select_by", type=str, default="macro",
                   choices=["macro", "overall"],
                   help="Checkpoint selection criterion (default: macro).")

    # Optional hard-negative stacking (Phase 6 + Phase 8 in one run).
    p.add_argument("--hard_negatives", type=str, nargs="*", default=None,
                   help="Optional: one or more mining JSONs to also include "
                        "as hard negatives. When present, this run is "
                        "Phase 6 + Phase 8 stacked. Without it, just "
                        "per-class focal γ on the standard training set.")
    p.add_argument("--isolate_classes", action="store_true",
                   help="PGI (per-class gradient isolation) on hard-neg "
                        "segments. Only meaningful with --hard_negatives.")
    p.add_argument("--oversample", type=int, default=PHASE8_DEFAULT_OVERSAMPLE,
                   help="Hard-neg oversample factor (default 5). Only "
                        "meaningful with --hard_negatives.")
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    wbu.seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Source-phase inference, identical convention to Phase 6.
    # ------------------------------------------------------------------
    source_short = Path(args.checkpoint).parent.name
    src_path = str(args.checkpoint)
    m = re.search(r"phase(\w+)_\d{8}_\d{6}", src_path)
    if m:
        source_phase = m.group(1)
    elif "hnm_D" in src_path:
        source_phase = "6_hnm_D"
    elif "whalevad" in src_path:
        source_phase = "baseline"
    else:
        source_phase = "unknown"
    print(f"  source phase: {source_phase}  (dir: {source_short})")

    # ------------------------------------------------------------------
    # Build per-class γ and α tensors.
    # ------------------------------------------------------------------
    gamma = build_gamma_tensor(args.gamma_bmabz, args.gamma_d, args.gamma_bp,
                               device)
    a_bmabz = args.alpha_bmabz if args.alpha_bmabz is not None else args.alpha
    a_d = args.alpha_d if args.alpha_d is not None else args.alpha
    a_bp = args.alpha_bp if args.alpha_bp is not None else args.alpha
    alpha = build_alpha_tensor(a_bmabz, a_d, a_bp, device)
    print(f"\nPer-class focal config:")
    for i, name in enumerate(cfg.CALL_TYPES_3):
        print(f"  {name:<6} γ={gamma[i].item():.2f}  α={alpha[i].item():.3f}")
    if (gamma == gamma[0]).all() and (alpha == alpha[0]).all():
        print(f"  WARNING: γ and α are identical across classes; this run is "
              f"equivalent to scalar focal. Set --gamma_d differently to "
              f"actually engage Phase 8.")

    # ------------------------------------------------------------------
    # Optional hard negatives (Phase 6 stacking)
    # ------------------------------------------------------------------
    use_hnm = bool(args.hard_negatives)
    if use_hnm:
        fp_records, hnm_meta_list = load_hard_negatives_json(args.hard_negatives)
        targets_used = sorted({m["target_class"] for m in hnm_meta_list})
        fps_per_class = {t: sum(1 for r in fp_records if r["target_class"] == t)
                         for t in targets_used}
        print(f"\nLoaded {len(fp_records)} hard negatives across "
              f"{len(args.hard_negatives)} JSON file(s)")
        for t, n_fps in fps_per_class.items():
            print(f"  {t}: {n_fps} FPs")
        print(f"  isolate_classes (PGI): {args.isolate_classes}")
    else:
        fp_records, hnm_meta_list, targets_used, fps_per_class = [], [], [], {}
        print("\nNo hard negatives: running per-class γ on standard data only.")

    # ------------------------------------------------------------------
    # Wandb init.
    # ------------------------------------------------------------------
    extra_tags = [f"gd_{args.gamma_d:g}", f"gbmabz_{args.gamma_bmabz:g}",
                  f"gbp_{args.gamma_bp:g}",
                  f"select_{args.select_by}",
                  f"source_{source_phase}"]
    if use_hnm:
        extra_tags.append("hnm_stacked")
        for t in targets_used:
            extra_tags.append(f"target_{t}")
        extra_tags.append("pgi_on" if args.isolate_classes else "pgi_off")
        extra_tags.append(f"oversample{args.oversample}")
    else:
        extra_tags.append("hnm_off")
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")

    config_payload = {
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": cfg.BATCH_SIZE,
        "weight_decay": cfg.WEIGHT_DECAY,
        "neg_ratio": cfg.NEG_RATIO,
        "use_3class": cfg.USE_3CLASS,
        "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
        "use_focal_loss": True,
        "phase8_gamma": {c: gamma[i].item()
                         for i, c in enumerate(cfg.CALL_TYPES_3)},
        "phase8_alpha": {c: alpha[i].item()
                         for i, c in enumerate(cfg.CALL_TYPES_3)},
        "train_sites": list(cfg.TRAIN_DATASETS),
        "val_sites": list(cfg.VAL_DATASETS),
        "early_stop_patience": PHASE8_EARLY_STOP,
        "resample_every": PHASE8_RESAMPLE_EVERY,
        "source_checkpoint": str(args.checkpoint),
        "source_short": source_short,
        "source_phase": source_phase,
        "select_by": args.select_by,
        "hnm_stacked": use_hnm,
    }
    if use_hnm:
        config_payload.update({
            "hard_negatives_jsons": [str(p) for p in args.hard_negatives],
            "n_hard_negs_total": len(fp_records),
            "n_hard_negs_per_class": fps_per_class,
            "mining_targets": targets_used,
            "isolate_classes": args.isolate_classes,
            "oversample": args.oversample,
            "mining_meta": hnm_meta_list,
        })

    run = wbu.init_phase("8", config=config_payload,
                         name_suffix=source_short, extra_tags=extra_tags)

    run_name = (args.run_name
                or (f"phase8_gd{args.gamma_d:g}_{source_short}"
                    + ("_hnm" if use_hnm else "")))
    run_dir = Path(cfg.OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------
    print("\nLoading training data...")
    train_anns = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(train_anns, train_manifest)
    manifest_idx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [
        extend_segment_to_fixed_length(
            s, PHASE0E_SEGMENT_S,
            float(manifest_idx.loc[(s.dataset, s.filename), "duration_s"]))
        for s in pos_segs
        if (s.dataset, s.filename) in manifest_idx.index
    ]
    print(f"  {len(pos_segs)} positive segments (30s extended)")

    if use_hnm:
        hard_segs, used_records = build_hard_negative_segments(
            fp_records, train_manifest, train_anns)
        print(f"  {len(hard_segs)} hard-neg segments × oversample "
              f"{args.oversample} = {len(hard_segs) * args.oversample} "
              f"effective copies/epoch")
        hard_neg_class_map = (build_hard_neg_class_map(hard_segs, used_records)
                              if args.isolate_classes else {})
        train_ds = HnmTrainingDataset(pos_segs, hard_segs, args.oversample,
                                      train_manifest, train_anns)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )
    else:
        # No HNM: assemble positives + resampled randoms ourselves. We can
        # reuse HnmTrainingDataset with an empty hard-neg list; that gives
        # us free periodic resampling of negatives.
        hard_segs, used_records, hard_neg_class_map = [], [], {}
        train_ds = HnmTrainingDataset(pos_segs, hard_segs, 0,
                                      train_manifest, train_anns)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(WhaleDataset(val_segs), batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=True)

    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}
    gt_events: list[Detection] = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  model type: {model_type}")
    if model_type == "bpn":
        print(f"  bpn_cfg: {ckpt.get('bpn_cfg')}")

    run.config.update({"model_type": model_type}, allow_val_change=True)
    if model_type == "bpn" and ckpt.get("bpn_cfg") is not None:
        run.config.update({"source_bpn_cfg": ckpt.get("bpn_cfg")},
                          allow_val_change=True)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))

    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False)
    if missing:
        non_bpn_missing = [k for k in missing if "bpn" not in k]
        if non_bpn_missing:
            print(f"  WARNING: missing non-BPN keys: {len(non_bpn_missing)}: "
                  f"{non_bpn_missing[:3]}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}: {unexpected[:3]}")
    print(f"  starting val F1 (per ckpt meta): "
          f"{ckpt.get('best_f1', 'unknown')}")

    pos_weight = (compute_class_weights().to(device)
                  if cfg.USE_WEIGHTED_BCE else None)
    if pos_weight is not None:
        print(f"  pos_weight: {pos_weight.tolist()}")
        run.config.update({"pos_weight": pos_weight.tolist()},
                          allow_val_change=True)

    n_classes = cfg.n_classes()

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY,
                      betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=4, min_lr=1e-7)

    # ------------------------------------------------------------------
    # Initial validation against the new (per-class focal) loss.
    # The pre-tune val F1 should be near the source ckpt's reported F1
    # since the model hasn't changed yet -- this is a sanity check.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nInitial validation (epoch 0)\n{'=' * 60}")
    val0 = validate_phase8(model, model_type, spec_extractor, val_loader,
                           device, gt_events, gamma, alpha, pos_weight,
                           n_classes, tune_thresholds=True)
    val0_log = dict(val0)
    val0_log["f1"] = val0[f"{args.select_by}_f1"]
    wbu.log_epoch_3class(0, float("nan"), val0_log)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = [{"epoch": 0, "train_loss": float("nan"),
                "f1": val0[f"{args.select_by}_f1"], "loss": val0["loss"],
                "per_class": val0["per_class"]}]
    best_f1 = val0[f"{args.select_by}_f1"]
    no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"Phase 8 fine-tune {args.epochs} epochs @ lr={args.lr}, "
          f"select_by={args.select_by}")
    print(f"  starting {args.select_by} F1: {best_f1:.3f}")
    print(f"  γ = {gamma.tolist()}  α = {alpha.tolist()}")
    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch > 1 and epoch % PHASE8_RESAMPLE_EVERY == 0:
            train_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(train_ds.segments)}")

        train_loss = train_epoch(
            model, model_type, spec_extractor, train_loader,
            gamma, alpha, pos_weight, optimizer, device,
            hard_neg_class_map, n_classes,
            use_class_mask=(use_hnm and args.isolate_classes),
        )
        val = validate_phase8(
            model, model_type, spec_extractor, val_loader, device,
            gt_events, gamma, alpha, pos_weight, n_classes,
            tune_thresholds=True,
        )

        selected_f1 = val[f"{args.select_by}_f1"]
        improved = selected_f1 > best_f1
        marker = " *** new best" if improved else ""
        print(f"\nEpoch {epoch:2d}/{args.epochs}  ({time.time()-t0:.0f}s)"
              f"{marker}")
        print(f"  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Val {args.select_by} F1: {selected_f1:.3f}  "
              f"Best: {best_f1:.3f}  "
              f"(macro={val['macro_f1']:.3f}, "
              f"overall={val['overall_f1']:.3f})")
        print(f"  Tuned thresholds: "
              f"{['%.2f' % t for t in val['thresholds']]}")

        scheduler.step(selected_f1)

        val_log = dict(val); val_log["f1"] = selected_f1
        wbu.log_epoch_3class(epoch, train_loss, val_log)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "f1": selected_f1, "loss": val["loss"],
                        "per_class": val["per_class"]})

        ckpt_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_f1": max(best_f1, selected_f1),
            "macro_f1": val["macro_f1"],
            "overall_f1": val["overall_f1"],
            "thresholds": torch.tensor(val["thresholds"]),
            "phase8_gamma": gamma.tolist(),
            "phase8_alpha": alpha.tolist(),
            "select_by": args.select_by,
            "source_checkpoint": str(args.checkpoint),
            "hnm_stacked": use_hnm,
        }
        if use_hnm:
            ckpt_save["hnm_meta"] = hnm_meta_list
            ckpt_save["isolate_classes"] = args.isolate_classes
        if model_type == "bpn":
            ckpt_save["bpn_cfg"] = ckpt.get("bpn_cfg")

        if improved:
            best_f1 = selected_f1
            torch.save(ckpt_save, run_dir / "best_model.pt")
            no_improve = 0
        else:
            no_improve += 1
        torch.save(ckpt_save, run_dir / "latest_model.pt")

        if no_improve >= PHASE8_EARLY_STOP:
            print(f"\n  Early stop: no improvement for {no_improve} epochs")
            break

    # ------------------------------------------------------------------
    # Per-class summary and verdict
    # ------------------------------------------------------------------
    import wandb
    for t in cfg.CALL_TYPES_3:
        start_t = val0["per_class"].get(t, {}).get("f1", 0.0)
        best_t = max(h["per_class"].get(t, {}).get("f1", 0.0)
                     for h in history)
        wandb.summary[f"start_f1_{t}"] = float(start_t)
        wandb.summary[f"best_f1_{t}"]  = float(best_t)
        wandb.summary[f"delta_f1_{t}"] = float(best_t - start_t)
    # Specific to phase 8: also stamp recall deltas since recall is the
    # whole point of the intervention.
    for t in cfg.CALL_TYPES_3:
        start_r = val0["per_class"].get(t, {}).get("recall", 0.0)
        best_r = max(h["per_class"].get(t, {}).get("recall", 0.0)
                     for h in history)
        wandb.summary[f"start_recall_{t}"] = float(start_r)
        wandb.summary[f"best_recall_{t}"]  = float(best_r)
        wandb.summary[f"delta_recall_{t}"] = float(best_r - start_r)

    wandb.summary["gamma_d"] = float(args.gamma_d)
    wandb.summary["gamma_bmabz"] = float(args.gamma_bmabz)
    wandb.summary["gamma_bp"] = float(args.gamma_bp)

    selected_delta = best_f1 - val0[f"{args.select_by}_f1"]
    d_recall_delta = wandb.summary["delta_recall_d"]
    if selected_delta > 0.005:
        verdict_text = (
            f"Phase 8 helped: {args.select_by} F1 "
            f"{val0[f'{args.select_by}_f1']:.3f} → {best_f1:.3f} "
            f"({selected_delta:+.3f}). D recall {d_recall_delta:+.3f} "
            f"with γ_d={args.gamma_d}. Source phase {source_phase}, "
            f"HNM stacked={use_hnm}."
        )
    elif selected_delta > -0.005:
        verdict_text = (
            f"Phase 8 neutral: {args.select_by} F1 "
            f"{val0[f'{args.select_by}_f1']:.3f} → {best_f1:.3f} "
            f"({selected_delta:+.3f}). D recall {d_recall_delta:+.3f}. "
            f"γ_d={args.gamma_d} did not move the needle past noise. "
            f"Try a different γ_d (currently {args.gamma_d})."
        )
    else:
        verdict_text = (
            f"Phase 8 hurt: {args.select_by} F1 "
            f"{val0[f'{args.select_by}_f1']:.3f} → {best_f1:.3f} "
            f"({selected_delta:+.3f}). γ_d={args.gamma_d} may be too "
            f"aggressive (overshoots into excessive FP-suppression), or "
            f"lr={args.lr} too high for the fine-tune from this source."
        )

    wbu.finalize_phase(history, verdict=verdict_text,
                       best_ckpt=run_dir / "best_model.pt")

    print(f"\nDone. Best {args.select_by} F1: {best_f1:.3f}")
    print(f"  starting:  {val0[f'{args.select_by}_f1']:.3f}")
    print(f"  delta:     {selected_delta:+.3f}")
    print(f"  D recall:  {val0['per_class'].get('d', {}).get('recall', 0):.3f}"
          f"  →  {max(h['per_class'].get('d', {}).get('recall', 0) for h in history):.3f}"
          f"  (Δ={d_recall_delta:+.3f})")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
