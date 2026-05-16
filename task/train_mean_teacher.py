"""
Phase 7: Mean Teacher Self-Training on the AADC Unlabeled Corpus
=================================================================

Continue training a converged WhaleVAD checkpoint (your HNM_D best, or
any other strong supervised model) with an additional consistency loss
on the unlabeled AADC corpus. This is the first application of the
DCASE Task 4 mean-teacher paradigm to whale call SED.

Pipeline (per training step)
----------------------------
1. Labeled batch: standard supervised forward through the student;
   focal BCE on the GT labels — identical to your existing recipe.
2. Unlabeled batch: same 30s clip, two augmentation views
     - weak view  → teacher  → soft probability targets (no_grad)
     - strong view → student → soft probability outputs
   MSE between the two = consistency loss.
3. Total loss = supervised_loss + λ(epoch) · consistency_loss
   λ ramps from 0 to ``--lambda-max`` over ``--lambda-ramp-epochs``.
4. Optimiser step on the student. Teacher then updated via EMA:
   θ' ← α(epoch) · θ' + (1 - α(epoch)) · θ.

Validation: the **teacher** is evaluated each epoch. EMA-averaging
makes the teacher more stable than the student; this is the same
choice DCASE submissions make. The best-teacher checkpoint is the
final output and drops directly into ``ensemble_predict.py``.

Site quarantine
---------------
The script refuses to start if any provided AADC site is in the
BioDCASE test set (kerguelen2020, ddu2021). Use the safe sites only:
Casey2018, Scott2019, Prydz2013, DDU2018, DDU2019, and any
Kerguelen year other than 2020.

Wandb phase tag
---------------
Uses the existing "baseline" phase with extra tags ``mean_teacher``
and ``ssl_pseudo`` so runs sort cleanly without needing to edit
``wandb_utils.PHASE_REGISTRY``. (If you'd rather register a proper
phase, add to ``PHASE_REGISTRY``:

    "7": dict(
        parent="hnm",
        hypothesis="Mean teacher consistency regularisation on AADC.",
        interventions=["mean_teacher", "ssl_consistency"],
    ),

and change ``init_phase("baseline", ...)`` to ``init_phase("7", ...)``
below.)

Usage
-----
::

    python train_mean_teacher.py \\
        --checkpoint runs/hnm_D_seed42/best_model.pt \\
        --aadc-root /home/matthias-nagl/BioDCASE/aadc_audio \\
        --aadc-sites Casey2018 Scott2019 Prydz2013 DDU2018 DDU2019 \\
        --epochs 30 \\
        --lambda-max 1.0 --lambda-ramp-epochs 5 \\
        --alpha-end 0.999 \\
        --output-dir runs/mean_teacher_seed42 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
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
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)

from ssl_dataset import build_pretrain_manifest, SSLClipDataset, collate_ssl
from mean_teacher_core import (
    EMATeacher,
    consistency_loss,
    sigmoid_ramp,
    cosine_alpha,
    align_lengths_pair,
    align_supervised_lengths,
    make_weak_view,
    make_strong_view,
    freeze_bn_running_stats,
)


# ======================================================================
# Test-set quarantine
# ======================================================================
# These two AADC sites ARE the BioDCASE 2025 Task 2 test set. Even
# though the challenge is closed and results are public, including
# them in pretraining would invalidate any comparison to other
# BioDCASE submissions. This check fires before any data is loaded.

_TEST_SET_SITES = {"kerguelen2020", "ddu2021"}


def quarantine_check(sites: list[str]) -> None:
    bad = [s for s in sites if s.lower() in _TEST_SET_SITES]
    if bad:
        raise SystemExit(
            f"REFUSING TO RUN: AADC sites {bad} are the BioDCASE test set. "
            f"Pretraining on them would be test-set leakage and invalidate "
            f"comparison to challenge submissions.\n"
            f"Safe alternatives: Casey2018, Scott2019, Prydz2013, "
            f"DDU2018, DDU2019, Kerguelen2016-2019/2021/2023/2024."
        )


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Warm start
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to converged supervised checkpoint (e.g. HNM_D best_model.pt). "
             "The student is initialised from these weights; the teacher is a "
             "deepcopy at construction time.",
    )

    # Unlabeled stream
    p.add_argument("--aadc-root", type=Path, required=True,
                   help="Root dir containing AADC site subfolders.")
    p.add_argument("--aadc-sites", nargs="+", required=True,
                   help="AADC site names. Must NOT include kerguelen2020 or ddu2021.")
    p.add_argument("--unlabeled-batch-size", type=int, default=None,
                   help="Batch size for the unlabeled stream. Defaults to cfg.BATCH_SIZE.")
    p.add_argument("--epoch-unlabeled-clips", type=int, default=10_000,
                   help="Random 30s clips drawn per epoch from the AADC pool. "
                        "Should be roughly the number of labeled steps × "
                        "unlabeled_batch_size so one full unlabeled-loader pass "
                        "covers the same number of steps as one labeled-loader pass.")
    p.add_argument("--include-train-audio-in-unlabeled", action="store_true",
                   help="Also draw unlabeled clips from labeled training-site "
                        "audio (no labels used). DCASE-standard practice.")

    # Mean-teacher hyperparameters
    p.add_argument("--lambda-max", type=float, default=1.0,
                   help="Final weight on the consistency loss. λ(epoch) ramps "
                        "from 0 to this over --lambda-ramp-epochs.")
    p.add_argument("--lambda-ramp-epochs", type=int, default=5,
                   help="Epochs over which λ ramps from 0 to lambda-max.")
    p.add_argument("--alpha-start", type=float, default=0.99,
                   help="Initial EMA decay. Lower α tracks the rapidly-changing "
                        "early student more closely.")
    p.add_argument("--alpha-end", type=float, default=0.999,
                   help="Steady-state EMA decay.")
    p.add_argument("--alpha-warmup-epochs", type=int, default=10,
                   help="Epochs over which α ramps from alpha-start to alpha-end.")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Low LR matches HNM fine-tuning — we're continuing from "
                        "a converged checkpoint, not retraining from scratch.")
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-7)

    # Output / reproducibility / logging
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Run output dir. Defaults to runs/mean_teacher_<timestamp>.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


# ======================================================================
# Reproducibility
# ======================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
# Validation (uses the teacher, not the student)
# ======================================================================

@torch.no_grad()
def validate_teacher(
    teacher_module: nn.Module,
    spec_extractor: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: torch.Tensor,
    val_annotations,
    file_start_dts: dict,
    tune_thresholds: bool = True,
) -> dict:
    """
    Validation pass on the EMA teacher. Same logic as ``train.validate``
    but factored here so we don't pull the whole train.py module just
    for one function.
    """
    teacher_module.eval()
    total_loss, n_batches = 0.0, 0
    all_probs: dict = {}

    for audio, targets, mask, metas in tqdm(val_loader, desc="Val (teacher)",
                                            leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = teacher_module(spec)
        targets_a, mask_a = align_supervised_lengths(logits, targets, mask)
        total_loss += criterion(logits, targets_a, mask_a).item()
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
            np.arange(0.20, 0.85, 0.05),                                        # bmabz
            np.concatenate([np.arange(0.05, 0.5, 0.05),
                            np.arange(0.5, 0.85, 0.10)]),                       # d
            np.concatenate([np.arange(0.05, 0.5, 0.05),
                            np.arange(0.5, 0.85, 0.10)]),                       # bp
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

    print(f"\n  Teacher event-level (tuned thresholds):")
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
# One training epoch
# ======================================================================

def train_epoch_mean_teacher(
    student: nn.Module,
    teacher: EMATeacher,
    spec_extractor: nn.Module,
    criterion: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    unlabeled_iter,
    optimizer,
    device: torch.device,
    epoch: int,
    lambda_weight: float,
    alpha: float,
) -> tuple[float, float, object]:
    """
    Run one mean-teacher epoch.

    Each step:
      - draws one labeled batch and one unlabeled batch
      - computes supervised loss on the labeled batch (student)
      - computes consistency loss between strong-aug student and
        weak-aug teacher on the unlabeled batch
      - sums and steps the optimiser
      - EMA-updates the teacher

    Returns
    -------
    (mean_sup_loss, mean_cons_loss, unlabeled_iter)
        ``unlabeled_iter`` is returned so the caller can keep iterating
        across epochs without re-creating it (avoids losing work when
        the unlabeled loader is longer than the labeled one).
    """
    student.train()
    freeze_bn_running_stats(student)
    total_sup, total_cons, n_steps = 0.0, 0.0, 0
    pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}", leave=False)

    for audio_l, targets, mask, _ in pbar:
        # ---- supervised batch ----
        audio_l = audio_l.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # ---- unlabeled batch ----
        try:
            ub = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            ub = next(unlabeled_iter)
        audio_u = ub["audio"].to(device, non_blocking=True)
        sites_u = ub["sites"]

        optimizer.zero_grad()

        # ---- supervised forward ----
        spec_l = spec_extractor(audio_l)
        logits_l = student(spec_l)
        targets_a, mask_a = align_supervised_lengths(logits_l, targets, mask)
        loss_sup = criterion(logits_l, targets_a, mask_a)

        # ---- consistency forward (student: strong view) ----
        spec_strong = make_strong_view(audio_u, sites_u, spec_extractor)
        logits_u_s = student(spec_strong)

        # ---- consistency forward (teacher: weak view, no grad) ----
        with torch.no_grad():
            spec_weak = make_weak_view(audio_u, sites_u, spec_extractor)
            logits_u_t = teacher.teacher(spec_weak)

        logits_u_s, logits_u_t = align_lengths_pair(logits_u_s, logits_u_t)
        loss_cons = consistency_loss(logits_u_s, logits_u_t)

        loss = loss_sup + lambda_weight * loss_cons

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  *** NaN/Inf loss at epoch {epoch}, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        # EMA update — AFTER the optimizer step, so the teacher always
        # lags by one step relative to the just-updated student.
        teacher.update(student, alpha=alpha)

        total_sup += loss_sup.item()
        total_cons += loss_cons.item()
        n_steps += 1

        pbar.set_postfix(
            sup=f"{loss_sup.item():.3f}",
            cons=f"{loss_cons.item():.3f}",
            **{"λ": f"{lambda_weight:.2f}", "α": f"{alpha:.4f}"},
        )

    return (
        total_sup / max(n_steps, 1),
        total_cons / max(n_steps, 1),
        unlabeled_iter,
    )


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    quarantine_check(args.aadc_sites)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dir
    if args.output_dir is None:
        args.output_dir = Path(cfg.OUTPUT_DIR) / (
            f"mean_teacher_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {args.output_dir}")

    unlabeled_bs = args.unlabeled_batch_size or cfg.BATCH_SIZE

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    if not args.no_wandb:
        wbu.init_phase(
            "baseline",
            extra_tags=["mean_teacher", "ssl_consistency",
                        f"seed_{args.seed}"],
            config={
                "checkpoint":              str(args.checkpoint),
                "aadc_sites":              list(args.aadc_sites),
                "epochs":                  args.epochs,
                "lr":                      args.lr,
                "lambda_max":              args.lambda_max,
                "lambda_ramp_epochs":      args.lambda_ramp_epochs,
                "alpha_start":             args.alpha_start,
                "alpha_end":               args.alpha_end,
                "alpha_warmup_epochs":     args.alpha_warmup_epochs,
                "unlabeled_batch_size":    unlabeled_bs,
                "epoch_unlabeled_clips":   args.epoch_unlabeled_clips,
                "include_train_audio":     args.include_train_audio_in_unlabeled,
                "seed":                    args.seed,
            },
        )

    # ------------------------------------------------------------------
    # Data: labeled (existing pipeline) + unlabeled (AADC manifest)
    # ------------------------------------------------------------------
    train_ds, labeled_loader, val_loader = build_dataloaders()

    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    print(f"\nBuilding unlabeled manifest from AADC sites: {args.aadc_sites}")
    train_audio_arg = (cfg.TRAIN_DATASETS
                       if args.include_train_audio_in_unlabeled else None)
    unlab_manifest = build_pretrain_manifest(
        train_datasets=train_audio_arg,
        aadc_sites=list(args.aadc_sites),
        aadc_root=args.aadc_root,
    )

    unlab_ds = SSLClipDataset(
        unlab_manifest,
        clip_seconds=30.0,
        sample_rate=cfg.SAMPLE_RATE,
        epoch_clips=args.epoch_unlabeled_clips,
    )
    unlab_loader = DataLoader(
        unlab_ds, batch_size=unlabeled_bs,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_ssl, pin_memory=True, drop_last=True,
    )
    unlab_iter = iter(unlab_loader)
    print(f"Unlabeled stream: {len(unlab_manifest)} files, "
          f"{args.epoch_unlabeled_clips} clips/epoch, "
          f"batch_size={unlabeled_bs}")

    # ------------------------------------------------------------------
    # Model: student initialised from the warm-start checkpoint;
    #        teacher = deepcopy(student) via EMATeacher.
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    student = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Force lazy-projection init before loading state dict
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        student(spec_extractor(dummy))

    print(f"\nWarm-starting student from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("encoder_state_dict", ckpt))
    missing, unexpected = student.load_state_dict(state, strict=False)
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Recover thresholds from the warm-start checkpoint if present —
    # this seeds the per-epoch threshold sweep so the first validation
    # doesn't pay a full coordinate-descent cost.
    if "thresholds" in ckpt:
        thresholds = ckpt["thresholds"].to(device).float()
        print(f"  Warm-start thresholds: {[f'{t:.2f}' for t in thresholds.tolist()]}")
    else:
        thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device).float()

    teacher = EMATeacher(student, alpha=args.alpha_start)
    teacher.to(device)

    # DataParallel: wrap only the student (teacher does no_grad forward
    # so DP isn't useful, and the EMA copy logic gets ugly with DP).
    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs (student only)")
        student = nn.DataParallel(student)

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")

    # ------------------------------------------------------------------
    # Optimiser, criterion, scheduler
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        student.parameters(), lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY, betas=(cfg.BETA1, cfg.BETA2),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = -1.0
    no_improve = 0
    print(f"\n{'='*70}\nMean teacher self-training for {args.epochs} epochs\n{'='*70}")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        lambda_w = args.lambda_max * sigmoid_ramp(epoch - 1, args.lambda_ramp_epochs)
        alpha = cosine_alpha(epoch - 1, args.alpha_start, args.alpha_end,
                             args.alpha_warmup_epochs)

        print(f"\n{'='*70}\nEpoch {epoch}/{args.epochs}  "
              f"lr={current_lr:.2e}  λ={lambda_w:.3f}  α={alpha:.4f}\n{'='*70}")

        sup_loss, cons_loss, unlab_iter = train_epoch_mean_teacher(
            student=student, teacher=teacher,
            spec_extractor=spec_extractor, criterion=criterion,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlab_loader, unlabeled_iter=unlab_iter,
            optimizer=optimizer, device=device, epoch=epoch,
            lambda_weight=lambda_w, alpha=alpha,
        )

        # Validate the TEACHER (more stable than the student)
        val = validate_teacher(
            teacher_module=teacher.teacher,
            spec_extractor=spec_extractor, val_loader=val_loader,
            criterion=criterion, device=device,
            thresholds=thresholds, val_annotations=val_annotations,
            file_start_dts=file_start_dts, tune_thresholds=True,
        )
        thresholds = torch.tensor(val["thresholds"], device=device).float()
        scheduler.step(val["mean_f1"])

        print(f"\n  Sup loss: {sup_loss:.4f}  Cons loss: {cons_loss:.4f}  "
              f"Val loss: {val['loss']:.4f}")
        print(f"  Teacher F1: {val['mean_f1']:.3f}  Best F1: {best_f1:.3f}")

        # Wandb log
        if not args.no_wandb:
            import wandb
            payload = {
                "epoch":          epoch,
                "lr":             current_lr,
                "lambda":         lambda_w,
                "alpha":          alpha,
                "train/sup_loss": sup_loss,
                "train/cons_loss": cons_loss,
                "val/loss":       val["loss"],
                "val/f1_macro":   val["mean_f1"],
            }
            for ci, cname in enumerate(cfg.CALL_TYPES_3):
                pc = val["per_class"].get(cname, {})
                payload[f"val/f1/{cname}"]        = pc.get("f1", 0.0)
                payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
                payload[f"val/recall/{cname}"]    = pc.get("recall", 0.0)
                payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
            wandb.log(payload, step=epoch)

        # Save checkpoints. ``model_state_dict`` is the TEACHER's weights
        # so this checkpoint drops straight into ensemble_predict.py.
        student_module = (student.module if isinstance(student, nn.DataParallel)
                          else student)
        ckpt_out = {
            "epoch":              epoch,
            "model_state_dict":   teacher.state_dict(),      # teacher = deployed
            "student_state_dict": student_module.state_dict(),  # for resume
            "best_f1":            best_f1,
            "thresholds":         thresholds.cpu(),
            "lambda":             lambda_w,
            "alpha":              alpha,
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt_out["best_f1"] = best_f1
            torch.save(ckpt_out, args.output_dir / "best_model.pt")
            print(f"  *** New best teacher F1: {best_f1:.3f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{args.early_stop_patience} epochs")

        torch.save(ckpt_out, args.output_dir / "latest_model.pt")

        if no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping: no improvement for "
                  f"{args.early_stop_patience} epochs")
            break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Done. Best teacher F1: {best_f1:.3f}")
    print(f"Output: {args.output_dir / 'best_model.pt'}")
    print(f"{'='*70}")
    print(f"\nNext step: drop into your existing ensemble pipeline, e.g.")
    print(f"  python ensemble_predict.py --per-model-eval \\")
    print(f"      --checkpoints {args.output_dir}/best_model.pt \\")
    print(f"                    runs/hnm_D_*/best_model.pt")

    if not args.no_wandb:
        wbu.finalize_phase(
            history=[],
            verdict=f"Mean teacher best F1 = {best_f1:.3f}",
            best_ckpt=args.output_dir / "best_model.pt",
        )


if __name__ == "__main__":
    main()
