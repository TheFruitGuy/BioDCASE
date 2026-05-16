"""
Phase 7B: Mean Teacher + HNM Hard-Negative Injection
=====================================================

Combines mean-teacher consistency regularisation on the unlabeled AADC
corpus with HNM hard-negative injection in the labeled stream. This is
the configuration that makes mean teacher actually work for this task:

  - HNM hard negatives in the labeled stream provide *new* training
    signal that the converged warm-start hasn't seen → supervised loss
    contributes productive gradient instead of random drift.
  - MT consistency on the unlabeled AADC stream adds a regularisation
    pressure that's compositional with the HNM signal.
  - PGI (per-class gradient isolation) on hard-neg segments preserves
    the cross-class non-interference property from HNM_D.

Why we need this combo
----------------------
Two mean-teacher runs without HNM injection both degraded the warm-start
in epoch 1 with λ≈0. Diagnosis: continued supervised training of a
converged checkpoint on data it has already seen produces random
parameter drift, not learning. The model wanders because its gradient
is below noise floor. HNM hard negatives are the missing ingredient —
they give the supervised loss something productive to do, which keeps
the student moving in useful directions, which keeps the EMA teacher a
meaningful target for the consistency loss.

Usage
-----
::

    python train_mean_teacher_hnm.py \\
        --checkpoint runs/hnm_D_whalevad_20260504_152450/best_model.pt \\
        --hard_negatives runs/hardnegs/d_<source>.json \\
                         runs/hardnegs/bmabz_<source>.json \\
        --isolate_classes \\
        --aadc-root ~/BioDCASE/task/data_pretrain/audio \\
        --aadc-sites Casey2018 DDU2018 DDU2019 Kerguelen2018 Kerguelen2019 \\
        --epochs 20 \\
        --lr 1e-5 \\
        --lambda-max 1.0 --lambda-ramp-epochs 3 \\
        --alpha-end 0.999 \\
        --output-dir runs/mt_hnm_seed42 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
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
    Segment, WhaleDataset, build_negative_segments, build_positive_segments,
    build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)

# HNM machinery — reused as-is from your existing pipeline
from train_phase_hnm import (
    load_hard_negatives_json,
    build_hard_negative_segments,
    build_hard_neg_class_map,
    build_class_mask,
    HnmTrainingDataset,
    logits_bce_focal_loss,
)
from train_phase0e import PHASE0E_SEGMENT_S

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
# Quarantine
# ======================================================================

_TEST_SET_SITES = {"kerguelen2020", "ddu2021"}


def quarantine_check(sites: list[str]) -> None:
    bad = [s for s in sites if s.lower() in _TEST_SET_SITES]
    if bad:
        raise SystemExit(
            f"REFUSING TO RUN: AADC sites {bad} are the BioDCASE test set."
        )


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Warm start (typically the HNM_D best, so HNM injection preserves
    # the operating point and MT adds on top)
    p.add_argument("--checkpoint", type=Path, required=True)

    # Hard negatives (same JSONs the HNM_D fine-tune used)
    p.add_argument("--hard_negatives", type=str, nargs="+", required=True,
                   help="One or more mine_hard_negatives.py JSON outputs.")
    p.add_argument("--isolate_classes", action="store_true",
                   help="Apply per-class gradient isolation (PGI).")
    p.add_argument("--oversample", type=int, default=5,
                   help="Hard-neg segments repeated this many times per epoch.")

    # Unlabeled AADC stream
    p.add_argument("--aadc-root", type=Path, required=True)
    p.add_argument("--aadc-sites", nargs="+", required=True)
    p.add_argument("--unlabeled-batch-size", type=int, default=None)
    p.add_argument("--epoch-unlabeled-clips", type=int, default=10_000)

    # Mean-teacher hyperparameters
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-ramp-epochs", type=int, default=3,
                   help="Faster ramp than the vanilla MT script — the HNM "
                        "supervised signal is strong enough to absorb the "
                        "consistency loss sooner.")
    p.add_argument("--alpha-start", type=float, default=0.99)
    p.add_argument("--alpha-end", type=float, default=0.999)
    p.add_argument("--alpha-warmup-epochs", type=int, default=5)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Standard HNM fine-tune LR. Do NOT raise this — "
                        "5e-5 destabilises the warm-start in epoch 1.")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--resample-every", type=int, default=5,
                   help="Resample random negatives every N epochs (HNM convention).")

    # Output / logging
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--select_by", type=str, default="macro",
                   choices=["macro", "overall"])
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
# Validation (teacher, identical to validate_teacher from MT script)
# ======================================================================

@torch.no_grad()
def validate_teacher(teacher_module, spec_extractor, val_loader, criterion,
                     device, thresholds, val_annotations, file_start_dts,
                     select_by="macro"):
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
    macro_f1 = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                              for c in cfg.CALL_TYPES_3]))
    selection_f1 = macro_f1 if select_by == "macro" else overall_f1

    print(f"\n  Teacher event-level (tuned thresholds):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name)
        if m is None:
            continue
        print(f"    {name.upper():6} t={used_thresholds[c]:.2f}  "
              f"TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}  MACRO F1={macro_f1:.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "selection_f1": selection_f1,
        "overall_f1": overall_f1,
        "macro_f1": macro_f1,
        "per_class": metrics,
        "thresholds": used_thresholds.tolist(),
    }


# ======================================================================
# Training epoch
# ======================================================================

def train_epoch_mt_hnm(
    student, teacher, spec_extractor, pos_weight,
    labeled_loader, unlabeled_loader, unlabeled_iter,
    optimizer, device, epoch,
    lambda_weight, alpha,
    hard_neg_class_map, n_classes, use_pgi,
):
    student.train()
    freeze_bn_running_stats(student)   # critical — see MT failure notes

    total_sup, total_cons, n_steps = 0.0, 0.0, 0
    pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}", leave=False)

    for audio, targets, mask, metas in pbar:
        # Labeled batch (includes HNM hard negatives, oversampled)
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        class_mask = build_class_mask(metas, hard_neg_class_map,
                                      n_classes, device)

        # Unlabeled batch (AADC)
        try:
            ub = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            ub = next(unlabeled_iter)
        audio_u = ub["audio"].to(device, non_blocking=True)
        sites_u = ub["sites"]

        optimizer.zero_grad()

        # Supervised forward + loss with PGI class masking
        spec_l = spec_extractor(audio)
        logits_l = student(spec_l)
        T = min(logits_l.size(1), targets.size(1))
        loss_sup = logits_bce_focal_loss(
            logits_l[:, :T], targets[:, :T], mask[:, :T],
            class_mask=class_mask, pos_weight=pos_weight,
            use_focal=cfg.USE_FOCAL_LOSS,
        )

        # Consistency forward (student: strong view)
        spec_strong = make_strong_view(audio_u, sites_u, spec_extractor)
        logits_u_s = student(spec_strong)

        # Consistency forward (teacher: weak view, no grad)
        with torch.no_grad():
            spec_weak = make_weak_view(audio_u, sites_u, spec_extractor)
            logits_u_t = teacher.teacher(spec_weak)

        logits_u_s, logits_u_t = align_lengths_pair(logits_u_s, logits_u_t)
        loss_cons = consistency_loss(logits_u_s, logits_u_t)

        loss = loss_sup + lambda_weight * loss_cons
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  *** NaN/Inf at epoch {epoch}, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        teacher.update(student, alpha=alpha)

        total_sup += loss_sup.item()
        total_cons += loss_cons.item()
        n_steps += 1

        pbar.set_postfix(
            sup=f"{loss_sup.item():.4f}",
            cons=f"{loss_cons.item():.4f}",
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

    if args.output_dir is None:
        args.output_dir = Path(cfg.OUTPUT_DIR) / (
            f"mt_hnm_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {args.output_dir}")

    unlabeled_bs = args.unlabeled_batch_size or cfg.BATCH_SIZE

    # ------------------------------------------------------------------
    # Load hard negatives (from HNM_D's mining outputs)
    # ------------------------------------------------------------------
    print(f"\nLoading hard negatives from {len(args.hard_negatives)} JSON(s)")
    fp_records, hnm_meta_list = load_hard_negatives_json(args.hard_negatives)
    fps_per_class: dict[str, int] = {}
    for r in fp_records:
        fps_per_class[r.get("target_class", "?")] = (
            fps_per_class.get(r.get("target_class", "?"), 0) + 1)
    print(f"  Total FP records: {len(fp_records)}")
    for cname, n in fps_per_class.items():
        print(f"    {cname}: {n}")
    targets_used = sorted(fps_per_class.keys())

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    if not args.no_wandb:
        extra_tags = ["mean_teacher", "hnm_injection",
                      f"seed_{args.seed}",
                      "pgi_on" if args.isolate_classes else "pgi_off"]
        for t in targets_used:
            extra_tags.append(f"target_{t}")
        wbu.init_phase(
            "baseline",
            extra_tags=extra_tags,
            config={
                "checkpoint":            str(args.checkpoint),
                "hard_negatives":        list(args.hard_negatives),
                "aadc_sites":            list(args.aadc_sites),
                "epochs":                args.epochs,
                "lr":                    args.lr,
                "lambda_max":            args.lambda_max,
                "lambda_ramp_epochs":    args.lambda_ramp_epochs,
                "alpha_start":           args.alpha_start,
                "alpha_end":             args.alpha_end,
                "alpha_warmup_epochs":   args.alpha_warmup_epochs,
                "isolate_classes":       args.isolate_classes,
                "oversample":            args.oversample,
                "select_by":             args.select_by,
                "n_hard_negs_total":     len(fp_records),
                "n_hard_negs_per_class": fps_per_class,
                "mining_targets":        targets_used,
                "mining_meta":           hnm_meta_list,
                "seed":                  args.seed,
            },
        )

    # ------------------------------------------------------------------
    # Labeled data: positives (30s-extended) + random negatives + HNM
    # ------------------------------------------------------------------
    print("\nLoading training data...")
    train_anns = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(train_anns, train_manifest)

    # Extend positives to fixed 30s (same as HNM_D pipeline)
    from train_phase0e import extend_segment_to_fixed_length
    manifest_idx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [
        extend_segment_to_fixed_length(
            s, PHASE0E_SEGMENT_S,
            float(manifest_idx.loc[(s.dataset, s.filename), "duration_s"]))
        for s in pos_segs
        if (s.dataset, s.filename) in manifest_idx.index
    ]
    print(f"  {len(pos_segs)} positive segments (30s)")

    hard_segs, used_records = build_hard_negative_segments(
        fp_records, train_manifest, train_anns)
    print(f"  {len(hard_segs)} hard-neg segments × oversample {args.oversample}"
          f" = {len(hard_segs) * args.oversample} copies/epoch")

    hard_neg_class_map = (build_hard_neg_class_map(hard_segs, used_records)
                          if args.isolate_classes else {})
    if args.isolate_classes:
        multi = sum(1 for v in hard_neg_class_map.values() if len(v) > 1)
        print(f"  PGI on: {len(hard_neg_class_map)} locations "
              f"({multi} multi-class)")
    else:
        print(f"  PGI off")

    labeled_ds = HnmTrainingDataset(pos_segs, hard_segs, args.oversample,
                                    train_manifest, train_anns)
    labeled_loader = DataLoader(
        labeled_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # Validation data
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }

    # Unlabeled AADC data
    print(f"\nBuilding unlabeled manifest: {args.aadc_sites}")
    unlab_manifest = build_pretrain_manifest(
        train_datasets=None,
        aadc_sites=list(args.aadc_sites),
        aadc_root=args.aadc_root,
    )
    unlab_ds = SSLClipDataset(
        unlab_manifest, clip_seconds=30.0,
        sample_rate=cfg.SAMPLE_RATE,
        epoch_clips=args.epoch_unlabeled_clips,
    )
    unlab_loader = DataLoader(
        unlab_ds, batch_size=unlabeled_bs,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_ssl, pin_memory=True, drop_last=True,
    )
    unlab_iter = iter(unlab_loader)
    print(f"Unlabeled: {len(unlab_manifest)} files, "
          f"{args.epoch_unlabeled_clips} clips/epoch, bs={unlabeled_bs}")

    # ------------------------------------------------------------------
    # Model + warm-start
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    student = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        student(spec_extractor(dummy))

    print(f"\nWarm-starting from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("encoder_state_dict", ckpt))
    missing, unexpected = student.load_state_dict(state, strict=False)
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")

    if "thresholds" in ckpt:
        thresholds = ckpt["thresholds"].to(device).float()
        print(f"  Warm-start thresholds: "
              f"{[f'{t:.2f}' for t in thresholds.tolist()]}")
    else:
        thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device).float()

    teacher = EMATeacher(student, alpha=args.alpha_start)
    teacher.to(device)

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs (student)")
        student = nn.DataParallel(student)

    # ------------------------------------------------------------------
    # Loss + optimizer + scheduler
    # ------------------------------------------------------------------
    pos_weight = (compute_class_weights().to(device)
                  if cfg.USE_WEIGHTED_BCE else None)
    if pos_weight is not None:
        print(f"  pos_weight: {pos_weight.tolist()}")

    baseline_criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)
    n_classes = cfg.n_classes()

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
    print(f"\n{'='*70}\nMT+HNM training for {args.epochs} epochs\n{'='*70}")

    for epoch in range(1, args.epochs + 1):
        # Periodic random-negative resample (HNM convention)
        if epoch > 1 and (epoch - 1) % args.resample_every == 0:
            print(f"  [epoch {epoch}] resampling random negatives")
            labeled_ds.resample_negatives()
            labeled_loader = DataLoader(
                labeled_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        lambda_w = args.lambda_max * sigmoid_ramp(epoch - 1,
                                                  args.lambda_ramp_epochs)
        alpha = cosine_alpha(epoch - 1, args.alpha_start, args.alpha_end,
                             args.alpha_warmup_epochs)

        print(f"\n{'='*70}\nEpoch {epoch}/{args.epochs}  "
              f"lr={current_lr:.2e}  λ={lambda_w:.3f}  α={alpha:.4f}\n"
              f"{'='*70}")

        sup_loss, cons_loss, unlab_iter = train_epoch_mt_hnm(
            student=student, teacher=teacher,
            spec_extractor=spec_extractor, pos_weight=pos_weight,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlab_loader, unlabeled_iter=unlab_iter,
            optimizer=optimizer, device=device, epoch=epoch,
            lambda_weight=lambda_w, alpha=alpha,
            hard_neg_class_map=hard_neg_class_map,
            n_classes=n_classes, use_pgi=args.isolate_classes,
        )

        val = validate_teacher(
            teacher_module=teacher.teacher,
            spec_extractor=spec_extractor, val_loader=val_loader,
            criterion=baseline_criterion, device=device,
            thresholds=thresholds, val_annotations=val_anns,
            file_start_dts=file_start_dts, select_by=args.select_by,
        )
        thresholds = torch.tensor(val["thresholds"], device=device).float()
        scheduler.step(val["selection_f1"])

        print(f"\n  Sup loss: {sup_loss:.4f}  Cons loss: {cons_loss:.4f}  "
              f"Val loss: {val['loss']:.4f}")
        print(f"  Teacher {args.select_by} F1: {val['selection_f1']:.3f}  "
              f"Best: {best_f1:.3f}")

        if not args.no_wandb:
            import wandb
            payload = {
                "epoch":           epoch,
                "lr":              current_lr,
                "lambda":          lambda_w,
                "alpha":           alpha,
                "train/sup_loss":  sup_loss,
                "train/cons_loss": cons_loss,
                "val/loss":        val["loss"],
                "val/f1_overall": val["overall_f1"],
                "val/f1_macro":    val["macro_f1"],
            }
            for ci, cname in enumerate(cfg.CALL_TYPES_3):
                pc = val["per_class"].get(cname, {})
                payload[f"val/f1/{cname}"]        = pc.get("f1", 0.0)
                payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
                payload[f"val/recall/{cname}"]    = pc.get("recall", 0.0)
                payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
            wandb.log(payload, step=epoch)

        student_module = (student.module if isinstance(student, nn.DataParallel)
                          else student)
        ckpt_out = {
            "epoch":              epoch,
            "model_state_dict":   teacher.state_dict(),
            "student_state_dict": student_module.state_dict(),
            "best_f1":            best_f1,
            "thresholds":         thresholds.cpu(),
            "lambda":             lambda_w,
            "alpha":              alpha,
            "select_by":          args.select_by,
        }

        if val["selection_f1"] > best_f1:
            best_f1 = val["selection_f1"]
            ckpt_out["best_f1"] = best_f1
            torch.save(ckpt_out, args.output_dir / "best_model.pt")
            print(f"  *** New best {args.select_by} F1: {best_f1:.3f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{args.early_stop_patience}")

        torch.save(ckpt_out, args.output_dir / "latest_model.pt")

        if no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping after {args.early_stop_patience} epochs "
                  f"without improvement")
            break

    print(f"\n{'='*70}")
    print(f"Done. Best teacher {args.select_by} F1: {best_f1:.3f}")
    print(f"Output: {args.output_dir / 'best_model.pt'}")
    print(f"{'='*70}")

    if not args.no_wandb:
        wbu.finalize_phase(
            history=[],
            verdict=f"MT+HNM best {args.select_by} F1 = {best_f1:.3f}",
            best_ckpt=args.output_dir / "best_model.pt",
        )


if __name__ == "__main__":
    main()
