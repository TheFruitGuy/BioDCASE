"""
FlatMatch + HNM Hard-Negative Injection
========================================

Cross-sharpness semi-supervised regularisation (FlatMatch, Huang et al.
NeurIPS 2023) on the unlabeled AADC corpus, layered on top of the same
HNM hard-negative + PGI labeled stream that ``train_mean_teacher_hnm``
uses. The skeleton (data loaders, validation, checkpointing) is cloned
from that script; the only substantive change is the unlabeled-side
training objective.

Mechanism (per step)
--------------------
::

    Labeled (HNM + positives + random negs) ──► Student(θ) ──► L_sup ──► backward
                                                                            │
                                                                  ∇L_l in .grad
                                                                            │
                                                          ε* = ρ · g/‖g‖₂  ◄┤
                                                              │
                                                              ▼
    Unlabeled AADC ──► Student(θ)      ── no_grad ──► f_u  ──► detach
                  ╲                                              ╲
                   ──► Student(θ + ε*) ── with grad ──► f̃_u ──► R_xs (MSE)

Two FlatMatch variants are selectable:

* ``--flatmatch-mode e``    (default) — ε* is read from an EMA buffer of
  past labeled gradients. No extra labeled work over plain SGD; the
  buffer is bootstrapped from the first step's ∇L_l and EMA-updated
  before the cross-sharpness backward (so contamination from ∇R_xs
  doesn't leak into M_{t+1}).
* ``--flatmatch-mode full`` — ε* is computed from this step's fresh
  ∇L_l. The labeled backward is reused for both the ε* source and the
  supervised contribution to the optimiser step (paper Eq. 2: the
  second-order correction is negligible), so cost matches mode "e".

The cross-sharpness term replaces the MT consistency loss by default,
matching Eq. 5 of the paper. Stacking the two is available via
``--lambda-mt > 0`` (default 0) — useful as an A/B sanity check, not
recommended by the paper.

Why we still keep an EMA teacher
--------------------------------
The validation pipeline (``validate_teacher``, threshold sweep,
``tune_thresholds_per_class``) is structured around an EMA-averaged
model, and EMA-averaged weights generally produce better detection
metrics than the bare student. We keep the EMA teacher purely as the
*evaluation* head — gradients still descend the FlatMatch objective.
Setting ``--alpha-end 0`` effectively disables the EMA and validates on
the student directly.

Usage
-----
::

    python train_flatmatch_hnm.py \\
        --checkpoint runs/hnm_D_whalevad_20260504_152450/best_model.pt \\
        --hard_negatives runs/hardnegs/d_<source>.json \\
                         runs/hardnegs/bmabz_<source>.json \\
        --isolate_classes \\
        --aadc-root ~/BioDCASE/task/data_pretrain/audio \\
        --aadc-sites Casey2018 DDU2018 DDU2019 Kerguelen2018 Kerguelen2019 \\
        --epochs 20 --lr 1e-5 \\
        --flatmatch-mode e --rho 0.1 \\
        --lambda-xs 1.0 --lambda-ramp-epochs 3 \\
        --output-dir runs/flatmatch_hnm_seed42 --seed 42
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
    WhaleDataset, build_positive_segments, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import Detection, collapse_probs_to_3class
from validation_core import (
    tune_thresholds_per_class, evaluate_with_thresholds, macro_f1,
)

from train_phase_hnm import (
    load_hard_negatives_json, build_hard_negative_segments,
    build_hard_neg_class_map, build_class_mask,
    HnmTrainingDataset, logits_bce_focal_loss,
)
from segment_length_core import PHASE0E_SEGMENT_S

from ssl_dataset import build_pretrain_manifest, SSLClipDataset, collate_ssl
from mean_teacher_core import (
    EMATeacher,
    consistency_loss, consistency_loss_asymmetric, consistency_loss_confident,
    sigmoid_ramp, cosine_alpha,
    align_lengths_pair, align_supervised_lengths,
    make_weak_view, make_strong_view,
    freeze_bn_running_stats,
)
from flatmatch_core import FlatMatchPerturbation, cross_sharpness_loss


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

    p.add_argument("--checkpoint", type=Path, required=True)

    p.add_argument("--hard_negatives", type=str, nargs="+", required=True)
    p.add_argument("--isolate_classes", action="store_true")
    p.add_argument("--oversample", type=int, default=5)

    p.add_argument("--aadc-root", type=Path, required=True)
    p.add_argument("--aadc-sites", nargs="+", required=True)
    p.add_argument("--unlabeled-batch-size", type=int, default=None)
    p.add_argument("--epoch-unlabeled-clips", type=int, default=10_000)

    # FlatMatch
    p.add_argument("--flatmatch-mode", type=str, default="e",
                   choices=["e", "full"],
                   help="ε* source. 'e': EMA buffer of past ∇L_l (paper's "
                        "efficient variant). 'full': this step's fresh ∇L_l.")
    p.add_argument("--rho", type=float, default=0.1,
                   help="Perturbation magnitude ‖ε*‖₂ (paper default 0.1).")
    p.add_argument("--xs-ema-alpha", type=float, default=0.999,
                   help="EMA decay for the labeled-gradient buffer "
                        "(mode 'e' only).")
    p.add_argument("--lambda-xs", type=float, default=1.0,
                   help="Max weight on the cross-sharpness term R_xs.")
    p.add_argument("--lambda-mt", type=float, default=0.0,
                   help="Max weight on the (optional) MT consistency term. "
                        "Default 0 = paper-faithful replacement. Set to "
                        "1.0 to stack MT alongside FlatMatch.")
    p.add_argument("--lambda-ramp-epochs", type=int, default=3,
                   help="Sigmoid ramp epochs for both λ_xs and λ_mt.")

    # EMA teacher (validation head only)
    p.add_argument("--alpha-start", type=float, default=0.99)
    p.add_argument("--alpha-end", type=float, default=0.999)
    p.add_argument("--alpha-warmup-epochs", type=int, default=5)

    # MT consistency variant (only used when --lambda-mt > 0)
    p.add_argument("--consistency-type", type=str, default="mse",
                   choices=["mse", "confident", "asymmetric_mse"])
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--neg-weight", type=float, default=1.0)
    p.add_argument("--conf-threshold", type=float, default=0.7)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--resample-every", type=int, default=5)

    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--select_by", type=str, default="macro",
                   choices=["macro", "overall"])
    p.add_argument("--val-workers", type=int, default=1)
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
# Validation (teacher head, same as MT script)
# ======================================================================

@torch.no_grad()
def validate_teacher(teacher_module, spec_extractor, val_loader, criterion,
                     device, thresholds, val_annotations, file_start_dts,
                     select_by="macro", val_workers=1):
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

    start_thresholds = np.asarray(thresholds.cpu().numpy(), dtype=np.float64)
    used_thresholds = tune_thresholds_per_class(
        all_probs, gt_events, start=start_thresholds,
        parallel_workers=val_workers,
    )

    metrics = evaluate_with_thresholds(all_probs, gt_events, used_thresholds)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
    macro = macro_f1(metrics)
    selection_f1 = macro if select_by == "macro" else overall_f1

    print(f"\n  Teacher event-level (tuned thresholds):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name)
        if m is None:
            continue
        print(f"    {name.upper():6} t={used_thresholds[c]:.2f}  "
              f"TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}  MACRO F1={macro:.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "selection_f1": selection_f1,
        "overall_f1": overall_f1,
        "macro_f1": macro,
        "per_class": metrics,
        "thresholds": used_thresholds.tolist(),
    }


# ======================================================================
# MT consistency dispatch (only used when λ_mt > 0)
# ======================================================================

def _mt_consistency(args, logits_u_s, logits_u_t):
    if args.consistency_type == "mse":
        return consistency_loss(logits_u_s, logits_u_t)
    if args.consistency_type == "confident":
        return consistency_loss_confident(
            logits_u_s, logits_u_t, conf_threshold=args.conf_threshold,
        )
    return consistency_loss_asymmetric(
        logits_u_s, logits_u_t,
        pos_weight=args.pos_weight, neg_weight=args.neg_weight,
        conf_threshold=args.conf_threshold,
    )


# ======================================================================
# Training epoch
# ======================================================================

def train_epoch_flatmatch_hnm(
    student, teacher, perturbation, spec_extractor, pos_weight,
    labeled_loader, unlabeled_loader, unlabeled_iter,
    optimizer, device, epoch,
    lambda_xs, lambda_mt, alpha,
    hard_neg_class_map, n_classes, use_pgi, args,
):
    """
    One training epoch.

    Per-step ordering (rationale documented in ``flatmatch_core``):

      1. Labeled forward + supervised loss.
      2. Unlabeled clean forward under ``no_grad`` → f_u (target).
         (Optionally also a weak-view teacher forward for the MT term.)
      3. Supervised backward — populates ``.grad`` with ∇L_l.
      4. Apply ε* (sourced from ``.grad`` in mode "full", from the EMA
         buffer in mode "e").
      5. In mode "e": EMA-update the buffer NOW, while ``.grad`` is still
         a clean ∇L_l. Doing it after the xs backward would contaminate
         M_{t+1} with ∇R_xs.
      6. Perturbed unlabeled forward → f̃_u.
      7. Restore params (undo ε*).
      8. λ_xs · R_xs(f̃_u, f_u.detach()) backward — accumulates ∇R_xs.
         If λ_mt > 0, also accumulates the MT consistency gradient on
         the strong-view student logits.
      9. Optimiser step + EMA teacher update.
    """
    student.train()
    freeze_bn_running_stats(student)

    total_sup = 0.0
    total_xs  = 0.0
    total_mt  = 0.0
    n_steps   = 0
    n_skipped_xs = 0   # mode "e" first step before buffer is warm

    use_mt = lambda_mt > 0.0

    pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}", leave=False)
    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        class_mask = build_class_mask(metas, hard_neg_class_map,
                                      n_classes, device)

        try:
            ub = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            ub = next(unlabeled_iter)
        audio_u = ub["audio"].to(device, non_blocking=True)
        sites_u = ub["sites"]

        optimizer.zero_grad()

        # --- (1) supervised forward + loss ---
        spec_l = spec_extractor(audio)
        logits_l = student(spec_l)
        T = min(logits_l.size(1), targets.size(1))
        loss_sup = logits_bce_focal_loss(
            logits_l[:, :T], targets[:, :T], mask[:, :T],
            class_mask=class_mask, pos_weight=pos_weight,
            use_focal=cfg.USE_FOCAL_LOSS,
        )

        # --- (2) unlabeled clean forward (target side, no grad) ---
        # One strong-aug view of the unlabeled batch is reused for both
        # f_u (no grad, at θ) and f̃_u (with grad, at θ+ε*). The paper's
        # Eq. 4 uses the same x_u in both branches.
        spec_u = make_strong_view(audio_u, sites_u, spec_extractor)
        with torch.no_grad():
            f_u = student(spec_u)

        # Optional MT consistency target (teacher on weak view, no grad)
        logits_u_t = None
        if use_mt:
            with torch.no_grad():
                spec_weak = make_weak_view(audio_u, sites_u, spec_extractor)
                logits_u_t = teacher.teacher(spec_weak)

        # --- (3) supervised backward → ∇L_l in .grad ---
        if torch.isnan(loss_sup) or torch.isinf(loss_sup):
            print(f"  *** NaN/Inf in L_sup at epoch {epoch}, skipping ***")
            continue
        loss_sup.backward()

        # --- (4) apply ε* (reads .grad if "full", buffer if "e") ---
        applied = perturbation.apply()

        # --- (5) EMA buffer update (mode "e" only), still on clean ∇L_l ---
        if args.flatmatch_mode == "e":
            perturbation.update_buffer()

        # --- (6) perturbed unlabeled forward, with grad ---
        if applied:
            f_tilde_u = student(spec_u)
        else:
            # First step in mode "e": buffer empty → no perturbation
            # available. Skip the cross-sharpness term for this step;
            # the buffer was just warmed up so subsequent steps will
            # apply normally.
            f_tilde_u = None
            n_skipped_xs += 1

        # --- (7) restore params (undo ε*) ---
        perturbation.restore()

        # --- (8) unlabeled-side loss + backward ---
        loss_xs_val = 0.0
        loss_mt_val = 0.0
        loss_u_terms = []

        if f_tilde_u is not None:
            f_tilde_u_a, f_u_a = align_lengths_pair(f_tilde_u, f_u)
            loss_xs = cross_sharpness_loss(f_tilde_u_a, f_u_a.detach())
            loss_u_terms.append(lambda_xs * loss_xs)
            loss_xs_val = loss_xs.item()

        if use_mt and logits_u_t is not None:
            # Re-use f_tilde_u as the "student strong" prediction if we
            # have it (gradient already attached to θ via θ+ε*), else
            # do a clean strong-view student forward.
            if f_tilde_u is not None:
                logits_u_s = f_tilde_u
            else:
                logits_u_s = student(spec_u)
            logits_u_s, logits_u_t_a = align_lengths_pair(
                logits_u_s, logits_u_t)
            loss_mt = _mt_consistency(args, logits_u_s, logits_u_t_a)
            loss_u_terms.append(lambda_mt * loss_mt)
            loss_mt_val = loss_mt.item()

        if loss_u_terms:
            loss_u = sum(loss_u_terms)
            if torch.isnan(loss_u) or torch.isinf(loss_u):
                print(f"  *** NaN/Inf in unlabeled loss at epoch {epoch}, "
                      f"skipping unlabeled backward ***")
            else:
                loss_u.backward()

        # --- (9) step + EMA teacher update ---
        nn.utils.clip_grad_norm_(student.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        teacher.update(student, alpha=alpha)

        total_sup += loss_sup.item()
        total_xs  += loss_xs_val
        total_mt  += loss_mt_val
        n_steps   += 1

        pbar.set_postfix(
            sup=f"{loss_sup.item():.4f}",
            xs=f"{loss_xs_val:.4f}",
            **({"mt": f"{loss_mt_val:.4f}"} if use_mt else {}),
            **{"λxs": f"{lambda_xs:.2f}", "α": f"{alpha:.4f}"},
        )

    if n_skipped_xs:
        print(f"  [mode 'e' warmup] cross-sharpness skipped on "
              f"{n_skipped_xs} step(s) before buffer was populated.")

    return (
        total_sup / max(n_steps, 1),
        total_xs  / max(n_steps, 1),
        total_mt  / max(n_steps, 1),
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
            f"flatmatch_hnm_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {args.output_dir}")

    unlabeled_bs = args.unlabeled_batch_size or cfg.BATCH_SIZE

    # ------------------------------------------------------------------
    # Hard negatives
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
        extra_tags = [
            "flatmatch", f"mode_{args.flatmatch_mode}",
            "hnm_injection", f"seed_{args.seed}",
            "pgi_on" if args.isolate_classes else "pgi_off",
            "stack_mt" if args.lambda_mt > 0 else "replace_mt",
        ]
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
                "flatmatch_mode":        args.flatmatch_mode,
                "rho":                   args.rho,
                "xs_ema_alpha":          args.xs_ema_alpha,
                "lambda_xs":             args.lambda_xs,
                "lambda_mt":             args.lambda_mt,
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
    # Labeled data
    # ------------------------------------------------------------------
    print("\nLoading training data...")
    train_anns = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(train_anns, train_manifest)

    from segment_length_core import extend_segment_to_fixed_length
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

    # Validation
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

    # Unlabeled AADC
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

    # FlatMatch perturbation operates on the *unwrapped* student so
    # `named_parameters()` keys match between apply() and restore() even
    # under DataParallel.
    perturbation = FlatMatchPerturbation(
        student, rho=args.rho, mode=args.flatmatch_mode,
        ema_alpha=args.xs_ema_alpha,
    )
    print(f"\nFlatMatch: mode={args.flatmatch_mode}  ρ={args.rho}  "
          f"λ_xs={args.lambda_xs}  λ_mt={args.lambda_mt}  "
          f"ema_α={args.xs_ema_alpha}")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs (student)")
        student = nn.DataParallel(student)

    # ------------------------------------------------------------------
    # Loss + optimiser + scheduler
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
    print(f"\n{'='*70}\nFlatMatch+HNM training for {args.epochs} epochs"
          f"\n{'='*70}")

    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and (epoch - 1) % args.resample_every == 0:
            print(f"  [epoch {epoch}] resampling random negatives")
            labeled_ds.resample_negatives()
            labeled_loader = DataLoader(
                labeled_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        ramp = sigmoid_ramp(epoch - 1, args.lambda_ramp_epochs)
        lambda_xs = args.lambda_xs * ramp
        lambda_mt = args.lambda_mt * ramp
        alpha = cosine_alpha(epoch - 1, args.alpha_start, args.alpha_end,
                             args.alpha_warmup_epochs)

        print(f"\n{'='*70}\nEpoch {epoch}/{args.epochs}  "
              f"lr={current_lr:.2e}  λ_xs={lambda_xs:.3f}  "
              f"λ_mt={lambda_mt:.3f}  α={alpha:.4f}\n{'='*70}")

        sup_loss, xs_loss, mt_loss, unlab_iter = train_epoch_flatmatch_hnm(
            student=student, teacher=teacher, perturbation=perturbation,
            spec_extractor=spec_extractor, pos_weight=pos_weight,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlab_loader, unlabeled_iter=unlab_iter,
            optimizer=optimizer, device=device, epoch=epoch,
            lambda_xs=lambda_xs, lambda_mt=lambda_mt, alpha=alpha,
            hard_neg_class_map=hard_neg_class_map,
            n_classes=n_classes, use_pgi=args.isolate_classes,
            args=args,
        )

        val = validate_teacher(
            teacher_module=teacher.teacher,
            spec_extractor=spec_extractor, val_loader=val_loader,
            criterion=baseline_criterion, device=device,
            thresholds=thresholds, val_annotations=val_anns,
            file_start_dts=file_start_dts, select_by=args.select_by,
            val_workers=args.val_workers,
        )
        thresholds = torch.tensor(val["thresholds"], device=device).float()
        scheduler.step(val["selection_f1"])

        print(f"\n  Sup loss: {sup_loss:.4f}  XS loss: {xs_loss:.4f}  "
              f"MT loss: {mt_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Teacher {args.select_by} F1: {val['selection_f1']:.3f}  "
              f"Best: {best_f1:.3f}")

        if not args.no_wandb:
            import wandb
            payload = {
                "epoch":            epoch,
                "lr":               current_lr,
                "lambda_xs":        lambda_xs,
                "lambda_mt":        lambda_mt,
                "alpha":            alpha,
                "train/sup_loss":   sup_loss,
                "train/xs_loss":    xs_loss,
                "train/mt_loss":    mt_loss,
                "val/loss":         val["loss"],
                "val/f1_overall":   val["overall_f1"],
                "val/f1_macro":     val["macro_f1"],
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
            "lambda_xs":          lambda_xs,
            "lambda_mt":          lambda_mt,
            "alpha":              alpha,
            "flatmatch_mode":     args.flatmatch_mode,
            "rho":                args.rho,
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
            verdict=f"FlatMatch+HNM best {args.select_by} F1 = {best_f1:.3f}",
            best_ckpt=args.output_dir / "best_model.pt",
        )


if __name__ == "__main__":
    main()
