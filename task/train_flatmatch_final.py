"""
FlatMatch + HNM — _final-native, 3-class & 7-class, with PGI
============================================================

Cross-sharpness semi-supervised regularisation (FlatMatch, Huang et al.
NeurIPS 2023) on the unlabeled AADC stream, layered on the SAME Block B
HNM hard-negative + PGI labeled stream that ``train_mt_final`` uses. This is
the FlatMatch sibling of ``train_mt_final.py``: byte-for-byte the same _final
stack, warm start, supervised half, validation, paper-F1 metric, and
checkpoint format — the ONLY thing that changes is the unlabeled-side
objective (cross-sharpness instead of mean-teacher consistency). That is what
makes its numbers directly comparable to the MT-from-base runs.

Per step
--------
  * supervised: student on the labeled batch (positives + oversampled hard-
    negs + randoms) -> masked weighted-BCE with the PGI class mask. Identical
    to ``train_hnm_final`` / ``train_mt_final`` (imported verbatim).
  * cross-sharpness: from the supervised gradient build a worst-case
    perturbation eps* = rho * grad(L_l) / ||grad(L_l)||_2 (mode 'e': EMA
    buffer of past grad(L_l); mode 'full': this step's fresh grad), then
    penalise the disagreement between student(theta + eps*) and
    student(theta) on a strong-augmented unlabeled clip (MSE on sigmoid
    probs). Eq. 4-5 of the paper.
  * total = L_sup + lambda_xs * R_xs; step the student; EMA-update the
    teacher. BN running stats frozen (cross-domain unlabeled), exactly as MT.

Gradient accumulation (for small GPUs)
--------------------------------------
``--grad-accum-steps K`` keeps the effective batch == cfg.BATCH_SIZE while
running micro-batches of size cfg.BATCH_SIZE // K, so the trainer fits on an
~11 GB card yet stays *mathematically identical* to the un-accumulated bs-32
run (and therefore comparable to the bs-32 MT runs). This is NOT a naive
"step every K": FlatMatch's eps* is a globally-normalised function of the
FULL-batch gradient, so each optimiser step is a TWO-PASS accumulation —

  Phase A (params = theta):  stream K labeled micro-batches, accumulate the
    full grad(L_l); for each paired unlabeled micro-batch store its strong
    view + clean target f_u = g_theta(x_u) (no grad).
  -> eps* built ONCE from the full accumulated grad(L_l) (mode 'full') or the
     EMA buffer (mode 'e'); applied once.
  Phase B (params = theta + eps*):  re-forward each stored unlabeled
    micro-batch, accumulate the full grad(R_xs).
  -> restore, step, EMA-teacher update.

Wrapping the per-batch loop with "step every K" would instead build K
different eps* from K noisier sub-gradients (micro-batch SAM) — a DIFFERENT
algorithm. K=1 reproduces the plain per-batch step exactly.

Validation runs on the TEACHER (EMA) and is scored with the exact Block
A/B/C paper-F1 (threshold sweep re-tuned from 0.5 each epoch), so the number
stays directly comparable to the MT runs. ``best_model.pt`` stores the TEACHER
weights (+ student) and the teacher's stitched 3-class posteriors.

Head (3c / 7c) is auto-detected from the checkpoint; PGI is subclass-level for
7-class. Loss matches the base (weighted BCE, pos_weight from the checkpoint,
focal off) so PGI / FlatMatch are the only things that vary.

Module dependencies
--------------------
Self-contained _final stack: ``flatmatch_core`` + ``mean_teacher_core_final``
(EMA teacher + view builders + schedules only) + ``ssl_dataset_final`` +
``ssl_augmentations_final``, plus the Block B supervised/validation pieces
from ``train_hnm_final`` and the metric primitives from ``rescore_base_epochs``.
The only runtime *data* dependency is the AADC unlabeled audio at --aadc-root.

Usage
-----
::

    # one 48 GB card (no accumulation):
    CUDA_VISIBLE_DEVICES=0 python train_flatmatch_final.py \
        --checkpoint runs/final_3c_s42_20260527_200054/paper_best.pt \
        --hard-negatives runs/hardnegs_final/3class/ensemble/{bmabz,d,bp}.json \
        --isolate-classes \
        --aadc-root /path/to/aadc --aadc-sites <site1> <site2> \
        --flatmatch-mode e --rho 0.1 --lambda-xs 1.0 \
        --seed 42 --run-name fmfb_3c_s42_ens_pgi

    # one ~11 GB card (micro-batch 16 x accum 2 = same effective batch 32):
    CUDA_VISIBLE_DEVICES=0 python train_flatmatch_final.py ... \
        --grad-accum-steps 2
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

import config_final as cfg
from dataset_final import (
    build_positive_segments, extend_segment_to_fixed_length, WhaleDataset,
    collate_fn, get_file_manifest, load_annotations, build_val_segments,
)

# Supervised HNM half + validation, reused verbatim from Block B (identical
# supervised loss, PGI, USE_3CLASS protocol, and paper-F1 metric as MT). This
# is the comparability backbone: the labeled side is the SAME code path the
# MT-from-base runs took.
from train_hnm_final import (
    load_hard_negatives_json, build_hard_negative_segments,
    build_hard_neg_class_map, build_class_mask, HnmTrainingDataset,
    masked_bce, save_posteriors, validate,
)
from rescore_base_epochs import build_model, build_gt_events

# Mean-teacher mechanics reused ONLY for: the EMA eval head, the weak/strong
# view builders, and the ramp/alpha schedules. The consistency losses are
# imported too but fire only when --lambda-mt > 0 (optional stacking arm).
from mean_teacher_core_final import (
    EMATeacher, consistency_loss, consistency_loss_confident,
    consistency_loss_asymmetric, sigmoid_ramp, cosine_alpha,
    align_lengths_pair, make_weak_view, make_strong_view,
    freeze_bn_running_stats,
)

# Unlabeled AADC stream (_final).
from ssl_dataset_final import build_pretrain_manifest, SSLClipDataset, collate_ssl

# FlatMatch cross-sharpness (codebase-agnostic; depends only on torch).
from flatmatch_core import FlatMatchPerturbation, cross_sharpness_loss

try:
    import wandb_utils as wbu
except Exception:
    wbu = None


# Defaults mirror train_mt_final.py (same warm start, same budget) so the only
# difference vs the MT runs is the unlabeled objective.
EPOCHS = 20
LR = 1e-5
OVERSAMPLE = 5
RESAMPLE_EVERY = 5
EARLY_STOP = 8
ALPHA_START = 0.99
ALPHA_END = 0.999
ALPHA_WARMUP_EPOCHS = 5
EPOCH_UNLABELED_CLIPS = 10_000
GRAD_ACCUM_STEPS = 1
# FlatMatch knobs (paper defaults).
FLATMATCH_MODE = "e"
RHO = 0.1
XS_EMA_ALPHA = 0.999
LAMBDA_XS = 1.0
LAMBDA_MT = 0.0
LAMBDA_RAMP_EPOCHS = 3


def quarantine_check(aadc_sites):
    overlap = set(aadc_sites) & set(cfg.VAL_DATASETS)
    if overlap:
        raise SystemExit(f"AADC unlabeled sites overlap VAL sites {sorted(overlap)} "
                         f"— refusing to leak val data into the unlabeled stream.")


def _mt_consistency(ctype, ls, lt, conf_thr, cpos, cneg):
    """Optional MT consistency term (only used when lambda_mt > 0)."""
    if ctype == "mse":
        return consistency_loss(ls, lt)
    if ctype == "confident":
        return consistency_loss_confident(ls, lt, conf_threshold=conf_thr)
    return consistency_loss_asymmetric(ls, lt, pos_weight=cpos, neg_weight=cneg,
                                       conf_threshold=conf_thr)


def train_epoch_flatmatch(student, teacher, perturbation, spec, pos_weight,
                          labeled_loader, unlab_loader, unlab_iter, optimizer,
                          device, epoch, lambda_xs, lambda_mt, alpha,
                          hard_neg_class_map, n_classes, use_focal,
                          flatmatch_mode, ctype, conf_thr, cpos, cneg,
                          accum_steps):
    """
    One FlatMatch+HNM epoch with optional two-pass gradient accumulation.

    accum_steps == 1 is the plain per-batch FlatMatch step. accum_steps > 1
    builds ONE optimiser step from accum_steps micro-batches such that the
    result is *bit-identical* to the un-accumulated step at the same effective
    batch:

      * supervised term: ``masked_bce`` is a MEAN over valid frame x class
        entries (denominator = the valid count over the first
        T=min(logits_len, target_len) frames), and that count varies per
        micro-batch (PGI). A flat 1/K would be wrong. Instead each micro-batch
        backward-s its UN-normalised weighted sum (masked_bce x its own count,
        the count taken AFTER the forward so T is exact), accumulating
        grad(sum_total) into .grad; then .grad is scaled once by 1/count_total.
        That equals grad(full-batch masked_bce) exactly.
      * cross-sharpness term: an unmasked mean over a fixed-shape tensor
        (constant element count per micro-batch), so 1/K is exact there.

    eps* is built ONCE from the full accumulated grad(L_l) (or the EMA buffer
    in mode 'e'); the perturbed forwards reuse the stored strong views.
    """
    student.train()
    freeze_bn_running_stats(student)        # critical (cross-domain unlabeled)
    tot_sup, tot_xs, tot_mt, n = 0.0, 0.0, 0.0, 0
    n_skipped_xs = 0
    use_mt = lambda_mt > 0.0

    def _run_step(batch_group):
        """batch_group: list of (audio, targets, mask, cm, audio_u, sites_u)."""
        nonlocal tot_sup, tot_xs, tot_mt, n, n_skipped_xs
        optimizer.zero_grad()

        # --- PHASE A (params = theta): un-normalised weighted sup grad + targets ---
        stored = []
        count_total = torch.zeros((), device=device)
        sup_sum = torch.zeros((), device=device)        # sum of (masked_bce_k * cnt_k)
        for audio, targets, mask, cm, audio_u, sites_u in batch_group:
            logits_l = student(spec(audio))
            T = min(logits_l.size(1), targets.size(1))
            raw_sup = masked_bce(logits_l[:, :T], targets[:, :T], mask[:, :T],
                                 cm, pos_weight, use_focal)
            # exact denominator masked_bce used (mask sliced to the real T)
            valid = mask[:, :T].unsqueeze(-1).float() * cm.unsqueeze(1)
            cnt = valid.sum().clamp(min=1.0).detach()
            if not (torch.isnan(raw_sup) or torch.isinf(raw_sup)):
                (raw_sup * cnt).backward()               # accumulates grad(sum_k)
                count_total = count_total + cnt
                sup_sum = sup_sum + raw_sup.detach() * cnt
            # one strong view per unlabeled micro-batch, reused for f_u (theta)
            # and f_tilde_u (theta+eps*), per the paper's Eq. 4
            spec_u = make_strong_view(audio_u, sites_u, spec).detach()
            with torch.no_grad():
                f_u = student(spec_u).detach()
                logits_u_t = (teacher.teacher(make_weak_view(audio_u, sites_u, spec)).detach()
                              if use_mt else None)
            stored.append((spec_u, f_u, logits_u_t))

        # normalise the accumulated supervised gradient by the TOTAL count ->
        # exactly grad(full-batch masked_bce). (eps* is scale-invariant, but the
        # supervised contribution to the step and the mode-'e' buffer need this.)
        inv = 1.0 / count_total.clamp(min=1.0)
        for p in student.parameters():
            if p.grad is not None:
                p.grad.mul_(inv)

        # --- eps* from the full accumulated grad(L_l) (or buffer in mode 'e') ---
        applied = perturbation.apply()
        if flatmatch_mode == "e":
            perturbation.update_buffer()                  # while .grad is clean grad(L_l)

        # --- PHASE B (params = theta+eps*): cross-sharpness grad, 1/K exact ---
        k = len(stored)
        xs_log, mt_log = 0.0, 0.0
        if applied:
            for spec_u, f_u, logits_u_t in stored:
                f_tilde = student(spec_u)
                fa, ua = align_lengths_pair(f_tilde, f_u)
                loss_xs = cross_sharpness_loss(fa, ua.detach())
                terms = [(lambda_xs * loss_xs) / k]
                xs_log += loss_xs.item()
                if use_mt and logits_u_t is not None:
                    ls, lt = align_lengths_pair(f_tilde, logits_u_t)
                    loss_mt = _mt_consistency(ctype, ls, lt, conf_thr, cpos, cneg)
                    terms.append((lambda_mt * loss_mt) / k)
                    mt_log += loss_mt.item()
                loss_u = sum(terms)
                if not (torch.isnan(loss_u) or torch.isinf(loss_u)):
                    loss_u.backward()                     # accumulates grad(R_xs)
        else:
            n_skipped_xs += k                             # mode 'e' before buffer is warm

        perturbation.restore()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        teacher.update(student, alpha=alpha)
        tot_sup += float((sup_sum / count_total.clamp(min=1.0)).item())
        tot_xs += xs_log / max(k, 1); tot_mt += mt_log / max(k, 1); n += 1

    batch_group = []
    pbar = tqdm(labeled_loader, desc=f"ep{epoch}", leave=False)
    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        cm = build_class_mask(metas, hard_neg_class_map, n_classes, device)

        try:
            ub = next(unlab_iter)
        except StopIteration:
            unlab_iter = iter(unlab_loader)
            ub = next(unlab_iter)
        audio_u = ub["audio"].to(device, non_blocking=True)
        sites_u = ub["sites"]

        batch_group.append((audio, targets, mask, cm, audio_u, sites_u))
        if len(batch_group) == accum_steps:
            _run_step(batch_group)
            batch_group = []
            pbar.set_postfix(sup=f"{tot_sup/max(n,1):.4f}", xs=f"{tot_xs/max(n,1):.4f}",
                             **({"mt": f"{tot_mt/max(n,1):.4f}"} if use_mt else {}),
                             **{"λxs": f"{lambda_xs:.2f}", "α": f"{alpha:.4f}"})

    if batch_group:                                       # trailing partial group
        _run_step(batch_group)

    if n_skipped_xs:
        print(f"  [mode 'e' warmup] cross-sharpness skipped on {n_skipped_xs} "
              f"micro-batch(es) before the gradient buffer was populated.")
    return tot_sup / max(n, 1), tot_xs / max(n, 1), tot_mt / max(n, 1), unlab_iter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hard-negatives", nargs="+", required=True)
    p.add_argument("--isolate-classes", action="store_true")
    p.add_argument("--oversample", type=int, default=OVERSAMPLE)
    # unlabeled AADC stream
    p.add_argument("--aadc-root", required=True)
    p.add_argument("--aadc-sites", nargs="+", required=True)
    p.add_argument("--unlabeled-batch-size", type=int, default=None)
    p.add_argument("--epoch-unlabeled-clips", type=int, default=EPOCH_UNLABELED_CLIPS)
    # memory / batch
    p.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS,
                   help="Two-pass gradient accumulation. Keeps the effective batch "
                        "== cfg.BATCH_SIZE while using micro-batches of "
                        "cfg.BATCH_SIZE // K (must divide evenly). K=1 = plain "
                        "per-batch step (default). Use 2 (micro 16) or 4 (micro 8) "
                        "to fit an ~11 GB card; the result is identical to bs-32.")
    # FlatMatch
    p.add_argument("--flatmatch-mode", default=FLATMATCH_MODE, choices=["e", "full"],
                   help="eps* source. 'e': EMA buffer of past grad(L_l) (paper "
                        "efficient variant); 'full': this step's fresh grad(L_l).")
    p.add_argument("--rho", type=float, default=RHO,
                   help="Perturbation magnitude ||eps*||_2 (paper default 0.1).")
    p.add_argument("--xs-ema-alpha", type=float, default=XS_EMA_ALPHA,
                   help="EMA decay for the labeled-gradient buffer (mode 'e').")
    p.add_argument("--lambda-xs", type=float, default=LAMBDA_XS,
                   help="Max weight on the cross-sharpness term R_xs.")
    p.add_argument("--lambda-mt", type=float, default=LAMBDA_MT,
                   help="Max weight on the OPTIONAL MT consistency term. 0 = "
                        "paper-faithful pure FlatMatch (default). Set >0 to stack "
                        "MT alongside FlatMatch (A/B sanity check only).")
    p.add_argument("--lambda-ramp-epochs", type=int, default=LAMBDA_RAMP_EPOCHS,
                   help="Sigmoid ramp epochs for lambda_xs (and lambda_mt if used).")
    # EMA teacher (evaluation head only)
    p.add_argument("--alpha-start", type=float, default=ALPHA_START)
    p.add_argument("--alpha-end", type=float, default=ALPHA_END)
    p.add_argument("--alpha-warmup-epochs", type=int, default=ALPHA_WARMUP_EPOCHS)
    # MT consistency variant (only used when --lambda-mt > 0)
    p.add_argument("--consistency-type", default="mse",
                   choices=["mse", "confident", "asymmetric_mse"])
    p.add_argument("--conf-threshold", type=float, default=0.7)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--neg-weight", type=float, default=1.0)
    # training
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--resample-every", type=int, default=RESAMPLE_EVERY)
    p.add_argument("--early-stop-patience", type=int, default=EARLY_STOP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=str(cfg.OUTPUT_DIR))
    p.add_argument("--val-workers", type=int, default=13)
    p.add_argument("--focal", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def seed_all(seed):
    if wbu is not None and hasattr(wbu, "seed_everything"):
        wbu.seed_everything(seed, deterministic=False)
    else:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_all(args.seed)
    quarantine_check(args.aadc_sites)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
    is_3class = (n_classes == 3)
    head = "3class" if is_3class else "7class"
    cfg.USE_3CLASS = is_3class
    print(f"Device: {device} | checkpoint head: {head} ({n_classes}-class)")

    # ---- effective batch / micro-batch (grad accumulation) ----------
    K = max(1, args.grad_accum_steps)
    if cfg.BATCH_SIZE % K != 0:
        raise SystemExit(f"--grad-accum-steps {K} must divide cfg.BATCH_SIZE "
                         f"({cfg.BATCH_SIZE}) evenly so the effective batch is "
                         f"preserved. Try K in {{1,2,4}}.")
    micro_bs = cfg.BATCH_SIZE // K
    unlab_eff = args.unlabeled_batch_size or cfg.BATCH_SIZE
    if unlab_eff % K != 0:
        raise SystemExit(f"--grad-accum-steps {K} must divide the unlabeled batch "
                         f"({unlab_eff}) evenly.")
    unlab_micro = unlab_eff // K
    if K > 1:
        print(f"Grad accumulation: K={K} | labeled micro-bs={micro_bs} "
              f"(effective {cfg.BATCH_SIZE}) | unlabeled micro-bs={unlab_micro} "
              f"(effective {unlab_eff}) — two-pass, identical to bs-{cfg.BATCH_SIZE}.")

    fp_records, hnm_meta = load_hard_negatives_json(args.hard_negatives)
    targets_used = sorted({r["target_class"] for r in fp_records})
    print(f"Loaded {len(fp_records)} hard negatives; targets={targets_used} | "
          f"PGI={args.isolate_classes} | FlatMatch mode={args.flatmatch_mode} "
          f"rho={args.rho} lambda_xs={args.lambda_xs} lambda_mt={args.lambda_mt}")

    run_name = args.run_name or (f"flatmatch_{head}_{Path(args.checkpoint).parent.name}"
                                 f"_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    run = None
    if not args.no_wandb and wbu is not None:
        try:
            tags = [f"target_{t}" for t in targets_used] + [
                "flatmatch_hnm_final", head,
                "pgi_on" if args.isolate_classes else "pgi_off",
                f"flatmatch_{args.flatmatch_mode}",
                ("pure_xs" if args.lambda_mt == 0 else "xs_plus_mt"),
                "focal" if args.focal else "weighted_bce"]
            run = wbu.init_phase("6", extra_tags=tags, job_type="flatmatch_hnm",
                                 config={"lr": args.lr, "epochs": args.epochs,
                                         "oversample": args.oversample, "seed": args.seed,
                                         "head": head, "n_classes": n_classes,
                                         "isolate_classes": args.isolate_classes,
                                         "grad_accum_steps": K,
                                         "effective_batch": cfg.BATCH_SIZE,
                                         "flatmatch_mode": args.flatmatch_mode,
                                         "rho": args.rho, "lambda_xs": args.lambda_xs,
                                         "lambda_mt": args.lambda_mt,
                                         "lambda_ramp_epochs": args.lambda_ramp_epochs,
                                         "alpha_start": args.alpha_start,
                                         "alpha_end": args.alpha_end,
                                         "aadc_sites": list(args.aadc_sites),
                                         "source_checkpoint": str(args.checkpoint),
                                         "hard_negatives": list(args.hard_negatives),
                                         "mining_targets": targets_used,
                                         "select_by": "macro_paper"})
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without logging.")
            run = None

    # ---- labeled data (built with the model's native flag) ----------
    train_anns = load_annotations(list(cfg.TRAIN_DATASETS))
    train_manifest = get_file_manifest(list(cfg.TRAIN_DATASETS))
    midx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [extend_segment_to_fixed_length(
                    s, cfg.TRAIN_SEGMENT_S,
                    float(midx.loc[(s.dataset, s.filename), "duration_s"]))
                for s in build_positive_segments(train_anns, train_manifest)
                if (s.dataset, s.filename) in midx.index]
    hard_segs, used = build_hard_negative_segments(fp_records, train_manifest, train_anns)
    print(f"  {len(pos_segs)} positives | {len(hard_segs)} hard-negs x{args.oversample}")
    hard_neg_class_map = (build_hard_neg_class_map(hard_segs, used, n_classes)
                          if args.isolate_classes else {})
    labeled_ds = HnmTrainingDataset(pos_segs, hard_segs, args.oversample,
                                    train_manifest, train_anns)
    labeled_loader = DataLoader(labeled_ds, batch_size=micro_bs, shuffle=True,
                                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                                pin_memory=True)

    # ---- validation -------------------------------------------------
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_loader = DataLoader(WhaleDataset(build_val_segments(val_manifest, val_anns)),
                            batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                            pin_memory=True)
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)

    # ---- unlabeled AADC stream --------------------------------------
    unlab_manifest = build_pretrain_manifest(train_datasets=None,
                                             aadc_sites=list(args.aadc_sites),
                                             aadc_root=args.aadc_root)
    unlab_ds = SSLClipDataset(unlab_manifest, clip_seconds=cfg.TRAIN_SEGMENT_S,
                              sample_rate=cfg.SAMPLE_RATE,
                              epoch_clips=args.epoch_unlabeled_clips)
    unlab_loader = DataLoader(unlab_ds, batch_size=unlab_micro, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_ssl,
                              pin_memory=True, drop_last=True)
    unlab_iter = iter(unlab_loader)
    print(f"  unlabeled: {len(unlab_manifest)} files, "
          f"{args.epoch_unlabeled_clips} clips/epoch, micro-bs={unlab_micro}")

    # ---- student (warm start) + EMA teacher -------------------------
    student, spec = build_model(n_classes, device)
    student.load_state_dict(ckpt["model_state_dict"], strict=False)
    pos_weight = (torch.tensor(ckpt["pos_weight"], dtype=torch.float32, device=device)
                  if ckpt.get("pos_weight") is not None else None)
    teacher = EMATeacher(student, alpha=args.alpha_start)
    teacher.to(device)

    # FlatMatch perturbation on the (unwrapped) student so named_parameters()
    # keys stay consistent between apply()/restore(). One visible GPU per run,
    # so no DataParallel (matches train_mt_final).
    perturbation = FlatMatchPerturbation(student, rho=args.rho,
                                         mode=args.flatmatch_mode,
                                         ema_alpha=args.xs_ema_alpha)

    optimizer = AdamW(student.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY, betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor,
                                  patience=args.lr_patience, min_lr=args.min_lr)

    # ---- epoch 0 (teacher == base) ----------------------------------
    print(f"\n{'='*60}\nInitial validation (teacher = base)\n{'='*60}")
    v0 = validate(teacher.teacher, spec, val_loader, device, gt_events, pos_weight,
                  n_classes, is_3class, args.val_workers, args.focal)
    print(f"  start paper-F1 = {v0['paper_f1']:.4f}")
    best_f1 = v0["paper_f1"]
    history = [{"epoch": 0, "paper_f1": best_f1, "per_class": v0["per_class"]}]
    save_posteriors(v0["probs3"], run_dir / "best_posteriors.npz")
    no_improve = 0

    print(f"\nFlatMatch+HNM {args.epochs} ep | mode={args.flatmatch_mode} rho={args.rho} "
          f"lambda_xs={args.lambda_xs} ramp={args.lambda_ramp_epochs} "
          f"lambda_mt={args.lambda_mt} | alpha {args.alpha_start}->{args.alpha_end} "
          f"| PGI={args.isolate_classes} | accum={K}")

    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and (epoch - 1) % args.resample_every == 0:
            labeled_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(labeled_ds.segments)}")
        ramp = sigmoid_ramp(epoch - 1, args.lambda_ramp_epochs)
        lambda_xs = args.lambda_xs * ramp
        lambda_mt = args.lambda_mt * ramp
        alpha = cosine_alpha(epoch - 1, args.alpha_start, args.alpha_end,
                             args.alpha_warmup_epochs)
        cfg.USE_3CLASS = is_3class       # labeled workers fork with the model's flag

        t0 = time.time()
        sup, xs, mt, unlab_iter = train_epoch_flatmatch(
            student, teacher, perturbation, spec, pos_weight, labeled_loader,
            unlab_loader, unlab_iter, optimizer, device, epoch,
            lambda_xs, lambda_mt, alpha, hard_neg_class_map, n_classes, args.focal,
            args.flatmatch_mode, args.consistency_type, args.conf_threshold,
            args.pos_weight, args.neg_weight, K)
        v = validate(teacher.teacher, spec, val_loader, device, gt_events, pos_weight,
                     n_classes, is_3class, args.val_workers, args.focal)

        improved = v["paper_f1"] > best_f1
        print(f"\nEpoch {epoch:2d}/{args.epochs} ({time.time()-t0:.0f}s)"
              f"{'  *** new best' if improved else ''}")
        print(f"  sup {sup:.4f} | xs {xs:.4f} | mt {mt:.4f} | val {v['loss']:.4f} | "
              f"teacher paper-F1 {v['paper_f1']:.4f} (best {best_f1:.4f}) | "
              f"lambda_xs={lambda_xs:.2f}")
        for c in cfg.CALL_TYPES_3:
            m = v["per_class"].get(c, {})
            print(f"    {c.upper():6} P={m.get('precision',0):.3f} "
                  f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")

        scheduler.step(v["paper_f1"])
        history.append({"epoch": epoch, "paper_f1": v["paper_f1"],
                        "per_class": v["per_class"]})
        if run is not None:
            try:
                run.log({"epoch": epoch, "train/sup": sup, "train/xs": xs,
                         "train/mt": mt, "val/loss": v["loss"],
                         "paper_f1": v["paper_f1"], "lambda_xs": lambda_xs,
                         "lambda_mt": lambda_mt, "alpha": alpha})
            except Exception:
                pass

        student_module = student.module if isinstance(student, nn.DataParallel) else student
        save = {"epoch": epoch, "model_state_dict": teacher.state_dict(),
                "student_state_dict": student_module.state_dict(),
                "paper_f1": v["paper_f1"], "thresholds": torch.tensor(v["thresholds"]),
                "n_classes": n_classes, "isolate_classes": args.isolate_classes,
                "grad_accum_steps": K, "effective_batch": cfg.BATCH_SIZE,
                "flatmatch_mode": args.flatmatch_mode, "rho": args.rho,
                "lambda_xs": lambda_xs, "lambda_mt": lambda_mt, "alpha": alpha,
                "source_checkpoint": str(args.checkpoint), "hnm_meta": hnm_meta,
                "aadc_sites": list(args.aadc_sites),
                "pos_weight": (pos_weight.cpu().tolist() if pos_weight is not None else None)}
        torch.save(save, run_dir / "latest_model.pt")
        if improved:
            best_f1 = v["paper_f1"]
            torch.save(save, run_dir / "best_model.pt")
            save_posteriors(v["probs3"], run_dir / "best_posteriors.npz")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.early_stop_patience:
            print(f"\n  Early stop after {no_improve} stale epochs")
            break

    # ---- per-class deltas + finalize --------------------------------
    delta = best_f1 - v0["paper_f1"]
    subs_of = {c3: {c7 for c7 in cfg.CALL_TYPES_7 if cfg.COLLAPSE_MAP[c7] == c3}
               for c3 in cfg.CALL_TYPES_3}
    print(f"\n{'='*60}")
    for c in cfg.CALL_TYPES_3:
        sc = v0["per_class"].get(c, {}).get("f1", 0.0)
        bc = max(h["per_class"].get(c, {}).get("f1", 0.0) for h in history)
        is_tgt = (c in targets_used) or bool(subs_of[c] & set(targets_used))
        print(f"  {c.upper():6} F1 {sc:.3f} -> {bc:.3f} ({bc-sc:+.3f})"
              f"{'  [target]' if is_tgt else ''}")
        if run is not None:
            try:
                run.summary[f"delta_f1_{c}"] = float(bc - sc)
                run.summary[f"is_target_{c}"] = bool(is_tgt)
            except Exception:
                pass
    verdict = (f"FlatMatch+HNM {head} PGI={'on' if args.isolate_classes else 'off'} "
               f"(mode={args.flatmatch_mode}, rho={args.rho}, accum={K}, "
               f"{'pure_xs' if args.lambda_mt == 0 else 'xs+mt'}): teacher paper-F1 "
               f"{v0['paper_f1']:.4f} -> {best_f1:.4f} ({delta:+.4f}).")
    print(f"{verdict}\nBest: {run_dir/'best_model.pt'}\n{'='*60}")
    if run is not None:
        try:
            run.summary["start_paper_f1"] = float(v0["paper_f1"])
            run.summary["best_paper_f1"] = float(best_f1)
            run.summary["delta_paper_f1"] = float(delta)
            run.summary["verdict"] = verdict
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
