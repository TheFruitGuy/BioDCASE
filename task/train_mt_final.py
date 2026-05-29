"""
Mean-Teacher + HNM — _final-native, 3-class & 7-class, with PGI (Block C)
=========================================================================

Adds mean-teacher consistency regularisation on the unlabeled AADC stream on
top of the Block B HNM fine-tuning. Per your MT notes, MT *alone* degrades the
warm start — the HNM hard negatives are what keep the student moving usefully,
so this trainer always runs MT and HNM together.

Per step
--------
  * supervised: student on the labeled batch (positives + oversampled hard-negs
    + randoms) -> masked weighted-BCE with the PGI class mask. Identical to
    Block B (imported from train_hnm_final).
  * consistency: student on a STRONG-augmented unlabeled clip vs the EMA teacher
    on a WEAK view (detached) -> MSE (or confident / asymmetric_mse), weighted
    by lambda (sigmoid ramp 0 -> lambda_max).
  * total = sup + lambda * cons; step the student; EMA-update the teacher
    (alpha cosine-ramped start -> end). BN running stats are frozen during
    training (the unlabeled domain differs from the labeled one).

Validation runs on the TEACHER and is scored with the exact Block A/B paper-F1
(threshold sweep re-tuned from 0.5 each epoch, same as the baseline — no
warm-start shortcut, so the number stays comparable). best_model.pt stores the
TEACHER weights (+ the student) and the teacher's stitched 3-class posteriors.

Head (3c / 7c) is auto-detected from the checkpoint; PGI is subclass-level for
7-class. Loss matches the base (weighted BCE, pos_weight from the checkpoint,
focal off) so PGI / MT are the only things that vary.

Module dependencies
--------------------
Self-contained _final stack: mean_teacher_core_final + ssl_dataset_final +
ssl_augmentations_final (all delivered alongside this file), plus the Block B
supervised/validation pieces from train_hnm_final and the metric primitives
from rescore_base_epochs. The only runtime *data* dependency is the AADC
unlabeled audio at --aadc-root (same data your train_mean_teacher_hnm.py uses).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 python train_mt_final.py \
        --checkpoint runs/final_3c_s42_20260527_200054/paper_best.pt \
        --hard-negatives runs/hardnegs_final/3class/ensemble/{bmabz,d,bp}.json \
        --isolate-classes \
        --aadc-root /path/to/aadc --aadc-sites <site1> <site2> \
        --seed 42 --run-name mt_3c_s42_ens_pgi
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

# Supervised HNM half + validation, reused verbatim from Block B (DRY; keeps
# the supervised loss, PGI, USE_3CLASS protocol, and paper-F1 metric identical).
from train_hnm_final import (
    load_hard_negatives_json, build_hard_negative_segments,
    build_hard_neg_class_map, build_class_mask, HnmTrainingDataset,
    masked_bce, save_posteriors, validate,
)
from rescore_base_epochs import build_model, build_gt_events

# Mean-teacher mechanics (_final; config-agnostic, spec extractor passed in).
from mean_teacher_core_final import (
    EMATeacher, consistency_loss, consistency_loss_confident,
    consistency_loss_asymmetric, sigmoid_ramp, cosine_alpha,
    align_lengths_pair, make_weak_view, make_strong_view,
    freeze_bn_running_stats,
    # Block E: SAT (self-adaptive per-class threshold) + marginal alignment.
    AdaptiveClassThreshold, consistency_loss_adaptive,
    marginal_alignment_penalty,
)

# Unlabeled AADC stream (_final). Needs the AADC audio at --aadc-root present
# on the box; ssl_augmentations_final is pulled in lazily by the view builders.
from ssl_dataset_final import build_pretrain_manifest, SSLClipDataset, collate_ssl

try:
    import wandb_utils as wbu
except Exception:
    wbu = None


# MT + HNM defaults (train_mean_teacher_hnm.py)
EPOCHS = 20
LR = 1e-5
OVERSAMPLE = 5
RESAMPLE_EVERY = 5
EARLY_STOP = 8
LAMBDA_MAX = 1.0
LAMBDA_RAMP_EPOCHS = 3
ALPHA_START = 0.99
ALPHA_END = 0.999
ALPHA_WARMUP_EPOCHS = 5
EPOCH_UNLABELED_CLIPS = 10_000


def quarantine_check(aadc_sites):
    overlap = set(aadc_sites) & set(cfg.VAL_DATASETS)
    if overlap:
        raise SystemExit(f"AADC unlabeled sites overlap VAL sites {sorted(overlap)} "
                         f"— refusing to leak val data into the consistency stream.")


@torch.no_grad()
def estimate_label_prior(labeled_loader, n_classes, device):
    """Mean per-class positive frame-rate over the labeled stream (one pass, no
    model). Used as the reference prior for the marginal-alignment 'cap'. The
    labeled stream is positive-enriched, so this is a generous upper bound —
    exactly the semantics the one-sided cap wants (penalize a class that fires
    MORE on unlabeled ocean than on positive-enriched labeled clips)."""
    pos = torch.zeros(n_classes, device=device)
    valid = torch.zeros(n_classes, device=device)
    for _, targets, mask, _ in tqdm(labeled_loader, desc="prior", leave=False):
        targets = targets.to(device)
        m = mask.to(device).unsqueeze(-1).float()
        pos += (targets * m).sum(dim=(0, 1))
        valid += m.expand_as(targets).sum(dim=(0, 1))
    return (pos / valid.clamp(min=1.0)).clamp(1e-4, 1.0)


def train_epoch_mt(student, teacher, spec, pos_weight, labeled_loader,
                   unlab_loader, unlab_iter, optimizer, device, epoch,
                   lambda_w, alpha, hard_neg_class_map, n_classes, use_focal,
                   ctype, conf_thr, cpos, cneg,
                   sat=None, da_prior=None, lambda_da=0.0, da_mode="cap",
                   vgate=None):
    student.train()
    freeze_bn_running_stats(student)        # critical (cross-domain unlabeled)
    tot_sup, tot_cons, tot_da, n = 0.0, 0.0, 0.0, 0
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

        optimizer.zero_grad()

        # supervised (masked weighted BCE + PGI), same as Block B
        logits_l = student(spec(audio))
        T = min(logits_l.size(1), targets.size(1))
        loss_sup = masked_bce(logits_l[:, :T], targets[:, :T], mask[:, :T],
                              cm, pos_weight, use_focal)

        # consistency: student(strong) vs detached teacher(weak)
        logits_u_s = student(make_strong_view(audio_u, sites_u, spec))
        with torch.no_grad():
            logits_u_t = teacher.teacher(make_weak_view(audio_u, sites_u, spec))
        ls, lt = align_lengths_pair(logits_u_s, logits_u_t)

        if ctype == "mse":
            loss_cons = consistency_loss(ls, lt)
        elif ctype == "confident":
            loss_cons = consistency_loss_confident(ls, lt, conf_threshold=conf_thr)
        elif ctype == "asymmetric_mse":
            loss_cons = consistency_loss_asymmetric(
                ls, lt, pos_weight=cpos, neg_weight=cneg, conf_threshold=conf_thr)
        else:  # "sat" — per-class adaptive threshold (+ optional verifier-gate)
            with torch.no_grad():
                t_probs = torch.sigmoid(lt)
                sat.update(t_probs)
            thr = sat.thresholds()
            gate = None
            if vgate is not None:
                g = vgate.gate_mask(audio_u, t_probs)        # (B, T_t, C) in {0,1}
                Tg = min(g.size(1), ls.size(1))
                ls, lt, gate = ls[:, :Tg], lt[:, :Tg], g[:, :Tg]
            loss_cons = consistency_loss_adaptive(ls, lt, thr, gate=gate)

        # anti-collapse marginal alignment on the unlabeled student prediction
        if lambda_da > 0.0 and da_prior is not None:
            loss_da = marginal_alignment_penalty(logits_u_s, da_prior, mode=da_mode)
        else:
            loss_da = logits_u_s.new_zeros(())

        # DA shares the consistency sigmoid ramp (lambda_w) so it phases in with MT.
        loss = loss_sup + lambda_w * (loss_cons + lambda_da * loss_da)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        teacher.update(student, alpha=alpha)

        tot_sup += loss_sup.item(); tot_cons += loss_cons.item()
        tot_da += float(loss_da); n += 1
        pbar.set_postfix(sup=f"{loss_sup.item():.4f}", cons=f"{loss_cons.item():.4f}",
                         da=f"{float(loss_da):.4f}",
                         **{"λ": f"{lambda_w:.2f}", "α": f"{alpha:.4f}"})
    return (tot_sup / max(n, 1), tot_cons / max(n, 1),
            tot_da / max(n, 1), unlab_iter)


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
    # MT schedules
    p.add_argument("--lambda-max", type=float, default=LAMBDA_MAX)
    p.add_argument("--lambda-ramp-epochs", type=int, default=LAMBDA_RAMP_EPOCHS)
    p.add_argument("--alpha-start", type=float, default=ALPHA_START)
    p.add_argument("--alpha-end", type=float, default=ALPHA_END)
    p.add_argument("--alpha-warmup-epochs", type=int, default=ALPHA_WARMUP_EPOCHS)
    # consistency variant
    p.add_argument("--consistency-type", default="mse",
                   choices=["mse", "confident", "asymmetric_mse", "sat"])
    p.add_argument("--conf-threshold", type=float, default=0.7)
    p.add_argument("--pos-weight", type=float, default=2.0, help="asymmetric_mse teacher-positive mult.")
    p.add_argument("--neg-weight", type=float, default=1.0, help="asymmetric_mse teacher-negative mult.")
    # SAT (self-adaptive per-class threshold) — used when --consistency-type sat
    p.add_argument("--sat-ema", type=float, default=0.999,
                   help="EMA decay for the SAT positive-side confidence tracker.")
    p.add_argument("--sat-tau", type=float, default=0.7,
                   help="SAT anchor gate for the most-confident class (== the old "
                        "fixed conf-threshold; SAT reduces to 'confident' if all "
                        "classes are equally confident).")
    p.add_argument("--sat-lo", type=float, default=0.30, help="Lower clamp on SAT gates.")
    p.add_argument("--sat-hi", type=float, default=0.90, help="Upper clamp on SAT gates.")
    # DA (marginal anti-collapse) — applied on the unlabeled stream, any ctype
    p.add_argument("--lambda-da", type=float, default=0.0,
                   help="Weight on the marginal-alignment penalty (0=off). Shares "
                        "the consistency sigmoid ramp, so peaks at lambda_da.")
    p.add_argument("--da-mode", default="cap", choices=["cap", "kl"],
                   help="'cap': one-sided excess-rate penalty (safe default, the "
                        "BMABZ-collapse guard). 'kl': symmetric Bernoulli KL.")
    p.add_argument("--da-prior", type=float, nargs="+", default=None,
                   help="Per-class reference positive-rate in MODEL class order "
                        "(3 or 7 values). Default: measured once from the labeled "
                        "stream via estimate_label_prior.")
    # Verifier-gate (optional stretch arm; 3-class heads only)
    p.add_argument("--verifier-gate", action="store_true",
                   help="Gate the target-class consistency through the stage-2 "
                        "SupCon verifier. Requires --consistency-type sat and a "
                        "3-class checkpoint.")
    p.add_argument("--verifier-checkpoint", default=None,
                   help="Verifier best.pt (required with --verifier-gate).")
    p.add_argument("--gate-classes", nargs="+", default=["d"],
                   help="CALL_TYPES_3 names whose positive pseudo-labels the "
                        "verifier gates (default: d).")
    p.add_argument("--verifier-accept-threshold", type=float, default=0.5,
                   help="Verifier P(real) below which a teacher event is masked out.")
    p.add_argument("--verifier-fire-threshold", type=float, default=0.5,
                   help="Teacher prob above which a frame is a candidate-event frame.")
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

    fp_records, hnm_meta = load_hard_negatives_json(args.hard_negatives)
    targets_used = sorted({r["target_class"] for r in fp_records})
    print(f"Loaded {len(fp_records)} hard negatives; targets={targets_used} | "
          f"PGI={args.isolate_classes} | consistency={args.consistency_type}")

    run_name = args.run_name or f"mt_{head}_{Path(args.checkpoint).parent.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    run = None
    if not args.no_wandb and wbu is not None:
        try:
            tags = [f"target_{t}" for t in targets_used] + [
                "mt_hnm_final", head, "pgi_on" if args.isolate_classes else "pgi_off",
                f"consistency_{args.consistency_type}",
                "focal" if args.focal else "weighted_bce"]
            if args.lambda_da > 0.0:
                tags.append(f"da_{args.da_mode}")
            if args.verifier_gate:
                tags.append("verifier_gate")
            run = wbu.init_phase("6", extra_tags=tags, job_type="mt_hnm",
                                 config={"lr": args.lr, "epochs": args.epochs,
                                         "oversample": args.oversample, "seed": args.seed,
                                         "head": head, "n_classes": n_classes,
                                         "isolate_classes": args.isolate_classes,
                                         "lambda_max": args.lambda_max,
                                         "lambda_ramp_epochs": args.lambda_ramp_epochs,
                                         "alpha_start": args.alpha_start,
                                         "alpha_end": args.alpha_end,
                                         "consistency_type": args.consistency_type,
                                         "lambda_da": args.lambda_da,
                                         "da_mode": args.da_mode,
                                         "sat": (args.consistency_type == "sat"),
                                         "verifier_gate": bool(args.verifier_gate),
                                         "gate_classes": (list(args.gate_classes)
                                                          if args.verifier_gate else None),
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
    labeled_loader = DataLoader(labeled_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
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
    unlab_bs = args.unlabeled_batch_size or cfg.BATCH_SIZE
    unlab_manifest = build_pretrain_manifest(train_datasets=None,
                                             aadc_sites=list(args.aadc_sites),
                                             aadc_root=args.aadc_root)
    unlab_ds = SSLClipDataset(unlab_manifest, clip_seconds=cfg.TRAIN_SEGMENT_S,
                              sample_rate=cfg.SAMPLE_RATE,
                              epoch_clips=args.epoch_unlabeled_clips)
    unlab_loader = DataLoader(unlab_ds, batch_size=unlab_bs, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_ssl,
                              pin_memory=True, drop_last=True)
    unlab_iter = iter(unlab_loader)
    print(f"  unlabeled: {len(unlab_manifest)} files, "
          f"{args.epoch_unlabeled_clips} clips/epoch, bs={unlab_bs}")

    # ---- student (warm start) + EMA teacher -------------------------
    student, spec = build_model(n_classes, device)
    student.load_state_dict(ckpt["model_state_dict"], strict=False)
    pos_weight = (torch.tensor(ckpt["pos_weight"], dtype=torch.float32, device=device)
                  if ckpt.get("pos_weight") is not None else None)
    teacher = EMATeacher(student, alpha=args.alpha_start)
    teacher.to(device)

    optimizer = AdamW(student.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY, betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor,
                                  patience=args.lr_patience, min_lr=args.min_lr)

    # ---- Block E: SAT / DA / verifier-gate setup --------------------
    # Defaults (mse, lambda_da=0, no gate) leave all three None/0 -> the loop
    # is byte-for-byte the old Block C and existing runs reproduce.
    if args.verifier_gate and args.consistency_type != "sat":
        raise SystemExit("--verifier-gate only applies with --consistency-type sat.")

    sat = None
    if args.consistency_type == "sat":
        sat = AdaptiveClassThreshold(n_classes, ema=args.sat_ema,
                                     tau_global=args.sat_tau, init=args.sat_tau,
                                     lo=args.sat_lo, hi=args.sat_hi, device=device)
        print(f"  SAT on: ema={args.sat_ema} tau={args.sat_tau} "
              f"clamp=[{args.sat_lo},{args.sat_hi}]")

    da_prior = None
    if args.lambda_da > 0.0:
        if args.da_prior is not None:
            da_prior = torch.tensor(args.da_prior, dtype=torch.float32, device=device)
            if da_prior.numel() != n_classes:
                raise SystemExit(f"--da-prior needs {n_classes} values "
                                 f"(model class count), got {da_prior.numel()}")
        else:
            cfg.USE_3CLASS = is_3class           # workers fork with the right flag
            da_prior = estimate_label_prior(labeled_loader, n_classes, device)
        print(f"  DA on: mode={args.da_mode} lambda_da={args.lambda_da} "
              f"prior={[round(x, 4) for x in da_prior.tolist()]}")

    vgate = None
    if args.verifier_gate:
        if not is_3class:
            raise SystemExit("--verifier-gate is 3-class only (verifier heads are in "
                             "CALL_TYPES_3 space). Drop it for a 7-class run.")
        if not args.verifier_checkpoint:
            raise SystemExit("--verifier-gate requires --verifier-checkpoint.")
        from verifier_gate import VerifierGate
        vgate = VerifierGate(args.verifier_checkpoint, device,
                             gate_classes=tuple(args.gate_classes),
                             accept_threshold=args.verifier_accept_threshold,
                             fire_threshold=args.verifier_fire_threshold)
        print(f"  Verifier-gate on: {args.verifier_checkpoint} | gate={args.gate_classes} "
              f"| accept>{args.verifier_accept_threshold} | crop_s={vgate.crop_s}")

    # ---- epoch 0 (teacher == base) ----------------------------------
    print(f"\n{'='*60}\nInitial validation (teacher = base)\n{'='*60}")
    v0 = validate(teacher.teacher, spec, val_loader, device, gt_events, pos_weight,
                  n_classes, is_3class, args.val_workers, args.focal)
    print(f"  start paper-F1 = {v0['paper_f1']:.4f}")
    best_f1 = v0["paper_f1"]
    history = [{"epoch": 0, "paper_f1": best_f1, "per_class": v0["per_class"]}]
    save_posteriors(v0["probs3"], run_dir / "best_posteriors.npz")
    no_improve = 0

    print(f"\nMT+HNM {args.epochs} ep | λ_max={args.lambda_max} ramp={args.lambda_ramp_epochs}"
          f" | α {args.alpha_start}->{args.alpha_end} | PGI={args.isolate_classes}")

    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and (epoch - 1) % args.resample_every == 0:
            labeled_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(labeled_ds.segments)}")
        lambda_w = args.lambda_max * sigmoid_ramp(epoch - 1, args.lambda_ramp_epochs)
        alpha = cosine_alpha(epoch - 1, args.alpha_start, args.alpha_end,
                             args.alpha_warmup_epochs)
        cfg.USE_3CLASS = is_3class       # labeled workers fork with the model's flag

        t0 = time.time()
        sup, cons, da, unlab_iter = train_epoch_mt(
            student, teacher, spec, pos_weight, labeled_loader, unlab_loader,
            unlab_iter, optimizer, device, epoch, lambda_w, alpha,
            hard_neg_class_map, n_classes, args.focal, args.consistency_type,
            args.conf_threshold, args.pos_weight, args.neg_weight,
            sat=sat, da_prior=da_prior, lambda_da=args.lambda_da,
            da_mode=args.da_mode, vgate=vgate)
        v = validate(teacher.teacher, spec, val_loader, device, gt_events, pos_weight,
                     n_classes, is_3class, args.val_workers, args.focal)

        improved = v["paper_f1"] > best_f1
        print(f"\nEpoch {epoch:2d}/{args.epochs} ({time.time()-t0:.0f}s)"
              f"{'  *** new best' if improved else ''}")
        print(f"  sup {sup:.4f} | cons {cons:.4f} | da {da:.4f} | val {v['loss']:.4f} | "
              f"teacher paper-F1 {v['paper_f1']:.4f} (best {best_f1:.4f}) | λ={lambda_w:.2f}")
        if sat is not None:
            print(f"  SAT thr {[f'{t:.2f}' for t in sat.thresholds().tolist()]} "
                  f"({'/'.join(cfg.CALL_TYPES_3)})")
        for c in cfg.CALL_TYPES_3:
            m = v["per_class"].get(c, {})
            print(f"    {c.upper():6} P={m.get('precision',0):.3f} "
                  f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")

        scheduler.step(v["paper_f1"])
        history.append({"epoch": epoch, "paper_f1": v["paper_f1"],
                        "per_class": v["per_class"]})
        if run is not None:
            try:
                run.log({"epoch": epoch, "train/sup": sup, "train/cons": cons,
                         "train/da": da, "val/loss": v["loss"],
                         "paper_f1": v["paper_f1"], "lambda": lambda_w, "alpha": alpha})
            except Exception:
                pass

        student_module = student.module if isinstance(student, nn.DataParallel) else student
        save = {"epoch": epoch, "model_state_dict": teacher.state_dict(),
                "student_state_dict": student_module.state_dict(),
                "paper_f1": v["paper_f1"], "thresholds": torch.tensor(v["thresholds"]),
                "n_classes": n_classes, "isolate_classes": args.isolate_classes,
                "lambda": lambda_w, "alpha": alpha, "consistency_type": args.consistency_type,
                "lambda_da": args.lambda_da, "da_mode": args.da_mode,
                "sat": (args.consistency_type == "sat"),
                "sat_thresholds": (sat.thresholds().cpu().tolist() if sat is not None else None),
                "verifier_gate": bool(args.verifier_gate),
                "gate_classes": (list(args.gate_classes) if args.verifier_gate else None),
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
    vbits = []
    if args.lambda_da > 0:
        vbits.append(f"da={args.da_mode}@{args.lambda_da}")
    if args.verifier_gate:
        vbits.append("vgate=" + ",".join(args.gate_classes))
    vsuffix = (", " + ", ".join(vbits)) if vbits else ""
    verdict = (f"MT+HNM {head} PGI={'on' if args.isolate_classes else 'off'} "
               f"({args.consistency_type}{vsuffix}): teacher paper-F1 "
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
