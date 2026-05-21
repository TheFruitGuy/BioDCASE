"""
Canonical Recipe Training — Email-corrected Reproduction
========================================================

Companion to ``train.py``. Implements the WhaleVAD training recipe as
clarified in Geldenhuys' email of 2026-05-21. The existing ``train.py``
diverges from the paper in several ways that grew up organically during
reproduction; this script collapses those deviations back to the
canonical setup.

What this script does differently from ``train.py``
---------------------------------------------------

1. **Loss = pure weighted BCE.** ``USE_FOCAL_LOSS`` is forced off, even
   if ``config.py`` has it on. The paper's Table 2 ``"+ Focal loss"`` row
   means focal *replaces* weighted BCE, not stacks on top of it. The
   stacked configuration is not validated by anyone.

2. **LR = fixed.** No ``ReduceLROnPlateau``, no warmup, no decay. Paper
   says LR is held constant. The previous LR bump from 1e-5 to 5e-5 was
   driven by the small-gradient symptom of focal-on-WBCE; with pure WBCE
   the paper's 1e-5 should be usable. Default is 1e-5 (``--recipe paper``)
   with ``--recipe desktop`` available for 1e-3 (Christiaan's smaller-batch
   suggestion).

3. **Epochs = fixed, no early stopping.** Default 20 epochs to match the
   paper budget. Early stopping is intentionally absent: with the canonical
   recipe the run length is the recipe.

4. **Multi-GPU = DDP + SyncBatchNorm.** Replaces ``nn.DataParallel`` with
   ``torch.nn.parallel.DistributedDataParallel`` (DDP). All ``BatchNorm2d``
   layers are converted to ``SyncBatchNorm`` so per-GPU statistics are
   aggregated across replicas — important on 4× A40 with small per-GPU
   batches where un-synced BN injects noise.

5. **pos_weight computed at frame level.** Replaces ``compute_class_weights``
   (which counts files) with ``compute_pos_weights_framewise`` from
   ``dataset_canonical.py`` (which counts the actual frames the BCE loss
   sees). Padded frames are not present at this stage and so cannot
   contaminate the count.

6. **Padding-mask audit on the first batch.** A defensive assertion that
   the boolean mask matches the implied segment lengths and that padded
   target positions are zero. Catches silent drift between collation and
   loss; runs once, costs microseconds.

7. **No bounding-box head.** Already absent in this codebase; mentioned
   here for completeness because the email explicitly confirmed the BB
   module is *removed* from the final 0.440 configuration (not just
   omitted at inference).

Launching
---------

Single-node, 4 GPUs (canonical run on 4× A40)::

    torchrun --standalone --nproc_per_node=4 train_canonical.py \\
        --recipe paper --epochs 20 --batch-per-gpu 32 --seed 42

Single GPU (smoke test or low-budget run)::

    torchrun --standalone --nproc_per_node=1 train_canonical.py \\
        --recipe paper --epochs 20 --batch-per-gpu 32 --seed 42

Multi-seed sweep on 4 GPUs (one seed per GPU, parallel)::

    for s in 42 1337 9999 7777; do
      CUDA_VISIBLE_DEVICES=$((s % 4)) torchrun --standalone --nproc_per_node=1 \\
          --master_port=$((29500 + s % 100)) train_canonical.py \\
          --recipe paper --epochs 20 --batch-per-gpu 64 --seed $s &
    done
    wait

The desktop recipe (``--recipe desktop``) uses ``lr=1e-3`` and ``epochs=15``
per Christiaan's smaller-batch suggestion.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss
from dataset import (
    get_file_manifest, load_annotations,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, TrainingDatasetWithResample, collate_fn,
)
from dataset_canonical import (
    compute_pos_weights_framewise, assert_mask_is_correct,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level, collapse_probs_to_3class,
)


# ======================================================================
# Recipe presets
# ======================================================================

RECIPES = {
    # Paper recipe: long training, very small LR, large effective batch.
    # The paper's "very large batch size and long segments" plus L40 VRAM
    # implied an effective batch well above 128. With 4× A40 at 32/GPU we
    # land at 128 effective; bump --batch-per-gpu to 64 for 256.
    "paper":   {"lr": 1e-5, "epochs": 20},

    # Desktop recipe (Christiaan's suggestion): 100× the paper LR with a
    # shorter run, validated on smaller batches. Useful as a fallback or
    # for sanity-checking the loss path on a single GPU.
    "desktop": {"lr": 1e-3, "epochs": 15},
}


# ======================================================================
# DDP plumbing
# ======================================================================

def ddp_setup() -> tuple[int, int, int, torch.device]:
    """
    Initialize ``torch.distributed`` from torchrun-provided env vars.

    torchrun sets ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``,
    and ``MASTER_PORT`` automatically. We honour them and bind the current
    process to the matching local GPU.

    Returns
    -------
    rank, local_rank, world_size, device
    """
    if "RANK" not in os.environ:
        # Not launched under torchrun. Fall back to single-process mode so
        # the script remains runnable as ``python train_canonical.py`` for
        # smoke tests, but with a clear notice.
        print("[ddp_setup] RANK not in env — single-process mode (no DDP)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[ddp_setup] world_size={world_size}, backend={backend}")
    return rank, local_rank, world_size, device


def ddp_cleanup():
    """Tear down the process group if we initialised one."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    """Convenience guard for things that should only run on rank 0."""
    return rank == 0


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", choices=list(RECIPES), default="paper",
                   help="Hyperparameter preset (paper or desktop).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the recipe's epoch count.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override the recipe's learning rate.")
    p.add_argument("--batch-per-gpu", type=int, default=32,
                   help="Per-GPU minibatch size (effective batch = "
                        "--batch-per-gpu × WORLD_SIZE).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for this run.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for checkpoints. Defaults to "
                        "OUTPUT_DIR/whalevad_canonical_<timestamp>_seed<S>.")
    p.add_argument("--resample-every", type=int, default=1,
                   help="Resample negative pool every N epochs (paper=1).")
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging entirely.")
    return p.parse_args()


# ======================================================================
# Loss / weight helpers
# ======================================================================

class CanonicalWBCELoss(nn.Module):
    """
    Pure weighted BCE with a padding mask. No focal modulation.

    Implementation is a stripped-down copy of ``WhaleVADLoss`` that
    explicitly cannot apply focal — this avoids any chance of a stray
    ``USE_FOCAL_LOSS=True`` in ``config.py`` quietly re-enabling focal
    via the shared loss module.

    Parameters
    ----------
    pos_weight : torch.Tensor, shape (num_classes,)
        Per-class positive-class weight for ``BCEWithLogitsLoss``.
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(
        self,
        logits: torch.Tensor,                  # (B, T, C)
        targets: torch.Tensor,                 # (B, T, C)
        padding_mask: torch.Tensor | None,     # (B, T) bool
    ) -> torch.Tensor:
        pw = self.pos_weight.view(1, 1, -1)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none",
        )
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            bce = bce * mask
            denom = mask.sum() * logits.size(-1) + 1e-8
            return bce.sum() / denom
        return bce.mean()


def align_lengths(logits, targets, mask):
    """Reconcile small T-mismatches between logits and (targets, mask)."""
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
# Train / validate
# ======================================================================

def train_epoch(model, spec, loader, criterion, optimizer, device, epoch,
                rank: int, total_epochs: int, audit_first_batch: bool):
    model.train()
    total_loss, n = 0.0, 0
    audited = not audit_first_batch
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}",
                disable=not is_main(rank), leave=False)

    for audio, targets, mask, metas in pbar:
        if not audited:
            # Run the assertion on a real batch before any GPU transfer.
            # Failure here aborts training with a clear error rather than
            # silently leaking gradient through padded positions.
            assert_mask_is_correct(audio, targets, mask, metas)
            if is_main(rank):
                print("[mask audit] first batch passed — padding mask "
                      "matches implied segment lengths and padded targets "
                      "are zero.")
            audited = True

        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        s = spec(audio)
        logits = model(s)
        targets, mask = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            if is_main(rank):
                print("*** NaN / Inf loss — skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        if is_main(rank):
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Reduce mean train loss across ranks for an honest log line.
    if dist.is_initialized():
        t = torch.tensor([total_loss, float(n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1].clamp(min=1)).item()
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, spec, loader, criterion, device,
             thresholds, val_annotations, file_start_dts):
    """
    Validation runs on rank 0 only (the val loader is constructed without
    a distributed sampler). Returns the same dict shape ``train.py`` uses
    so existing tuning helpers keep working.
    """
    model.eval()
    total_loss, nb = 0.0, 0
    all_probs: dict = {}

    for audio, targets, mask, metas in tqdm(loader, desc="Validate",
                                            leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        s = spec(audio)
        logits = model(s)
        targets, mask = align_lengths(logits, targets, mask)
        total_loss += criterion(logits, targets, mask).item()
        nb += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        hop = spec.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    all_probs = collapse_probs_to_3class(all_probs)

    gt_events = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # Per-class threshold sweep (same coarse grid as train.py).
    used = np.asarray(thresholds.cpu().numpy(), dtype=np.float64).copy()
    grids = [
        np.arange(0.20, 0.85, 0.05),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
    ]
    for c, name in enumerate(cfg.CALL_TYPES_3):
        best_f1, best_t = -1.0, used[c]
        for t in grids[c]:
            trial = used.copy()
            trial[c] = t
            preds = postprocess_predictions(all_probs, trial)
            m = compute_metrics(preds, gt_events, iou_threshold=0.3)
            f1 = m.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        used[c] = best_t

    pred_events = postprocess_predictions(all_probs, used)
    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    print(f"\n  Event-level F1 (tuned thresholds):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name)
        if m is None:
            continue
        print(f"    {name.upper():6} t={used[c]:.2f}  "
              f"TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}")

    return {
        "loss": total_loss / max(nb, 1),
        "mean_f1": overall_f1,
        "per_class": metrics,
        "thresholds": used.tolist(),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    recipe = RECIPES[args.recipe]
    lr = args.lr if args.lr is not None else recipe["lr"]
    epochs = args.epochs if args.epochs is not None else recipe["epochs"]

    rank, local_rank, world_size, device = ddp_setup()
    try:
        # Deterministic seeding per rank: same model init across ranks
        # (which DDP requires), but different worker/shuffle seeds so each
        # replica sees a different slice each epoch.
        wbu.seed_everything(args.seed, deterministic=False)

        # ------------------------------------------------------------------
        # Wandb (rank 0 only)
        # ------------------------------------------------------------------
        run = None
        if is_main(rank) and not args.no_wandb:
            run = wbu.init_phase(
                "canonical",
                extra_tags=["wbce_only", "fixed_lr", "ddp",
                            f"recipe_{args.recipe}", f"seed_{args.seed}"],
                config={
                    "recipe":          args.recipe,
                    "lr":              lr,
                    "epochs":          epochs,
                    "batch_per_gpu":   args.batch_per_gpu,
                    "world_size":      world_size,
                    "effective_batch": args.batch_per_gpu * world_size,
                    "seed":            args.seed,
                    "use_focal_loss":  False,
                    "use_weighted_bce": True,
                    "pos_weight_unit": "frame",
                    "resample_every":  args.resample_every,
                    "use_3class":      cfg.USE_3CLASS,
                    "n_classes":       cfg.n_classes(),
                    "lstm_hidden":     cfg.LSTM_HIDDEN,
                    "lstm_layers":     cfg.LSTM_LAYERS,
                },
            )

        # ------------------------------------------------------------------
        # Run directory (rank 0 only)
        # ------------------------------------------------------------------
        if is_main(rank):
            if args.output_dir:
                run_dir = Path(args.output_dir)
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                run_dir = (Path(cfg.OUTPUT_DIR) /
                           f"whalevad_canonical_{ts}_seed{args.seed}")
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Run dir: {run_dir}")
        else:
            run_dir = None

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        if is_main(rank):
            print("Loading training manifest...")
        train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
        train_annotations = load_annotations(cfg.TRAIN_DATASETS,
                                             manifest=train_manifest)
        if is_main(rank):
            print(f"  {len(train_manifest)} train files, "
                  f"{len(train_annotations)} annotations")

        pos_segs = build_positive_segments(train_annotations, train_manifest)
        train_ds = TrainingDatasetWithResample(
            pos_segs, train_manifest, train_annotations,
        )
        if is_main(rank):
            print(f"Training: {len(pos_segs)} pos + "
                  f"{len(train_ds.negative_segments)} neg")

        # Distributed sampler shards the dataset across ranks every epoch.
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed,
        ) if world_size > 1 else None

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_per_gpu,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        )

        # Validation: only rank 0 needs it; we evaluate event-level F1 on
        # the full validation set so sharding it across ranks would just
        # complicate the post-processing path.
        val_loader = None
        val_annotations = None
        file_start_dts = None
        if is_main(rank):
            val_manifest = get_file_manifest(cfg.VAL_DATASETS)
            val_annotations = load_annotations(cfg.VAL_DATASETS,
                                               manifest=val_manifest)
            val_segs = build_val_segments(val_manifest, val_annotations)
            val_ds = WhaleDataset(val_segs)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_per_gpu, shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn, pin_memory=True,
            )
            file_start_dts = {
                (r.dataset, r.filename): r.start_dt
                for _, r in val_manifest.iterrows()
            }
            print(f"Validation: {len(val_segs)} segments")

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        spec_extractor = SpectrogramExtractor().to(device)
        model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

        # Dummy forward to materialize the lazy projection layer BEFORE
        # wrapping in DDP — otherwise DDP would record a parameter set
        # that excludes the lazy layer and complain when it appears.
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            _ = model(spec_extractor(dummy))

        # SyncBatchNorm: turn every BN into one that reduces statistics
        # across DDP replicas. Critical when batch-per-GPU is small —
        # otherwise each replica's BN running stats see only its local
        # 32 samples and inject per-replica noise.
        if world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank],
                        find_unused_parameters=False)

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        if is_main(rank):
            print(f"Parameters: {n_params:,}")

        # ------------------------------------------------------------------
        # Loss with frame-level pos_weight
        # ------------------------------------------------------------------
        # Compute on the current (positives + negatives) set. Negatives
        # will be resampled later but the weight ratios are stable enough
        # across resamples that re-computing every epoch isn't needed —
        # the dominant signal is the positive-class scarcity, which is
        # fixed by the data.
        if is_main(rank):
            print("\nComputing frame-level pos_weight (canonical):")
        pos_weight = compute_pos_weights_framewise(
            train_ds.segments, verbose=is_main(rank),
        ).to(device)

        # Broadcast to all ranks so every replica uses the exact same
        # weight tensor (it depends on the negative sample, which differs
        # by random state per rank).
        if world_size > 1:
            dist.broadcast(pos_weight, src=0)
        criterion = CanonicalWBCELoss(pos_weight=pos_weight).to(device)

        # ------------------------------------------------------------------
        # Optimizer — fixed LR, no scheduler
        # ------------------------------------------------------------------
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(cfg.BETA1, cfg.BETA2),
        )
        if is_main(rank):
            print(f"Optimizer: AdamW lr={lr:.0e} wd={cfg.WEIGHT_DECAY} "
                  f"(no scheduler, no warmup)")
            print(f"Effective batch: {args.batch_per_gpu * world_size} "
                  f"({args.batch_per_gpu}/GPU × {world_size} GPUs)")
            print(f"Epochs: {epochs} (no early stopping)\n")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        best_f1 = 0.0
        thresholds = torch.tensor([0.5, 0.5, 0.5], device=device)

        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Resample negatives (paper says every epoch). Rebuilding the
            # sampler is required because train_ds.segments was replaced.
            if (epoch - 1) % args.resample_every == 0:
                if is_main(rank):
                    print(f"\n[epoch {epoch}] resampling negatives")
                # Only rank 0 actually draws negatives; all ranks rebuild
                # the loader against the new segments. To keep ranks in
                # sync we re-seed deterministically from (seed, epoch).
                if is_main(rank):
                    train_ds.resample_negatives()
                # Sync: replace segments on every rank from rank 0's copy.
                # The simplest portable approach is to have each rank do
                # its own resample with the same seed.
                seed_for_epoch = args.seed + epoch
                import random
                random.seed(seed_for_epoch)
                np.random.seed(seed_for_epoch)
                if not is_main(rank):
                    train_ds.resample_negatives()
                # Rebuild sampler + loader against the new segment list.
                train_sampler = DistributedSampler(
                    train_ds, num_replicas=world_size, rank=rank,
                    shuffle=True, seed=args.seed + epoch,
                ) if world_size > 1 else None
                train_sampler.set_epoch(epoch) if train_sampler else None
                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_per_gpu,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    num_workers=args.num_workers,
                    collate_fn=collate_fn, pin_memory=True,
                )

            if is_main(rank):
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"\n{'=' * 60}\nEpoch {epoch}/{epochs}  "
                      f"LR={current_lr:.2e}\n{'=' * 60}")

            train_loss = train_epoch(
                model, spec_extractor, train_loader, criterion, optimizer,
                device, epoch, rank, epochs,
                audit_first_batch=(epoch == 1),
            )

            # Validation on rank 0 only; other ranks wait at the barrier.
            if is_main(rank):
                val = validate(
                    model.module if isinstance(model, DDP) else model,
                    spec_extractor, val_loader, criterion, device,
                    thresholds, val_annotations, file_start_dts,
                )
                thresholds = torch.tensor(val["thresholds"], device=device,
                                          dtype=torch.float32)
                print(f"\n  Train loss: {train_loss:.4f}  "
                      f"Val loss: {val['loss']:.4f}")
                print(f"  Mean F1: {val['mean_f1']:.3f}  "
                      f"Best F1: {best_f1:.3f}")

                if run is not None:
                    import wandb
                    payload = {
                        "epoch":         epoch,
                        "lr":            optimizer.param_groups[0]["lr"],
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
                    wandb.log(payload, step=epoch)

                # Checkpointing — always unwrap DDP first.
                model_state = (model.module.state_dict()
                               if isinstance(model, DDP)
                               else model.state_dict())
                ckpt = {
                    "epoch":            epoch,
                    "model_state_dict": model_state,
                    "best_f1":          best_f1,
                    "thresholds":       thresholds.cpu(),
                    "recipe":           args.recipe,
                    "lr":               lr,
                    "seed":             args.seed,
                }
                torch.save(ckpt, run_dir / "latest_model.pt")
                if val["mean_f1"] > best_f1:
                    best_f1 = val["mean_f1"]
                    ckpt["best_f1"] = best_f1
                    torch.save(ckpt, run_dir / "best_model.pt")
                    print(f"  *** New best F1: {best_f1:.3f}")

            # Barrier so non-rank-0 workers don't race ahead.
            if dist.is_initialized():
                dist.barrier()

        # ------------------------------------------------------------------
        # Post-training threshold tuning + final checkpoint (rank 0 only)
        # ------------------------------------------------------------------
        if is_main(rank):
            print(f"\n{'=' * 60}\nTuning thresholds on best model"
                  f"\n{'=' * 60}")
            best_ckpt = torch.load(run_dir / "best_model.pt",
                                   map_location=device, weights_only=False)
            base_model = (model.module if isinstance(model, DDP) else model)
            base_model.load_state_dict(best_ckpt["model_state_dict"])

            tuned = tune_thresholds_event_level(
                base_model, spec_extractor, val_loader, device,
                val_annotations, file_start_dts,
            )
            print(f"Tuned thresholds: {tuned.tolist()}")

            torch.save({
                "model_state_dict": base_model.state_dict(),
                "thresholds":       torch.tensor(tuned),
                "best_f1":          best_f1,
                "recipe":           args.recipe,
                "lr":               lr,
                "seed":             args.seed,
            }, run_dir / "final_model.pt")

            print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
            print(f"Run dir: {run_dir}")

            if run is not None:
                import wandb
                wandb.summary["best_f1"]          = float(best_f1)
                wandb.summary["final_thresholds"] = list(map(float, tuned))
                wandb.summary["epochs_run"]       = epochs
                wandb.summary["effective_batch"]  = args.batch_per_gpu * world_size
                wandb.summary["verdict"] = (
                    f"Canonical (WBCE-only, fixed LR={lr:.0e}, "
                    f"{epochs} epochs, effective batch "
                    f"{args.batch_per_gpu * world_size}) finished at "
                    f"F1 {best_f1:.3f}."
                )
                art = wandb.Artifact(
                    f"canonical-{run.name}", type="model",
                    metadata={
                        "best_f1":  float(best_f1),
                        "recipe":   args.recipe,
                        "lr":       float(lr),
                        "seed":     int(args.seed),
                        "epochs":   int(epochs),
                    },
                )
                art.add_file(str(run_dir / "best_model.pt"))
                art.add_file(str(run_dir / "final_model.pt"))
                run.log_artifact(art, aliases=["canonical", "best"])
                wandb.finish()
    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
