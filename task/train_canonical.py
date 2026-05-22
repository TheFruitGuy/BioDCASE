"""
Canonical Recipe Training — Email-corrected Reproduction
========================================================

Companion to ``train.py``. Implements the WhaleVAD training recipe as
clarified in Geldenhuys' email of 2026-05-21, with CLI flags to adapt
the same recipe across hardware setups (large-VRAM cluster GPUs, small
desktop GPUs, mixed-precision-capable cards, etc.).

Recipe differences from ``train.py``
------------------------------------

1. **Loss = pure weighted BCE.** Focal modulation is hard-disabled.
2. **LR = fixed.** No scheduler, no warmup, no decay.
3. **Epochs = fixed.** No early stopping; the recipe sets the length.
4. **Multi-GPU = DDP + (optional) SyncBatchNorm.**
5. **pos_weight = frame-level.** Computed from annotation frames, not files.
6. **Padding-mask audit** runs on the first training batch.
7. **No bounding-box head.** Confirmed absent from the paper's 0.440 config.

Hardware-adaptation flags
-------------------------

``--amp {none,fp16,bf16}``
    Mixed precision via ``torch.amp.autocast``. ``fp16`` for Turing
    (2080 Ti and older — no bf16 support); ``bf16`` for Ampere+
    (A40 / A100 / 3000-series and newer) — same fp32 exponent range,
    so no GradScaler needed. Default ``none`` (full fp32).

``--grad-accum N``
    Accumulate gradients over N micro-batches before each optimizer step.
    Effective batch = ``--batch-per-gpu × WORLD_SIZE × --grad-accum``.
    In DDP, non-boundary micro-batches use ``model.no_sync()`` to avoid
    wasted gradient all-reduces.

``--no-sync-bn``
    Skip the ``SyncBatchNorm`` conversion that runs by default when
    ``WORLD_SIZE > 1``. Useful when per-GPU batch is already large
    (≥64) and local BN stats are low-noise — sync then is pure overhead.

Launching
---------

4× A40 (48 GB) canonical run::

    torchrun --standalone --nproc_per_node=4 train_canonical.py \\
        --recipe paper --batch-per-gpu 64 --amp bf16 --seed 42

8× 2080 Ti (11 GB) canonical run via fp16 + accumulation::

    torchrun --standalone --nproc_per_node=8 train_canonical.py \\
        --recipe paper --batch-per-gpu 16 --grad-accum 2 \\
        --amp fp16 --seed 42
    # effective batch = 16 × 8 × 2 = 256

8× 2080 Ti multi-seed sweep (one seed per card, parallel)::

    seeds=(42 1337 9999 7777 11 23 31 47)
    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i torchrun --standalone --nproc_per_node=1 \\
          --master_port=$((29500 + i)) train_canonical.py \\
          --recipe desktop --batch-per-gpu 16 --amp fp16 \\
          --no-sync-bn --seed ${seeds[$i]} &
    done
    wait

Single-GPU smoke test (any hardware)::

    torchrun --standalone --nproc_per_node=1 train_canonical.py \\
        --recipe paper --epochs 2 --batch-per-gpu 16 --amp fp16 --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from datetime import timedelta
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
from model import WhaleVAD
from dataset import (
    get_file_manifest, load_annotations,
    build_positive_segments, build_val_segments,
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
# Recipe and AMP presets
# ======================================================================

RECIPES = {
    "paper":   {"lr": 1e-5, "epochs": 20},
    "desktop": {"lr": 1e-3, "epochs": 15},
}

AMP_DTYPES = {
    "none": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# ======================================================================
# Recipe / batch sanity check
# ======================================================================
# The email gives us two validated operating points, not a continuous
# scaling law. These thresholds bracket "obviously wrong" combinations
# of recipe and effective batch size — between them is an underspecified
# middle ground that the email does not cover. The warnings nudge but do
# not block; pass --lr to override the recipe LR explicitly if you know
# what you're doing.

PAPER_MIN_EFFECTIVE_BATCH = 128    # below this, "very large batch" is a stretch
DESKTOP_MAX_EFFECTIVE_BATCH = 64   # above this, the desktop LR is likely too aggressive


def warn_if_recipe_batch_mismatch(
    recipe: str, effective_batch: int, lr: float, lr_overridden: bool,
) -> None:
    """Print a startup warning if the recipe / effective_batch combo is suspect."""
    if lr_overridden:
        # User explicitly chose an LR; assume they know what they want.
        return
    if recipe == "paper" and effective_batch < PAPER_MIN_EFFECTIVE_BATCH:
        print("*** WARNING: --recipe paper sets lr=1e-5, which Christiaan "
              "validated only with 'very large' batches.")
        print(f"    Your effective batch is {effective_batch}, "
              f"below the heuristic floor of {PAPER_MIN_EFFECTIVE_BATCH}.")
        print("    Consider --recipe desktop (lr=1e-3) for small-batch "
              "regimes, or pass --lr explicitly if you've validated a "
              "middle ground yourself.\n")
    elif recipe == "desktop" and effective_batch > DESKTOP_MAX_EFFECTIVE_BATCH:
        print("*** WARNING: --recipe desktop sets lr=1e-3, which Christiaan "
              "validated only on 'smaller batches'.")
        print(f"    Your effective batch is {effective_batch}, "
              f"above the heuristic ceiling of {DESKTOP_MAX_EFFECTIVE_BATCH}.")
        print("    Consider --recipe paper (lr=1e-5) for large-batch "
              "regimes, or pass --lr explicitly.\n")


# ======================================================================
# DDP plumbing
# ======================================================================

def ddp_setup() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        print("[ddp_setup] RANK not in env — single-process mode (no DDP)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # Default NCCL watchdog timeout is 10 min, which is too short here: only
    # rank 0 runs validation while ranks 1+ wait at dist.barrier(). With ~80k
    # val segments and per-class threshold sweeping that grows as the model
    # learns to predict positives, validation can take >10 min. Bump the
    # process-group timeout to 60 min so the barrier survives.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=60),
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[ddp_setup] world_size={world_size}, backend={backend}")
    return rank, local_rank, world_size, device


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Recipe / training duration
    p.add_argument("--recipe", choices=list(RECIPES), default="paper",
                   help="Hyperparameter preset.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the recipe's epoch count.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override the recipe's learning rate.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")

    # Hardware-shape flags
    p.add_argument("--batch-per-gpu", type=int, default=32,
                   help="Per-GPU minibatch size.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = "
                        "--batch-per-gpu × WORLD_SIZE × --grad-accum.")
    p.add_argument("--amp", choices=list(AMP_DTYPES), default="none",
                   help="Mixed precision dtype. fp16 for Turing (2080 Ti); "
                        "bf16 for Ampere+ (A40/A100/3000-series).")
    p.add_argument("--no-sync-bn", action="store_true",
                   help="Skip SyncBatchNorm conversion. Default is to "
                        "convert when WORLD_SIZE > 1.")
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS,
                   help="DataLoader worker processes per rank.")

    # Bookkeeping
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. Defaults to "
                        "OUTPUT_DIR/whalevad_canonical_<timestamp>_seed<S>.")
    p.add_argument("--resample-every", type=int, default=1,
                   help="Resample negative pool every N epochs (paper=1).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging entirely.")
    return p.parse_args()


# ======================================================================
# Loss
# ======================================================================

class CanonicalWBCELoss(nn.Module):
    """Pure weighted BCE with a padding mask. No focal modulation, ever."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets, padding_mask):
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
# Train epoch with AMP + gradient accumulation
# ======================================================================

def train_epoch(model, spec, loader, criterion, optimizer, device, epoch,
                rank: int, total_epochs: int, audit_first_batch: bool,
                amp_dtype: torch.dtype | None, scaler,
                grad_accum: int):
    """
    One training epoch with optional AMP and gradient accumulation.

    Gradient accumulation:
      - ``optimizer.zero_grad()`` runs only at the start of each cycle.
      - Per-micro-batch loss is divided by ``grad_accum`` so the sum
        across a cycle equals the mean over the full effective batch.
      - In DDP, ``model.no_sync()`` is used on non-boundary micro-batches
        to skip the gradient all-reduce until the cycle completes.

    Mixed precision:
      - ``torch.amp.autocast`` wraps the model forward and the loss.
      - For ``fp16`` we use a ``GradScaler`` to prevent gradient underflow.
      - For ``bf16`` no scaler is needed (bf16 has fp32's exponent range).
      - Spectrogram extraction stays in fp32 — STFT can be numerically
        sensitive and is cheap, so there's no reason to cast it.
    """
    model.train()
    total_loss, n = 0.0, 0
    audited = not audit_first_batch
    use_amp = amp_dtype is not None
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}",
                disable=not is_main(rank), leave=False)

    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)

    for batch_idx, (audio, targets, mask, metas) in enumerate(pbar):
        if not audited:
            assert_mask_is_correct(audio, targets, mask, metas)
            if is_main(rank):
                print("[mask audit] first batch passed.")
            audited = True

        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        is_step_boundary = (
            ((batch_idx + 1) % grad_accum == 0)
            or (batch_idx == n_batches - 1)
        )
        sync_ctx = (
            model.no_sync()
            if (isinstance(model, DDP) and not is_step_boundary)
            else nullcontext()
        )

        with sync_ctx:
            # STFT in fp32; only the network forward + loss run under AMP.
            s = spec(audio)
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp else nullcontext()
            )
            with amp_ctx:
                logits = model(s)
                targets_a, mask_a = align_lengths(logits, targets, mask)
                loss = criterion(logits, targets_a, mask_a) / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main(rank):
                    print(f"*** NaN/Inf loss at batch {batch_idx} — "
                          f"dropping accumulation cycle ***")
                if is_step_boundary:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_step_boundary:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Multiply back by grad_accum so logged loss is comparable across
        # different --grad-accum settings.
        total_loss += loss.item() * grad_accum
        n += 1
        if is_main(rank):
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

    if dist.is_initialized():
        t = torch.tensor([total_loss, float(n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1].clamp(min=1)).item()
    return total_loss / max(n, 1)


# ======================================================================
# Validation
# ======================================================================

@torch.no_grad()
def validate(model, spec, loader, criterion, device,
             thresholds, val_annotations, file_start_dts,
             amp_dtype: torch.dtype | None):
    model.eval()
    total_loss, nb = 0.0, 0
    all_probs: dict = {}
    use_amp = amp_dtype is not None

    for audio, targets, mask, metas in tqdm(loader, desc="Validate",
                                            leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        s = spec(audio)
        amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                   if use_amp else nullcontext())
        with amp_ctx:
            logits = model(s)
            targets_a, mask_a = align_lengths(logits, targets, mask)
            loss = criterion(logits, targets_a, mask_a)
        total_loss += loss.item()
        nb += 1

        # Cast probs back to fp32 for numerically stable post-processing.
        probs = torch.sigmoid(logits.float()).cpu().numpy()
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

    used = np.asarray(thresholds.cpu().numpy(), dtype=np.float64).copy()
    grids = [
        np.arange(0.20, 0.85, 0.05),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
    ]
    # Threshold sweep has two nested loops (3 classes × ~14 grid points each
    # = ~42 iterations). Each calls postprocess_predictions + compute_metrics
    # over the whole val set, so the wall-clock can be a few minutes once the
    # model starts predicting positives. Surface progress so the user knows
    # it's not hung.
    total_iters = sum(len(g) for g in grids)
    sweep_pbar = tqdm(total=total_iters, desc="Tuning thresholds",
                      leave=False)
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
            sweep_pbar.update(1)
            sweep_pbar.set_postfix(cls=name, t=f"{t:.2f}",
                                   best_f1=f"{best_f1:.3f}")
        used[c] = best_t
    sweep_pbar.close()

    pred_events = postprocess_predictions(all_probs, used)
    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    # BioDCASE scores on **macro F1** (mean of per-class F1s), not micro.
    # The "overall" entry returned by compute_metrics is computed from
    # total TP/FP/FN across classes which is the micro number — useful
    # but not the official metric. Compute macro explicitly and treat it
    # as the headline.
    per_class_f1s = []
    for cname in cfg.CALL_TYPES_3:
        m = metrics.get(cname)
        if m is not None:
            per_class_f1s.append(m.get("f1", 0.0))
    macro_f1 = (sum(per_class_f1s) / len(per_class_f1s)
                if per_class_f1s else 0.0)

    print(f"\n  Event-level F1 (tuned thresholds):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name)
        if m is None:
            continue
        print(f"    {name.upper():6} t={used[c]:.2f}  "
              f"TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"    MACRO F1   ={macro_f1:.3f}  (official BioDCASE metric)")
    print(f"    OVERALL F1 ={overall_f1:.3f}  (micro — TP/FP/FN summed)")

    return {
        "loss":       total_loss / max(nb, 1),
        "mean_f1":    macro_f1,    # headline for best-model tracking (= macro)
        "macro_f1":   macro_f1,
        "overall_f1": overall_f1,
        "per_class":  metrics,
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
    amp_dtype = AMP_DTYPES[args.amp]

    # Warn early if bf16 is requested on hardware that doesn't support it.
    if args.amp == "bf16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("*** WARNING: --amp bf16 requested but device does not "
                  "support bf16. Use --amp fp16 on Turing (2080 Ti / older).")

    rank, local_rank, world_size, device = ddp_setup()
    try:
        wbu.seed_everything(args.seed, deterministic=False)

        effective_batch = args.batch_per_gpu * world_size * args.grad_accum

        # Heuristic warning if recipe and effective batch look mismatched.
        # The email validates two endpoints (paper: 1e-5 with very large
        # batch; desktop: 1e-3 with smaller batch). The space between is
        # underspecified — flag it loudly so users can decide consciously.
        if is_main(rank):
            warn_if_recipe_batch_mismatch(
                args.recipe, effective_batch, lr,
                lr_overridden=(args.lr is not None),
            )

        # ------------------------------------------------------------------
        # Wandb (rank 0 only)
        # ------------------------------------------------------------------
        run = None
        if is_main(rank) and not args.no_wandb:
            run = wbu.init_phase(
                "canonical",
                extra_tags=["wbce_only", "fixed_lr", "ddp",
                            f"recipe_{args.recipe}", f"seed_{args.seed}",
                            f"amp_{args.amp}"],
                config={
                    "recipe":           args.recipe,
                    "lr":               lr,
                    "epochs":           epochs,
                    "batch_per_gpu":    args.batch_per_gpu,
                    "grad_accum":       args.grad_accum,
                    "world_size":       world_size,
                    "effective_batch":  effective_batch,
                    "amp":              args.amp,
                    "sync_bn":          (not args.no_sync_bn) and world_size > 1,
                    "seed":             args.seed,
                    "use_focal_loss":   False,
                    "use_weighted_bce": True,
                    "pos_weight_unit":  "frame",
                    "resample_every":   args.resample_every,
                    "use_3class":       cfg.USE_3CLASS,
                    "n_classes":        cfg.n_classes(),
                    "lstm_hidden":      cfg.LSTM_HIDDEN,
                    "lstm_layers":      cfg.LSTM_LAYERS,
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

        # Materialize the lazy projection layer before DDP wrapping.
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            _ = model(spec_extractor(dummy))

        use_sync_bn = (world_size > 1) and (not args.no_sync_bn)
        if use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank],
                        find_unused_parameters=False)

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        if is_main(rank):
            print(f"Parameters: {n_params:,}")

        # ------------------------------------------------------------------
        # Loss with frame-level pos_weight
        # ------------------------------------------------------------------
        if is_main(rank):
            print("\nComputing frame-level pos_weight (canonical):")
        pos_weight = compute_pos_weights_framewise(
            train_ds.segments, verbose=is_main(rank),
        ).to(device)
        if world_size > 1:
            dist.broadcast(pos_weight, src=0)
        criterion = CanonicalWBCELoss(pos_weight=pos_weight).to(device)

        # ------------------------------------------------------------------
        # Optimizer + GradScaler (fp16 only)
        # ------------------------------------------------------------------
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(cfg.BETA1, cfg.BETA2),
        )

        # bf16 has fp32's exponent range so it doesn't need scaling;
        # fp32 obviously doesn't either. Only fp16 needs the scaler.
        scaler = (torch.amp.GradScaler("cuda") if args.amp == "fp16"
                  else None)

        if is_main(rank):
            print(f"\n{'=' * 60}")
            print("Canonical recipe configuration")
            print(f"{'=' * 60}")
            print(f"  recipe:           {args.recipe}")
            print(f"  loss:             pure weighted BCE (no focal)")
            print(f"  pos_weight unit:  frame")
            print(f"  lr:               {lr:.0e}  (fixed, no scheduler)")
            print(f"  epochs:           {epochs}  (no early stopping)")
            print(f"  batch/gpu:        {args.batch_per_gpu}")
            print(f"  grad accum:       {args.grad_accum}")
            print(f"  world size:       {world_size}")
            print(f"  effective batch:  {effective_batch}")
            print(f"  AMP:              {args.amp} "
                  f"({'with scaler' if scaler else 'no scaler'})")
            print(f"  SyncBatchNorm:    {use_sync_bn}")
            print(f"  seed:             {args.seed}")
            print(f"{'=' * 60}\n")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        best_f1 = 0.0
        thresholds = torch.tensor([0.5, 0.5, 0.5], device=device)

        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if (epoch - 1) % args.resample_every == 0:
                if is_main(rank):
                    print(f"\n[epoch {epoch}] resampling negatives")
                seed_for_epoch = args.seed + epoch
                import random
                random.seed(seed_for_epoch)
                np.random.seed(seed_for_epoch)
                train_ds.resample_negatives()

                train_sampler = DistributedSampler(
                    train_ds, num_replicas=world_size, rank=rank,
                    shuffle=True, seed=args.seed + epoch,
                ) if world_size > 1 else None
                if train_sampler:
                    train_sampler.set_epoch(epoch)
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
                amp_dtype=amp_dtype, scaler=scaler,
                grad_accum=args.grad_accum,
            )

            if is_main(rank):
                val = validate(
                    model.module if isinstance(model, DDP) else model,
                    spec_extractor, val_loader, criterion, device,
                    thresholds, val_annotations, file_start_dts,
                    amp_dtype=amp_dtype,
                )
                thresholds = torch.tensor(val["thresholds"], device=device,
                                          dtype=torch.float32)
                print(f"\n  Train loss: {train_loss:.4f}  "
                      f"Val loss: {val['loss']:.4f}")
                print(f"  Macro F1: {val['macro_f1']:.3f}  "
                      f"(Overall/micro: {val['overall_f1']:.3f})  "
                      f"Best Macro F1: {best_f1:.3f}")

                if run is not None:
                    import wandb
                    payload = {
                        "epoch":          epoch,
                        "lr":             optimizer.param_groups[0]["lr"],
                        "train/loss":     train_loss,
                        "val/loss":       val["loss"],
                        "val/f1_macro":   val["macro_f1"],    # BioDCASE metric
                        "val/f1_overall": val["overall_f1"],  # micro, for reference
                    }
                    for ci, cname in enumerate(cfg.CALL_TYPES_3):
                        pc = val["per_class"].get(cname, {})
                        payload[f"val/f1/{cname}"]        = pc.get("f1", 0.0)
                        payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
                        payload[f"val/recall/{cname}"]    = pc.get("recall", 0.0)
                        payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
                    wandb.log(payload, step=epoch)

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
                    "amp":              args.amp,
                    "grad_accum":       args.grad_accum,
                    "effective_batch":  effective_batch,
                    "seed":             args.seed,
                }
                torch.save(ckpt, run_dir / "latest_model.pt")
                if val["mean_f1"] > best_f1:
                    best_f1 = val["mean_f1"]
                    ckpt["best_f1"] = best_f1
                    torch.save(ckpt, run_dir / "best_model.pt")
                    print(f"  *** New best F1: {best_f1:.3f}")

            if dist.is_initialized():
                dist.barrier()

        # ------------------------------------------------------------------
        # Post-training threshold tuning + final checkpoint (rank 0)
        # ------------------------------------------------------------------
        if is_main(rank):
            print(f"\n{'=' * 60}\nTuning thresholds on best model"
                  f"\n{'=' * 60}")
            print("(This runs a finer-grained sweep over the full val set "
                  "and can take several minutes — no progress bar from the "
                  "external tuner, so be patient.)")
            best_ckpt = torch.load(run_dir / "best_model.pt",
                                   map_location=device, weights_only=False)
            base_model = (model.module if isinstance(model, DDP) else model)
            base_model.load_state_dict(best_ckpt["model_state_dict"])

            tune_start = time.time()
            tuned = tune_thresholds_event_level(
                base_model, spec_extractor, val_loader, device,
                val_annotations, file_start_dts,
            )
            tune_elapsed = time.time() - tune_start
            print(f"Tuned thresholds: {tuned.tolist()} "
                  f"(took {tune_elapsed:.1f}s)")

            torch.save({
                "model_state_dict": base_model.state_dict(),
                "thresholds":       torch.tensor(tuned),
                "best_f1":          best_f1,
                "recipe":           args.recipe,
                "lr":               lr,
                "amp":              args.amp,
                "grad_accum":       args.grad_accum,
                "effective_batch":  effective_batch,
                "seed":             args.seed,
            }, run_dir / "final_model.pt")

            print(f"\nDone. Best Macro F1 (BioDCASE official): {best_f1:.3f}")
            print(f"Run dir: {run_dir}")

            if run is not None:
                import wandb
                wandb.summary["best_f1_macro"]    = float(best_f1)
                wandb.summary["final_thresholds"] = list(map(float, tuned))
                wandb.summary["epochs_run"]       = epochs
                wandb.summary["effective_batch"]  = effective_batch
                wandb.summary["amp"]              = args.amp
                wandb.summary["verdict"] = (
                    f"Canonical (WBCE-only, fixed LR={lr:.0e}, "
                    f"{epochs} epochs, effective batch {effective_batch}, "
                    f"AMP={args.amp}) → Macro F1 {best_f1:.3f}."
                )
                art = wandb.Artifact(
                    f"canonical-{run.name}", type="model",
                    metadata={
                        "best_f1":  float(best_f1),
                        "recipe":   args.recipe,
                        "lr":       float(lr),
                        "seed":     int(args.seed),
                        "epochs":   int(epochs),
                        "amp":      args.amp,
                        "effective_batch": int(effective_batch),
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
