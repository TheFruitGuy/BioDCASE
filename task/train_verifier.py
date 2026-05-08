"""
Train the Stage-2 Verifier
==========================

v2 (2026-05-08): smaller crop, augmentation enabled, stronger
regularization, product-only combination. v1 peaked at epoch 1 and
overfit; this version slows that down.

Trains the multi-head ``WhaleVerifier`` on candidate events extracted by
``extract_candidates.py``. The training task is binary classification
per class: given an audio crop centred on a candidate event of class
``c``, predict whether it is a real call of class ``c`` (TP, label=1)
or a false alarm (FP, label=0).

Train/val split
---------------
By default, the candidates parquet is split 80/20 *by recording*: every
candidate from one ``(dataset, filename)`` goes entirely to train or
entirely to val. This avoids leakage from per-file noise patterns.

Key v2 changes vs v1
--------------------
    crop_s              30.0  → 15.0
    train augmentations none  → time shift ±2s, vol scale 0.7-1.3,
                                time mask 50% prob × ≤1s
    backbone dropout    0.3   → 0.5
    head dropout        0.2   → 0.3
    learning rate       1e-3  → 3e-4
    weight decay        1e-4  → 1e-3
    score combination   max(prod, mean)  → product only
                          (mean was masking collapsed verifier heads)

Diagnostic the script reports
-----------------------------
After each epoch, three F1 numbers per class:

    1. ``F1(stage1_score)``    — what stage-1 alone would give at the
       best binary threshold. The floor.
    2. ``F1(verifier_score)``  — verifier alone, useful for spotting
       collapsed heads (drops to 0).
    3. ``F1(stage1 × verifier)`` — combined score, the operational number.

Usage
-----
::

    python train_verifier.py \\
        --candidates candidates_val.parquet \\
        --output_dir runs_verifier/v2_seed42 \\
        --epochs 30 \\
        --seed 42

To replicate v1 behaviour for comparison: ``--crop_s 30 --no_augment
--lr 1e-3 --weight_decay 1e-4 --backbone_dropout 0.3 --head_dropout 0.2``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset_verifier import (
    CandidateRecord, VerifierDataset, load_candidates, verifier_collate_fn,
)
from model_verifier import WhaleVerifier, count_parameters
from spectrogram import SpectrogramExtractor


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=str, required=True,
                   help="Path to candidates parquet from extract_candidates.py.")
    p.add_argument("--val_candidates", type=str, default=None,
                   help="Optional separate val parquet. If omitted, the "
                        "single --candidates parquet is split internally "
                        "by recording.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="v2 default 3e-4 (v1 was 1e-3).")
    p.add_argument("--weight_decay", type=float, default=1e-3,
                   help="v2 default 1e-3 (v1 was 1e-4).")
    p.add_argument("--internal_val_frac", type=float, default=0.2)
    p.add_argument("--samples_per_epoch", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--patience", type=int, default=12,
                   help="Epochs without val F1 improvement before early "
                        "stop. v2 default 12 — slower training needs "
                        "more patience.")

    # v2 dataset options
    p.add_argument("--crop_s", type=float, default=15.0,
                   help="Audio crop length in seconds (v1 used 30).")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable training-time augmentation (for "
                        "reproducing v1 baseline).")
    p.add_argument("--time_shift_max_s", type=float, default=2.0)
    p.add_argument("--volume_lo", type=float, default=0.7)
    p.add_argument("--volume_hi", type=float, default=1.3)
    p.add_argument("--time_mask_max_s", type=float, default=1.0)
    p.add_argument("--time_mask_prob", type=float, default=0.5)

    # v2 model options
    p.add_argument("--backbone_dropout", type=float, default=0.5,
                   help="v2 default 0.5 (v1 was 0.3).")
    p.add_argument("--head_dropout", type=float, default=0.3,
                   help="v2 default 0.3 (v1 was 0.2).")

    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ======================================================================
# Reproducibility
# ======================================================================

def seed_everything(seed: int):
    """Seed Python / NumPy / PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed: int):
    """Per-worker numpy seeding so augmentation differs across workers
    but stays reproducible run-to-run."""
    def _init(worker_id: int):
        worker_seed = (seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _init


# ======================================================================
# Train/val split by recording
# ======================================================================

def split_by_recording(
    records: list[CandidateRecord],
    val_frac: float,
    seed: int,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """Split candidates by ``(dataset, filename)``."""
    groups: dict[tuple[str, str], list[CandidateRecord]] = {}
    for r in records:
        groups.setdefault((r.dataset, r.filename), []).append(r)

    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_val = max(1, int(round(len(keys) * val_frac)))
    val_keys = set(keys[:n_val])

    train_records, val_records = [], []
    for k, recs in groups.items():
        (val_records if k in val_keys else train_records).extend(recs)
    return train_records, val_records


def summarise_split(name: str, records: list[CandidateRecord]):
    print(f"\n  {name} split: {len(records)} candidates")
    for cls_name in cfg.CALL_TYPES_3:
        cls_idx = cfg.CALL_TYPES_3.index(cls_name)
        sub = [r for r in records if r.class_idx == cls_idx]
        n_tp = sum(1 for r in sub if r.label == 1)
        n_fp = sum(1 for r in sub if r.label == 0)
        print(f"    {cls_name}: TP={n_tp:5d}  FP={n_fp:5d}")


# ======================================================================
# Train / validate epochs
# ======================================================================

def train_one_epoch(
    model, spec_extractor, loader, optimizer, scheduler, device, epoch,
):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    losses = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for audio, class_idx, label, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        aux = torch.tensor(
            [m["stage1_score"] for m in metas],
            device=device, dtype=torch.float32,
        ).unsqueeze(-1)

        spec = spec_extractor(audio)
        logits = model(spec, class_idx, aux)
        loss = bce(logits, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_val_scores(model, spec_extractor, loader, device):
    """Gather verifier and stage-1 scores for every val candidate."""
    model.eval()
    all_v, all_s, all_y, all_c = [], [], [], []
    for audio, class_idx, label, metas in tqdm(loader, desc="Val",
                                               leave=False):
        audio = audio.to(device, non_blocking=True)
        class_idx_d = class_idx.to(device, non_blocking=True)
        aux = torch.tensor(
            [m["stage1_score"] for m in metas],
            device=device, dtype=torch.float32,
        ).unsqueeze(-1)
        spec = spec_extractor(audio)
        probs = torch.sigmoid(model(spec, class_idx_d, aux)).cpu().numpy()
        all_v.append(probs)
        all_s.append(np.array([m["stage1_score"] for m in metas],
                              dtype=np.float32))
        all_y.append(label.numpy())
        all_c.append(class_idx.numpy())
    return {
        "verifier": np.concatenate(all_v),
        "stage1":   np.concatenate(all_s),
        "label":    np.concatenate(all_y).astype(np.int32),
        "class":    np.concatenate(all_c).astype(np.int32),
    }


# ======================================================================
# Per-class binary F1 sweep
# ======================================================================

def best_binary_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Brute-force best F1 on a fine threshold grid."""
    if len(labels) == 0 or labels.sum() == 0:
        return 0.0, 0.5
    grid = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5
    pos = labels == 1
    for t in grid:
        pred_pos = scores >= t
        tp = int(np.sum(pred_pos & pos))
        fp = int(np.sum(pred_pos & ~pos))
        fn = int(np.sum(~pred_pos & pos))
        if tp + fp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def per_class_summary(arrs: dict[str, np.ndarray]) -> dict:
    """
    For each class, compute best binary F1 of stage-1 alone, verifier
    alone, and the product. v2 drops the product-vs-mean fallback that
    was masking collapsed verifier heads in v1.
    """
    out: dict = {}
    macro_combined_f1 = 0.0
    n_classes_seen = 0
    for cls_idx, cls_name in enumerate(cfg.CALL_TYPES_3):
        mask = arrs["class"] == cls_idx
        if not mask.any():
            continue
        n_classes_seen += 1
        s1 = arrs["stage1"][mask]
        sv = arrs["verifier"][mask]
        y  = arrs["label"][mask]

        f1_s1, t_s1 = best_binary_f1(s1, y)
        f1_v,  t_v  = best_binary_f1(sv, y)
        comb = s1 * sv
        f1_comb, t_comb = best_binary_f1(comb, y)

        out[cls_name] = {
            "n_tp": int(y.sum()),
            "n_fp": int(len(y) - y.sum()),
            "f1_stage1":         f1_s1,    "thr_stage1":   t_s1,
            "f1_verifier":       f1_v,     "thr_verifier": t_v,
            "f1_combined":       f1_comb,  "thr_combined": t_comb,
            "delta":             f1_comb - f1_s1,
            "mean_v_tp": float(sv[y == 1].mean()) if (y == 1).any() else float("nan"),
            "mean_v_fp": float(sv[y == 0].mean()) if (y == 0).any() else float("nan"),
        }
        macro_combined_f1 += f1_comb
    out["macro_combined_f1"] = macro_combined_f1 / max(n_classes_seen, 1)
    return out


def print_summary(summary: dict, header: str):
    print(f"\n{header}")
    print(f"  {'class':<8}{'TPs':>6}{'FPs':>6}"
          f"{'F1(stage1)':>14}{'F1(verif)':>13}{'F1(prod)':>13}"
          f"{'Δ vs s1':>10}{'v̄(TP)':>9}{'v̄(FP)':>9}")
    for cls_name in cfg.CALL_TYPES_3:
        if cls_name not in summary:
            continue
        s = summary[cls_name]
        delta_sign = "+" if s["delta"] >= 0 else ""
        print(f"  {cls_name:<8}{s['n_tp']:>6}{s['n_fp']:>6}"
              f"{s['f1_stage1']:>14.4f}{s['f1_verifier']:>13.4f}"
              f"{s['f1_combined']:>13.4f}"
              f"{delta_sign}{s['delta']:>9.4f}"
              f"{s['mean_v_tp']:>9.3f}{s['mean_v_fp']:>9.3f}")
    print(f"  macro F1 (combined): {summary['macro_combined_f1']:.4f}")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load and split candidates
    # ------------------------------------------------------------------
    print(f"\nLoading candidates: {args.candidates}")
    all_records = load_candidates(args.candidates)
    print(f"  {len(all_records)} total")

    if args.val_candidates:
        print(f"Loading val candidates: {args.val_candidates}")
        train_records = all_records
        val_records = load_candidates(args.val_candidates)
        max_train_id = max((r.cand_id for r in train_records), default=-1)
        for i, r in enumerate(val_records):
            r.cand_id = max_train_id + 1 + i
    else:
        print(f"Splitting {args.candidates} by recording "
              f"({1 - args.internal_val_frac:.0%}/{args.internal_val_frac:.0%})")
        train_records, val_records = split_by_recording(
            all_records, args.internal_val_frac, args.seed,
        )
    summarise_split("train", train_records)
    summarise_split("val", val_records)

    # ------------------------------------------------------------------
    # DataLoaders — augmentation ON for train, OFF for val
    # ------------------------------------------------------------------
    aug_kwargs = dict(
        crop_s=args.crop_s,
        time_shift_max_s=args.time_shift_max_s,
        volume_scale_range=(args.volume_lo, args.volume_hi),
        time_mask_max_s=args.time_mask_max_s,
        time_mask_prob=args.time_mask_prob,
    )
    train_ds = VerifierDataset(
        train_records, train=(not args.no_augment), **aug_kwargs,
    )
    val_ds = VerifierDataset(val_records, crop_s=args.crop_s, train=False)

    samples_per_epoch = args.samples_per_epoch or len(train_records)
    train_sampler = train_ds.make_balanced_sampler(
        num_samples=samples_per_epoch, replacement=True,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=verifier_collate_fn,
        pin_memory=True, drop_last=True,
        worker_init_fn=make_worker_init_fn(args.seed), generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=verifier_collate_fn,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVerifier(
        n_classes=len(cfg.CALL_TYPES_3),
        backbone_dropout=args.backbone_dropout,
        head_dropout=args.head_dropout,
    ).to(device)
    total, trainable = count_parameters(model)
    print(f"\nVerifier params: total={total:,}  trainable={trainable:,}")
    print(f"  backbone_dropout={args.backbone_dropout}  "
          f"head_dropout={args.head_dropout}")
    print(f"  crop_s={args.crop_s}  augment={not args.no_augment}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    steps_per_epoch = math.ceil(samples_per_epoch / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Optional wandb
    # ------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                entity="bio-dcase",
                project="biodcase26-task2-whale-sed",
                group="verifier_v2",
                job_type="verifier_train",
                name=f"verifier_v2__seed{args.seed}__{time.strftime('%m%d-%H%M')}",
                config=vars(args),
                tags=["verifier", "two_stage", "v2"],
            )
        except Exception as e:
            print(f"  wandb init failed ({e}) — continuing without")
            use_wandb = False

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_macro = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, spec_extractor, train_loader,
            optimizer, scheduler, device, epoch,
        )
        val_arrs = collect_val_scores(
            model, spec_extractor, val_loader, device,
        )
        summary = per_class_summary(val_arrs)
        macro = summary["macro_combined_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "macro_combined_f1": macro,
            **{f"f1_combined_{k}": v["f1_combined"]
               for k, v in summary.items() if isinstance(v, dict)},
            **{f"f1_stage1_{k}": v["f1_stage1"]
               for k, v in summary.items() if isinstance(v, dict)},
        })

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  macro_F1={macro:.4f}  "
              f"({elapsed:.1f}s)")
        print_summary(summary, f"  Per-class (val) — epoch {epoch}")

        if use_wandb:
            payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/macro_combined_f1": macro,
                "val/lr": optimizer.param_groups[0]["lr"],
            }
            for cls_name in cfg.CALL_TYPES_3:
                if cls_name not in summary:
                    continue
                s = summary[cls_name]
                payload[f"val/f1_stage1/{cls_name}"]   = s["f1_stage1"]
                payload[f"val/f1_verifier/{cls_name}"] = s["f1_verifier"]
                payload[f"val/f1_combined/{cls_name}"] = s["f1_combined"]
                payload[f"val/delta/{cls_name}"]       = s["delta"]
                payload[f"val/mean_v_tp/{cls_name}"]   = s["mean_v_tp"]
                payload[f"val/mean_v_fp/{cls_name}"]   = s["mean_v_fp"]
            wandb.log(payload, step=epoch)

        improved = macro > best_macro
        if improved:
            best_macro = macro
            best_epoch = epoch
            epochs_without_improvement = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "macro_combined_f1": macro,
                "summary": summary,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stop: no improvement for "
                      f"{args.patience} epochs.")
                break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n=== Best epoch: {best_epoch}  macro F1 (combined): "
          f"{best_macro:.4f} ===")
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_arrs = collect_val_scores(model, spec_extractor, val_loader, device)
    final_summary = per_class_summary(final_arrs)
    print_summary(final_summary, "FINAL (best checkpoint, val):")

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(_jsonable({
            "best_epoch": best_epoch,
            "best_macro_combined_f1": best_macro,
            "summary": final_summary,
            "history": history,
            "args": vars(args),
        }), f, indent=2)
    print(f"\nWrote {out_dir / 'final_summary.json'}")
    print(f"Wrote {out_dir / 'best.pt'}")

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_macro_combined_f1"] = best_macro
        for cls_name in cfg.CALL_TYPES_3:
            if cls_name in final_summary:
                wandb.summary[f"final/f1_combined/{cls_name}"] = \
                    final_summary[cls_name]["f1_combined"]
                wandb.summary[f"final/delta/{cls_name}"] = \
                    final_summary[cls_name]["delta"]
        wandb.finish()


if __name__ == "__main__":
    main()
