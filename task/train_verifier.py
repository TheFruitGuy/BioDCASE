"""
Train the Stage-2 Verifier
==========================

Trains the multi-head ``WhaleVerifier`` on candidate events extracted by
``extract_candidates.py``. The training task is binary classification per
class: given a 30-second audio crop centred on a candidate event of class
``c``, predict whether it is a real call of class ``c`` (TP, label=1) or a
false alarm (FP, label=0).

Train/val split
---------------
By default, the candidates parquet is split 80/20 *by recording*: every
candidate from one ``(dataset, filename)`` goes entirely to train or
entirely to val. This avoids the leakage that would happen if two
candidates from the same wav file were split across the boundary —
hydrophone noise, recording-level distribution shift, and within-call
fragmentation are all per-file effects that a pure random split would let
the model memorise.

Diagnostic the script reports
-----------------------------
After training, three F1 curves are computed on the held-out val
candidates per class:

    1. ``F1(stage1_score)``    — what stage-1 alone would give at the best
       binary threshold. This is the floor.
    2. ``F1(verifier_score)``  — what the verifier alone would give at
       the best binary threshold. This says whether the verifier *can*
       discriminate, independent of stage-1.
    3. ``F1(stage1 × verifier)`` — what their product gives at the best
       binary threshold. This is the operational number — what we'd ship
       if we re-ranked candidates by combined score.

If (3) doesn't beat (1), the verifier didn't help and we stop here. If it
does, we proceed to ``inference_two_stage.py`` for end-to-end event-level
evaluation.

Note that these are binary-classification F1 on candidates, not the
event-level 1D-IoU F1 you'd see in the leaderboard — the IoU matching has
already happened in ``extract_candidates.py``. Binary F1 is the right
metric for the verifier's training objective; event-level F1 will be
computed end-to-end by ``inference_two_stage.py`` in the next step.

Usage
-----
::

    python train_verifier.py \\
        --candidates candidates_val.parquet \\
        --output_dir runs_verifier/v1 \\
        --epochs 30 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write checkpoints and logs.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--internal_val_frac", type=float, default=0.2,
                   help="Fraction of recordings sent to val when using "
                        "internal split. Ignored if --val_candidates given.")
    p.add_argument("--samples_per_epoch", type=int, default=None,
                   help="Total samples per training epoch. Defaults to "
                        "len(train_records) — fine for ~10-15k candidates.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--patience", type=int, default=8,
                   help="Epochs without val F1 improvement before early stop.")
    p.add_argument("--no_wandb", action="store_true",
                   help="Skip wandb logging (useful for quick local tests).")
    return p.parse_args()


# ======================================================================
# Reproducibility
# ======================================================================

def seed_everything(seed: int):
    """Seed Python / NumPy / PyTorch in one line. Local copy so this script
    has no hard dependency on wandb_utils (which the user might not have
    installed in every env)."""
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
# Train/val split by recording
# ======================================================================

def split_by_recording(
    records: list[CandidateRecord],
    val_frac: float,
    seed: int,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """
    Split candidates into train and val by recording, keeping all
    candidates from one ``(dataset, filename)`` together.

    Parameters
    ----------
    records : list of CandidateRecord
    val_frac : float, in (0, 1)
    seed : int

    Returns
    -------
    train_records, val_records : list of CandidateRecord
    """
    # Group by (dataset, filename).
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
    """Print per-class TP/FP counts for one half of the split."""
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
    model: nn.Module,
    spec_extractor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """Single training epoch. Returns mean BCE loss."""
    model.train()
    bce = nn.BCEWithLogitsLoss()
    losses = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for audio, class_idx, label, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # Build aux features tensor: (B, 1) with stage1 score per sample.
        aux = torch.tensor(
            [m["stage1_score"] for m in metas],
            device=device, dtype=torch.float32,
        ).unsqueeze(-1)

        spec = spec_extractor(audio)
        logits = model(spec, class_idx, aux)
        loss = bce(logits, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Mild grad clip — same value train.py uses for stage 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_val_scores(
    model: nn.Module,
    spec_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Run the verifier on every val candidate and return arrays keyed by
    ``cand_id``-aligned index:

        verifier_score, stage1_score, label, class_idx
    """
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
    """
    Brute-force best F1 over a fine threshold grid.

    Returns
    -------
    best_f1 : float
    best_threshold : float
    """
    if len(labels) == 0 or labels.sum() == 0:
        return 0.0, 0.5

    # Slightly finer than the stage-1 grid since this is just numpy.
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
    For each class, compute best binary F1 of:
      - stage-1 score alone
      - verifier score alone
      - stage1 × verifier (combined)

    Plus mean scores per (class, label) for diagnostic.
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
        # Combined = product. Also try mean of scores and pick whichever
        # is higher per-class (cheap and robust to score-scale mismatch).
        comb_prod = s1 * sv
        comb_mean = 0.5 * (s1 + sv)
        f1_comb_prod, t_cp = best_binary_f1(comb_prod, y)
        f1_comb_mean, t_cm = best_binary_f1(comb_mean, y)
        if f1_comb_prod >= f1_comb_mean:
            f1_comb, t_comb, comb_kind = f1_comb_prod, t_cp, "product"
        else:
            f1_comb, t_comb, comb_kind = f1_comb_mean, t_cm, "mean"

        out[cls_name] = {
            "n_tp": int(y.sum()),
            "n_fp": int(len(y) - y.sum()),
            "f1_stage1":         f1_s1,    "thr_stage1":   t_s1,
            "f1_verifier":       f1_v,     "thr_verifier": t_v,
            "f1_combined":       f1_comb,  "thr_combined": t_comb,
            "combined_kind":     comb_kind,
            "delta":             f1_comb - f1_s1,
            "mean_v_tp": float(sv[y == 1].mean()) if (y == 1).any() else float("nan"),
            "mean_v_fp": float(sv[y == 0].mean()) if (y == 0).any() else float("nan"),
        }
        macro_combined_f1 += f1_comb

    out["macro_combined_f1"] = macro_combined_f1 / max(n_classes_seen, 1)
    return out


def print_summary(summary: dict, header: str):
    """Pretty-print the per-class summary."""
    print(f"\n{header}")
    print(f"  {'class':<8}{'TPs':>6}{'FPs':>6}"
          f"{'F1(stage1)':>14}{'F1(verif)':>13}{'F1(combined)':>16}"
          f"{'Δ vs s1':>10}{'kind':>10}")
    for cls_name in cfg.CALL_TYPES_3:
        if cls_name not in summary:
            continue
        s = summary[cls_name]
        delta_sign = "+" if s["delta"] >= 0 else ""
        print(f"  {cls_name:<8}{s['n_tp']:>6}{s['n_fp']:>6}"
              f"{s['f1_stage1']:>14.4f}{s['f1_verifier']:>13.4f}"
              f"{s['f1_combined']:>16.4f}"
              f"{delta_sign}{s['delta']:>9.4f}{s['combined_kind']:>10}")
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
        # cand_id can collide between two parquets — make val ids globally
        # unique by offsetting them. (Only matters if downstream code uses
        # the id as a primary key.)
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
    # DataLoaders
    # ------------------------------------------------------------------
    train_ds = VerifierDataset(train_records)
    val_ds = VerifierDataset(val_records)

    samples_per_epoch = args.samples_per_epoch or len(train_records)
    train_sampler = train_ds.make_balanced_sampler(
        num_samples=samples_per_epoch, replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=verifier_collate_fn,
        pin_memory=True, drop_last=True,
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
    model = WhaleVerifier(n_classes=len(cfg.CALL_TYPES_3)).to(device)

    total, trainable = count_parameters(model)
    print(f"\nVerifier params: total={total:,}  trainable={trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    # Cosine schedule with 1-epoch warmup. Steps per epoch = batches.
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
                group="verifier_v1",
                job_type="verifier_train",
                name=f"verifier__seed{args.seed}__{time.strftime('%m%d-%H%M')}",
                config=vars(args),
                tags=["verifier", "two_stage"],
            )
        except Exception as e:
            print(f"  wandb init failed ({e}) — continuing without logging")
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
              f"loss={train_loss:.4f}  macro_F1_combined={macro:.4f}  "
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

        # Best-model selection
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
    # Final summary using the best checkpoint
    # ------------------------------------------------------------------
    print(f"\n=== Best epoch: {best_epoch}  macro F1 (combined): "
          f"{best_macro:.4f} ===")

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_arrs = collect_val_scores(model, spec_extractor, val_loader, device)
    final_summary = per_class_summary(final_arrs)
    print_summary(final_summary, "FINAL (best checkpoint, val):")

    # Save final summary as JSON for downstream analysis.
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
