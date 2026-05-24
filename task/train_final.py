"""
Final Pipeline - Training Entry Point
=====================================

Clean, self-contained training script for the reproduced Whale-VAD recipe.
It depends only on the sibling ``*_final`` modules (and, optionally,
``wandb_utils`` for experiment tracking) -- none of the exploratory
``train_phase0*.py`` scripts.

Recipe
------
- 8 Antarctic training sites, official 3-site validation split.
- Paper-sized BiLSTM (hidden 128, 2 layers).
- 7 fine-grained call types as training targets, collapsed to 3 coarse
  classes at evaluation via max-over-subclasses.
- Weighted BCE with segment-count-normalised per-class ``pos_weight``
  (``w_c = N / P_c``), computed once from the annotation table.
- Per-epoch negative resampling: positives are fixed for the whole run; a
  fresh random no-call subset is drawn at the start of every epoch.
- Training segments extended to a fixed 30 s window, matching the validation
  window length.

Experiment tracking
-------------------
Weights & Biases logging is opt-in via ``--wandb``. Without that flag the
script imports nothing from ``wandb`` and logs only to stdout.

Usage
-----
::

    python train_final.py                 # train, no W&B
    python train_final.py --wandb         # train and log to W&B
    python train_final.py --wandb --wandb-mode offline
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from dataset_final import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    extend_all_segments, WhaleDataset, collate_fn,
)
from model_final import WhaleVAD
from spectrogram_final import SpectrogramExtractor
from postprocess_final import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)


# ======================================================================
# Reproducibility helpers (standalone -- no dependency on wandb_utils)
# ======================================================================

def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Call once at the top of ``main()`` before any model, dataset, or
    DataLoader is built, since DataLoader workers inherit the RNG state at
    construction time.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seeded_dataloader_kwargs(seed: int) -> dict:
    """Return DataLoader kwargs giving reproducible shuffle and worker RNG."""
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {"generator": g, "worker_init_fn": _worker_init}


# ======================================================================
# Class weights
# ======================================================================

def compute_pos_weight(
    sites: list[str], device: torch.device, verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Per-class ``pos_weight`` for the 7-class head, segment-count normalised.

    Uses ``w_c = N / P_c`` where ``P_c`` is the number of positive segments
    (= annotations) for fine-grained class c and ``N`` is the total positive
    segment count across all classes. Computed from the raw annotation table,
    before any dataloader exists, so padded frames cannot contaminate the
    per-class counts.

    Returns
    -------
    pos_weight : torch.Tensor, shape (7,)
        Weights in ``cfg.CALL_TYPES_7`` order, moved to ``device``.
    info : dict
        Diagnostics (raw counts, weight values, max/min ratio).
    """
    annotations = load_annotations(sites)

    p_c = [max(int((annotations["annotation"] == c).sum()), 1)
           for c in cfg.CALL_TYPES_7]
    n_total = sum(p_c)
    weights = [n_total / pc for pc in p_c]

    info = {
        "annotation_counts": dict(zip(cfg.CALL_TYPES_7, p_c)),
        "pos_weight": dict(zip(cfg.CALL_TYPES_7, weights)),
        "n_total_pos": n_total,
        "min_weight": min(weights),
        "max_weight": max(weights),
        "weight_ratio": max(weights) / min(weights),
    }

    if verbose:
        print("\n  Per-class positive weights (w_c = N / P_c):")
        print(f"  {'class':12} {'P_c (segments)':>15} {'w_c':>10}")
        for c_name, pc, w in zip(cfg.CALL_TYPES_7, p_c, weights):
            print(f"  {c_name:12} {pc:>15,} {w:>10.3f}")
        print(f"  {'total':12} {n_total:>15,}")
        print(f"  Weight ratio (max/min): {info['weight_ratio']:.2f}x")

    return torch.tensor(weights, dtype=torch.float32).to(device), info


# ======================================================================
# Model construction
# ======================================================================

def build_model(device: torch.device):
    """
    Build the 7-class Whale-VAD model and its spectrogram extractor.

    A single dummy forward pass materialises the lazily-created projection
    layer so the model is immediately ready for training or checkpoint
    loading.
    """
    model = WhaleVAD(num_classes=7).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


# ======================================================================
# Train / validate
# ======================================================================

def train_one_epoch(model, spec_extractor, loader, criterion, optimizer, device):
    """Run one training pass over ``loader``; return the mean batch loss."""
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits = model(spec_extractor(audio))

        # Trim model output to the target length (small STFT boundary drift).
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        # Mask padded frames before reduction; normalise over valid
        # frame-class elements.
        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             val_annotations, file_start_dts, threshold: float):
    """
    Validation pass: 7-class inference, collapse to 3 coarse classes, score
    event-level F1 against the 3-class ground truth.

    Returns a dict with ``loss``, micro ``f1``, and a ``per_class`` breakdown.
    """
    model.eval()
    losses, n = 0.0, 0
    all_probs_7 = {}
    hop = spec_extractor.hop_length

    for audio, targets, mask, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits = model(spec_extractor(audio))
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)
        losses += loss.item()
        n += 1

        probs7 = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs7[j].shape[0])
            all_probs_7[key] = probs7[j, :n_frames, :]

    # Collapse 7 -> 3 (active because cfg.USE_3CLASS is False here).
    all_probs_3 = collapse_probs_to_3class(all_probs_7)

    # Post-processing labels channels via cfg.class_names(); flip to 3-class
    # for the duration of the call so the arrays are labelled [bmabz, d, bp],
    # then restore so the next training epoch sees 7-class targets.
    cfg.USE_3CLASS = True
    try:
        thresholds = np.array([threshold] * 3)
        pred_events = postprocess_predictions(all_probs_3, thresholds)
    finally:
        cfg.USE_3CLASS = False

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

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    per_class = {
        name: {
            "f1": metrics.get(name, {}).get("f1", 0.0),
            "precision": metrics.get(name, {}).get("precision", 0.0),
            "recall": metrics.get(name, {}).get("recall", 0.0),
            "tp": metrics.get(name, {}).get("tp", 0),
            "fp": metrics.get(name, {}).get("fp", 0),
            "fn": metrics.get(name, {}).get("fn", 0),
        }
        for name in cfg.CALL_TYPES_3
    }
    return {
        "loss": losses / max(n, 1),
        "f1": metrics.get("overall", {}).get("f1", 0.0),
        "per_class": per_class,
    }


# ======================================================================
# Per-epoch negative resampling
# ======================================================================

def resample_negatives_for_epoch(
    pos_segs_extended: list,
    train_annotations,
    train_manifest,
    n_neg: int,
    segment_s: float,
    epoch: int,
    rng: random.Random,
    verbose: bool = False,
):
    """
    Draw a fresh negative segment set for one epoch and return the combined
    (fixed positives + new negatives) training segment list.

    Sampling is driven by the dedicated ``rng`` instance rather than the
    global ``random`` stream. Reusing the same instance across epochs makes it
    advance naturally between calls -- so each epoch draws a different subset
    -- while keeping the whole run reproducible from the master seed and
    independent of anything else (e.g. ``wandb.init``) that touches the global
    stream.
    """
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg, rng=rng,
    )
    neg_segs = extend_all_segments(neg_segs, train_manifest, segment_s)

    if verbose and neg_segs:
        first = neg_segs[0]
        print(f"    epoch {epoch}: resampled {len(neg_segs)} negatives "
              f"[first: {first.filename} @ {first.start_sample} samp]")

    return pos_segs_extended + neg_segs


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the final Whale-VAD model.")
    p.add_argument("--wandb", action="store_true",
                   help="Log this run to Weights & Biases. Off by default; "
                        "without it the run logs only to stdout.")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B run mode (only used when --wandb is set).")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                   help=f"Number of training epochs (default {cfg.EPOCHS}).")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help=f"Master random seed (default {cfg.SEED}).")
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    # 7-class targets: keep cfg.USE_3CLASS False so WhaleDataset emits
    # 7-channel targets. The validation path toggles it internally.
    cfg.USE_3CLASS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = seed_everything(args.seed, deterministic=False)

    # Dedicated RNG for all segment sampling (positive collars + per-epoch
    # negatives). Isolating this from the global ``random`` stream makes the
    # draws identical whether or not W&B is enabled, since nothing else can
    # perturb the stream between calls. random.Random(seed) yields the same
    # sequence the global module would after random.seed(seed).
    sampling_rng = random.Random(seed)

    train_sites = list(cfg.TRAIN_DATASETS)
    val_sites = list(cfg.VAL_DATASETS)

    # --- pos_weight from annotations, before any dataloader exists ---
    print(f"\nComputing 7-class pos_weight over {len(train_sites)} sites...")
    pos_weight, weight_info = compute_pos_weight(train_sites, device, verbose=True)

    # --- optional W&B run ---
    run = None
    if args.wandb:
        import wandb_utils as wbu
        run = wbu.init_phase("final", config={
            "lr": cfg.LR,
            "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE,
            "threshold": cfg.THRESHOLD,
            "seed": seed,
            "neg_ratio": cfg.NEG_RATIO,
            "neg_resample_each_epoch": True,
            "segment_s": cfg.TRAIN_SEGMENT_S,
            "epochs": args.epochs,
            "train_sites": train_sites,
            "val_sites": val_sites,
            "lstm_hidden": cfg.LSTM_HIDDEN,
            "lstm_layers": cfg.LSTM_LAYERS,
            "pos_weight": weight_info["pos_weight"],
            "pos_weight_counts": weight_info["annotation_counts"],
            "pos_weight_ratio": weight_info["weight_ratio"],
        }, mode=args.wandb_mode)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"final_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration:")
    print(f"  Training sites:   {train_sites}  ({len(train_sites)})")
    print(f"  Validation sites: {val_sites}")
    print(f"  Output:           7-class training, collapsed to 3-class at eval")
    print(f"  Loss:             weighted BCE (segment-count normalised)")
    print(f"  Negatives:        resampled at the start of every epoch")
    print(f"  LSTM:             hidden={cfg.LSTM_HIDDEN}, layers={cfg.LSTM_LAYERS}")
    print(f"  LR={cfg.LR}, batch={cfg.BATCH_SIZE}, epochs={args.epochs}")

    # --- fixed positives + static validation set ---
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    print(f"  {len(train_manifest)} files, {len(train_annotations)} annotations")

    pos_segs = build_positive_segments(
        train_annotations, train_manifest, rng=sampling_rng,
    )
    pos_segs = extend_all_segments(pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    print(f"  Positive segments (fixed): {len(pos_segs)}")
    print(f"  Negative segments per epoch: {n_neg}")

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, {len(val_annotations)} "
          f"annotations, {len(val_segments)} segments")

    # --- model, loss, optimizer ---
    # Re-seed torch immediately before weight init so the model is identical
    # regardless of any earlier torch RNG consumption (e.g. by wandb.init).
    # This is a no-op for the no-W&B path and makes the two paths agree.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, spec_extractor = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    if run is not None:
        run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weight,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # --- training loop with per-epoch negative resampling ---
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {args.epochs} epochs (per-epoch negative resampling)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_segments = resample_negatives_for_epoch(
            pos_segs_extended=pos_segs,
            train_annotations=train_annotations,
            train_manifest=train_manifest,
            n_neg=n_neg,
            segment_s=cfg.TRAIN_SEGMENT_S,
            epoch=epoch,
            rng=sampling_rng,
            verbose=True,
        )
        train_loader = DataLoader(
            WhaleDataset(train_segments), batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
            **seeded_dataloader_kwargs(seed),
        )

        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=cfg.THRESHOLD,
        )
        epoch_time = time.time() - t0

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]
        marker = " *** new best" if improved else ""

        print(f"\nEpoch {epoch:2d}/{args.epochs}  ({epoch_time:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        macro = sum(val["per_class"][n]["f1"] for n in cfg.CALL_TYPES_3) / 3
        print(f"    OVERALL F1={val['f1']:.3f}  MACRO F1={macro:.3f}")

        if run is not None:
            wbu.log_epoch_3class(epoch, train_loss, val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "macro_f1": macro,
            "per_class": val["per_class"],
        })

        ckpt = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
            "pos_weight": pos_weight.detach().cpu().tolist(),
        }
        torch.save(ckpt, run_dir / f"final_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "final_best.pt")

    # --- summary ---
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    print(f"\nMicro F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Macro F1 by epoch: {[f'{m:.3f}' for m in macros]}")
    print(f"\nBest micro F1: {max(f1s):.3f}  (epoch {f1s.index(max(f1s)) + 1})")
    print(f"Best macro F1: {max(macros):.3f}")
    for name in cfg.CALL_TYPES_3:
        best = max(h["per_class"][name]["f1"] for h in history)
        print(f"  best {name}: {best:.3f}")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability: mean swing {mean_swing:.3f}, "
          f"max swing {max_swing:.3f}")

    verdict = (f"Best micro F1 {max(f1s):.3f}, best macro F1 {max(macros):.3f}; "
               f"second-half mean F1 swing {mean_swing:.3f}.")
    if run is not None:
        wbu.finalize_phase(history, verdict=verdict,
                           best_ckpt=run_dir / "final_best.pt")


if __name__ == "__main__":
    main()
