"""
Phase 12a: Frequency Dynamic Convolution on the Final Recipe
=============================================================

Single-axis architecture change to the consolidated ``train_final.py``
recipe (the ``final`` phase in the wandb ladder): replace the first
and third depthwise 3x3 convolutions in the residual aggregation block
with ``FDConv2d`` (K=4 basis kernels, attention reduction=4 — both
paper-standard). Everything else inherits from ``train_final.py``:
8-site training, paper BiLSTM, 7-class targets collapsed to 3 at eval,
segment-count-normalised weighted BCE, per-epoch negative resampling,
~30 s segments matching the validation window.

Why this matters
----------------
Standard 2D convs assume "the same pattern at 27 Hz means the same
thing as at 100 Hz" — wrong for whale data, where calls are *defined*
by their frequency band (Bm-Z at ~27 Hz, D above 20 Hz). Diagnostic
ablations (Phases 1-2) confirmed the CNN frontend is the bottleneck;
this phase attacks the wrong frequency-equivariance inductive bias
directly.

Reference: Nam et al. 2022 (FDY conv, arXiv:2203.15296). Original
"FDY conv" formulation: +7.56% on DESED over a standard-conv CRNN
baseline. The dilated and multi-dilated follow-ups in the same family
report +9.27% (DFD) and +10.98% (MDFD) respectively but are out of
scope for phase 12a — those are 12b/12c candidates if 12a beats final.

Standard hyperparameters
------------------------
- ``K = 4`` basis kernels (paper default; range 4-8)
- ``reduction = 4`` in the attention head (paper default)
- Replace 2 of 3 depthwise convs (first at index 1, last at index 7
  of the aggregation Sequential; middle stays standard). Matches the
  conservative "one or two of the depthwise convs" recommendation.

Parameter cost: ~+16k params (+1.57% on a ~1.03M-param model). See
``model_fdconv_final.py``'s self-test for the exact delta.

Code reuse
----------
This script imports the heavy lifting from ``train_final``:
``seed_everything``, ``seeded_dataloader_kwargs``, ``compute_pos_weight``,
``train_one_epoch``, ``validate``, ``resample_negatives_for_epoch``.
Only ``build_model`` and ``main`` are local — and ``main`` differs from
``train_final.main`` in exactly two ways:
  1. Calls the local ``build_model`` (which returns the FDConv variant)
  2. Calls ``wbu.init_phase("12a", ...)`` with FDConv-specific config
     keys, not ``init_phase("final", ...)``.

If ``train_final.py`` ever grows a ``model_factory`` argument the way
``phase1_baseline.run_phase1_training`` has one, this script collapses
to about 30 lines.

Usage
-----
::

    python train_phase12a.py
    python train_phase12a.py --epochs 40 --seed 1337

W&B logging is unconditional — every run lands on the dashboard. To
run offline (e.g. on a node without internet), set
``WANDB_MODE=offline`` in the environment; wandb honours that natively.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config_final as cfg
from dataset_final import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_val_segments,
    extend_all_segments, WhaleDataset, collate_fn,
)
from spectrogram_final import SpectrogramExtractor
from model_fdconv_final import WhaleVAD_FDConv, DEFAULT_FDCONV_POSITIONS

# Reuse the canonical recipe's helpers verbatim. If any of these change
# in train_final, phase 12a picks up the change automatically.
from train_final import (
    seed_everything,
    seeded_dataloader_kwargs,
    compute_pos_weight,
    train_one_epoch,
    validate,
    resample_negatives_for_epoch,
)

# W&B logging is unconditional: every phase 12a run lands on the
# dashboard. Set ``WANDB_MODE=offline`` in the environment if you need
# offline behaviour (wandb honours that natively).
import wandb_utils as wbu


# ======================================================================
# FDConv hyperparameters (paper-standard values)
# ======================================================================

PHASE12A_K = 4
PHASE12A_REDUCTION = 4
PHASE12A_POSITIONS = DEFAULT_FDCONV_POSITIONS  # (1, 7) — first and last


# ======================================================================
# Model construction (the only real divergence from train_final)
# ======================================================================

def build_model(device: torch.device):
    """
    Build the 7-class FDConv variant and its spectrogram extractor.

    Mirrors ``train_final.build_model`` but instantiates
    ``WhaleVAD_FDConv`` with the phase 12a hyperparameters. The dummy
    forward pass materialises the lazily-created projection layer so
    the model is immediately ready for training or checkpoint loading.
    """
    model = WhaleVAD_FDConv(
        num_classes=7,
        K=PHASE12A_K,
        reduction=PHASE12A_REDUCTION,
        positions=PHASE12A_POSITIONS,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train phase 12a (FDConv on the final recipe)."
    )
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
    sampling_rng = random.Random(seed)

    train_sites = list(cfg.TRAIN_DATASETS)
    val_sites = list(cfg.VAL_DATASETS)

    # --- pos_weight from annotations, before any dataloader exists ---
    print(f"\nComputing 7-class pos_weight over {len(train_sites)} sites...")
    pos_weight, weight_info = compute_pos_weight(
        train_sites, device, verbose=True,
    )

    # --- W&B run ---------------------------------------------------------
    # Always logged. Same config payload as train_final, plus FDConv-
    # specific keys so the wandb dashboard shows what changed vs. the
    # final baseline.
    run = wbu.init_phase("12a", config={
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
        # FDConv-specific config — single-axis change vs. "final".
        "arch_change":         "fdconv_aggregation",
        "fdconv_K":            PHASE12A_K,
        "fdconv_reduction":    PHASE12A_REDUCTION,
        "fdconv_positions":    list(PHASE12A_POSITIONS),
        "fdconv_layer_count":  len(PHASE12A_POSITIONS),
        "fdconv_total_layers": 3,  # there are 3 depthwise convs
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase12a_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration:")
    print(f"  Phase:            12a (FDConv on the final recipe)")
    print(f"  Training sites:   {train_sites}  ({len(train_sites)})")
    print(f"  Validation sites: {val_sites}")
    print(f"  Output:           7-class training, collapsed to 3-class at eval")
    print(f"  Loss:             weighted BCE (segment-count normalised)")
    print(f"  Negatives:        resampled at the start of every epoch")
    print(f"  LSTM:             hidden={cfg.LSTM_HIDDEN}, "
          f"layers={cfg.LSTM_LAYERS}")
    print(f"  FDConv:           K={PHASE12A_K}, reduction={PHASE12A_REDUCTION}, "
          f"positions={PHASE12A_POSITIONS} (of {(1, 4, 7)})")
    print(f"  LR={cfg.LR}, batch={cfg.BATCH_SIZE}, epochs={args.epochs}")

    # --- fixed positives + static validation set ------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    pos_segs = build_positive_segments(
        train_annotations, train_manifest, rng=sampling_rng,
    )
    pos_segs = extend_all_segments(
        pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S,
    )
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    print(f"  Positive segments (fixed): {len(pos_segs)}")
    print(f"  Negative segments per epoch: {n_neg}")

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments),
        batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    # --- model, loss, optimizer ------------------------------------------
    # Re-seed torch immediately before weight init so the model is
    # identical regardless of any earlier torch RNG consumption
    # (e.g. by wandb.init). Same trick train_final uses.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, spec_extractor = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}  "
          f"(FDConv variant of model_final.WhaleVAD)")
    run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weight,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # --- training loop with per-epoch negative resampling ----------------
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
            WhaleDataset(train_segments),
            batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True, **seeded_dataloader_kwargs(seed),
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
        print(f"  Train loss: {train_loss:.4f}   "
              f"Val loss: {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        macro = sum(val["per_class"][n]["f1"] for n in cfg.CALL_TYPES_3) / 3
        print(f"    OVERALL F1={val['f1']:.3f}  MACRO F1={macro:.3f}")

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
            "fdconv": {
                "K": PHASE12A_K,
                "reduction": PHASE12A_REDUCTION,
                "positions": list(PHASE12A_POSITIONS),
            },
        }
        torch.save(ckpt, run_dir / f"phase12a_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "phase12a_best.pt")

    # --- summary ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 12a SUMMARY")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    print(f"\nMicro F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Macro F1 by epoch: {[f'{m:.3f}' for m in macros]}")
    print(f"\nBest micro F1: {max(f1s):.3f}  "
          f"(epoch {f1s.index(max(f1s)) + 1})")
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

    verdict = (
        f"Phase 12a (FDConv K={PHASE12A_K}, "
        f"positions={PHASE12A_POSITIONS}): "
        f"best micro F1 {max(f1s):.3f}, best macro F1 {max(macros):.3f}; "
        f"second-half mean F1 swing {mean_swing:.3f}."
    )
    wbu.finalize_phase(
        history, verdict=verdict,
        best_ckpt=run_dir / "phase12a_best.pt",
    )


if __name__ == "__main__":
    main()
