"""
Phase 0r: 7-Class Training + Correct Weighted BCE (Segment-Count)
==================================================================

Phase 0m reproduced the paper's headline F1 (0.442 vs 0.443) using
plain BCE on 7-class targets. Christiaan (the paper's first author)
later confirmed by email that:

  1. The paper's final 0.440 model used weighted BCE *only*.
     Table 2's "+ Focal loss" row represents focal REPLACING weighted
     BCE, not stacked on top of it. The previous reading of Section
     2.6 was ambiguous.

  2. Weights are normalised by the number of **segments**, not files
     or durations: ``w_c = N / P_c`` where P_c is the count of
     positive segments belonging to class c, and N is the total
     number of negative segments per epoch.

  3. Christiaan explicitly flagged "padded frames being counted in
     the per-class positive/negative counts used for BCE weighting"
     as a frequent silent failure mode.

The existing ``compute_class_weights_for_sites`` (used by phases
0h/0i/0o) counts FILES, not segments. A file with 1 BmA call and a
file with 50 BmA calls were treated identically. This phase fixes
that and applies the correct recipe.

What changes vs Phase 0m
------------------------
A single change: ``nn.BCEWithLogitsLoss(reduction="none")`` becomes
``nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_7)``.

The 7-element ``pos_weight_7`` tensor is computed from the training
manifest BEFORE any dataloader is constructed, so segment counts
come from the annotation table directly — no padded frames can leak
in. Each annotation maps 1:1 to a positive training segment via
``build_positive_segments``, so ``Pc = count(annotations of class c)``
is the segment-count Christiaan describes.

Everything else is identical to 0m: same 8 sites, same paper LSTM
(hidden=128 layers=2), same 7-class output, same 7→3 collapse at
eval, same LR=PHASE0_LR (=1e-3, the working desktop LR Christiaan
recommended), same 30 epochs, same seed=42.

Expected outcomes
-----------------
1. Macro F1 climbs from 0m's 0.357 toward ~0.38-0.40. Micro F1 may
   move only marginally (still dominated by bmabz). This would
   confirm the bug class explains the 0h/0i/0o negative results
   and validate Christiaan's recipe at the 7-class scale.

2. F1 holds near 0m's 0.442 with shifted per-class profile (d/bp
   up, bmabz down). The weighting trades majority for minority.
   Useful for ensembling with 0m even if the headline doesn't move.

3. F1 drops materially. The 7-class weights are stronger than the
   3-class case (max ~36× vs max ~5×). If training destabilises,
   the next move is normalising weights to min=1 (the existing 0h
   code did this, the paper does not specify).

Reference table for the report
-------------------------------
- 0m (plain BCE, 7-class):     0.442 micro, 0.357 macro  ← baseline
- 0r (weighted BCE, 7-class):  TBD                        ← this phase
- Paper (weighted BCE, 3-class): 0.443 micro, 0.377 macro
- BPN paper baseline (post-proc only): 0.422 macro

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0r.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import PHASE0F_VAL_SITES, PHASE0F_EPOCHS
from train_phase0g import train_one_epoch_3class
from train_phase0m import (
    PHASE0M_TRAIN_SITES, PHASE0M_LSTM_HIDDEN, PHASE0M_LSTM_LAYERS,
    build_phase0m_model, validate_7to3,
)


# ======================================================================
# Class weight computation
# ======================================================================

def compute_pos_weight_7class_segments(
    sites: list[str], verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Per-class ``pos_weight`` for the 7-class head, segment-count
    normalised per Christiaan's email.

    Formula (paper Section 2.6, author-confirmed):

        w_c = N / P_c

    where ``P_c`` is the number of positive segments for fine-grained
    class c, and N is the total number of negative segments per epoch.

    Why this is segment-count not file-count
    ----------------------------------------
    In ``build_positive_segments`` (dataset.py) one positive segment
    is constructed per annotation. A file with 50 BmA annotations
    yields 50 positive BmA segments, not 1. So:

        P_c = count(annotations of fine-grained class c)

    This is the smallest unit of "positive presence" the loss sees
    during training.

    Why this is computed from the manifest, not the dataloader
    ----------------------------------------------------------
    The annotation table has no padding. If we instead counted
    positives over batched, padded targets, we would over- or
    under-count rare classes depending on which segments happened to
    be the longest in each batch. Christiaan flagged exactly this
    failure mode. Computing from raw annotations bypasses padding
    entirely.

    Negative count
    --------------
    Per Section 2.5 (and confirmed in the email), the trainer
    undersamples negatives to roughly match positives per mini-batch,
    so over an epoch::

        N ≈ Σ_c P_c   (total positive segments across all classes)

    Some annotations overlap in time so the per-class P_c counts can
    exceed the number of distinct positive segments, but the formula
    is robust to this: w_c just scales linearly with the chosen N
    convention, and the *relative* per-class weights are unchanged.

    Returns
    -------
    pos_weight : torch.Tensor of shape (7,)
        Weights in ``cfg.CALL_TYPES_7`` order.
    info : dict
        Diagnostics for logging (raw counts, weight values, ratio
        between max and min weight).
    """
    annotations = load_annotations(sites)

    p_c = []
    for c_name in cfg.CALL_TYPES_7:
        n = int((annotations["annotation"] == c_name).sum())
        p_c.append(max(n, 1))   # clamp to 1 to avoid div-by-zero

    n_total = sum(p_c)
    weights = [n_total / pc for pc in p_c]

    pos_weight = torch.tensor(weights, dtype=torch.float32)

    info = {
        "annotation_counts": dict(zip(cfg.CALL_TYPES_7, p_c)),
        "pos_weight":        dict(zip(cfg.CALL_TYPES_7, weights)),
        "n_total_pos":       n_total,
        "min_weight":        min(weights),
        "max_weight":        max(weights),
        "weight_ratio":      max(weights) / min(weights),
    }

    if verbose:
        print(f"\n  Per-class positive weights (wc = N / Pc, "
              f"segment-count normalised):")
        print(f"  {'class':12} {'Pc (segments)':>15} {'wc':>10}")
        for c_name, pc, w in zip(cfg.CALL_TYPES_7, p_c, weights):
            print(f"  {c_name:12} {pc:>15,} {w:>10.3f}")
        print(f"  {'total':12} {n_total:>15,}")
        print(f"  Weight ratio (max/min): {info['weight_ratio']:.2f}×")

    return pos_weight, info


# ======================================================================
# Main
# ======================================================================

def main():
    """Run Phase 0r end-to-end."""
    assert not cfg.USE_3CLASS, (
        "Phase 0r requires cfg.USE_3CLASS=False (same as 0m). Flip it "
        "in config.py before running so WhaleDataset emits 7-channel "
        "targets."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Seed first — DataLoader workers inherit the RNG at construction.
    # Same seed as 0m for an apples-to-apples loss-only comparison.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    # ------------------------------------------------------------------
    # Compute pos_weight BEFORE any dataloader exists, so padding can
    # not contaminate the per-class segment counts. This is the
    # author-flagged bug class.
    # ------------------------------------------------------------------
    print(f"\nComputing 7-class pos_weight over {len(PHASE0M_TRAIN_SITES)} "
          f"training sites...")
    pos_weight, weight_info = compute_pos_weight_7class_segments(
        PHASE0M_TRAIN_SITES, verbose=True,
    )
    pos_weight = pos_weight.to(device)

    # ------------------------------------------------------------------
    # Wandb run setup. Stamp the full weight vector and diagnostics
    # so the run page shows the actual values used — no need to grep
    # stdout to debug a bad weighting later.
    # ------------------------------------------------------------------
    run = wbu.init_phase("0r", config={
        "lr":              PHASE0_LR,
        "weight_decay":    PHASE0_WEIGHT_DECAY,
        "batch_size":      PHASE0_BATCH_SIZE,
        "threshold":       PHASE0_THRESHOLD,
        "seed":            SEED,
        "neg_ratio":       cfg.NEG_RATIO,
        "segment_s":       PHASE0E_SEGMENT_S,
        "epochs":          PHASE0F_EPOCHS,
        "train_sites":     PHASE0M_TRAIN_SITES,
        "val_sites":       PHASE0F_VAL_SITES,
        "lstm_hidden":     PHASE0M_LSTM_HIDDEN,
        "lstm_layers":     PHASE0M_LSTM_LAYERS,
        "pos_weight":      weight_info["pos_weight"],
        "pos_weight_counts": weight_info["annotation_counts"],
        "pos_weight_ratio":  weight_info["weight_ratio"],
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0r_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0r configuration:")
    print(f"  Training sites: {PHASE0M_TRAIN_SITES}  (8 sites)")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 7-class (collapsed to 3-class at eval)")
    print(f"  Loss: weighted BCE (segment-count normalised, no focal)")
    print(f"  LSTM: hidden={PHASE0M_LSTM_HIDDEN}, "
          f"layers={PHASE0M_LSTM_LAYERS} (paper config)")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0M_TRAIN_SITES)
    train_annotations = load_annotations(
        PHASE0M_TRAIN_SITES, manifest=train_manifest,
    )
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs
    print(f"  Training segments: {len(pos_segs)} pos + {len(neg_segs)} neg")

    # Cross-check: P_c sum from annotations should match positive segment
    # count, modulo overlap. Useful sanity check that segment ≈ annotation.
    p_c_sum = sum(weight_info["annotation_counts"].values())
    pct_diff = abs(len(pos_segs) - p_c_sum) / max(p_c_sum, 1) * 100
    print(f"  Sanity: Σ P_c = {p_c_sum:,}, positive segments = "
          f"{len(pos_segs):,} (Δ {pct_diff:.1f}%)")

    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(
        PHASE0F_VAL_SITES, manifest=val_manifest,
    )
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

    sample_audio, sample_target, sample_mask, _ = train_ds[0]
    assert sample_target.shape[1] == 7, (
        f"Expected 7-channel targets, got shape {sample_target.shape}. "
        f"WhaleDataset is still using 3-class mode."
    )
    print(f"  Sanity: target shape per sample = {sample_target.shape}")

    train_loader = DataLoader(
        train_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        **wbu.seeded_dataloader_kwargs(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model + loss + optimizer.
    #
    # The loss is the only change vs Phase 0m:
    #   - Same nn.BCEWithLogitsLoss(reduction="none") shell
    #   - Adds pos_weight (the segment-count-normalised 7-tensor)
    #
    # ``train_one_epoch_3class`` already applies the sequence mask to
    # the per-frame loss before reduction (see train_phase0g.py), so
    # padded frames don't enter the loss numerator regardless of how
    # large pos_weight makes the per-positive loss values. That handles
    # the "padding masked out of the loss computation" half of
    # Christiaan's warning. The "padding masked out of per-class counts"
    # half is handled by computing pos_weight from raw annotations
    # above, before any dataloader exists.
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0m_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weight,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # One-batch sanity check: confirm that weighted loss on a real
    # batch is finite, positive, and substantially larger than plain
    # BCE on the same batch (since rare-class positives get upweighted).
    # If this fires NaN/Inf we'd rather know now than at epoch 30.
    # ------------------------------------------------------------------
    print(f"\nLoss sanity check (one batch):")
    model.eval()
    with torch.no_grad():
        sanity_audio, sanity_targets, sanity_mask, _ = next(iter(train_loader))
        sanity_audio = sanity_audio.to(device)
        sanity_targets = sanity_targets.to(device)
        sanity_mask = sanity_mask.to(device)
        spec = spec_extractor(sanity_audio)
        logits = model(spec)
        T = min(logits.size(1), sanity_targets.size(1))
        logits = logits[:, :T]
        sanity_targets = sanity_targets[:, :T]
        sanity_mask = sanity_mask[:, :T]

        plain = nn.functional.binary_cross_entropy_with_logits(
            logits, sanity_targets, reduction="none",
        )
        weighted = criterion(logits, sanity_targets)
        valid = sanity_mask.unsqueeze(-1).float()
        plain_mean = (plain * valid).sum() / (valid.sum() * 7).clamp(min=1)
        weighted_mean = (weighted * valid).sum() / (valid.sum() * 7).clamp(min=1)
        print(f"  Plain BCE (masked):    {plain_mean.item():.4f}")
        print(f"  Weighted BCE (masked): {weighted_mean.item():.4f}")
        print(f"  Ratio: {weighted_mean.item() / max(plain_mean.item(), 1e-9):.2f}×")
        assert torch.isfinite(weighted_mean), \
            "Weighted BCE produced NaN/Inf on first batch — abort."

    model.train()

    # ------------------------------------------------------------------
    # Training loop. Same structure and helper as 0m — the helper is
    # class-count agnostic and already masks padding before reduction.
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (7-class train + "
          f"weighted BCE, 3-class eval)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_3class(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_7to3(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=PHASE0_THRESHOLD,
        )
        epoch_time = time.time() - t0

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]
        marker = " *** new best" if improved else ""

        print(f"\nEpoch {epoch:2d}/{PHASE0F_EPOCHS}  ({epoch_time:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        macro = sum(val["per_class"][n]["f1"] for n in cfg.CALL_TYPES_3) / 3
        print(f"    OVERALL F1={val['f1']:.3f}  MACRO F1={macro:.3f}")

        wbu.log_epoch_3class(epoch, train_loss, val)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val["loss"],
            "f1":         val["f1"],
            "macro_f1":   macro,
            "per_class":  val["per_class"],
        })

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
            "pos_weight": pos_weight.detach().cpu().tolist(),
        }, run_dir / f"phase0r_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
                "pos_weight": pos_weight.detach().cpu().tolist(),
            }, run_dir / "phase0r_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0r VERDICT")
    print(f"{'=' * 60}")

    f1s        = [h["f1"]        for h in history]
    macros     = [h["macro_f1"]  for h in history]
    bmabz_f1s  = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s      = [h["per_class"]["d"]["f1"]     for h in history]
    bp_f1s     = [h["per_class"]["bp"]["f1"]    for h in history]

    print(f"\nOverall (micro) F1 by epoch:")
    print(f"  {[f'{f:.3f}' for f in f1s]}")
    print(f"\nMacro F1 by epoch:")
    print(f"  {[f'{m:.3f}' for m in macros]}")

    print(f"\nBest F1 by class (vs 0m plain BCE):")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0m: 0.539 tuned)")
    print(f"  d:     {max(d_f1s):.3f}  (0m: 0.130 tuned)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0m: 0.402 tuned)")
    print(f"  Best overall (micro): {max(f1s):.3f}  (0m: 0.447 tuned, "
          f"paper: 0.443)")
    print(f"  Best macro:           {max(macros):.3f}  (0m: 0.357, "
          f"paper: 0.377)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}  (0m: 0.042)")
    print(f"  Max swing:  {max_swing:.3f}  (0m: 0.130)")

    macro_delta = max(macros) - 0.357
    micro_delta = max(f1s) - 0.442

    if macro_delta > 0.015:
        verdict_text = (
            f"Correct weighted BCE helped macro F1: "
            f"{max(macros):.3f} (+{macro_delta:.3f} vs 0m). "
            f"Bug class explained 0h/0i/0o negative results."
        )
        print(f"\n→ Macro F1 improved by +{macro_delta:.3f} vs 0m.")
        print("  Segment-count weighting (vs the file-count bug) was")
        print("  the missing ingredient. Recipe now matches the paper.")
    elif macro_delta > -0.015 and micro_delta > -0.015:
        verdict_text = (
            f"Correct weighted BCE neutral: micro {max(f1s):.3f} "
            f"({micro_delta:+.3f}), macro {max(macros):.3f} "
            f"({macro_delta:+.3f}). Loss choice not the bottleneck."
        )
        print(f"\n→ Weighted BCE didn't change the headline.")
        print("  Plain BCE (0m) and correct weighted BCE arrive at the")
        print("  same operating point. The bottleneck is elsewhere.")
    else:
        verdict_text = (
            f"Correct weighted BCE hurt: micro {max(f1s):.3f} "
            f"({micro_delta:+.3f}), macro {max(macros):.3f} "
            f"({macro_delta:+.3f}). 7-class weights too aggressive."
        )
        print(f"\n→ Even with correct segment-count weights, training")
        print("  destabilised. Try normalising weights to min=1, or")
        print("  clamp max weight to ~10× to soften the rare-class push.")

    # ------------------------------------------------------------------
    # Wandb finalize
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0r_best.pt",
    )


if __name__ == "__main__":
    main()
