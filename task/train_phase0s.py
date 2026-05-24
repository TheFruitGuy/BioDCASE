"""
Phase 0s: 7-Class Weighted BCE + Per-Epoch Negative Resampling
================================================================

Phase 0r (correct segment-count weighted BCE on top of 0m) reached
the same macro F1 as 0m (0.358 vs 0.357) but with **4.5× worse
training stability** — mean epoch-to-epoch swing 0.190 vs 0m's 0.042,
driven by catastrophic D-class FP explosions at epochs 19, 21, 25,
29 (D FPs hit 58k, 157k, 163k, 113k respectively).

The hypothesis: weighted BCE upweights rare-class positives strongly
enough that the D detector overfits to specific noise patterns in
the fixed negative set. When the LSTM weights drift in a direction
that aligns with that overfit, the detector fires on most of val.
The paper's Section 2.5 explicitly pairs weighted BCE with **per-
epoch negative resampling** — drawing a fresh subset of no-call
segments at the start of each epoch. Christiaan confirmed this is
not optional: "after each epoch of training, a different subset of
negative segments is sampled. The set of positive calls remains
consistent for each epoch during finetuning."

Without resampling, the negative set is whatever was sampled at
phase startup, and the model can memorize those specific clips.
With resampling, "negative" means something different every epoch,
which forces the rare-class detectors to learn invariances rather
than memorize.

What changes vs Phase 0r
------------------------
A single change: ``build_negative_segments`` moves from being called
once before the training loop to being called at the start of every
epoch with a per-epoch RNG seed. The positive segment set is built
once and reused. The dataloader is rebuilt each epoch to pick up the
new negatives.

Everything else is identical to 0r: same 8 sites, same paper LSTM
(hidden=128 layers=2), same 7-class output, same pos_weight
(computed once from annotations — independent of the negative set),
same LR=PHASE0_LR, same 30 epochs, same seed=42.

Per-epoch seed derivation
-------------------------
``epoch_seed = SEED + epoch * 1000``. Choosing 1000 as the step
gives clean separation in the RNG state space — different epochs
draw from non-overlapping regions of the seed space. Setting both
``np.random.seed`` and ``random.seed`` before calling
``build_negative_segments`` ensures the resampling is deterministic
given the master seed but distinct between epochs.

Expected outcomes
-----------------
1. Stability recovers: mean swing drops from 0.190 back toward 0m's
   0.042. The D-class FP explosions disappear. **This is the
   headline test.** If true, the paper recipe is a coupled system
   and resampling is the load-bearing ingredient for stability.

2. Macro F1 lifts modestly above 0r/0m's 0.357. Cross-site negative
   diversity is a known generalisation lever. Expected range
   0.37-0.40 if the hypothesis holds.

3. Both improvements together would mean phase0s is the new best
   plain-recipe baseline, and the reproduction story tightens:
   "we matched the paper recipe (weighted BCE + per-epoch
   resampling) at F1=??? and identified that the resampling was
   the load-bearing ingredient for training stability."

4. No improvement (stability still bad, macro still 0.36): the
   instability has a different cause than the fixed negative set,
   and the paper recipe doesn't transfer cleanly to our setup.
   Still informative — rules out the resampling hypothesis.

Reference table
---------------
- 0m   (plain BCE):                    F1=0.447 micro, 0.357 macro, swing 0.042
- 0r   (weighted BCE, fixed negs):     F1=0.440 micro, 0.358 macro, swing 0.190
- 0s   (weighted BCE + resample):      TBD                          ← this phase
- Paper:                               F1=0.443 micro, 0.377 macro

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0s.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
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
from train_phase0r import compute_pos_weight_7class_segments


# ======================================================================
# Per-epoch negative resampling helper
# ======================================================================

def resample_negatives_for_epoch(
    pos_segs_extended: list,
    train_annotations,
    train_manifest,
    n_neg: int,
    segment_s: float,
    epoch_seed: int,
    verbose: bool = False,
):
    """
    Draw a fresh negative segment set for one epoch and build the
    training segment list.

    Why we re-seed Python's ``random`` and NumPy: ``build_negative_
    segments`` uses NumPy's global RNG for the no-call subsample.
    Some segment-construction helpers also touch Python's ``random``.
    Seeding both makes the resampling deterministic given the master
    seed but different across epochs.

    Parameters
    ----------
    pos_segs_extended : list
        Already-extended positive segments. Reused across epochs.
    n_neg : int
        Number of negative segments to draw. Fixed across epochs at
        ``cfg.NEG_RATIO × len(pos_segs)`` so batch composition stays
        consistent.
    epoch_seed : int
        Per-epoch seed. Derived as ``SEED + epoch * 1000``.
    verbose : bool
        Print a short fingerprint (count + first segment ID) so it's
        visible in the training log that resampling happened.

    Returns
    -------
    train_segments : list
        Combined pos + new neg segment list ready for WhaleDataset.
    """
    np.random.seed(epoch_seed)
    random.seed(epoch_seed)

    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    neg_segs = extend_all_segments(neg_segs, train_manifest, segment_s)

    if verbose:
        # Quick fingerprint to verify in the log that the negative set
        # is actually drifting between epochs. If two epochs print the
        # same fingerprint, seeding is broken.
        if neg_segs:
            first = neg_segs[0]
            # Segments are objects (likely a dataclass), not dicts —
            # getattr with defaults is the safe way to inspect fields
            # without knowing the exact schema.
            filename = getattr(first, "filename", "?")
            start = getattr(first, "start_time", None)
            if isinstance(start, (int, float)):
                fp = f"file={filename}, start={start:.1f}s"
            else:
                fp = f"file={filename}, start={start}"
        else:
            fp = "<empty>"
        print(f"    resampled {len(neg_segs)} negatives "
              f"[seed={epoch_seed}; first: {fp}]")

    return pos_segs_extended + neg_segs


# ======================================================================
# Main
# ======================================================================

def main():
    """Run Phase 0s end-to-end."""
    # ------------------------------------------------------------------
    # Same cfg flip as 0r — 7-channel targets.
    # ------------------------------------------------------------------
    if cfg.USE_3CLASS:
        print("Setting cfg.USE_3CLASS = False for Phase 0s (7-class "
              "training, was True in config.py)")
    cfg.USE_3CLASS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    # ------------------------------------------------------------------
    # pos_weight computed from annotations BEFORE any dataloader exists
    # (same as 0r). Resampling doesn't change the annotation table, so
    # the weights are identical to 0r's by construction.
    # ------------------------------------------------------------------
    print(f"\nComputing 7-class pos_weight over {len(PHASE0M_TRAIN_SITES)} "
          f"training sites...")
    pos_weight, weight_info = compute_pos_weight_7class_segments(
        PHASE0M_TRAIN_SITES, verbose=True,
    )
    pos_weight = pos_weight.to(device)

    # ------------------------------------------------------------------
    # Wandb. Note the ``neg_resample_per_epoch`` tag added via the
    # registry entry for 0s — that's the differentiator from 0r.
    # ------------------------------------------------------------------
    run = wbu.init_phase("0s", config={
        "lr":                PHASE0_LR,
        "weight_decay":      PHASE0_WEIGHT_DECAY,
        "batch_size":        PHASE0_BATCH_SIZE,
        "threshold":         PHASE0_THRESHOLD,
        "seed":              SEED,
        "neg_ratio":         cfg.NEG_RATIO,
        "neg_resample_each_epoch": True,
        "neg_seed_step":     1000,
        "segment_s":         PHASE0E_SEGMENT_S,
        "epochs":            PHASE0F_EPOCHS,
        "train_sites":       PHASE0M_TRAIN_SITES,
        "val_sites":         PHASE0F_VAL_SITES,
        "lstm_hidden":       PHASE0M_LSTM_HIDDEN,
        "lstm_layers":       PHASE0M_LSTM_LAYERS,
        "pos_weight":        weight_info["pos_weight"],
        "pos_weight_counts": weight_info["annotation_counts"],
        "pos_weight_ratio":  weight_info["weight_ratio"],
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0s_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0s configuration:")
    print(f"  Training sites: {PHASE0M_TRAIN_SITES}  (8 sites)")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 7-class (collapsed to 3-class at eval)")
    print(f"  Loss: weighted BCE (segment-count normalised)")
    print(f"  Negatives: resampled at start of every epoch")
    print(f"  LSTM: hidden={PHASE0M_LSTM_HIDDEN}, "
          f"layers={PHASE0M_LSTM_LAYERS} (paper config)")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Build positives ONCE — they don't get resampled per Section 2.5.
    # Build val set ONCE — val is static throughout training.
    # ------------------------------------------------------------------
    print(f"\nLoading training data (positives + manifest)...")
    train_manifest = get_file_manifest(PHASE0M_TRAIN_SITES)
    train_annotations = load_annotations(
        PHASE0M_TRAIN_SITES, manifest=train_manifest,
    )
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    print(f"  Positive segments (fixed across epochs): {len(pos_segs)}")
    print(f"  Negative segments per epoch: {n_neg}")

    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(
        PHASE0F_VAL_SITES, manifest=val_manifest,
    )
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_ds = WhaleDataset(val_segments)
    val_loader = DataLoader(
        val_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    # ------------------------------------------------------------------
    # Model + loss + optimizer — identical to 0r.
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
    # Training loop with per-epoch negative resampling.
    #
    # The rebuild cost is small: ``build_negative_segments`` is a
    # subsample over the negative-pool manifest, and ``DataLoader``
    # construction is O(n_segments) work that happens before workers
    # spin up. On the 0r profile (~7 min/epoch early, ~15 min/epoch
    # late due to system contention), this adds <10 seconds.
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (per-epoch neg resampling)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()

        # ------------------------------------------------------------
        # Resample negatives. Per-epoch seed = SEED + epoch * 1000.
        # The fingerprint print verifies in the log that the negative
        # set is actually different every epoch.
        # ------------------------------------------------------------
        epoch_seed = SEED + epoch * 1000
        train_segments = resample_negatives_for_epoch(
            pos_segs_extended=pos_segs,
            train_annotations=train_annotations,
            train_manifest=train_manifest,
            n_neg=n_neg,
            segment_s=PHASE0E_SEGMENT_S,
            epoch_seed=epoch_seed,
            verbose=True,
        )
        train_ds = WhaleDataset(train_segments)
        train_loader = DataLoader(
            train_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True,
            **wbu.seeded_dataloader_kwargs(epoch_seed),
        )

        # ------------------------------------------------------------
        # Train + validate. Same helpers as 0r.
        # ------------------------------------------------------------
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
            "epoch":       epoch,
            "epoch_seed":  epoch_seed,
            "train_loss":  train_loss,
            "val_loss":    val["loss"],
            "f1":          val["f1"],
            "macro_f1":    macro,
            "per_class":   val["per_class"],
        })

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
            "pos_weight": pos_weight.detach().cpu().tolist(),
            "epoch_seed": epoch_seed,
        }, run_dir / f"phase0s_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
                "pos_weight": pos_weight.detach().cpu().tolist(),
                "epoch_seed": epoch_seed,
            }, run_dir / "phase0s_best.pt")

    # ------------------------------------------------------------------
    # Verdict — explicit comparison to both 0m and 0r since this is
    # the third step in a controlled three-phase ladder.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0s VERDICT")
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

    print(f"\nBest F1 by class (vs 0m plain BCE, vs 0r fixed-neg):")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0m: 0.539 tuned, 0r: 0.531 tuned)")
    print(f"  d:     {max(d_f1s):.3f}  (0m: 0.130 tuned, 0r: 0.115 tuned)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0m: 0.402 tuned, 0r: 0.428 tuned)")
    print(f"  Best overall (micro): {max(f1s):.3f}  "
          f"(0m: 0.447, 0r: 0.440 tuned, paper: 0.443)")
    print(f"  Best macro:           {max(macros):.3f}  "
          f"(0m: 0.357, 0r: 0.358 tuned, paper: 0.377)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability (the headline metric for 0s):")
    print(f"  Mean swing: {mean_swing:.3f}  (0m: 0.042, 0r: 0.190)")
    print(f"  Max swing:  {max_swing:.3f}  (0m: 0.130, 0r: 0.365)")

    # Count catastrophic FP explosions (D FPs > 50k) — the failure
    # mode that defined 0r's instability.
    explosions = sum(
        1 for h in history
        if h["per_class"]["d"]["fp"] > 50_000
    )
    print(f"  D-class FP explosions (>50k FPs): {explosions}  "
          f"(0r had 4: epochs 19, 21, 25, 29)")

    macro_delta_vs_0m = max(macros) - 0.357
    macro_delta_vs_0r = max(macros) - 0.358
    swing_delta_vs_0r = mean_swing - 0.190

    if explosions == 0 and swing_delta_vs_0r < -0.10:
        if macro_delta_vs_0m > 0.015:
            verdict_text = (
                f"Per-epoch resampling fixed BOTH stability and macro: "
                f"swing {mean_swing:.3f} (0r: 0.190), macro {max(macros):.3f} "
                f"(+{macro_delta_vs_0m:.3f} vs 0m, "
                f"+{macro_delta_vs_0r:.3f} vs 0r). Recipe is now coupled."
            )
            print(f"\n→ Both interventions converged: resampling restored")
            print(f"  stability AND lifted macro. Paper recipe is a system,")
            print(f"  and 0s is the new plain-recipe baseline.")
        else:
            verdict_text = (
                f"Per-epoch resampling fixed stability but macro "
                f"unchanged: swing {mean_swing:.3f}, macro "
                f"{max(macros):.3f} ({macro_delta_vs_0m:+.3f} vs 0m). "
                f"Resampling is the load-bearing ingredient for "
                f"stability under weighted BCE, but doesn't improve "
                f"the F1 ceiling at this scale."
            )
            print(f"\n→ Stability restored by resampling — confirms the")
            print(f"  D-class explosions were noise-set memorization. But")
            print(f"  macro F1 didn't move past 0m. The ~0.36 macro ceiling")
            print(f"  is robust across loss/sampling variations.")
    elif explosions >= 2:
        verdict_text = (
            f"Per-epoch resampling did NOT fix stability: {explosions} "
            f"D-class FP explosions, swing {mean_swing:.3f}. The "
            f"instability has a different root cause than the fixed "
            f"negative set."
        )
        print(f"\n→ Resampling didn't help: still {explosions} D-class")
        print(f"  explosions. The hypothesis was wrong; the instability")
        print(f"  is in the weights themselves or in the LSTM dynamics,")
        print(f"  not in the negative set composition.")
    else:
        verdict_text = (
            f"Partial stability gain: swing {mean_swing:.3f} "
            f"(0r: 0.190), {explosions} explosions, macro "
            f"{max(macros):.3f}. Resampling helps but doesn't "
            f"fully resolve the failure mode."
        )
        print(f"\n→ Resampling helped stability partially but didn't")
        print(f"  fully resolve it. The bug is real but not solely")
        print(f"  about the fixed negative set.")

    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0s_best.pt",
    )


if __name__ == "__main__":
    main()
