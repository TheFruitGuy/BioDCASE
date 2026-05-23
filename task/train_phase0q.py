"""
Phase 0q: 3-Class Training with Frame-Level BCE Pos-Weight
==========================================================

Branches off Phase 0m's convergence-node recipe — 8-site data + paper-
config LSTM (hidden=128, layers=2) — but takes a different turn:

  * Phase 0m trained 7-class and collapsed to 3 at eval, in keeping with
    the paper's published recipe.
  * Phase 0q trains directly on 3 classes (same setup as 0k/0l) and
    reintroduces per-class BCE weighting, this time using the canonical
    recipe's *frame-level* positive weight rather than the file-level
    proxy that earlier phase scripts used.

Why this is worth a separate phase
----------------------------------
Earlier weighting experiments (Phase 0g–0i) hurt F1, so we dropped
weighting from 0k onward. The hypothesis for Phase 0q is that the
*form* of the weight was wrong rather than the idea itself:
``model.compute_class_weights`` counts annotated files per class, which
is a coarse proxy that does not match the per-frame BCE loss the model
is actually trained against. The canonical helper
``compute_pos_weights_framewise`` (from ``dataset_canonical.py``)
counts the actual frame-level positive vs. negative ratio across all
training segments — exactly the quantity
``BCEWithLogitsLoss(pos_weight=...)`` expects.

If frame-level weighting works where file-level weighting failed, that
isolates the 0g–0i drop to a stats bug, not to the conceptual choice
of weighting BCE. If it still hurts, the prior is reaffirmed and we
can confidently move on without weighting.

What changes vs Phase 0m
------------------------
- Model output: 3 channels (not 7). ``cfg.USE_3CLASS`` is forced to True
  before any dataset is built, mirroring 0m's ``False`` flip.
- Targets: 3-class straight through — the ``USE_3CLASS`` flip-flop hack
  inside ``validate_7to3`` is gone.
- Validation: reuses ``train_phase0g.validate_3class`` directly.
- Loss: ``BCEWithLogitsLoss(pos_weight=w, reduction="none")``, with
  ``w`` computed by ``compute_pos_weights_framewise`` over the
  (positive + negative) training segments after the 30s extension.
  Frames per class are counted on ``Segment`` objects pre-collation,
  so the padding mask cannot bias the ratio.

The pos_weight is computed once at the start (segments are built once
in Phase 0 — no between-epoch resampling at this stage of the ladder),
so the weight is stable across the run and is persisted in every
checkpoint for downstream reproducibility.

Three possible outcomes
-----------------------
1. F1 climbs above 0.43 — frame-level weighting wins where file-level
   failed. The 0g–0i drop was a stats bug, not a conceptual problem.
   This becomes a candidate top-of-ladder from-scratch number.
2. F1 lands in 0.40–0.42 (similar to 0k/0l/0m, ~0.418) — the canonical
   weighting is mathematically correct but doesn't move the needle on
   this dataset; imbalance isn't the binding constraint at this scale.
3. F1 drops below 0.38 — weighted BCE genuinely hurts at this scale,
   regardless of weight form. Reaffirms the 0k/0l choice of plain BCE.

Compute
-------
8 sites x paper-config LSTM x 3-class output. Almost identical cost to
Phase 0k. The extra work is the one-time pos_weight computation before
training, which is O(annotations) on the CPU and finishes in seconds.
Expect ~16 min/epoch x 30 epochs ~= 8 hours, same as 0m.

Prerequisites
-------------
``dataset_canonical.py`` must sit alongside the standard ``dataset.py``
in this directory — it's a companion module, not a replacement, per
its own docstring.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0q.py
"""

import time
from pathlib import Path

import numpy as np
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
from dataset_canonical import compute_pos_weights_framewise
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import PHASE0F_VAL_SITES, PHASE0F_EPOCHS
from train_phase0g import train_one_epoch_3class, validate_3class


#: Full 8-site training set, matching 0k/0l/0m and the paper.
PHASE0Q_TRAIN_SITES = list(cfg.TRAIN_DATASETS)

#: Paper-config LSTM — matches 0l/0m. The corresponding cfg fields are
#: checked at model-build time, so flipping cfg.LSTM_HIDDEN/LAYERS down
#: by accident would trip the assertion rather than silently downgrading.
PHASE0Q_LSTM_HIDDEN = 128
PHASE0Q_LSTM_LAYERS = 2


def build_phase0q_model(device: torch.device):
    """
    Build a 3-class WhaleVAD with the paper-config LSTM.

    Mirrors ``build_phase0m_model`` from Phase 0m but for 3 classes: the
    USE_3CLASS assertion is inverted, and the classifier is sized to 3
    channels so no collapse step is needed at eval.
    """
    assert cfg.USE_3CLASS, (
        "Phase 0q requires cfg.USE_3CLASS=True. Flip it before constructing "
        "the dataset so target tensors come out 3-wide."
    )
    assert cfg.LSTM_HIDDEN >= PHASE0Q_LSTM_HIDDEN, (
        f"cfg.LSTM_HIDDEN={cfg.LSTM_HIDDEN}; expected >={PHASE0Q_LSTM_HIDDEN}"
    )
    assert cfg.LSTM_LAYERS >= PHASE0Q_LSTM_LAYERS, (
        f"cfg.LSTM_LAYERS={cfg.LSTM_LAYERS}; expected >={PHASE0Q_LSTM_LAYERS}"
    )
    model = WhaleVAD(num_classes=3).to(device)
    spec = SpectrogramExtractor().to(device)
    # Forward one dummy batch so the lazy projection layer materialises
    # its weights before any checkpoint load or parameter inspection.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


def main():
    """Run Phase 0q end-to-end."""
    # --- CRITICAL: flip the global flag BEFORE building any dataset ---
    # WhaleDataset reads cfg.USE_3CLASS at __init__ to decide target
    # tensor width, and compute_pos_weights_framewise reads it to choose
    # between the fine ("label") and coarse ("label_3class") annotation
    # keys. Both must see True before any segment-touching code runs.
    cfg.USE_3CLASS = True
    print(f"cfg.USE_3CLASS forced to {cfg.USE_3CLASS} for 3-class training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # Same seed as 0g/0k/0l/0m for fair comparison across the ladder.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0q", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
        "epochs": PHASE0F_EPOCHS,
        "train_sites": PHASE0Q_TRAIN_SITES,
        "val_sites": PHASE0F_VAL_SITES,
        "lstm_hidden": cfg.LSTM_HIDDEN,
        "lstm_layers": cfg.LSTM_LAYERS,
        "training_classes": 3,
        "eval_classes": 3,
        "loss": "BCEWithLogitsLoss",
        "pos_weight_source": "framewise_canonical_raw",
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0q_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0q configuration:")
    print(f"  Training sites: {PHASE0Q_TRAIN_SITES}  ({len(PHASE0Q_TRAIN_SITES)})")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class training, 3-class eval (no collapse)")
    print(f"  Loss: weighted BCE, frame-level pos_weight (canonical recipe)")
    print(f"  LSTM: hidden={cfg.LSTM_HIDDEN}, layers={cfg.LSTM_LAYERS} "
          f"(paper config)")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Training data — full 8 sites
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0Q_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0Q_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    # 3-class breakdown — this is what the model trains against, so it's
    # the only one that matters for 0q (no fine-class detour).
    print(f"  3-class breakdown of training annotations:")
    coarse_counts = {}
    for name in cfg.CALL_TYPES_3:
        c = int((train_annotations["label_3class"] == name).sum())
        coarse_counts[name] = c
        print(f"    {name}: {c}")

    run.config.update({
        "train_counts_3class": coarse_counts,
        "n_train_files":       len(train_manifest),
        "n_train_annotations": len(train_annotations),
    }, allow_val_change=True)

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    # 30s extension — same fix used since Phase 0e/0f to align training
    # segment length with the 30s validation tile length.
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs
    print(f"  Training segments: {len(pos_segs)} pos + {len(neg_segs)} neg")

    # ------------------------------------------------------------------
    # Validation — official BioDCASE split
    # ------------------------------------------------------------------
    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    # WhaleDataset with USE_3CLASS=True -> 3-channel targets in batches.
    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

    # Confirm target width — quick sanity check before consuming hours.
    sample_audio, sample_target, sample_mask, _ = train_ds[0]
    assert sample_target.shape[1] == 3, (
        f"Expected 3-channel targets, got shape {sample_target.shape}. "
        f"WhaleDataset is still using 7-class mode."
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
    # Frame-level pos_weight — the headline change vs Phase 0m
    # ------------------------------------------------------------------
    # Count positive/negative frames per class across the full training
    # segment set (positives + sampled negatives, all already 30s-extended),
    # then pass the resulting ratio to BCEWithLogitsLoss. The canonical
    # helper does this directly from Segment objects, so collation-time
    # padding cannot bias the count.
    #
    # ``normalize_min_to_one=False`` keeps the raw N_neg/N_pos ratio per
    # class — the strict canonical pos_weight as defined by the
    # BCEWithLogitsLoss docs. The min-to-one rescaling that
    # model.compute_class_weights does is a workaround for the file-level
    # proxy producing odd magnitudes, not part of the canonical recipe.
    # With raw ratios the rare classes (d, bp) will get aggressively
    # up-weighted — that's the whole point: if frame-level weighting
    # works, this is the form in which it should work. The trade-off is
    # that loss magnitudes are larger overall, so watch for training
    # instability in the second-half stability metric below.
    print(f"\n{'=' * 60}")
    print("Computing frame-level pos_weight (canonical recipe, raw ratio)")
    print(f"{'=' * 60}")
    pos_weight = compute_pos_weights_framewise(
        train_segments,
        n_classes=3,
        normalize_min_to_one=False,
        verbose=True,
    ).to(device)
    print(f"\n  pos_weight (device tensor): {pos_weight.cpu().tolist()}")
    run.config.update({
        "pos_weight": [float(w) for w in pos_weight.cpu().tolist()],
    }, allow_val_change=True)

    # ------------------------------------------------------------------
    # Model + loss + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0q_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    run.config.update({"n_params": n_params}, allow_val_change=True)

    # BCE with the frame-level pos_weight. reduction='none' is required
    # because train_one_epoch_3class / validate_3class do their own
    # mask-aware mean over the per-frame loss tensor (the padded frames
    # must be excluded before the mean is taken).
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none",
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop. The 0g training/validation helpers are class-count-
    # agnostic and use whatever criterion they're given, so the new
    # pos_weight flows through without further plumbing. Reusing
    # validate_3class directly (rather than 0m's custom validate_7to3)
    # also eliminates the USE_3CLASS flip-flop hack — predictions and
    # GT are both on the 3-class axis throughout.
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (3-class, frame-level pos_weight)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_3class(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_3class(
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
        print(f"    OVERALL F1={val['f1']:.3f}")

        wbu.log_epoch_3class(epoch, train_loss, val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "per_class": val["per_class"],
        })

        # Persist pos_weight in every checkpoint — without it, anyone
        # re-running eval on the checkpoint later would have to recompute
        # the weight from segment manifests, which is annoying and
        # error-prone. Stashing it next to the weights makes the loss
        # function reproducible from the .pt alone.
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
            "pos_weight": pos_weight.cpu(),
        }, run_dir / f"phase0q_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
                "pos_weight": pos_weight.cpu(),
            }, run_dir / "phase0q_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0q VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0k 8-site 3-class: 0.502)")
    print(f"  d:     {max(d_f1s):.3f}  (0k 8-site 3-class: 0.113)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0k 8-site 3-class: 0.390)")
    print(f"  Best overall: {max(f1s):.3f}  "
          f"(0k: 0.418, 0l: 0.417, 0m: ~0.418, paper: 0.443)")

    # Second-half stability — same diagnostic as 0m. Big swings in the
    # back half are a sign the weighting destabilised training and the
    # F1 reported above may be a lucky-epoch fluke rather than a stable
    # operating point.
    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}")
    print(f"  Max swing:  {max_swing:.3f}")

    if max(f1s) > 0.43:
        verdict_text = (
            f"Frame-level pos_weight matched/beat the paper: best F1 "
            f"{max(f1s):.3f} (paper: 0.443). The 0g-0i drop was a stats "
            f"bug (file-level counts), not a problem with weighted BCE."
        )
        print("\n→ Frame-level pos_weight matched/beat the paper.")
        print("  The 0g-0i drop was a stats-bug (file-level counts),")
        print("  not a conceptual problem with weighted BCE on this task.")
        print("  Add threshold tuning (Phase 0j-style) on this checkpoint")
        print("  for any final marginal gain.")
    elif max(f1s) > 0.40:
        verdict_text = (
            f"Frame-level pos_weight comparable to plain-BCE baselines: "
            f"best F1 {max(f1s):.3f} (0k/0l/0m: ~0.418). Canonical weight "
            f"is mathematically correct but doesn't move the needle."
        )
        print("\n→ Frame-level pos_weight is correct on its own terms but")
        print("  doesn't beat plain BCE at this dataset scale. Class")
        print("  imbalance is not the binding constraint — the bottleneck")
        print("  is elsewhere (likely the CNN frontend representation).")
    else:
        verdict_text = (
            f"Frame-level pos_weight underperformed: best F1 {max(f1s):.3f}. "
            f"Weighted BCE hurts regardless of weight form."
        )
        print("\n→ Frame-level weighting still hurts. Weighted BCE is")
        print("  genuinely the wrong choice on this task — the 0g-0i")
        print("  drop reproduces with the corrected weight form, so the")
        print("  prior holds: stick with plain BCE (0k/0l/0m).")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0q_best.pt",
    )


if __name__ == "__main__":
    main()
