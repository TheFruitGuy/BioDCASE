"""
Phase 0f: Multi-Site Training, Real Validation Split
====================================================

Phase 0e (single-site, 30s training segments) hit F1=0.426 on within-site
validation but with high epoch-to-epoch swings (max 0.21). Our diagnosis
was that the small validation set (145 annotations) made it noise-
sensitive — single-file noise patterns can shift many false positives
across the threshold each epoch.

Phase 0f scales up the data while keeping every other complexity axis
fixed:

  - **Training sites**: 4 Antarctic sites (kerguelen2005, maudrise2014,
    rosssea2014, elephantisland2014) — chosen for geographic similarity
    to the official validation sites (also Antarctic). ~10× more data
    than Phase 0e's single-site training.

  - **Validation sites**: the official BioDCASE val split
    (casey2017, kerguelen2014, kerguelen2015). This is the *real* test:
    if we hit F1=0.30+ on this split, we have a credible baseline to
    add complexity to.

  - All Phase 0 simplifications kept: single class (bmabz), single
    output channel, plain BCE, no class weighting, no focal loss,
    smaller LSTM, 30s training segments.

What this tests
---------------
1. Does multi-site training data smooth out the val-loss spikes seen
   in 0c-0e? If yes, those spikes were small-data artifacts.
2. What F1 ceiling does the bmabz class hit on the official val split
   when training is stable? This is the number to beat in subsequent
   phases that add complexity.

Three possible outcomes
-----------------------
1. F1 climbs steadily to 0.30+ on official val, with low oscillation.
   → Pipeline scales. Move to Phase 0g (3-class output) and beyond
   confident that the foundation is solid.

2. F1 plateaus around 0.15-0.25 with reduced oscillation.
   → Multi-site smooths noise but cross-site generalization to the
   official val sites is genuinely hard. We'd need more sites or
   site-aware training tricks before adding classes.

3. F1 stays low and unstable.
   → Multi-site doesn't help on its own. Something else is broken;
   investigate input normalisation across sites, or the LSTM hidden
   state propagation.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0f.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    collate_fn,
)
from train_phase0 import (
    SingleClassDataset, build_phase0_model, validate_one_class,
    train_one_epoch,
    TARGET_CLASS_IDX, TARGET_CLASS_NAME,
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE,
    PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S


#: Antarctic sites used for training. Selected to match the geographic
#: profile of the official val split (also Antarctic). Excludes
#: ballenyislands2015 and casey2014 to keep training-data growth
#: incremental, and excludes greenwich2015 because its annotation
#: density per file is much lower than the others (would be nearly all
#: negatives, an unhelpful noise floor for diagnostics).
PHASE0F_TRAIN_SITES = [
    "kerguelen2005",
    "maudrise2014",
    "rosssea2014",
    "elephantisland2014",
]

#: The official BioDCASE 2025 validation split — what we ultimately
#: care about reproducing. Phase 0f's headline number is its F1 on
#: this set after stable training.
PHASE0F_VAL_SITES = ["casey2017", "kerguelen2014", "kerguelen2015"]

#: We need more epochs than Phase 0e because the model has more data
#: to absorb. 30 is enough to see whether F1 stabilises and where it
#: plateaus, without committing to a marathon run.
PHASE0F_EPOCHS = 30


def main():
    """Run Phase 0f end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0f expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0f_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0f configuration:")
    print(f"  Training sites: {PHASE0F_TRAIN_SITES}")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Target class: {TARGET_CLASS_NAME}")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s (matches val length)")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Training data — multi-site
    # ------------------------------------------------------------------
    print(f"\nLoading training data ({len(PHASE0F_TRAIN_SITES)} sites)...")
    train_manifest = get_file_manifest(PHASE0F_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0F_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} train files, "
          f"{len(train_annotations)} train annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )

    # Phase 0e fix carried over: every training segment is 30s.
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs

    pos_durs = [(s.end_sample - s.start_sample) / cfg.SAMPLE_RATE
                for s in pos_segs[:200]]
    print(f"  Training: {len(pos_segs)} positive + {len(neg_segs)} negative "
          f"segments")
    print(f"  Segment durations (first 200 positives): "
          f"min={min(pos_durs):.1f}s, max={max(pos_durs):.1f}s, "
          f"mean={sum(pos_durs)/len(pos_durs):.1f}s")

    # Class distribution sanity check — is bmabz actually well-represented?
    bmabz_count = (train_annotations["label_3class"] == TARGET_CLASS_NAME).sum()
    print(f"  {TARGET_CLASS_NAME} annotations across training set: {bmabz_count}")

    # ------------------------------------------------------------------
    # Validation data — the official split
    # ------------------------------------------------------------------
    print(f"\nLoading validation data ({len(PHASE0F_VAL_SITES)} sites, "
          f"OFFICIAL split)...")
    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"  {len(val_manifest)} val files, "
          f"{len(val_annotations)} val annotations, "
          f"{len(val_segments)} val segments")

    bmabz_val_count = (val_annotations["label_3class"] == TARGET_CLASS_NAME).sum()
    print(f"  {TARGET_CLASS_NAME} annotations on val split: {bmabz_val_count}")

    train_ds = SingleClassDataset(train_segments, TARGET_CLASS_IDX)
    val_ds = SingleClassDataset(val_segments, TARGET_CLASS_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=PHASE0_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
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
    # Model + loss + optimizer (same Phase 0 small variant)
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0_model(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (multi-site, official val)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_one_class(
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
        print(f"  {TARGET_CLASS_NAME}: TP={val['tp']:5} FP={val['fp']:6} "
              f"FN={val['fn']:5}  P={val['precision']:.3f} "
              f"R={val['recall']:.3f} F1={val['f1']:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "precision": val["precision"],
            "recall": val["recall"],
        })

        # Save checkpoint, marking the best one for downstream use.
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history, "is_best": improved,
        }, run_dir / f"phase0f_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0f_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0f VERDICT")
    print(f"{'=' * 60}")
    f1s = [h["f1"] for h in history]
    print(f"F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Best F1: {max(f1s):.3f} at epoch {f1s.index(max(f1s)) + 1}")
    print(f"Final F1: {f1s[-1]:.3f}")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0

    print(f"\nSecond-half stability:")
    print(f"  Mean epoch-to-epoch F1 swing: {mean_swing:.3f}")
    print(f"  Max  epoch-to-epoch F1 swing: {max_swing:.3f}")
    print(f"  Reference (Phase 0e single-site): mean 0.13, max 0.21")

    # Reference target: Geldenhuys' bmabz F1 on this val split, weighted
    # by site, comes out to about 0.62 across casey2017, kerguelen2014,
    # kerguelen2015. We're aiming much lower than that here because we
    # have a much smaller model, less training data per site, and no
    # threshold tuning. F1=0.20-0.30 would be a credible single-class
    # baseline; F1=0.40+ would be remarkable.
    if max(f1s) > 0.30 and max_swing < 0.15:
        print("→ Strong baseline. Pipeline scales to real val split with")
        print("  acceptable stability. Phase 0g: add second class.")
    elif max(f1s) > 0.20:
        print("→ Reasonable baseline. F1 lower than within-site (expected for")
        print("  cross-site eval) but stable enough to build on.")
        print("  Phase 0g: add second class, monitor whether F1 holds.")
    elif max(f1s) > 0.10:
        print("→ Marginal. Multi-site training partially helps but cross-site")
        print("  generalization is genuinely hard at this scale.")
        print("  Consider adding more training sites before adding classes.")
    else:
        print("→ Weak. Multi-site didn't help. Something else is broken.")
        print("  Investigate: per-site BatchNorm stats, input normalisation")
        print("  differences across sites, or whether sites have radically")
        print("  different signal-to-noise characteristics.")


if __name__ == "__main__":
    main()
