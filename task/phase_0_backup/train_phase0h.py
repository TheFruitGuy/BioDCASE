"""
Phase 0h: Add Weighted BCE to the 3-Class Baseline
==================================================

Phase 0g showed that 3-class plain BCE on the official val split
trains stably and rare classes start firing slowly:

  Epoch 5  bmabz=0.426  d=0.024  bp=0.046  overall=0.298

The slow rare-class climb suggests that with enough epochs the model
will learn d and bp from gradient signal alone. But if we want to push
those classes faster (and match the paper's setup), the next axis to
add is **weighted BCE**: per-class positive weights ``wc = N/Pc`` so
that loss on rare-class positives counts more heavily.

The earlier full-pipeline runs combined weighted BCE with focal loss,
class-imbalance scaling, the full 8-site training data, and the full
LSTM in one go — and we saw rare classes either stuck at zero or
training collapsing entirely. We could never tell which ingredient
was breaking things.

Phase 0h isolates weighted BCE alone: same architecture as 0g, same
4 training sites, same 30s segments, same plain BCE-with-logits — but
now with per-class positive weights computed on the Phase 0h training
sites only.

What this tests
---------------
1. Does weighted BCE *help* d and bp F1 vs plain BCE? Comparing 0h
   final F1 against 0g final F1 per class is the headline.
2. Does weighted BCE *destabilize* training? If 0h's bmabz F1 craters
   below 0g's, the weighting is hurting more than it helps. If 0h's
   F1 oscillates wildly, weighted BCE was a major cause of the full-
   pipeline instability.
3. What's a reasonable weight magnitude? With 4 Antarctic training
   sites the weights are different from the full 8-site run we saw
   earlier (which had bmz at 25.9× and bp20plus at 19.7×). Three-class
   weights from a 4-site subset should be much milder.

Three possible outcomes
-----------------------
1. d and bp F1 climb materially faster than 0g, bmabz F1 holds near
   0.50, training is stable. → Move to Phase 0i (add focal loss).
2. d and bp F1 climb a little but bmabz F1 drops. → Weighting works
   but is too aggressive; consider clamping max weight.
3. Training oscillates / collapses. → Weighted BCE is the destabiliser.
   Stay with plain BCE for the baseline; investigate why rare classes
   need different treatment.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0h.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import (
    PHASE0F_TRAIN_SITES, PHASE0F_VAL_SITES, PHASE0F_EPOCHS,
)
from train_phase0g import (
    build_phase0g_model, train_one_epoch_3class, validate_3class,
)


def compute_class_weights_for_sites(sites: list[str]) -> torch.Tensor:
    """
    Compute per-class positive weights ``wc = N / Pc`` over a specific
    set of training sites.

    The codebase already has ``model.compute_class_weights()`` but it
    pulls from ``cfg.TRAIN_DATASETS`` (the full 8-site list). Phase 0h
    only trains on 4 sites, so the relevant ``N`` and ``Pc`` counts
    must be recomputed over that subset. Otherwise the weights would
    reflect class proportions from data we're not actually using.

    Parameters
    ----------
    sites : list of str
        Training site names whose annotations are used for the weight
        calculation.

    Returns
    -------
    torch.Tensor
        Per-class weight tensor of shape ``(3,)`` ordered by
        ``cfg.CALL_TYPES_3``. The minimum weight is normalised to 1.
    """
    annotations = load_annotations(sites)
    total_files = annotations.groupby(["dataset", "filename"]).ngroups
    weights = []
    for c_name in cfg.CALL_TYPES_3:
        # All fine-grained labels that collapse to this coarse class.
        orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
        class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        p_c = max(class_annots.groupby(["dataset", "filename"]).ngroups, 1)
        n_neg = max(total_files - p_c, 1)
        weights.append(n_neg / p_c)
    result = torch.tensor(weights, dtype=torch.float32)
    result = result / result.min()
    return result


def main():
    """Run Phase 0h end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0h expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0h_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0h configuration:")
    print(f"  Training sites: {PHASE0F_TRAIN_SITES}")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class (bmabz, d, bp)")
    print(f"  Loss: weighted BCE (wc = N/Pc, computed on training subset)")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Per-class weights — over the 4-site training subset, NOT the full
    # cfg.TRAIN_DATASETS. This matters: the full 8-site weights are
    # different (bmabz appears in proportionally more files than in our
    # Antarctic subset) so the class balance is different.
    # ------------------------------------------------------------------
    print(f"\nComputing class weights over Phase 0h training sites...")
    pos_weight = compute_class_weights_for_sites(PHASE0F_TRAIN_SITES)
    print(f"Class weights (wc = N/Pc, normalized to min=1):")
    for name, w in zip(cfg.CALL_TYPES_3, pos_weight.tolist()):
        print(f"  {name}: {w:.3f}")

    # ------------------------------------------------------------------
    # Training data — same 4 sites, 30s segments
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0F_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0F_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")
    for name in cfg.CALL_TYPES_3:
        c = (train_annotations["label_3class"] == name).sum()
        print(f"    {name}: {c} train annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
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

    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

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
    # Model + loss + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0g_model(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Weighted BCE — the single change vs Phase 0g. ``pos_weight`` is
    # broadcast across (B, T) and applied per-class, so each frame's
    # positive-class loss is scaled by that class's weight.
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none",
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (3-class, weighted BCE)")
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

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "per_class": val["per_class"],
        })

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
        }, run_dir / f"phase0h_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0h_best.pt")

    # ------------------------------------------------------------------
    # Verdict — compares per-class trajectories vs Phase 0g
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0h VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (Phase 0g reference: ~0.45)")
    print(f"  d:     {max(d_f1s):.3f}  (Phase 0g reference: ~0.05)")
    print(f"  bp:    {max(bp_f1s):.3f}  (Phase 0g reference: ~0.10)")
    print(f"  Best overall: {max(f1s):.3f}")

    # Stability metric: max epoch-to-epoch swing in second half.
    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half max F1 swing: {max_swing:.3f}")
    print(f"  (Phase 0f reference: ~0.06, Phase 0g reference: similar)")

    if max_swing > 0.20:
        print("\n→ Weighted BCE destabilised training. Roll back: stay with")
        print("  plain BCE for the rare-class fix and look elsewhere")
        print("  (e.g. clamp weights to max=5, or use focal loss instead).")
    elif max(d_f1s) > 0.10 and max(bp_f1s) > 0.10:
        print("\n→ Weighted BCE pushed both rare classes meaningfully.")
        print("  Phase 0i: add focal loss on top.")
    else:
        print("\n→ Marginal gain on rare classes. Weighted BCE alone isn't")
        print("  enough at this scale. Move to Phase 0i (focal loss) or")
        print("  scale up training data.")


if __name__ == "__main__":
    main()
