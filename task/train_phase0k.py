"""
Phase 0k: Scale Up to 8 Training Sites
======================================

Phase 0g (plain BCE, 4 sites, 30s segments) gave us a stable F1=0.385
on the official val split. Threshold tuning didn't move it. The
remaining single-axis change we haven't tried is **more training data**.

Phase 0k uses the same recipe as 0g but trains on all 8 sites in
``cfg.TRAIN_DATASETS`` instead of the 4-site Antarctic subset. Same
plain BCE, same 3-class output, same 30s training segments, same
LR=5e-5, same small LSTM, same official 3-site validation.

Per the threshold-tuning analysis, bmabz is our biggest gap from
Geldenhuys: we're at F1≈0.46 vs his 0.62 averaged across val sites.
The hypothesis here: bmabz benefits most from more training data
because (a) it's the most common class so more examples means a
stronger signal, and (b) the 4 added sites all have bmabz annotations
so the absolute count grows roughly 2x.

Expected vs risks
-----------------
- Best case: F1 climbs to 0.42-0.45, bmabz approaches 0.55+
- Likely case: F1 lands in 0.36-0.42 range, modest improvement
- Worst case: distribution shift from new sites hurts performance
  on the held-out val sites; F1 drops to 0.30-0.35

The new sites (ballenyislands2015, casey2014, elephantisland2013,
greenwich2015) are still Antarctic, similar acoustic environments.
Probably safe to scale.

Compute
-------
Training set ~2× larger than Phase 0g (about 50k positive segments
vs 26k). Epochs will take ~16 minutes instead of ~8. 30 epochs ≈ 8
hours. If GPU is shared, expect closer to 10-12 hours.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0k.py
"""

import time
from pathlib import Path

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
from train_phase0g import (
    build_phase0g_model, train_one_epoch_3class, validate_3class,
)


#: Full 8-site training list from cfg.TRAIN_DATASETS, made explicit
#: here so the script's intent is readable without indirection.
PHASE0K_TRAIN_SITES = list(cfg.TRAIN_DATASETS)


def main():
    """Run Phase 0k end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0k expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # Same seed as 0g for fair "data scaling" comparison.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0k", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
        "epochs": PHASE0F_EPOCHS,
        "train_sites": PHASE0K_TRAIN_SITES,
        "val_sites": PHASE0F_VAL_SITES,
        "n_train_sites": len(PHASE0K_TRAIN_SITES),
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0k_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0k configuration:")
    print(f"  Training sites: {PHASE0K_TRAIN_SITES}  ({len(PHASE0K_TRAIN_SITES)})")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class (bmabz, d, bp)")
    print(f"  Loss: plain BCE (no weighting, no focal)")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Training data — all 8 sites
    # ------------------------------------------------------------------
    print(f"\nLoading training data (8 sites)...")
    train_manifest = get_file_manifest(PHASE0K_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0K_TRAIN_SITES,
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
    print(f"  Training segments: {len(pos_segs)} pos + {len(neg_segs)} neg "
          f"(2x larger than Phase 0g's 26k)")

    # Stamp dataset-size info onto the wandb config so scaling effects
    # are queryable in the UI.
    run.config.update({
        "n_train_files":       len(train_manifest),
        "n_train_annotations": len(train_annotations),
        "n_train_segments":    len(train_segments),
        "n_pos_segments":      len(pos_segs),
        "n_neg_segments":      len(neg_segs),
    }, allow_val_change=True)

    # ------------------------------------------------------------------
    # Validation — official BioDCASE split, unchanged
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
    # Model + loss + optimizer (identical to Phase 0g)
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0g_model(device)
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
    print(f"Training {PHASE0F_EPOCHS} epochs (8 sites, plain BCE)")
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

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
        }, run_dir / f"phase0k_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0k_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0k VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0g 4-site: 0.468)")
    print(f"  d:     {max(d_f1s):.3f}  (0g 4-site: 0.098)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0g 4-site: 0.346)")
    print(f"  Best overall: {max(f1s):.3f}  (0g 4-site: 0.385)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}  (0g: 0.05)")
    print(f"  Max swing:  {max_swing:.3f}  (0g: 0.29)")

    delta_overall = max(f1s) - 0.385  # 0g best
    if delta_overall > 0.03:
        verdict_text = (
            f"Data scaling helped: best F1 {max(f1s):.3f} "
            f"(+{delta_overall:.3f} vs 0g). Consider larger LSTM next."
        )
        print(f"\n→ Data scaling helped (+{delta_overall:.3f} F1).")
        print("  Phase 0l: scale up the LSTM (paper config) for further gains.")
    elif delta_overall > -0.02:
        verdict_text = (
            f"Data scaling break-even: best F1 {max(f1s):.3f} "
            f"({delta_overall:+.3f} vs 0g). Model likely parameter-starved."
        )
        print(f"\n→ Data scaling roughly break-even ({delta_overall:+.3f} F1).")
        print("  Model is parameter-starved. Phase 0l (larger LSTM) is the")
        print("  next likely lever.")
    else:
        verdict_text = (
            f"Data scaling hurt: best F1 {max(f1s):.3f} "
            f"({delta_overall:+.3f} vs 0g). Distribution shift from new sites."
        )
        print(f"\n→ Data scaling hurt ({delta_overall:+.3f} F1). Distribution")
        print("  shift from new sites likely. Investigate per-site contribution")
        print("  before scaling further.")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0k_best.pt",
    )


if __name__ == "__main__":
    main()
