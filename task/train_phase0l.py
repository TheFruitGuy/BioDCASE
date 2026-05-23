"""
Phase 0l: Scale Up the LSTM to Paper Config
============================================

Phase 0g (plain BCE, 4 sites, small LSTM) plateaued at F1=0.385 with
threshold tuning unable to move it. Phase 0j showed bmabz is our
biggest gap from Geldenhuys (0.46 vs 0.62). Two possible bottlenecks:
data quantity and model capacity. Phase 0k tests data scaling. This
script (Phase 0l) tests model capacity.

The Phase 0 model used a tiny LSTM (hidden=32, single layer) for fast
iteration. The paper config is hidden=128, 2 layers — 4× wider, 2×
deeper. Total parameter count goes from 462k to ~1.0M, matching the
paper's checkpoint we already verified loads and reproduces F1=0.443.

Single change vs Phase 0g
-------------------------
- LSTM hidden: 32 → 128
- LSTM layers: 1 → 2
- Dropout: 0.5 between layers (was inactive at layers=1)

Everything else identical: 4 training sites (kerguelen2005,
maudrise2014, rosssea2014, elephantisland2014), 30s training segments,
plain BCE, official 3-site validation, LR=5e-5, batch=32, 30 epochs.

Why this is the right test
--------------------------
The official Geldenhuys checkpoint, evaluated through our pipeline,
hit F1=0.443 — and that checkpoint has the paper-config LSTM. Our
training pipeline can already reach F1=0.385 with a 14× smaller
recurrent component. Closing the 0.06 gap is plausibly explained by
LSTM capacity, since the BiLSTM is what aggregates temporal context
for event detection.

Compute
-------
The 2-layer LSTM doubles the recurrent compute, but it's a small
fraction of total epoch time (the spectrogram extractor and CNN are
the bottleneck). Expect ~10 min/epoch × 30 epochs ≈ 5 hours.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0l.py
"""

import time
from pathlib import Path

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
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import (
    PHASE0F_TRAIN_SITES, PHASE0F_VAL_SITES, PHASE0F_EPOCHS,
)
from train_phase0g import train_one_epoch_3class, validate_3class


#: Paper-config LSTM dimensions, matching the values already in
#: ``cfg.LSTM_HIDDEN`` / ``cfg.LSTM_LAYERS``. Made explicit here so
#: the script's intent is readable without indirection.
PHASE0L_LSTM_HIDDEN = 128
PHASE0L_LSTM_LAYERS = 2


def build_phase0l_model(device: torch.device):
    """
    Build a 3-class WhaleVAD with the paper-config LSTM.

    Unlike ``train_phase0g.build_phase0g_model`` which monkey-patches
    cfg to shrink the LSTM, this function uses the cfg defaults
    directly — they already match the paper. The function exists for
    parallel structure with the other phase scripts and to make the
    intent explicit.
    """
    # Sanity check that nothing else has touched these values.
    assert cfg.LSTM_HIDDEN == PHASE0L_LSTM_HIDDEN, (
        f"cfg.LSTM_HIDDEN is {cfg.LSTM_HIDDEN}, expected "
        f"{PHASE0L_LSTM_HIDDEN}. Phase 0l requires paper-config LSTM."
    )
    assert cfg.LSTM_LAYERS == PHASE0L_LSTM_LAYERS, (
        f"cfg.LSTM_LAYERS is {cfg.LSTM_LAYERS}, expected "
        f"{PHASE0L_LSTM_LAYERS}. Phase 0l requires paper-config LSTM."
    )

    model = WhaleVAD(num_classes=3).to(device)
    spec = SpectrogramExtractor().to(device)
    # Materialize the lazy projection layer with a dummy 30s input.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


def main():
    """Run Phase 0l end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0l expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # Same seed as 0g/0k for fair "model capacity" comparison.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0l", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
        "epochs": PHASE0F_EPOCHS,
        "train_sites": PHASE0F_TRAIN_SITES,
        "val_sites": PHASE0F_VAL_SITES,
        "lstm_hidden": PHASE0L_LSTM_HIDDEN,
        "lstm_layers": PHASE0L_LSTM_LAYERS,
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0l_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0l configuration:")
    print(f"  Training sites: {PHASE0F_TRAIN_SITES}  (4 sites, like 0g)")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class (bmabz, d, bp)")
    print(f"  Loss: plain BCE (no weighting, no focal)")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  *** LSTM: hidden={PHASE0L_LSTM_HIDDEN}, "
          f"layers={PHASE0L_LSTM_LAYERS} (paper config) ***")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Training data — 4 sites (same as Phase 0g)
    # ------------------------------------------------------------------
    print(f"\nLoading training data (4 sites, same as Phase 0g)...")
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
    # Model + loss + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0l_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}  "
          f"(Phase 0g small LSTM: 462,723; ratio: {n_params/462723:.2f}x)")

    # Stamp model size onto the wandb config so capacity-vs-F1 is
    # queryable across the ladder (0g vs 0l vs 0k vs anything else).
    run.config.update({
        "n_params": n_params,
        "n_params_ratio_vs_phase0g": n_params / 462723,
    }, allow_val_change=True)

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
    print(f"Training {PHASE0F_EPOCHS} epochs (paper-config LSTM, plain BCE)")
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
        }, run_dir / f"phase0l_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0l_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0l VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0g small LSTM: 0.468)")
    print(f"  d:     {max(d_f1s):.3f}  (0g small LSTM: 0.098)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0g small LSTM: 0.346)")
    print(f"  Best overall: {max(f1s):.3f}  (0g small LSTM: 0.385)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}  (0g: 0.05)")
    print(f"  Max swing:  {max_swing:.3f}  (0g: 0.29)")

    # Reference: Geldenhuys' checkpoint with this LSTM config and his
    # full 8-site training hit F1=0.443 on this val split. The gap
    # between our F1 and 0.443 is the residual from training data
    # quantity, training duration, and any other recipe details we
    # haven't matched.
    delta_overall = max(f1s) - 0.385  # 0g best
    if delta_overall > 0.05:
        verdict_text = (
            f"Larger LSTM helped substantially: best F1 {max(f1s):.3f} "
            f"(+{delta_overall:.3f} vs 0g). Combine with 8-site data."
        )
        print(f"\n→ Larger LSTM helped substantially (+{delta_overall:.3f} F1).")
        print("  Model capacity was a real bottleneck. If 0k also helped,")
        print("  combine: 8-site training + paper LSTM should push toward")
        print("  paper's 0.443.")
    elif delta_overall > 0.02:
        verdict_text = (
            f"Larger LSTM helped modestly: best F1 {max(f1s):.3f} "
            f"(+{delta_overall:.3f} vs 0g). Combine with data scaling."
        )
        print(f"\n→ Larger LSTM helped modestly (+{delta_overall:.3f} F1).")
        print("  Combine with 0k's data scaling for full effect.")
    elif delta_overall > -0.02:
        verdict_text = (
            f"Larger LSTM break-even: best F1 {max(f1s):.3f} "
            f"({delta_overall:+.3f} vs 0g). Capacity not the bottleneck."
        )
        print(f"\n→ Larger LSTM ~break-even ({delta_overall:+.3f} F1).")
        print("  Model capacity isn't the bottleneck at this data scale.")
        print("  Data quantity (Phase 0k) is the more likely lever.")
    else:
        verdict_text = (
            f"Larger LSTM hurt: best F1 {max(f1s):.3f} "
            f"({delta_overall:+.3f} vs 0g). Possibly overfitting at 4 sites."
        )
        print(f"\n→ Larger LSTM hurt ({delta_overall:+.3f} F1). Possibly")
        print("  overfitting at the 4-site scale. Try 0l with 8 sites.")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0l_best.pt",
    )


if __name__ == "__main__":
    main()
