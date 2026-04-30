"""
Phase 0g: Add 3-Class Output to the Stable Baseline
===================================================

Phase 0f hit F1=0.495 on bmabz alone with a stable training trajectory
(no collapse, no val-loss spikes, no wild oscillation). That's the
foundation. Phase 0g adds the next axis: full 3-class output.

What changes vs Phase 0f
------------------------
- Model outputs 3 channels (bmabz, d, bp) instead of 1.
- Targets are 3-class (the model's full ``__getitem__`` output, not
  the SingleClassDataset slice).
- Loss is plain BCE summed over 3 classes — no class weighting, no
  focal loss. We're testing whether multi-class training itself works,
  not the loss-function choices.
- Validation reports per-class F1 plus overall F1.

Everything else identical to Phase 0f: same 4 training sites, same
official val split, same 30s training segments, same small LSTM, same
LR=5e-5.

What this tests
---------------
1. Does adding rare-class outputs (d, bp) destroy bmabz training, or
   do they coexist? In the full pipeline runs we kept seeing bmabz
   work fine but d and bp stay at zero — but that was confounded by
   class weighting amplifying gradient noise. Phase 0g checks the
   plain-BCE case to find out whether the multi-class architecture
   itself is the problem or whether it was the weighting.

2. What's the realistic F1 ceiling for d and bp on the official val
   split *without* class weighting? If it's near zero, weighted BCE
   becomes an obvious next step (Phase 0h). If it's already in the
   0.05-0.15 range, that's a useful baseline.

Three possible outcomes
-----------------------
1. bmabz F1 stays near 0.50, d and bp F1 are nonzero (any positive value)
   → Multi-class training works. Add weighted BCE in Phase 0h to push
   rare-class F1 higher.

2. bmabz F1 stays near 0.50, d and bp F1 are exactly 0
   → Architecture handles multi-class fine but the rare classes need
   class weighting to learn. Move to Phase 0h.

3. bmabz F1 drops below 0.40
   → Multi-class training disrupts what worked single-class. Investigate
   loss aggregation or class balance in batches.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0g.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
)
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
    PHASE0_LSTM_HIDDEN, PHASE0_LSTM_LAYERS,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import (
    PHASE0F_TRAIN_SITES, PHASE0F_VAL_SITES, PHASE0F_EPOCHS,
)


def build_phase0g_model(device: torch.device):
    """
    Build a 3-class WhaleVAD with the small Phase 0 LSTM.

    Same monkey-patching pattern as ``train_phase0.build_phase0_model``
    but with ``num_classes=3`` so the classifier head produces one
    probability per coarse class.
    """
    orig_hidden = cfg.LSTM_HIDDEN
    orig_layers = cfg.LSTM_LAYERS

    cfg.LSTM_HIDDEN = PHASE0_LSTM_HIDDEN
    cfg.LSTM_LAYERS = PHASE0_LSTM_LAYERS
    try:
        model = WhaleVAD(num_classes=3).to(device)
        spec = SpectrogramExtractor().to(device)
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            model(spec(dummy))
    finally:
        cfg.LSTM_HIDDEN = orig_hidden
        cfg.LSTM_LAYERS = orig_layers

    return model, spec


def train_one_epoch_3class(model, spec_extractor, loader, criterion,
                           optimizer, device):
    """
    Training pass with 3-class targets.

    Almost identical to ``train_phase0.train_one_epoch`` but the targets
    are kept as their full 3-class shape (we use the regular
    ``WhaleDataset``, not ``SingleClassDataset``).
    """
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)

        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        # Mask out padded frames; mask broadcasts across the class dim.
        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


@torch.no_grad()
def validate_3class(model, spec_extractor, loader, criterion, device,
                   val_annotations, file_start_dts, threshold: float):
    """
    Validation pass for 3-class outputs.

    Builds detections at a single fixed threshold per class (same value
    for all three to keep the test simple), computes event-level F1
    against the ground-truth annotations of all three classes.
    """
    model.eval()
    losses, n = 0.0, 0
    all_probs = {}
    hop = spec_extractor.hop_length

    for audio, targets, mask, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec)
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)
        losses += loss.item()
        n += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Single threshold across all 3 classes for simplicity. Per-class
    # tuning happens in later phases.
    thresholds = np.array([threshold] * 3)
    pred_events = postprocess_predictions(all_probs, thresholds)

    # GT for all three classes.
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    # Per-class breakdown for logging.
    per_class = {}
    for name in cfg.CALL_TYPES_3:
        m = metrics.get(name, {})
        per_class[name] = {
            "f1": m.get("f1", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
        }

    return {
        "loss": losses / max(n, 1),
        "f1": overall_f1,
        "per_class": per_class,
    }


def main():
    """Run Phase 0g end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0g expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0g_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0g configuration:")
    print(f"  Training sites: {PHASE0F_TRAIN_SITES}")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class (bmabz, d, bp)")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Training data — same 4 sites as Phase 0f, same 30s extension
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0F_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0F_TRAIN_SITES,
                                         manifest=train_manifest)
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    # Per-class annotation counts: useful for spotting if a class has
    # vanishingly few examples in the training data.
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
    for name in cfg.CALL_TYPES_3:
        c = (val_annotations["label_3class"] == name).sum()
        print(f"    {name}: {c} val annotations")

    # Use the regular WhaleDataset — no SingleClassDataset wrapper. This
    # is what makes Phase 0g 3-class.
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

    # Plain BCE — no class weighting, no focal. The point of Phase 0g
    # is to test multi-class output without confounding by loss tweaks.
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
    print(f"Training {PHASE0F_EPOCHS} epochs (3-class)")
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
        }, run_dir / f"phase0g_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0g_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0g VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"Overall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (Phase 0f reference: 0.495)")
    print(f"  d:     {max(d_f1s):.3f}")
    print(f"  bp:    {max(bp_f1s):.3f}")
    print(f"  Best overall: {max(f1s):.3f}")

    # Stability check: bmabz shouldn't have collapsed when we added
    # output channels.
    if max(bmabz_f1s) < 0.40:
        print("\n→ bmabz performance dropped vs Phase 0f. Multi-class output")
        print("  disrupts what worked single-class. Investigate loss")
        print("  aggregation or class balance in mini-batches.")
    elif max(d_f1s) > 0.05 or max(bp_f1s) > 0.05:
        print("\n→ All three classes train to nonzero F1. Pipeline scales to")
        print("  multi-class. Phase 0h: add weighted BCE to push rare-class F1.")
    else:
        print("\n→ bmabz preserved but rare classes stuck at zero. Architecture")
        print("  handles multi-class but rare-class learning needs class")
        print("  weighting. Move to Phase 0h.")


if __name__ == "__main__":
    main()
