"""
Phase 0p: 3-Class Direct Training (No 7-Class Collapse)
========================================================

Phase 0m used 7-class training with collapse-at-eval and got F1=0.442.
The patched ``train.py`` used 3-class direct training (USE_3CLASS=True)
and got F1=0.474 — but with weighted BCE and 150 epochs, so the 3-class
contribution can't be isolated.

Phase 0p closes that ambiguity: same 30-epoch schedule as Phase 0m,
same 8 sites, same paper-config full LSTM, same plain BCE, same seed,
but the model outputs 3 channels and trains directly on the coarse
(bmabz, d, bp) targets. No collapse-at-eval is needed — the model
output dimension already matches the eval class count.

The 7-vs-3 question matters
---------------------------
The paper claims the 7-class fine-grained training helps because it
gives the model richer subclass signal during learning, even if you
only care about the coarse 3-class metric at deployment. Phase 0m's
F1=0.442 supports that — it's slightly above what we got from 3-class
training at smaller scales (0g 4-site small-LSTM hit 0.385, and 0k
8-site small-LSTM hit 0.427). But we never compared 3-class direct
vs 7-class collapse *at the same architecture/data scale*.

Phase 0p is that comparison:

  - Same data (8 sites)
  - Same model (paper-config full LSTM)
  - Same loss (plain BCE)
  - Same epochs (30)
  - Same seed (42)

Only the output dimension differs: 3 channels here, 7 channels in 0m.

Possible outcomes
-----------------
1. F1 ≈ 0.442 ± 0.01: 7-class collapse and 3-class direct are
   equivalent at this scale. The paper's choice of 7-class is
   pedagogically nice but doesn't actually deliver F1 gains. Keep
   3-class direct for simplicity.
2. F1 > 0.46: 3-class direct is *better*. The fine-grained subclass
   training in 0m was actually hurting somehow, possibly by spreading
   gradient signal across rare subclasses that never converged.
3. F1 < 0.43: 3-class direct is worse. Confirms the paper's claim
   that 7-class fine-grained training helps. Keep collapse-at-eval.

This phase is independent of Phase 0o (focal loss). The two together
fully cover the loss × output-dimensionality 2x2 grid that
distinguishes Phase 0m from the patched ``train.py``:

  - 0m:   plain BCE,    7-class collapse  →  F1=0.442
  - 0o:   weighted+focal, 7-class collapse →  F1=?
  - 0p:   plain BCE,    3-class direct    →  F1=?
  - train.py patched: weighted BCE, 3-class direct, 150 epochs → F1=0.474

If 0o and 0p both come in around 0.44, the +0.032 gap from 0m to
``train.py`` is mostly from the 150-epoch schedule, not from loss or
output dim.

What changes vs Phase 0m
------------------------
Just two things:
  - cfg.USE_3CLASS = True (so WhaleDataset produces 3-channel targets)
  - Model built with num_classes=3
  - Validation skips the 7→3 collapse step (3-class probs go straight
    to postprocess_predictions)

Everything else — data loading, augmentation (none), wandb logging —
mirrors 0m.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0p.py
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
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
)
from train_phase0 import (
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE, PHASE0_THRESHOLD,
)
from train_phase0e import extend_all_segments, PHASE0E_SEGMENT_S
from train_phase0f import PHASE0F_VAL_SITES, PHASE0F_EPOCHS
from train_phase0g import train_one_epoch_3class
from train_phase0m import PHASE0M_TRAIN_SITES


def build_phase0p_model(device: torch.device):
    """
    Build a 3-class WhaleVAD with the paper-config full LSTM.

    Mirrors ``train_phase0m.build_phase0m_model`` but with
    ``num_classes=3`` and asserts ``cfg.USE_3CLASS=True`` so the
    dataset produces matching 3-channel targets. The full LSTM
    (hidden=128, layers=2) is the same as 0m — only the output head
    differs.
    """
    assert cfg.USE_3CLASS, (
        "Phase 0p requires cfg.USE_3CLASS=True. Flip it before "
        "constructing the dataset so target tensors come out 3-wide."
    )
    model = WhaleVAD(num_classes=3).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


@torch.no_grad()
def validate_3class_direct(model, spec_extractor, loader, criterion, device,
                            val_annotations, file_start_dts,
                            threshold: float):
    """
    Validation for 3-class direct training.

    Identical to ``train_phase0m.validate_7to3`` minus the collapse
    step. Model produces 3-channel probs which feed straight into
    ``postprocess_predictions``. Because ``cfg.USE_3CLASS=True``,
    ``cfg.class_names()`` returns the coarse names, so postprocess
    labels detections correctly without any toggling needed.
    """
    from tqdm import tqdm
    model.eval()
    losses, n = 0.0, 0
    all_probs = {}
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

        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    thresholds = np.array([threshold] * 3)
    pred_events = postprocess_predictions(all_probs, thresholds)

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
    """Run Phase 0p end-to-end."""
    # 3-class direct mode. WhaleDataset reads cfg.USE_3CLASS at
    # __init__, so we have to flip BEFORE constructing the dataset.
    cfg.USE_3CLASS = True
    print(f"cfg.USE_3CLASS = {cfg.USE_3CLASS} (3-class direct training)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Same seed as 0m for fair "output-dim only"
    # comparison.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0p", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
        "epochs": PHASE0F_EPOCHS,
        "train_sites": PHASE0M_TRAIN_SITES,
        "val_sites": PHASE0F_VAL_SITES,
        "lstm_hidden": cfg.LSTM_HIDDEN,
        "lstm_layers": cfg.LSTM_LAYERS,
        "training_classes": 3,
        "eval_classes": 3,
        # Loss flags so wandb diff is unambiguous.
        "use_weighted_bce": False,
        "use_focal_loss":   False,
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0p_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0p configuration:")
    print(f"  Training sites: {PHASE0M_TRAIN_SITES}  ({len(PHASE0M_TRAIN_SITES)})")
    print(f"  Validation sites: {PHASE0F_VAL_SITES}")
    print(f"  Output: 3-class direct (no 7→3 collapse)")
    print(f"  Loss: plain BCE")
    print(f"  LSTM: hidden={cfg.LSTM_HIDDEN}, layers={cfg.LSTM_LAYERS}")
    print(f"  Training segments: {PHASE0E_SEGMENT_S}s")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0F_EPOCHS}")

    # ------------------------------------------------------------------
    # Data — same 8 sites as Phase 0m
    # ------------------------------------------------------------------
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(PHASE0M_TRAIN_SITES)
    train_annotations = load_annotations(PHASE0M_TRAIN_SITES,
                                         manifest=train_manifest)
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

    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    print(f"\n  Val: {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    train_ds = WhaleDataset(train_segments)
    val_ds = WhaleDataset(val_segments)

    # Sanity check: targets should be 3-wide, not 7-wide.
    sample_audio, sample_target, sample_mask, _ = train_ds[0]
    assert sample_target.shape[1] == 3, (
        f"Expected 3-channel targets, got {sample_target.shape}. "
        f"WhaleDataset is in 7-class mode."
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

    run.config.update({
        "n_train_files":       len(train_manifest),
        "n_train_annotations": len(train_annotations),
        "n_train_segments":    len(train_segments),
    }, allow_val_change=True)

    # ------------------------------------------------------------------
    # Model + loss + optimizer
    # ------------------------------------------------------------------
    model, spec_extractor = build_phase0p_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
    optimizer = AdamW(
        model.parameters(), lr=PHASE0_LR, weight_decay=PHASE0_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # ------------------------------------------------------------------
    # Training loop. train_one_epoch_3class is class-count-agnostic
    # so it works for both 3-class and 7-class.
    # ------------------------------------------------------------------
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {PHASE0F_EPOCHS} epochs (3-class direct, plain BCE)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, PHASE0F_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_3class(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_3class_direct(
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
        }, run_dir / f"phase0p_epoch_{epoch:02d}.pt")
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history,
            }, run_dir / "phase0p_best.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0p VERDICT")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    bmabz_f1s = [h["per_class"]["bmabz"]["f1"] for h in history]
    d_f1s = [h["per_class"]["d"]["f1"] for h in history]
    bp_f1s = [h["per_class"]["bp"]["f1"] for h in history]

    print(f"\nOverall F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"\nBest F1 by class:")
    print(f"  bmabz: {max(bmabz_f1s):.3f}  (0m 7→3: 0.531)")
    print(f"  d:     {max(d_f1s):.3f}  (0m 7→3: 0.149)")
    print(f"  bp:    {max(bp_f1s):.3f}  (0m 7→3: 0.400)")
    print(f"  Best overall: {max(f1s):.3f}  (0m: 0.442, paper: 0.443)")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability:")
    print(f"  Mean swing: {mean_swing:.3f}  (0m: 0.042)")
    print(f"  Max swing:  {max_swing:.3f}  (0m: 0.130)")

    delta = max(f1s) - 0.442
    if delta > 0.015:
        verdict_text = (
            f"3-class direct beat 7→3 collapse: best F1 {max(f1s):.3f} "
            f"(+{delta:.3f} vs 0m). Drop fine-grained training."
        )
        print(f"\n→ 3-class direct beat 7-class collapse "
              f"(+{delta:.3f} F1 vs 0m).")
        print("  Fine-grained subclass training was hurting, not helping.")
        print("  Drop the 7-class step from future training.")
    elif delta > -0.015:
        verdict_text = (
            f"3-class ≈ 7→3: best F1 {max(f1s):.3f} ({delta:+.3f} vs 0m). "
            f"Output dim choice is a wash; pick simpler."
        )
        print(f"\n→ 3-class ≈ 7-class ({delta:+.3f} F1).")
        print("  The output dimensionality choice doesn't matter at this")
        print("  scale. Pick whichever is simpler — 3-class direct.")
    else:
        verdict_text = (
            f"3-class direct lost to 7→3: best F1 {max(f1s):.3f} "
            f"({delta:+.3f} vs 0m). Keep 7-class training."
        )
        print(f"\n→ 3-class direct lost to 7-class collapse "
              f"({delta:+.3f} F1).")
        print("  Confirms paper's claim that fine-grained subclass")
        print("  training helps. Keep 7-class for future runs.")

    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=run_dir / "phase0p_best.pt",
    )


if __name__ == "__main__":
    main()
