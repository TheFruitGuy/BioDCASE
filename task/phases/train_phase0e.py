"""
Phase 0e: Match Training Segment Length to Validation
=====================================================

Phase 0c showed the model can hit F1=0.443 within-site, but with wild
oscillation (0.44 → 0.004 → 0.18 → 0.39 across consecutive epochs).
Phase 0d ruled out BatchNorm running statistics as the primary cause.

Diagnostic of segment lengths revealed:

  - Training positive segments: median 9.5s, range 3-30s
  - Training negative segments: median 16.1s, range 5-29s
  - Validation segments: ALL exactly 30s

The model is being trained on short clips and evaluated on clips 3×
longer. This mismatches the BiLSTM's hidden state dynamics, the
BatchNorm spatial statistics, and effectively asks the model to
extrapolate to a context length it never saw during training.

Phase 0e tests the fix: **extend every training segment to a fixed
30 seconds** (the same length validation uses). The annotations stay
where they are within the segment, only the surrounding context grows.
A 7-second call+collar segment becomes a 30s segment with the call
sitting somewhere inside; targets are 1 only at frames where the call
is, 0 elsewhere. The expectation is the same as it always was, but
now the model sees 30s of audio with sparse positive frames during
training, exactly matching what it sees at evaluation time.

Three possible outcomes
-----------------------
1. F1 stabilises at 0.30+ across multiple consecutive epochs. Best F1
   similar to or higher than Phase 0c. → Train/val length mismatch
   was the cause. Apply this fix to the full pipeline.

2. F1 still oscillates but is smoother (max swing 0.10-0.20 instead
   of 0.40+). → Length is part of the problem; combine with BN
   momentum reduction in Phase 0f.

3. F1 oscillates as before. → Length wasn't it. Look elsewhere
   (positive/negative ratio per batch, optimizer choice, or whether
   the BiLSTM hidden state initialization is the issue).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0e.py
"""

import time
import random as random_module
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from dataset import (
    Segment, load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    collate_fn,
)
from train_phase0 import (
    SingleClassDataset, build_phase0_model, validate_one_class,
    train_one_epoch,
    TARGET_CLASS_IDX, TARGET_CLASS_NAME,
    PHASE0_LR, PHASE0_WEIGHT_DECAY, PHASE0_BATCH_SIZE,
    PHASE0_EPOCHS, PHASE0_THRESHOLD,
)
from train_phase0c import (
    PHASE0C_SITE, PHASE0C_VAL_FRACTION, PHASE0C_SEED,
    split_manifest_by_file,
)


#: Target segment length in seconds, matching validation tiles.
PHASE0E_SEGMENT_S = 30.0


def extend_segment_to_fixed_length(
    seg: Segment,
    target_seconds: float,
    file_duration_s: float,
    sample_rate: int = cfg.SAMPLE_RATE,
    rng: random_module.Random = None,
) -> Segment:
    """
    Return a copy of ``seg`` whose ``[start_sample, end_sample)`` window
    is exactly ``target_seconds`` long.

    The original segment is grown to the target length by adding context
    on both sides. The position of the original content within the new
    window is randomised so the model doesn't learn a fixed
    "calls-always-near-frame-N" prior. If the original segment is already
    at or above the target length, it's returned unchanged.

    The segment's ``annotations`` field is NOT modified — annotation
    times are file-relative, so they remain valid regardless of how the
    segment window moves. The frame-level target tensor is rebuilt by
    ``WhaleDataset.__getitem__`` based on which annotations intersect
    the new ``[start_sample, end_sample)`` range.

    Parameters
    ----------
    seg : Segment
        The original variable-length training segment.
    target_seconds : float
        Desired final length, e.g. 30.0 for matching validation tiles.
    file_duration_s : float
        Total duration of the underlying audio file, used to clamp the
        extended window to file boundaries.
    sample_rate : int
    rng : random.Random, optional
        For reproducible randomised positioning. ``None`` uses the
        module-level ``random``.

    Returns
    -------
    Segment
        A new Segment dataclass instance; original is not mutated.
    """
    if rng is None:
        rng = random_module

    target_samples = int(target_seconds * sample_rate)
    file_samples = int(file_duration_s * sample_rate)
    cur_length = seg.end_sample - seg.start_sample

    # Already at or above target; nothing to do.
    if cur_length >= target_samples:
        return seg

    extra = target_samples - cur_length

    # File is shorter than target — center on midpoint, clamp ends.
    if file_samples <= target_samples:
        return replace(seg, start_sample=0, end_sample=file_samples)

    # How much can we add on each side without leaving the file?
    pre_room = seg.start_sample
    post_room = file_samples - seg.end_sample

    # Randomise the split between pre and post when both sides have room.
    # Bound by what each side actually has available, then by what we need.
    pre_extra = min(pre_room, rng.randint(0, extra))
    post_extra = min(post_room, extra - pre_extra)

    # If one side runs out of room, push the leftover to the other side.
    deficit = extra - pre_extra - post_extra
    if deficit > 0:
        if pre_room - pre_extra >= deficit:
            pre_extra += deficit
        else:
            post_extra += deficit

    new_start = max(0, seg.start_sample - pre_extra)
    new_end = min(file_samples, new_start + target_samples)
    # Final guarantee on length.
    new_start = max(0, new_end - target_samples)

    return replace(seg, start_sample=new_start, end_sample=new_end)


def extend_all_segments(segments, manifest, target_seconds: float):
    """
    Apply ``extend_segment_to_fixed_length`` to every segment in a list.

    Looks up each segment's ``duration_s`` from the file manifest so the
    extension can clamp to file boundaries.
    """
    rng = random_module.Random(0xC0FFEE)
    duration_lookup = {
        (r["dataset"], r["filename"]): r["duration_s"]
        for _, r in manifest.iterrows()
    }
    extended = []
    for seg in segments:
        dur = duration_lookup.get((seg.dataset, seg.filename))
        if dur is None:
            continue
        extended.append(
            extend_segment_to_fixed_length(seg, target_seconds, dur, rng=rng)
        )
    return extended


def main():
    """Run Phase 0e end-to-end."""
    assert cfg.USE_3CLASS, (
        "Phase 0e expects USE_3CLASS=True. Set it in config.py before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Wandb run setup. Seeding everything FIRST so DataLoader workers
    # and model weights are reproducible across runs of this phase.
    # ------------------------------------------------------------------
    SEED = 42
    wbu.seed_everything(SEED, deterministic=False)

    run = wbu.init_phase("0e", config={
        "lr": PHASE0_LR,
        "weight_decay": PHASE0_WEIGHT_DECAY,
        "batch_size": PHASE0_BATCH_SIZE,
        "threshold": PHASE0_THRESHOLD,
        "seed": SEED,
        "neg_ratio": cfg.NEG_RATIO,
        "segment_s": PHASE0E_SEGMENT_S,
    })

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase0e_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nPhase 0e configuration:")
    print(f"  Site: {PHASE0C_SITE} (same-site split)")
    print(f"  Val fraction: {PHASE0C_VAL_FRACTION}")
    print(f"  Target class: {TARGET_CLASS_NAME}")
    print(f"  *** Training segments forced to {PHASE0E_SEGMENT_S}s ***")
    print(f"  LR: {PHASE0_LR}, batch: {PHASE0_BATCH_SIZE}, "
          f"epochs: {PHASE0_EPOCHS}")

    # ------------------------------------------------------------------
    # Data — like Phase 0c but with the segment-extension step
    # ------------------------------------------------------------------
    print(f"\nLoading data...")
    full_manifest = get_file_manifest([PHASE0C_SITE])
    full_annotations = load_annotations([PHASE0C_SITE], manifest=full_manifest)

    train_manifest, val_manifest = split_manifest_by_file(
        full_manifest, PHASE0C_VAL_FRACTION, seed=PHASE0C_SEED,
    )

    val_filenames = set(val_manifest["filename"])
    train_annotations = full_annotations[
        ~full_annotations["filename"].isin(val_filenames)
    ].reset_index(drop=True)
    val_annotations = full_annotations[
        full_annotations["filename"].isin(val_filenames)
    ].reset_index(drop=True)

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )

    # Phase 0e intervention: extend every training segment to 30s.
    pos_segs = extend_all_segments(pos_segs, train_manifest, PHASE0E_SEGMENT_S)
    neg_segs = extend_all_segments(neg_segs, train_manifest, PHASE0E_SEGMENT_S)
    train_segments = pos_segs + neg_segs

    # Quick verification that the extension actually worked.
    pos_durs = [(s.end_sample - s.start_sample) / cfg.SAMPLE_RATE
                for s in pos_segs[:200]]
    print(f"Training segment durations after extension (first 200 positives):")
    print(f"  min={min(pos_durs):.1f}s  max={max(pos_durs):.1f}s  "
          f"mean={sum(pos_durs)/len(pos_durs):.1f}s")

    val_segments = build_val_segments(val_manifest, val_annotations)

    train_ds = SingleClassDataset(train_segments, TARGET_CLASS_IDX)
    val_ds = SingleClassDataset(val_segments, TARGET_CLASS_IDX)

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
    print(f"Training {PHASE0_EPOCHS} epochs (30s training segments)")
    print(f"{'=' * 60}")

    for epoch in range(1, PHASE0_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate_one_class(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=PHASE0_THRESHOLD,
        )
        epoch_time = time.time() - t0

        print(f"\nEpoch {epoch:2d}/{PHASE0_EPOCHS}  ({epoch_time:.0f}s)")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        print(f"  {TARGET_CLASS_NAME}: TP={val['tp']:4} FP={val['fp']:5} "
              f"FN={val['fn']:4}  P={val['precision']:.3f} "
              f"R={val['recall']:.3f} F1={val['f1']:.3f}")

        wbu.log_epoch(epoch, train_loss, val)


        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "precision": val["precision"],
            "recall": val["recall"],
        })

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "history": history,
        }, run_dir / f"phase0e_epoch_{epoch:02d}.pt")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 0e VERDICT")
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
    print(f"  Phase 0c reference: max swing 0.44, mean swing 0.20")
    print(f"  Phase 0d reference: max swing 0.37, mean swing 0.18")

    if max_swing < 0.10:
        print("→ STABILITY FIXED. Train/val length mismatch was the cause.")
        print("  Update dataset.py to extend training segments to 30s, then")
        print("  scale up to multi-site / multi-class.")
    elif max_swing < 0.20:
        print("→ Improved but still wobbling. Combine with BN momentum")
        print("  reduction in Phase 0f, or investigate batch composition.")
    else:
        print("→ Same oscillation. Length wasn't the primary cause.")
        print("  Investigate: positive-fraction-per-batch invariance, LSTM")
        print("  hidden initialization, or whether input normalization")
        print("  differs between training and validation pipelines.")


    # ------------------------------------------------------------------
    # Wandb: stamp summary metrics + verdict and log best checkpoint
    # ------------------------------------------------------------------
    verdict_text = (
        f"Phase 0e: best F1 {max(f1s):.3f} at epoch "
        f"{f1s.index(max(f1s)) + 1}, final F1 {f1s[-1]:.3f}."
    )
    wbu.finalize_phase(
        history,
        verdict=verdict_text,
        best_ckpt=None,
    )

if __name__ == "__main__":
    main()
