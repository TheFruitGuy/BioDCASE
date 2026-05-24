"""
BPN Ladder - Rung 1: Minimal Working Gate
=========================================

Second rung of the WhaleVAD-BPN ladder. It adds the smallest possible boundary
proposal branch on top of rung 0's dilated backbone + focal recipe, to verify
that the gating mechanism wires up and trains end-to-end without collapsing the
classifier.

Minimal configuration (later rungs relax each of these):
- single tap (the last depthwise tap, h2),
- a single ROI per head (``R = 1``, so no multi-ROI proposal conv-transpose),
- a BiLSTM ROI processor producing a per-frame, per-class gate,
- the gate initialised (bias = ``BPN_GATE_INIT_BIAS``) so the mask starts ~0.98
  and the model begins as a near-pass-through of the classifier, learning only
  to suppress false positives over time.

The gated output is a probability (classifier probability x mask), so training
uses focal loss on probabilities. Optionally warm-start the backbone from a
rung-0 checkpoint with ``--init-from``; the fresh BPN parameters load as new.

What this rung answers
----------------------
Does the gate train stably and does suppression help precision without
destroying recall? It establishes the mechanics before sweeping the unknown
design values (tap count, R, weighted-mean form, gate supervision) in later
rungs.

Usage
-----
::

    python train_bpn1.py
    python train_bpn1.py --init-from runs/bpn0_<timestamp>/bpn0_best.pt
    python train_bpn1.py --wandb
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from dataset_final import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_negative_segments, build_val_segments,
    extend_all_segments, WhaleDataset, collate_fn,
)
from spectrogram_final import SpectrogramExtractor
from model_bpn_final import WhaleVADBPN, focal_loss_with_probs
from postprocess_final import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)


# ======================================================================
# Reproducibility helpers
# ======================================================================

def seed_everything(seed: int = 42) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def seeded_dataloader_kwargs(seed: int) -> dict:
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {"generator": g, "worker_init_fn": _worker_init}


# ======================================================================
# Model
# ======================================================================

def build_model(device: torch.device, init_from: str | None = None):
    """
    BPN model (minimal gate) + spectrogram extractor.

    If ``init_from`` is given, the backbone/classifier weights are loaded from
    a rung-0 checkpoint (strict=False); the fresh BPN parameters are left at
    their gate-pass-through initialisation.
    """
    model = WhaleVADBPN(
        num_classes=7, use_bpn=True,
        bpn_taps=(2,), bpn_R=1, bpn_use_bilstm=True,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))

    if init_from:
        ckpt = torch.load(init_from, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Warm-started from {init_from}")
        print(f"    fresh (BPN) params: {len(missing)}; ignored: {len(unexpected)}")
    return model, spec


# ======================================================================
# Train / validate
# ======================================================================

def train_one_epoch(model, spec_extractor, loader, optimizer, device):
    """One training pass with masked focal loss on gated probabilities."""
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        probs = model(spec_extractor(audio))   # already in (0, 1)
        T = min(probs.size(1), targets.size(1))
        probs, targets, mask = probs[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = focal_loss_with_probs(probs, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


@torch.no_grad()
def validate(model, spec_extractor, loader, device,
             val_annotations, file_start_dts, threshold: float):
    """7-class gated inference -> collapse to 3 -> event-level F1."""
    model.eval()
    all_probs_7 = {}
    hop = spec_extractor.hop_length

    for audio, _, _, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        probs = model(spec_extractor(audio)).cpu().numpy()   # gated probs
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs_7[key] = probs[j, :n_frames, :]

    all_probs_3 = collapse_probs_to_3class(all_probs_7)

    cfg.USE_3CLASS = True
    try:
        pred_events = postprocess_predictions(
            all_probs_3, np.array([threshold] * 3))
    finally:
        cfg.USE_3CLASS = False

    gt_events = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    per_class = {
        name: {
            "f1": metrics.get(name, {}).get("f1", 0.0),
            "precision": metrics.get(name, {}).get("precision", 0.0),
            "recall": metrics.get(name, {}).get("recall", 0.0),
            "tp": metrics.get(name, {}).get("tp", 0),
            "fp": metrics.get(name, {}).get("fp", 0),
            "fn": metrics.get(name, {}).get("fn", 0),
        }
        for name in cfg.CALL_TYPES_3
    }
    return {"f1": metrics.get("overall", {}).get("f1", 0.0), "per_class": per_class}


# ======================================================================
# Per-epoch negative resampling
# ======================================================================

def resample_negatives_for_epoch(pos_segs, train_annotations, train_manifest,
                                 n_neg, segment_s, epoch, rng, verbose=False):
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg, rng=rng)
    neg_segs = extend_all_segments(neg_segs, train_manifest, segment_s)
    if verbose and neg_segs:
        first = neg_segs[0]
        print(f"    epoch {epoch}: resampled {len(neg_segs)} negatives "
              f"[first: {first.filename} @ {first.start_sample} samp]")
    return pos_segs + neg_segs


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BPN ladder rung 1: minimal working gate.")
    p.add_argument("--wandb", action="store_true",
                   help="Log to Weights & Biases (off by default).")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--init-from", default=None,
                   help="Optional rung-0 checkpoint to warm-start the backbone.")
    p.add_argument("--epochs", type=int, default=cfg.BPN_EPOCHS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    cfg.USE_3CLASS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = seed_everything(args.seed)
    sampling_rng = random.Random(seed)

    train_sites = list(cfg.TRAIN_DATASETS)
    val_sites = list(cfg.VAL_DATASETS)

    run = None
    if args.wandb:
        import wandb_utils as wbu
        run = wbu.init_phase("bpn1", config={
            "lr": cfg.BPN_LR, "weight_decay": cfg.BPN_WEIGHT_DECAY,
            "batch_size": cfg.BPN_BATCH_SIZE, "threshold": cfg.THRESHOLD,
            "seed": seed, "neg_ratio": cfg.NEG_RATIO,
            "segment_s": cfg.TRAIN_SEGMENT_S, "epochs": args.epochs,
            "loss": "focal", "focal_alpha": cfg.FOCAL_ALPHA,
            "focal_gamma": cfg.FOCAL_GAMMA,
            "bpn_taps": (2,), "bpn_R": 1, "bpn_use_bilstm": True,
            "bpn_gate_init_bias": cfg.BPN_GATE_INIT_BIAS,
            "init_from": args.init_from,
            "train_sites": train_sites, "val_sites": val_sites,
        }, mode=args.wandb_mode)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"bpn1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration (BPN rung 1 - minimal gate):")
    print(f"  Taps: (2,)   R: 1   ROI processor: BiLSTM")
    print(f"  Gate init bias: {cfg.BPN_GATE_INIT_BIAS} (mask starts ~pass-through)")
    print(f"  Loss: focal (alpha={cfg.FOCAL_ALPHA}, gamma={cfg.FOCAL_GAMMA})")
    print(f"  LR={cfg.BPN_LR}, wd={cfg.BPN_WEIGHT_DECAY}, "
          f"batch={cfg.BPN_BATCH_SIZE}, epochs={args.epochs}")

    # --- data ---
    print("\nLoading training data...")
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    pos_segs = build_positive_segments(
        train_annotations, train_manifest, rng=sampling_rng)
    pos_segs = extend_all_segments(pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    print(f"  {len(train_manifest)} files, {len(train_annotations)} annotations")
    print(f"  Positive segments (fixed): {len(pos_segs)}; negatives/epoch: {n_neg}")

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BPN_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, {len(val_segments)} segments")

    # --- model + optimizer ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, spec_extractor = build_model(device, init_from=args.init_from)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    if run is not None:
        run.config.update({"n_params": n_params}, allow_val_change=True)

    optimizer = AdamW(
        model.parameters(), lr=cfg.BPN_LR, weight_decay=cfg.BPN_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2))

    # --- training loop ---
    history = []
    print(f"\n{'=' * 60}\nTraining {args.epochs} epochs\n{'=' * 60}")
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_segments = resample_negatives_for_epoch(
            pos_segs, train_annotations, train_manifest, n_neg,
            cfg.TRAIN_SEGMENT_S, epoch, sampling_rng, verbose=True)
        train_loader = DataLoader(
            WhaleDataset(train_segments), batch_size=cfg.BPN_BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True, **seeded_dataloader_kwargs(seed))

        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, optimizer, device)
        val = validate(model, spec_extractor, val_loader, device,
                       val_annotations, file_start_dts, threshold=cfg.THRESHOLD)
        epoch_time = time.time() - t0

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]
        marker = " *** new best" if improved else ""
        macro = sum(val["per_class"][n]["f1"] for n in cfg.CALL_TYPES_3) / 3

        print(f"\nEpoch {epoch:2d}/{args.epochs}  ({epoch_time:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        print(f"    OVERALL F1={val['f1']:.3f}  MACRO F1={macro:.3f}")

        if run is not None:
            import wandb_utils as wbu
            wbu.log_epoch_3class(epoch, train_loss,
                                 {"loss": 0.0, "f1": val["f1"],
                                  "per_class": val["per_class"]})

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "f1": val["f1"], "macro_f1": macro,
                        "per_class": val["per_class"]})
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "f1": val["f1"], "history": history}
        torch.save(ckpt, run_dir / f"bpn1_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "bpn1_best.pt")

    # --- summary ---
    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    print(f"\n{'=' * 60}\nBPN RUNG 1 SUMMARY\n{'=' * 60}")
    print(f"Best micro F1: {max(f1s):.3f} (epoch {f1s.index(max(f1s)) + 1})")
    print(f"Best macro F1: {max(macros):.3f}")
    for name in cfg.CALL_TYPES_3:
        print(f"  best {name}: {max(h['per_class'][name]['f1'] for h in history):.3f}")

    if run is not None:
        import wandb_utils as wbu
        wbu.finalize_phase(
            history,
            verdict=(f"Minimal BPN gate: best micro {max(f1s):.3f}, "
                     f"best macro {max(macros):.3f}."),
            best_ckpt=run_dir / "bpn1_best.pt")


if __name__ == "__main__":
    main()
