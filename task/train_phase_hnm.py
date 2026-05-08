"""
Phase HNM: Hard-Negative Mining Fine-Tune
==========================================

Continue training a converged Whale-VAD checkpoint with the explicit
hard-negative segments produced by ``mine_hard_negatives.py`` mixed into
the training pool.

The hard-neg segments are 30s windows centered on FPs the model currently
fires on. Their frame-level targets for the mined class (D) are zero by
construction (no GT D-call there); any other annotations that fall in
the window are preserved as positives so we never accidentally train the
model to suppress real bmabz/bp events that happen to sit near a ship
transient.

Usage
-----
::

    python train_phase_hnm.py \\
        --checkpoint runs/baseline_seed42/best_model.pt \\
        --hard_negatives runs/hardnegs/d_top1500.json \\
        --epochs 15 --lr 1e-5 --oversample 5
"""

from __future__ import annotations
import argparse, time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from dataset import (
    Segment, WhaleDataset, build_negative_segments, build_positive_segments,
    build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from spectrogram import SpectrogramExtractor
from train import validate  # reuse existing validate(), threshold tuning, etc.
from train_phase0e import extend_segment_to_fixed_length, PHASE0E_SEGMENT_S
import json


HNM_DEFAULT_LR = 1e-5
HNM_DEFAULT_EPOCHS = 15
HNM_DEFAULT_OVERSAMPLE = 5
HNM_RESAMPLE_EVERY = 5  # match train.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hard_negatives", type=str, required=True,
                   help="JSON output of mine_hard_negatives.py")
    p.add_argument("--epochs", type=int, default=HNM_DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=HNM_DEFAULT_LR,
                   help="Fine-tuning LR. Default 1e-5: low enough to not "
                        "destroy the converged representation.")
    p.add_argument("--oversample", type=int, default=HNM_DEFAULT_OVERSAMPLE,
                   help="Hard-neg segments are repeated this many times "
                        "per epoch so they're not drowned by the standard "
                        "positives + random negatives.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def load_hard_negatives_json(path: str) -> tuple[list[dict], dict]:
    """Read mining output, return (records, metadata)."""
    with open(path) as f:
        payload = json.load(f)
    return payload["fps"], {k: v for k, v in payload.items() if k != "fps"}


def build_hard_negative_segments(
    fp_records: list[dict],
    manifest, annotations,
    segment_s: float = PHASE0E_SEGMENT_S,
) -> list[Segment]:
    """
    Materialize each FP record into a 30s ``Segment`` centered on it.

    The frame-level targets are built by ``WhaleDataset.__getitem__`` from
    the segment's ``annotations`` field — we attach any GT annotation that
    intersects the window so co-occurring real calls (bmabz/bp typically)
    retain their positive labels. The mined-class region itself is zero
    by construction since the FP record is, by definition, in a region
    with no same-class GT.
    """
    sample_rate = cfg.SAMPLE_RATE
    target_samples = int(segment_s * sample_rate)

    # Index manifest for O(1) lookup.
    manifest_idx = manifest.set_index(["dataset", "filename"])

    # Pre-bucket annotations per file and convert to file-relative seconds.
    ann_by_file: dict = {}
    for _, a in annotations.iterrows():
        key = (a["dataset"], a["filename"])
        if key not in manifest_idx.index:
            continue
        fsd = manifest_idx.loc[key, "start_dt"]
        if fsd is None:
            continue
        ann_by_file.setdefault(key, []).append({
            "start_s": (a["start_datetime"] - fsd).total_seconds(),
            "end_s":   (a["end_datetime"]   - fsd).total_seconds(),
            "label":        a["annotation"],
            "label_3class": a["label_3class"],
        })

    segments: list[Segment] = []
    skipped = 0
    for r in fp_records:
        key = (r["dataset"], r["filename"])
        if key not in manifest_idx.index:
            skipped += 1
            continue
        file_row = manifest_idx.loc[key]
        file_dur_s = file_row["duration_s"]
        if file_dur_s < segment_s:
            skipped += 1
            continue

        # Center the 30s window on the FP midpoint, clamping to file ends.
        mid_s = 0.5 * (r["start_s"] + r["end_s"])
        seg_start_s = max(0.0, mid_s - segment_s / 2)
        seg_end_s = seg_start_s + segment_s
        if seg_end_s > file_dur_s:
            seg_end_s = file_dur_s
            seg_start_s = seg_end_s - segment_s

        # Annotations that intersect this window keep their positive labels.
        file_anns = ann_by_file.get(key, [])
        inter_anns = [
            a for a in file_anns
            if a["end_s"] > seg_start_s and a["start_s"] < seg_end_s
        ]

        segments.append(Segment(
            dataset=r["dataset"], filename=r["filename"], path=file_row["path"],
            start_sample=int(seg_start_s * sample_rate),
            end_sample=int(seg_end_s * sample_rate),
            file_start_dt=file_row["start_dt"],
            annotations=inter_anns,
            is_positive=False,  # informational; loss doesn't use this
        ))

    if skipped:
        print(f"  skipped {skipped} FP records (file too short or missing)")
    return segments


class HnmTrainingDataset(WhaleDataset):
    """
    Training dataset that mixes positives + resampled randoms + a fixed
    pool of oversampled hard-negatives.

    The hard-neg pool is computed once from the JSON; randoms are
    resampled each epoch (same protocol as ``TrainingDatasetWithResample``
    plus the Phase 0e 30s extension).
    """
    def __init__(self, positive_segments, hard_neg_segments, oversample,
                 manifest, annotations):
        self.positive_segments = positive_segments
        self.hard_neg_segments = list(hard_neg_segments) * max(1, oversample)
        self.manifest = manifest
        self.annotations = annotations
        self.negative_segments = []
        self.resample_negatives()
        super().__init__(self._all_segments())

    def _all_segments(self):
        return (self.positive_segments
                + self.negative_segments
                + self.hard_neg_segments)

    def resample_negatives(self):
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        neg = build_negative_segments(self.annotations, self.manifest, n_neg)
        # Phase 0e: extend randoms to 30s to match val tile length.
        manifest_idx = self.manifest.set_index(["dataset", "filename"])
        extended = []
        for s in neg:
            try:
                file_dur_s = float(manifest_idx.loc[
                    (s.dataset, s.filename), "duration_s"])
            except KeyError:
                continue
            extended.append(extend_segment_to_fixed_length(
                s, PHASE0E_SEGMENT_S, file_dur_s))
        self.negative_segments = extended
        self.segments = self._all_segments()


def train_epoch(model, spec_extractor, loader, criterion, optimizer, device):
    """Standard per-frame BCE epoch — same shape as train.train_epoch."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        spec = spec_extractor(audio)
        logits = model(spec)
        T = min(logits.size(1), targets.size(1))
        loss = criterion(logits[:, :T], targets[:, :T], mask[:, :T])
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = args.run_name or f"hnm_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Hard negatives
    # ------------------------------------------------------------------
    fp_records, hnm_meta = load_hard_negatives_json(args.hard_negatives)
    print(f"\nLoaded {len(fp_records)} hard negatives "
          f"(target='{hnm_meta['target_class']}', "
          f"threshold={hnm_meta['threshold']})")

    # ------------------------------------------------------------------
    # Standard data
    # ------------------------------------------------------------------
    print("\nLoading training data...")
    train_anns = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(train_anns, train_manifest)
    # Extend positives to 30s as well (Phase 0e).
    manifest_idx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [
        extend_segment_to_fixed_length(
            s, PHASE0E_SEGMENT_S,
            float(manifest_idx.loc[(s.dataset, s.filename), "duration_s"]))
        for s in pos_segs
        if (s.dataset, s.filename) in manifest_idx.index
    ]
    print(f"  {len(pos_segs)} positive segments (30s extended)")

    hard_segs = build_hard_negative_segments(fp_records, train_manifest,
                                              train_anns)
    print(f"  {len(hard_segs)} hard-neg segments × oversample {args.oversample}"
          f" = {len(hard_segs) * args.oversample} effective copies/epoch")

    train_ds = HnmTrainingDataset(
        pos_segs, hard_segs, args.oversample, train_manifest, train_anns)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"  starting val F1 (per ckpt): {ckpt.get('best_f1', 'unknown')}")

    # Tuned thresholds from the source ckpt drive in-loop validation.
    if "thresholds" in ckpt:
        thresholds = ckpt["thresholds"].to(device).float()
    else:
        thresholds = torch.tensor(cfg.DEFAULT_THRESHOLDS, device=device)
    print(f"  starting thresholds: {thresholds.tolist()}")

    # ------------------------------------------------------------------
    # Loss + optimizer (same recipe as train.py, but lower LR)
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY,
                      betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=4, min_lr=1e-7)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch > 1 and epoch % HNM_RESAMPLE_EVERY == 0:
            train_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(train_ds.segments)}")

        train_loss = train_epoch(model, spec_extractor, train_loader,
                                  criterion, optimizer, device)
        val = validate(model, spec_extractor, val_loader, criterion, device,
                        thresholds, val_anns, file_start_dts,
                        tune_thresholds=True)
        thresholds = torch.tensor(val["thresholds"], device=device).float()

        improved = val["mean_f1"] > best_f1
        marker = " *** new best" if improved else ""
        print(f"\nEpoch {epoch:2d}/{args.epochs}  "
              f"({time.time()-t0:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Val F1: {val['mean_f1']:.3f}  Best: {best_f1:.3f}")

        scheduler.step(val["mean_f1"])

        ckpt_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_f1": max(best_f1, val["mean_f1"]),
            "thresholds": thresholds.cpu(),
            "hnm_meta": hnm_meta,
        }
        if improved:
            best_f1 = val["mean_f1"]
            torch.save(ckpt_save, run_dir / "best_model.pt")
            no_improve = 0
        else:
            no_improve += 1
        torch.save(ckpt_save, run_dir / "latest_model.pt")

        if no_improve >= 8:
            print(f"\n  Early stop: no improvement for {no_improve} epochs")
            break

    print(f"\nDone. Best F1: {best_f1:.3f}")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()