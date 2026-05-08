"""
Phase HNM: Hard-Negative Mining Fine-Tune (model-agnostic)
============================================================

Continue training a converged WhaleVAD or WhaleVAD-BPN checkpoint with
the explicit hard-negative segments produced by ``mine_hard_negatives.py``
mixed into the training pool.

Auto-detects the checkpoint's architecture using the same
``ensemble_predict.build_model_for_ckpt`` helper used by ensembling.
Handles both:

  - Baseline ``WhaleVAD``: forward returns logits → sigmoid → BCE-with-
    logits + optional focal + per-class pos_weight (matches train.py).
  - ``WhaleVADBPN``: forward returns ``{"probs": ...}`` already in
    [0, 1] from the gating multiplication → BCE-on-probs + matching
    focal/weighted variant (matches train_bpn.py).

The hard-neg segments are 30-second windows centered on the FPs the
source model fires on. Their frame-level targets for the mined class
are zero by construction (no GT call there); any other GT annotations
that overlap the window are preserved as positives so we never train
the model to suppress real co-occurring calls.

Usage
-----
::

    python train_phase_hnm.py \\
        --checkpoint runs/phase5_20260507_211504/best_model.pt \\
        --hard_negatives runs/hardnegs/d_phase5_20260507_211504.json \\
        --epochs 15 --lr 1e-5 --oversample 5

Run once per checkpoint, then re-ensemble the resulting ``best_model.pt``
files via ``ensemble_predict.py``.
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model import WhaleVADLoss, compute_class_weights
from postprocess import (
    Detection, collapse_probs_to_3class, compute_metrics,
    postprocess_predictions,
)
from spectrogram import SpectrogramExtractor
from train_phase0e import extend_segment_to_fixed_length, PHASE0E_SEGMENT_S

# Shared model/inference plumbing for baseline + BPN.
from ensemble_predict import (
    build_model_for_ckpt, predict_probabilities,
    tune_thresholds_on_probs, evaluate_with_thresholds,
)


# ======================================================================
# Constants
# ======================================================================

HNM_DEFAULT_LR = 1e-5
HNM_DEFAULT_EPOCHS = 15
HNM_DEFAULT_OVERSAMPLE = 5
HNM_RESAMPLE_EVERY = 5     # match train.py's negative-resample cadence
HNM_EARLY_STOP = 8


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Source checkpoint to continue training from. "
                        "Architecture is auto-detected.")
    p.add_argument("--hard_negatives", type=str, required=True,
                   help="JSON output of mine_hard_negatives.py")
    p.add_argument("--epochs", type=int, default=HNM_DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=HNM_DEFAULT_LR,
                   help="Fine-tuning LR. Default 1e-5: low enough not to "
                        "destroy the converged representation.")
    p.add_argument("--oversample", type=int, default=HNM_DEFAULT_OVERSAMPLE,
                   help="Hard-neg segments are repeated this many times "
                        "per epoch so they're not drowned by the standard "
                        "positives + random negatives.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


# ======================================================================
# Hard-negative loading + segment construction
# ======================================================================

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
    intersects the window so co-occurring real calls (typically bmabz/bp)
    retain their positive labels. The mined-class region itself is zero
    by construction since the FP record is, by definition, in a region
    with no same-class GT.
    """
    sample_rate = cfg.SAMPLE_RATE
    manifest_idx = manifest.set_index(["dataset", "filename"])

    # Pre-bucket annotations per file in file-relative seconds.
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
        file_dur_s = float(file_row["duration_s"])
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
            is_positive=False,
        ))

    if skipped:
        print(f"  skipped {skipped} FP records (file too short or missing)")
    return segments


# ======================================================================
# Training dataset (positives + resampled randoms + oversampled hard-negs)
# ======================================================================

class HnmTrainingDataset(WhaleDataset):
    """
    Mixes positives + resampled random negatives + a fixed pool of
    oversampled hard negatives.

    The hard-neg pool is computed once from the JSON; randoms are
    resampled every HNM_RESAMPLE_EVERY epochs (same protocol as
    ``TrainingDatasetWithResample``) plus the Phase 0e 30s extension
    so train and val tile lengths match.
    """
    def __init__(self, positive_segments, hard_neg_segments, oversample,
                 manifest, annotations):
        self.positive_segments = positive_segments
        self.hard_neg_segments = list(hard_neg_segments) * max(1, oversample)
        self.manifest = manifest
        self.annotations = annotations
        self._manifest_idx = manifest.set_index(["dataset", "filename"])
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
        extended = []
        for s in neg:
            try:
                file_dur_s = float(self._manifest_idx.loc[
                    (s.dataset, s.filename), "duration_s"])
            except KeyError:
                continue
            extended.append(extend_segment_to_fixed_length(
                s, PHASE0E_SEGMENT_S, file_dur_s))
        self.negative_segments = extended
        self.segments = self._all_segments()


# ======================================================================
# Loss: BCE on probs (BPN) — matches the WhaleVADLoss recipe but on
# already-sigmoided outputs from the BPN's gating step
# ======================================================================

def probs_bce_focal_loss(
    probs: torch.Tensor,            # (B, T, C) in [0, 1]
    targets: torch.Tensor,          # (B, T, C) in {0, 1}
    mask: torch.Tensor,             # (B, T) bool/float
    pos_weight: torch.Tensor | None = None,  # (C,) or None
    use_focal: bool = True,
    focal_alpha: float = cfg.FOCAL_ALPHA,
    focal_gamma: float = cfg.FOCAL_GAMMA,
) -> torch.Tensor:
    """
    BCE on probabilities, with the same focal/weighted variant the
    baseline ``WhaleVADLoss`` applies on logits.

    The BPN model returns probabilities (post-sigmoid, post-gate), so
    ``F.binary_cross_entropy_with_logits`` cannot be used. We compute
    the BCE term explicitly in numerically-stable form via a clamp,
    then optionally apply the focal weight ``(1 - p_t)^gamma`` and the
    alpha balancing term as in Lin et al. (2018).

    The ``pos_weight`` scaling matches BCEWithLogitsLoss semantics:
    pos_weight multiplies the positive term only.
    """
    eps = 1e-7
    p = probs.clamp(eps, 1.0 - eps)

    # Standard BCE per element.
    pos_term = targets * torch.log(p)
    neg_term = (1.0 - targets) * torch.log(1.0 - p)

    # Per-class positive weighting. Broadcasts (C,) → (B, T, C).
    if pos_weight is not None:
        pos_term = pos_term * pos_weight.view(1, 1, -1)

    if use_focal:
        # p_t is the probability assigned to the *true* class label.
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        focal_w = (1.0 - p_t).pow(focal_gamma)
        # Alpha balancing: weight positives by alpha, negatives by 1-alpha.
        alpha_t = targets * focal_alpha + (1.0 - targets) * (1.0 - focal_alpha)
        weight = focal_w * alpha_t
        pos_term = pos_term * weight
        neg_term = neg_term * weight

    per_elem = -(pos_term + neg_term)        # (B, T, C)

    valid = mask.unsqueeze(-1).float()       # (B, T, 1)
    per_elem = per_elem * valid
    return per_elem.sum() / valid.sum().clamp(min=1.0) / per_elem.size(-1)


# ======================================================================
# Forward + loss dispatch (baseline vs BPN)
# ======================================================================

def forward_and_loss(model, model_type: str, spec_extractor,
                     audio, targets, mask,
                     baseline_criterion: WhaleVADLoss,
                     bpn_pos_weight: torch.Tensor | None):
    """
    Run the model on a batch and compute the appropriate loss.

    For baseline checkpoints we use the existing ``WhaleVADLoss`` exactly
    as ``train.py`` does; for BPN checkpoints we compute BCE on the gated
    probabilities with the same focal+weighted recipe.
    """
    spec = spec_extractor(audio)
    out = model(spec)

    if model_type == "bpn":
        probs = out["probs"]                                # (B, T, C)
        T = min(probs.size(1), targets.size(1))
        return probs_bce_focal_loss(
            probs[:, :T], targets[:, :T], mask[:, :T],
            pos_weight=bpn_pos_weight,
            use_focal=cfg.USE_FOCAL_LOSS,
        )
    else:
        logits = out                                        # (B, T, C)
        T = min(logits.size(1), targets.size(1))
        return baseline_criterion(
            logits[:, :T], targets[:, :T], mask[:, :T])


def train_epoch(model, model_type, spec_extractor, loader,
                baseline_criterion, bpn_pos_weight, optimizer, device):
    """One fine-tuning epoch — same shape as train.train_epoch."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = forward_and_loss(
            model, model_type, spec_extractor,
            audio, targets, mask,
            baseline_criterion, bpn_pos_weight,
        )
        # Skip occasional NaN batches rather than crashing (rare BiLSTM
        # instability seen in train.py too).
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1)


# ======================================================================
# Validation (model-agnostic)
# ======================================================================

@torch.no_grad()
def validate_hnm(model, model_type, spec_extractor, val_loader, device,
                 gt_events, baseline_criterion, bpn_pos_weight,
                 tune_thresholds: bool = True):
    """
    Run model on val set, optionally tune per-class thresholds, return
    metrics + tracking loss + tuned thresholds.

    Reuses ``predict_probabilities`` so BPN's dict-return is handled, and
    ``tune_thresholds_on_probs`` so the threshold sweep matches the
    grids used at training time.
    """
    model.eval()

    # 1. Tracking loss across val batches (just for logging — does not
    # drive checkpointing or threshold selection).
    total_loss, n = 0.0, 0
    for audio, targets, mask, _ in val_loader:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        loss = forward_and_loss(
            model, model_type, spec_extractor,
            audio, targets, mask,
            baseline_criterion, bpn_pos_weight,
        )
        total_loss += loss.item()
        n += 1
    val_loss = total_loss / max(n, 1)

    # 2. Re-run inference to collect per-window probabilities (needed
    # in the right format for stitch + threshold sweep). predict_probabilities
    # handles baseline-vs-BPN return-value differences.
    all_probs = predict_probabilities(
        model, model_type, spec_extractor, val_loader, device)
    all_probs = collapse_probs_to_3class(all_probs)

    # 3. Per-class threshold tune (coordinate descent on the same grid
    # as train.py / ensemble_predict.py).
    if tune_thresholds:
        thresholds = tune_thresholds_on_probs(all_probs, gt_events)
    else:
        thresholds = np.array(cfg.DEFAULT_THRESHOLDS, dtype=np.float64)

    # 4. Final metrics with the chosen thresholds.
    metrics = evaluate_with_thresholds(all_probs, gt_events, thresholds)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)

    # Per-class print, mirroring train.py's epoch summary.
    print(f"  Val (loss={val_loss:.4f}):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):6} "
              f"FN={m.get('fn', 0):6}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}")

    return {
        "loss": val_loss,
        "mean_f1": overall_f1,
        "per_class": metrics,
        "thresholds": thresholds.tolist(),
    }


# ======================================================================
# Main
# ======================================================================

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
    # Extend positives to 30s as well (Phase 0e protocol).
    manifest_idx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [
        extend_segment_to_fixed_length(
            s, PHASE0E_SEGMENT_S,
            float(manifest_idx.loc[(s.dataset, s.filename), "duration_s"]))
        for s in pos_segs
        if (s.dataset, s.filename) in manifest_idx.index
    ]
    print(f"  {len(pos_segs)} positive segments (30s extended)")

    hard_segs = build_hard_negative_segments(
        fp_records, train_manifest, train_anns)
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

    # GT events for IoU-based metrics.
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    gt_events = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # ------------------------------------------------------------------
    # Model + checkpoint (auto-detect baseline vs BPN)
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  model type: {model_type}")
    if model_type == "bpn":
        print(f"  bpn_cfg: {ckpt.get('bpn_cfg')}")

    # Materialize lazy projection layer before load_state_dict.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))

    # strict=False because the BPN dummy forward may attach extra buffers
    # not present in older checkpoints; non-BPN keys must still load.
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False)
    if missing:
        non_bpn_missing = [k for k in missing if "bpn" not in k]
        if non_bpn_missing:
            print(f"  WARNING: missing non-BPN keys: {len(non_bpn_missing)}: "
                  f"{non_bpn_missing[:3]}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}: {unexpected[:3]}")
    print(f"  starting val F1 (per ckpt): {ckpt.get('best_f1', 'unknown')}")

    # ------------------------------------------------------------------
    # Loss + optimizer (same recipe as train.py / train_bpn.py at low LR)
    # ------------------------------------------------------------------
    pos_weight = (compute_class_weights().to(device)
                  if cfg.USE_WEIGHTED_BCE else None)
    if pos_weight is not None:
        print(f"  pos_weight: {pos_weight.tolist()}")

    # Baseline criterion (used iff model_type == 'baseline'). For BPN
    # we route through probs_bce_focal_loss inside forward_and_loss.
    baseline_criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY,
                      betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=4, min_lr=1e-7)

    # ------------------------------------------------------------------
    # Initial validation — sanity baseline before any HNM gradient.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nInitial validation (epoch 0)\n{'=' * 60}")
    val0 = validate_hnm(
        model, model_type, spec_extractor, val_loader, device,
        gt_events, baseline_criterion, pos_weight, tune_thresholds=True)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    best_f1 = val0["mean_f1"]
    no_improve = 0
    print(f"\n{'=' * 60}")
    print(f"HNM fine-tune {args.epochs} epochs @ lr={args.lr}")
    print(f"  starting F1: {best_f1:.3f}")
    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch > 1 and epoch % HNM_RESAMPLE_EVERY == 0:
            train_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(train_ds.segments)}")

        train_loss = train_epoch(
            model, model_type, spec_extractor, train_loader,
            baseline_criterion, pos_weight, optimizer, device)
        val = validate_hnm(
            model, model_type, spec_extractor, val_loader, device,
            gt_events, baseline_criterion, pos_weight, tune_thresholds=True)

        improved = val["mean_f1"] > best_f1
        marker = " *** new best" if improved else ""
        print(f"\nEpoch {epoch:2d}/{args.epochs}  "
              f"({time.time()-t0:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Val F1: {val['mean_f1']:.3f}  Best: {best_f1:.3f}")
        print(f"  Tuned thresholds: "
              f"{['%.2f' % t for t in val['thresholds']]}")

        scheduler.step(val["mean_f1"])

        # Build save payload — round-trip bpn_cfg so the result re-loads
        # correctly via build_model_for_ckpt downstream.
        ckpt_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_f1": max(best_f1, val["mean_f1"]),
            "thresholds": torch.tensor(val["thresholds"]),
            "hnm_meta": hnm_meta,
            "source_checkpoint": str(args.checkpoint),
        }
        if model_type == "bpn":
            ckpt_save["bpn_cfg"] = ckpt.get("bpn_cfg")

        if improved:
            best_f1 = val["mean_f1"]
            torch.save(ckpt_save, run_dir / "best_model.pt")
            no_improve = 0
        else:
            no_improve += 1
        torch.save(ckpt_save, run_dir / "latest_model.pt")

        if no_improve >= HNM_EARLY_STOP:
            print(f"\n  Early stop: no improvement for "
                  f"{no_improve} epochs")
            break

    print(f"\nDone. Best F1: {best_f1:.3f}")
    print(f"  starting:  {val0['mean_f1']:.3f}")
    print(f"  delta:     {best_f1 - val0['mean_f1']:+.3f}")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()