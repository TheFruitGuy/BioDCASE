"""
Phase 6: Hard-Negative Mining Fine-Tune (multi-class with PGI)
================================================================

Continue training a converged WhaleVAD or WhaleVAD-BPN checkpoint with
explicit hard-negative segments mined from the source model. Supports:

  - Multiple mining targets (D + BMABZ): pass multiple JSON files via
    --hard_negatives.
  - Per-segment class-conditional loss masking (--isolate_classes):
    when enabled, hard-negative segments contribute loss ONLY for their
    target class. Non-target classes receive zero gradient on those
    segments, eliminating cross-class interference where D-suppression
    gradient would otherwise nudge bmabz/bp predictions through the
    shared backbone. (PGI = per-class gradient isolation.)
  - Checkpoint selection by macro or overall F1 (--select_by). Macro
    matches the BPN paper's reporting convention and is the default.

Auto-detects baseline vs BPN architecture from the checkpoint and
routes the loss accordingly.

Usage
-----
::

    # Multi-class HNM with PGI:
    python train_phase_hnm.py \\
        --checkpoint runs/phase5_20260507_211504/best_model.pt \\
        --hard_negatives runs/hardnegs/d_phase5_20260507_211504.json \\
                         runs/hardnegs/bmabz_phase5_20260507_211504.json \\
        --isolate_classes \\
        --epochs 15 --lr 1e-5 --oversample 5

    # Ablation C: multi-class HNM without isolation (omit --isolate_classes)
    # Ablation B: D-only HNM (single JSON, omit --isolate_classes)
"""

from __future__ import annotations
import argparse
import json
import re
import time
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
from model import WhaleVADLoss, compute_class_weights
from postprocess import Detection, collapse_probs_to_3class
from spectrogram import SpectrogramExtractor
from train_phase0e import extend_segment_to_fixed_length, PHASE0E_SEGMENT_S
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
HNM_RESAMPLE_EVERY = 5
HNM_EARLY_STOP = 8


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Source checkpoint to continue training from. "
                        "Architecture is auto-detected.")
    p.add_argument("--hard_negatives", type=str, nargs="+", required=True,
                   help="One or more JSON files from mine_hard_negatives.py.")
    p.add_argument("--isolate_classes", action="store_true",
                   help="Apply per-segment class-conditional loss masking "
                        "(PGI). When set, a hard-neg segment's loss only "
                        "includes its target class; other classes' outputs "
                        "receive zero gradient on that segment.")
    p.add_argument("--epochs", type=int, default=HNM_DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=HNM_DEFAULT_LR,
                   help="Fine-tuning LR. Default 1e-5: low enough not to "
                        "destroy the converged representation.")
    p.add_argument("--oversample", type=int, default=HNM_DEFAULT_OVERSAMPLE,
                   help="Hard-neg segments are repeated this many times per "
                        "epoch so they're not drowned by standard data.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--select_by", type=str, default="macro",
                   choices=["macro", "overall"],
                   help="Checkpoint selection criterion (default: macro).")
    return p.parse_args()


# ======================================================================
# Hard-negative loading + segment construction
# ======================================================================

def load_hard_negatives_json(paths):
    """Read one or more mining outputs. Returns (records, list_of_metas)."""
    if isinstance(paths, str):
        paths = [paths]
    fp_records, metas = [], []
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        target_class = payload.get("target_class")
        for r in payload["fps"]:
            r.setdefault("target_class", target_class)
        fp_records.extend(payload["fps"])
        metas.append({k: v for k, v in payload.items() if k != "fps"})
    return fp_records, metas


def build_hard_negative_segments(fp_records, manifest, annotations,
                                 segment_s=PHASE0E_SEGMENT_S):
    """
    Materialize each FP record into a 30s ``Segment`` centered on it.

    Returns BOTH the segments and the matching subset of fp_records that
    produced segments. The two lists are 1:1 aligned, so they can be
    zipped to build the per-segment class map for PGI.
    """
    sample_rate = cfg.SAMPLE_RATE
    manifest_idx = manifest.set_index(["dataset", "filename"])

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

    segments, used_records, skipped = [], [], 0
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

        mid_s = 0.5 * (r["start_s"] + r["end_s"])
        seg_start_s = max(0.0, mid_s - segment_s / 2)
        seg_end_s = seg_start_s + segment_s
        if seg_end_s > file_dur_s:
            seg_end_s = file_dur_s
            seg_start_s = seg_end_s - segment_s

        file_anns = ann_by_file.get(key, [])
        inter_anns = [a for a in file_anns
                      if a["end_s"] > seg_start_s and a["start_s"] < seg_end_s]

        segments.append(Segment(
            dataset=r["dataset"], filename=r["filename"], path=file_row["path"],
            start_sample=int(seg_start_s * sample_rate),
            end_sample=int(seg_end_s * sample_rate),
            file_start_dt=file_row["start_dt"],
            annotations=inter_anns,
            is_positive=False,
        ))
        used_records.append(r)

    if skipped:
        print(f"  skipped {skipped} FP records (file too short or missing)")
    return segments, used_records


def build_hard_neg_class_map(hard_segs, used_records):
    """Map (dataset, filename, start_sample) → set of target class indices."""
    if len(hard_segs) != len(used_records):
        raise ValueError(
            f"Length mismatch: {len(hard_segs)} segments vs "
            f"{len(used_records)} fp records.")
    out: dict[tuple, set[int]] = {}
    for seg, rec in zip(hard_segs, used_records):
        key = (seg.dataset, seg.filename, seg.start_sample)
        idx = cfg.CALL_TYPES_3.index(rec["target_class"])
        out.setdefault(key, set()).add(idx)
    return out


def build_class_mask(metas, hard_neg_class_map, n_classes, device):
    """Per-sample (B, C) loss-contribution mask. Empty map → all-ones."""
    B = len(metas)
    class_mask = torch.ones(B, n_classes, device=device)
    if not hard_neg_class_map:
        return class_mask
    for i, m in enumerate(metas):
        key = (m["dataset"], m["filename"], m["start_sample"])
        idxs = hard_neg_class_map.get(key)
        if idxs is None:
            continue
        class_mask[i] = 0.0
        for idx in idxs:
            class_mask[i, idx] = 1.0
    return class_mask


# ======================================================================
# Training dataset
# ======================================================================

class HnmTrainingDataset(WhaleDataset):
    """Positives + resampled randoms + oversampled fixed hard-neg pool."""
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
# Loss functions
# ======================================================================

def probs_bce_focal_loss(probs, targets, mask, class_mask,
                         pos_weight=None, use_focal=True,
                         focal_alpha=cfg.FOCAL_ALPHA,
                         focal_gamma=cfg.FOCAL_GAMMA):
    """BCE on probs with focal/weighted variant + per-segment class masking."""
    eps = 1e-7
    p = probs.clamp(eps, 1.0 - eps)
    pos_term = targets * torch.log(p)
    neg_term = (1.0 - targets) * torch.log(1.0 - p)
    if pos_weight is not None:
        pos_term = pos_term * pos_weight.view(1, 1, -1)
    if use_focal:
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        focal_w = (1.0 - p_t).pow(focal_gamma)
        alpha_t = targets * focal_alpha + (1.0 - targets) * (1.0 - focal_alpha)
        weight = focal_w * alpha_t
        pos_term = pos_term * weight
        neg_term = neg_term * weight
    per_elem = -(pos_term + neg_term)
    time_valid = mask.unsqueeze(-1).float()
    class_valid = class_mask.unsqueeze(1)
    valid = time_valid * class_valid
    per_elem = per_elem * valid
    return per_elem.sum() / valid.sum().clamp(min=1.0)


def logits_bce_focal_loss(logits, targets, mask, class_mask,
                          pos_weight=None, use_focal=True,
                          focal_alpha=cfg.FOCAL_ALPHA,
                          focal_gamma=cfg.FOCAL_GAMMA):
    """BCE on logits with focal/weighted variant + per-segment class masking."""
    p = torch.sigmoid(logits)
    return probs_bce_focal_loss(p, targets, mask, class_mask,
                                pos_weight=pos_weight, use_focal=use_focal,
                                focal_alpha=focal_alpha, focal_gamma=focal_gamma)


# ======================================================================
# Forward + loss dispatch (baseline vs BPN)
# ======================================================================

def forward_and_loss(model, model_type, spec_extractor, audio, targets, mask,
                     class_mask, baseline_criterion, pos_weight, use_class_mask):
    """Run the model on a batch and compute the appropriate loss."""
    spec = spec_extractor(audio)
    out = model(spec)
    if model_type == "bpn":
        probs = out["probs"]
        T = min(probs.size(1), targets.size(1))
        return probs_bce_focal_loss(
            probs[:, :T], targets[:, :T], mask[:, :T], class_mask=class_mask,
            pos_weight=pos_weight, use_focal=cfg.USE_FOCAL_LOSS)
    logits = out
    T = min(logits.size(1), targets.size(1))
    if use_class_mask:
        return logits_bce_focal_loss(
            logits[:, :T], targets[:, :T], mask[:, :T], class_mask=class_mask,
            pos_weight=pos_weight, use_focal=cfg.USE_FOCAL_LOSS)
    return baseline_criterion(logits[:, :T], targets[:, :T], mask[:, :T])


def train_epoch(model, model_type, spec_extractor, loader,
                baseline_criterion, pos_weight, optimizer, device,
                hard_neg_class_map, n_classes, use_class_mask):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        class_mask = build_class_mask(metas, hard_neg_class_map, n_classes,
                                      device)
        optimizer.zero_grad()
        loss = forward_and_loss(model, model_type, spec_extractor, audio,
                                targets, mask, class_mask, baseline_criterion,
                                pos_weight, use_class_mask)
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
# Validation
# ======================================================================

@torch.no_grad()
def validate_hnm(model, model_type, spec_extractor, val_loader, device,
                 gt_events, baseline_criterion, pos_weight, n_classes,
                 tune_thresholds=True):
    model.eval()
    total_loss, n = 0.0, 0
    for audio, targets, mask, _ in val_loader:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        class_mask = torch.ones(audio.size(0), n_classes, device=device)
        loss = forward_and_loss(model, model_type, spec_extractor, audio,
                                targets, mask, class_mask, baseline_criterion,
                                pos_weight, use_class_mask=True)
        total_loss += loss.item()
        n += 1
    val_loss = total_loss / max(n, 1)

    all_probs = predict_probabilities(model, model_type, spec_extractor,
                                      val_loader, device)
    all_probs = collapse_probs_to_3class(all_probs)

    if tune_thresholds:
        thresholds = tune_thresholds_on_probs(all_probs, gt_events)
    else:
        thresholds = np.array(cfg.DEFAULT_THRESHOLDS, dtype=np.float64)

    metrics = evaluate_with_thresholds(all_probs, gt_events, thresholds)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
    macro_f1 = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                              for c in cfg.CALL_TYPES_3]))

    print(f"  Val (loss={val_loss:.4f}):")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):6} "
              f"FN={m.get('fn', 0):6}  P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} F1={m.get('f1', 0):.3f}")
    print(f"    OVERALL F1={overall_f1:.3f}  MACRO F1={macro_f1:.3f}")

    per_class_only = {k: v for k, v in metrics.items()
                      if k in cfg.CALL_TYPES_3}

    return {"loss": val_loss, "overall_f1": overall_f1, "macro_f1": macro_f1,
            "per_class": per_class_only, "thresholds": thresholds.tolist()}


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    wbu.seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Hard negatives — load FIRST so wandb config can include their stats.
    # ------------------------------------------------------------------
    fp_records, hnm_meta_list = load_hard_negatives_json(args.hard_negatives)
    targets_used = sorted({m["target_class"] for m in hnm_meta_list})
    fps_per_class = {t: sum(1 for r in fp_records if r["target_class"] == t)
                     for t in targets_used}
    print(f"\nLoaded {len(fp_records)} hard negatives across "
          f"{len(args.hard_negatives)} JSON file(s)")
    for t, n_fps in fps_per_class.items():
        print(f"  {t}: {n_fps} FPs")
    print(f"  isolate_classes (PGI): {args.isolate_classes}")
    print(f"  select_by: {args.select_by}")

    # ------------------------------------------------------------------
    # Source-phase inference. Same convention as train.py / train_bpn.py:
    # phase5_<ts>/best_model.pt → "5"; whalevad_<ts>/best_model.pt →
    # "baseline"; otherwise "unknown". Becomes both a tag (source_<phase>)
    # and a config field for the runs table.
    # ------------------------------------------------------------------
    source_short = Path(args.checkpoint).parent.name
    src_path = str(args.checkpoint)
    m = re.search(r"phase(\w+)_\d{8}_\d{6}", src_path)
    if m:
        source_phase = m.group(1)
    elif "whalevad" in src_path:
        source_phase = "baseline"
    else:
        source_phase = "unknown"
    print(f"  source phase: {source_phase}  (dir: {source_short})")

    # ------------------------------------------------------------------
    # Wandb run init. Phase 6 = HNM. Tags carry mining targets, PGI
    # flag, source phase, oversample, and selection criterion so the
    # ablation table groups easily.
    # ------------------------------------------------------------------
    extra_tags = [f"target_{t}" for t in targets_used]
    extra_tags.append("pgi_on" if args.isolate_classes else "pgi_off")
    extra_tags.append(f"select_{args.select_by}")
    extra_tags.append(f"source_{source_phase}")
    extra_tags.append(f"oversample{args.oversample}")
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")
    extra_tags.append("multi_target" if len(targets_used) > 1
                      else "single_target")

    run = wbu.init_phase(
        "6",
        config={
            "lr": args.lr,
            "epochs": args.epochs,
            "oversample": args.oversample,
            "seed": args.seed,
            "batch_size": cfg.BATCH_SIZE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "neg_ratio": cfg.NEG_RATIO,
            "use_3class": cfg.USE_3CLASS,
            "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
            "use_focal_loss": getattr(cfg, "USE_FOCAL_LOSS", False),
            "focal_alpha": getattr(cfg, "FOCAL_ALPHA", None),
            "focal_gamma": getattr(cfg, "FOCAL_GAMMA", None),
            "train_sites": list(cfg.TRAIN_DATASETS),
            "val_sites": list(cfg.VAL_DATASETS),
            "early_stop_patience": HNM_EARLY_STOP,
            "resample_every": HNM_RESAMPLE_EVERY,
            # Source / mining lineage.
            "source_checkpoint": str(args.checkpoint),
            "source_short": source_short,
            "source_phase": source_phase,
            "hard_negatives_jsons": [str(p) for p in args.hard_negatives],
            "n_hard_negs_total": len(fp_records),
            "n_hard_negs_per_class": fps_per_class,
            "mining_targets": targets_used,
            "n_mining_targets": len(targets_used),
            "mining_meta": hnm_meta_list,
            # Methodological knobs.
            "isolate_classes": args.isolate_classes,
            "select_by": args.select_by,
        },
        name_suffix=source_short,
        extra_tags=extra_tags,
    )

    run_name = args.run_name or f"hnm_{source_short}"
    run_dir = Path(cfg.OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Standard training data
    # ------------------------------------------------------------------
    print("\nLoading training data...")
    train_anns = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    pos_segs = build_positive_segments(train_anns, train_manifest)
    manifest_idx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [
        extend_segment_to_fixed_length(
            s, PHASE0E_SEGMENT_S,
            float(manifest_idx.loc[(s.dataset, s.filename), "duration_s"]))
        for s in pos_segs
        if (s.dataset, s.filename) in manifest_idx.index
    ]
    print(f"  {len(pos_segs)} positive segments (30s extended)")

    hard_segs, used_records = build_hard_negative_segments(
        fp_records, train_manifest, train_anns)
    print(f"  {len(hard_segs)} hard-neg segments × oversample {args.oversample}"
          f" = {len(hard_segs) * args.oversample} effective copies/epoch")

    # Build the per-segment class map only when isolation is requested.
    hard_neg_class_map = (build_hard_neg_class_map(hard_segs, used_records)
                          if args.isolate_classes else {})
    if args.isolate_classes:
        multi = sum(1 for v in hard_neg_class_map.values() if len(v) > 1)
        print(f"  PGI on: {len(hard_neg_class_map)} unique hard-neg locations "
              f"({multi} mined for >1 class)")
        run.config.update({"n_pgi_locations": len(hard_neg_class_map),
                           "n_pgi_multi_class": multi},
                          allow_val_change=True)
    else:
        print(f"  PGI off: standard multi-class loss on all segments")

    train_ds = HnmTrainingDataset(pos_segs, hard_segs, args.oversample,
                                  train_manifest, train_anns)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True)

    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(WhaleDataset(val_segs), batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=True)

    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  model type: {model_type}")
    if model_type == "bpn":
        print(f"  bpn_cfg: {ckpt.get('bpn_cfg')}")

    run.config.update({"model_type": model_type}, allow_val_change=True)
    if model_type == "bpn" and ckpt.get("bpn_cfg") is not None:
        run.config.update({"source_bpn_cfg": ckpt.get("bpn_cfg")},
                          allow_val_change=True)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))

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
    # Loss + optimizer
    # ------------------------------------------------------------------
    pos_weight = (compute_class_weights().to(device)
                  if cfg.USE_WEIGHTED_BCE else None)
    if pos_weight is not None:
        print(f"  pos_weight: {pos_weight.tolist()}")
        run.config.update({"pos_weight": pos_weight.tolist()},
                          allow_val_change=True)

    baseline_criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)
    n_classes = cfg.n_classes()

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY,
                      betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=4, min_lr=1e-7)

    # ------------------------------------------------------------------
    # Initial validation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nInitial validation (epoch 0)\n{'=' * 60}")
    val0 = validate_hnm(model, model_type, spec_extractor, val_loader, device,
                        gt_events, baseline_criterion, pos_weight, n_classes,
                        tune_thresholds=True)

    val0_log = dict(val0)
    val0_log["f1"] = val0[f"{args.select_by}_f1"]
    wbu.log_epoch_3class(0, float("nan"), val0_log)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = [{"epoch": 0, "train_loss": float("nan"),
                "f1": val0[f"{args.select_by}_f1"], "loss": val0["loss"],
                "per_class": val0["per_class"]}]
    best_f1 = val0[f"{args.select_by}_f1"]
    no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"HNM fine-tune {args.epochs} epochs @ lr={args.lr}, "
          f"select_by={args.select_by}")
    print(f"  starting {args.select_by} F1: {best_f1:.3f}")
    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch > 1 and epoch % HNM_RESAMPLE_EVERY == 0:
            train_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(train_ds.segments)}")

        train_loss = train_epoch(
            model, model_type, spec_extractor, train_loader,
            baseline_criterion, pos_weight, optimizer, device,
            hard_neg_class_map, n_classes,
            use_class_mask=args.isolate_classes)
        val = validate_hnm(model, model_type, spec_extractor, val_loader,
                           device, gt_events, baseline_criterion, pos_weight,
                           n_classes, tune_thresholds=True)

        selected_f1 = val[f"{args.select_by}_f1"]
        improved = selected_f1 > best_f1
        marker = " *** new best" if improved else ""
        print(f"\nEpoch {epoch:2d}/{args.epochs}  ({time.time()-t0:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Val {args.select_by} F1: {selected_f1:.3f}  "
              f"Best: {best_f1:.3f}  "
              f"(macro={val['macro_f1']:.3f}, "
              f"overall={val['overall_f1']:.3f})")
        print(f"  Tuned thresholds: "
              f"{['%.2f' % t for t in val['thresholds']]}")

        scheduler.step(selected_f1)

        val_log = dict(val); val_log["f1"] = selected_f1
        wbu.log_epoch_3class(epoch, train_loss, val_log)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "f1": selected_f1, "loss": val["loss"],
                        "per_class": val["per_class"]})

        ckpt_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_f1": max(best_f1, selected_f1),
            "macro_f1": val["macro_f1"],
            "overall_f1": val["overall_f1"],
            "thresholds": torch.tensor(val["thresholds"]),
            "hnm_meta": hnm_meta_list,
            "isolate_classes": args.isolate_classes,
            "select_by": args.select_by,
            "source_checkpoint": str(args.checkpoint),
        }
        if model_type == "bpn":
            ckpt_save["bpn_cfg"] = ckpt.get("bpn_cfg")

        if improved:
            best_f1 = selected_f1
            torch.save(ckpt_save, run_dir / "best_model.pt")
            no_improve = 0
        else:
            no_improve += 1
        torch.save(ckpt_save, run_dir / "latest_model.pt")

        if no_improve >= HNM_EARLY_STOP:
            print(f"\n  Early stop: no improvement for {no_improve} epochs")
            break

    # ------------------------------------------------------------------
    # Per-class summary metrics. The headline question for HNM is "did
    # fine-tuning on FPs of class X improve F1 on class X?", so we
    # stamp delta_f1_<class> for every class. ``is_target_<class>`` is
    # True only for classes actually targeted in this run, so the runs
    # table can group/filter on "all d-target runs" cleanly.
    # ------------------------------------------------------------------
    import wandb
    for t in cfg.CALL_TYPES_3:
        start_t = val0["per_class"].get(t, {}).get("f1", 0.0)
        best_t = max(h["per_class"].get(t, {}).get("f1", 0.0)
                     for h in history)
        wandb.summary[f"start_f1_{t}"] = float(start_t)
        wandb.summary[f"best_f1_{t}"]  = float(best_t)
        wandb.summary[f"delta_f1_{t}"] = float(best_t - start_t)
        wandb.summary[f"is_target_{t}"] = (t in targets_used)

    wandb.summary["n_targets"]      = len(targets_used)
    wandb.summary["mining_targets"] = list(targets_used)

    # Verdict text — multi-target aware.
    selected_delta = best_f1 - val0[f"{args.select_by}_f1"]
    per_target_str = ", ".join(
        f"{t}={wandb.summary[f'delta_f1_{t}']:+.3f}"
        for t in targets_used
    )
    pgi_str = "on" if args.isolate_classes else "off"

    if selected_delta > 0.005:
        verdict_text = (
            f"HNM helped: {args.select_by} F1 "
            f"{val0[f'{args.select_by}_f1']:.3f} → {best_f1:.3f} "
            f"({selected_delta:+.3f}). Targets={targets_used}, "
            f"PGI={pgi_str}, source={source_phase}. "
            f"Per-target deltas: {per_target_str}."
        )
    elif selected_delta > -0.005:
        verdict_text = (
            f"HNM neutral: {args.select_by} F1 "
            f"{val0[f'{args.select_by}_f1']:.3f} → {best_f1:.3f} "
            f"({selected_delta:+.3f}). Targets={targets_used}, "
            f"PGI={pgi_str}. Source phase {source_phase} appears to "
            f"already handle these FPs."
        )
    else:
        verdict_text = (
            f"HNM hurt: {args.select_by} F1 "
            f"{val0[f'{args.select_by}_f1']:.3f} → {best_f1:.3f} "
            f"({selected_delta:+.3f}). LR={args.lr} or "
            f"oversample={args.oversample} may be too aggressive."
        )

    wbu.finalize_phase(history, verdict=verdict_text,
                       best_ckpt=run_dir / "best_model.pt")

    print(f"\nDone. Best {args.select_by} F1: {best_f1:.3f}")
    print(f"  starting:  {val0[f'{args.select_by}_f1']:.3f}")
    print(f"  delta:     {best_f1 - val0[f'{args.select_by}_f1']:+.3f}")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
