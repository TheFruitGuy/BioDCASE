"""
HNM Fine-Tuning — _final-native, 3-class & 7-class, with PGI
=============================================================

The Block B centerpiece. Fine-tunes a re-scored base checkpoint
(``paper_best.pt``) on its mined hard-negative false-positives, optionally
with Per-class Gradient Isolation (PGI), and selects the epoch by the paper
metric (F1 of macro-P and macro-R).

Why this instead of train_phase_hnm*.py
----------------------------------------
The existing HNM trainers import the OLD module set (``config`` / ``dataset``
/ ``postprocess`` / ``model`` / ``ensemble_predict``), not ``_final``. Feeding
them ``_final`` checkpoints + ``_final``-mined JSONs risks the same
``USE_3CLASS``-collapse corruption we avoided in the rescore and the miner.
This trainer is ``_final``-pure and reuses the rescore's validated metric
primitives, so a Block B run is scored on EXACTLY the same paper-F1 as the
Block A baseline — the two are directly comparable.

Head is auto-detected from the checkpoint's classifier width (3 -> 3-class,
7 -> 7-class); no flag. One script handles both.

Faithful to train_phase_hnm.py mechanics
-----------------------------------------
  * fine-tune from the base checkpoint at a low LR (1e-5)
  * materialise each FP as a 30s segment, oversample it (default x5)
  * positives + per-epoch-resampled randoms + the fixed oversampled FP pool
  * AdamW + ReduceLROnPlateau(max, 0.5, patience 4); grad-clip 1.0
  * per-epoch threshold-tuned validation; select best by macro-paper F1;
    early stop after 8 stale epochs

Differences, on purpose
------------------------
  * Loss matches how the base was trained (train_final.py): masked weighted
    BCE with the per-class ``pos_weight`` loaded FROM the checkpoint, focal
    OFF by default. PGI is then the ONLY thing that differs between the
    pgi_on / pgi_off arms — a clean ablation. (``--focal`` is available but
    diverges from the base; leave it off for Block B.)
  * Saves the stitched 3-class val posteriors of the best epoch
    (``best_posteriors.npz``, float16) for Block D / ensembling [plan 2.1].
  * 7-class PGI is subclass-level: a JSON whose ``target_class`` is a
    CALL_TYPES_7 name isolates that single channel; a 3-class group name
    isolates the channels that collapse to it [plan 2.3].

Usage
-----
::

    # 3-class, ensemble hard-negs, PGI on (skip GPUs 2 & 7 = Blackwell)
    CUDA_VISIBLE_DEVICES=5 python train_hnm_final.py \
        --checkpoint runs/final_3c_s42_20260527_200054/paper_best.pt \
        --hard-negatives runs/hardnegs_final/3class/ensemble/{bmabz,d,bp}.json \
        --isolate-classes --seed 42 --run-name hnm_3c_s42_ens_pgi

    # 7-class, that seed's solo subclass hard-negs, PGI on (subclass-level)
    CUDA_VISIBLE_DEVICES=6 python train_hnm_final.py \
        --checkpoint runs/final_7c_s42_20260528_204414/paper_best.pt \
        --hard-negatives runs/hardnegs_final/7class/seed42/*.json \
        --isolate-classes --seed 42 --run-name hnm_7c_s42_solo_pgi
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from dataset_final import (
    Segment, WhaleDataset, collate_fn, build_positive_segments,
    build_negative_segments, extend_segment_to_fixed_length,
    get_file_manifest, load_annotations, build_val_segments,
)
from model_final import WhaleVAD  # noqa: F401  (built via rescore.build_model)
from spectrogram_final import SpectrogramExtractor
import postprocess_final as pp
from postprocess_final import Detection  # noqa: F401

# Metric single-source: the rescore's _final primitives. Importing guarantees
# Block B is scored identically to Block A. (rescore_base_epochs.py must sit in
# the same directory — it is the Block-A re-scoring script.)
try:
    from rescore_base_epochs import (
        tune_thresholds, evaluate, paper_f1, collapse_to_3class,
        build_gt_events, build_model,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        f"Could not import metric primitives from rescore_base_epochs.py "
        f"({e}). Put rescore_base_epochs.py in this directory.")

try:
    import wandb_utils as wbu
except Exception:
    wbu = None


# ----------------------------------------------------------------------
# HNM hyperparameters (train_phase_hnm.py defaults; plan section 3)
# ----------------------------------------------------------------------
HNM_LR = 1e-5
HNM_EPOCHS = 15
HNM_OVERSAMPLE = 5
HNM_RESAMPLE_EVERY = 5
HNM_EARLY_STOP = 8


# ======================================================================
# Hard negatives -> segments
# ======================================================================

def load_hard_negatives_json(paths):
    fp_records, metas = [], []
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        tc = payload.get("target_class")
        for r in payload["fps"]:
            r.setdefault("target_class", tc)
        fp_records.extend(payload["fps"])
        metas.append({k: v for k, v in payload.items() if k != "fps"})
    return fp_records, metas


def build_hard_negative_segments(fp_records, manifest, annotations,
                                 segment_s=cfg.TRAIN_SEGMENT_S):
    """Each FP -> a 30s Segment centered on it, with intersecting GT attached.
    Returns (segments, used_records) 1:1 aligned for the PGI class map."""
    sr = cfg.SAMPLE_RATE
    midx = manifest.set_index(["dataset", "filename"])

    ann_by_file: dict = {}
    for _, a in annotations.iterrows():
        key = (a["dataset"], a["filename"])
        if key not in midx.index:
            continue
        fsd = midx.loc[key, "start_dt"]
        if fsd is None:
            continue
        ann_by_file.setdefault(key, []).append({
            "start_s": (a["start_datetime"] - fsd).total_seconds(),
            "end_s":   (a["end_datetime"] - fsd).total_seconds(),
            "label":        a["annotation"],
            "label_3class": a["label_3class"],
        })

    segs, used, skipped = [], [], 0
    for r in fp_records:
        key = (r["dataset"], r["filename"])
        if key not in midx.index:
            skipped += 1
            continue
        row = midx.loc[key]
        dur = float(row["duration_s"])
        if dur < segment_s:
            skipped += 1
            continue
        mid = 0.5 * (r["start_s"] + r["end_s"])
        s0 = max(0.0, mid - segment_s / 2)
        s1 = s0 + segment_s
        if s1 > dur:
            s1, s0 = dur, dur - segment_s
        inter = [a for a in ann_by_file.get(key, [])
                 if a["end_s"] > s0 and a["start_s"] < s1]
        segs.append(Segment(
            dataset=r["dataset"], filename=r["filename"], path=row["path"],
            start_sample=int(s0 * sr), end_sample=int(s1 * sr),
            file_start_dt=row["start_dt"], annotations=inter, is_positive=False))
        used.append(r)
    if skipped:
        print(f"  skipped {skipped} FP records (file too short / missing)")
    return segs, used


# ======================================================================
# PGI: per-segment class map + (B, C) loss mask
# ======================================================================

def _group_to_7c_indices() -> dict[str, set[int]]:
    """3-class group -> the set of 7-class indices that collapse to it."""
    return {c3: {i for i, c7 in enumerate(cfg.CALL_TYPES_7)
                 if cfg.COLLAPSE_MAP[c7] == c3}
            for c3 in cfg.CALL_TYPES_3}


def build_hard_neg_class_map(hard_segs, used_records, n_classes):
    """(dataset, filename, start_sample) -> set of class indices to isolate.

    3-class head: target_class is a CALL_TYPES_3 name -> its single index.
    7-class head: a CALL_TYPES_7 target -> that one channel (subclass PGI);
                  a CALL_TYPES_3 target -> the channels collapsing to it
                  (group PGI). Mixed granularities allowed in one run."""
    if len(hard_segs) != len(used_records):
        raise ValueError(f"{len(hard_segs)} segs vs {len(used_records)} records")
    g2i = _group_to_7c_indices() if n_classes == 7 else None
    out: dict[tuple, set[int]] = {}
    for seg, rec in zip(hard_segs, used_records):
        key = (seg.dataset, seg.filename, seg.start_sample)
        tc = rec["target_class"]
        if n_classes == 3:
            if tc not in cfg.CALL_TYPES_3:
                raise ValueError(f"3-class head got target {tc!r}")
            idxs = {cfg.CALL_TYPES_3.index(tc)}
        else:
            if tc in cfg.CALL_TYPES_7:
                idxs = {cfg.CALL_TYPES_7.index(tc)}
            elif tc in cfg.CALL_TYPES_3:
                idxs = set(g2i[tc])
            else:
                raise ValueError(f"7-class head got target {tc!r}")
        out.setdefault(key, set()).update(idxs)
    return out


def build_class_mask(metas, hard_neg_class_map, n_classes, device):
    """(B, C) loss mask. Non-hard-neg rows -> all ones (full gradient).
    Hard-neg rows -> zeros except the isolated class indices."""
    B = len(metas)
    cm = torch.ones(B, n_classes, device=device)
    if not hard_neg_class_map:
        return cm
    for i, m in enumerate(metas):
        idxs = hard_neg_class_map.get(
            (m["dataset"], m["filename"], m["start_sample"]))
        if idxs is None:
            continue
        cm[i] = 0.0
        for idx in idxs:
            cm[i, idx] = 1.0
    return cm


# ======================================================================
# Training dataset: positives + resampled randoms + oversampled FP pool
# ======================================================================

class HnmTrainingDataset(WhaleDataset):
    def __init__(self, positive_segments, hard_neg_segments, oversample,
                 manifest, annotations):
        self.positive_segments = positive_segments
        self.hard_neg_segments = list(hard_neg_segments) * max(1, oversample)
        self.manifest = manifest
        self.annotations = annotations
        self._midx = manifest.set_index(["dataset", "filename"])
        self.negative_segments = []
        self.resample_negatives()
        super().__init__(self._all())

    def _all(self):
        return (self.positive_segments + self.negative_segments
                + self.hard_neg_segments)

    def resample_negatives(self):
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        neg = build_negative_segments(self.annotations, self.manifest, n_neg)
        ext = []
        for s in neg:
            try:
                dur = float(self._midx.loc[(s.dataset, s.filename), "duration_s"])
            except KeyError:
                continue
            ext.append(extend_segment_to_fixed_length(s, cfg.TRAIN_SEGMENT_S, dur))
        self.negative_segments = ext
        self.segments = self._all()


# ======================================================================
# Loss: masked weighted BCE (+ optional focal) with the PGI class mask
# ======================================================================

def masked_bce(logits, targets, mask, class_mask, pos_weight,
               use_focal=False, focal_alpha=cfg.FOCAL_ALPHA,
               focal_gamma=cfg.FOCAL_GAMMA):
    eps = 1e-7
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pos = targets * torch.log(p)
    neg = (1.0 - targets) * torch.log(1.0 - p)
    if pos_weight is not None:
        pos = pos * pos_weight.view(1, 1, -1)
    if use_focal:
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        w = (1.0 - p_t).pow(focal_gamma) * (
            targets * focal_alpha + (1.0 - targets) * (1.0 - focal_alpha))
        pos, neg = pos * w, neg * w
    per_elem = -(pos + neg)
    valid = mask.unsqueeze(-1).float() * class_mask.unsqueeze(1)
    return (per_elem * valid).sum() / valid.sum().clamp(min=1.0)


def forward_loss(model, spec, audio, targets, mask, class_mask, pos_weight,
                 use_focal):
    logits = model(spec(audio))
    T = min(logits.size(1), targets.size(1))
    return masked_bce(logits[:, :T], targets[:, :T], mask[:, :T],
                      class_mask, pos_weight, use_focal)


# ======================================================================
# Train / validate
# ======================================================================

def train_epoch(model, spec, loader, optimizer, pos_weight, device,
                hard_neg_class_map, n_classes, use_focal):
    model.train()
    tot, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        cm = build_class_mask(metas, hard_neg_class_map, n_classes, device)
        optimizer.zero_grad()
        loss = forward_loss(model, spec, audio, targets, mask, cm, pos_weight,
                            use_focal)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return tot / max(n, 1)


@torch.no_grad()
def validate(model, spec, loader, device, gt_events, pos_weight, n_classes,
             is_3class, val_workers, use_focal):
    """One pass over the val loader for loss + raw probs, then collapse to
    3-class and score with the rescore's tuned paper-F1.

    USE_3CLASS protocol: the loader is iterated with the model's native flag
    (False for 7c) so targets/labels match; the flag is only flipped (to pool
    7->3 and label in 3-class space) AFTER the loader drains; restored before
    returning so the next train epoch's workers fork with the right flag."""
    cfg.USE_3CLASS = is_3class
    model.eval()
    hop = spec.hop_length
    tot, n = 0.0, 0
    probs = {}
    for audio, targets, mask, metas in tqdm(loader, desc="val", leave=False):
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(spec(audio))
        T = min(logits.size(1), targets.size(1))
        cm = torch.ones(audio.size(0), n_classes, device=device)
        tot += masked_bce(logits[:, :T], targets[:, :T], mask[:, :T], cm,
                          pos_weight, use_focal).item()
        n += 1
        p = torch.sigmoid(logits).cpu().numpy()
        for j, m in enumerate(metas):
            nf = min((m["end_sample"] - m["start_sample"]) // hop, p[j].shape[0])
            probs[(m["dataset"], m["filename"], m["start_sample"])] = p[j, :nf, :]
    val_loss = tot / max(n, 1)

    probs3 = collapse_to_3class(probs, n_classes)        # flips flag; ends True
    thresholds = tune_thresholds(probs3, gt_events, val_workers)
    metrics = evaluate(probs3, gt_events, thresholds)
    mp = paper_f1(metrics)
    per_class = {c: metrics.get(c, {}) for c in cfg.CALL_TYPES_3}
    cfg.USE_3CLASS = is_3class                            # restore for next epoch

    return {"loss": val_loss, "paper_f1": mp, "thresholds": thresholds,
            "per_class": per_class, "probs3": probs3}


def save_posteriors(probs3, path: Path):
    stitched = pp.stitch_segments(probs3)
    flat = {f"{ds}|{fn}": arr.astype(np.float16) for (ds, fn), arr in stitched.items()}
    np.savez_compressed(path, **flat)


# ======================================================================
# Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Base paper_best.pt.")
    p.add_argument("--hard-negatives", nargs="+", required=True,
                   help="Mined JSON(s) for this run (one head's group/subclass set).")
    p.add_argument("--isolate-classes", action="store_true", help="PGI on.")
    p.add_argument("--epochs", type=int, default=HNM_EPOCHS)
    p.add_argument("--lr", type=float, default=HNM_LR)
    p.add_argument("--oversample", type=int, default=HNM_OVERSAMPLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=str(cfg.OUTPUT_DIR))
    p.add_argument("--val-workers", type=int, default=13,
                   help="Fork workers for the per-class threshold sweep (~13 grid pts).")
    p.add_argument("--focal", action="store_true",
                   help="Use focal loss (DIVERGES from the base; off for Block B).")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def seed_all(seed):
    if wbu is not None and hasattr(wbu, "seed_everything"):
        wbu.seed_everything(seed, deterministic=False)
    else:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- head from checkpoint ---------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
    is_3class = (n_classes == 3)
    head = "3class" if is_3class else "7class"
    cfg.USE_3CLASS = is_3class
    print(f"Device: {device} | checkpoint head: {head} ({n_classes}-class)")

    # ---- hard negatives ---------------------------------------------
    fp_records, hnm_meta = load_hard_negatives_json(args.hard_negatives)
    targets_used = sorted({r["target_class"] for r in fp_records})
    per_t = {t: sum(1 for r in fp_records if r["target_class"] == t) for t in targets_used}
    print(f"Loaded {len(fp_records)} hard negatives from {len(args.hard_negatives)} "
          f"JSON(s); targets={targets_used}")
    for t, c in per_t.items():
        print(f"  {t}: {c}")
    print(f"  PGI: {args.isolate_classes} | focal: {args.focal}")

    # ---- run dir + wandb --------------------------------------------
    run_name = args.run_name or f"hnm_{head}_{Path(args.checkpoint).parent.name}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    run = None
    if not args.no_wandb and wbu is not None:
        try:
            tags = [f"target_{t}" for t in targets_used] + [
                "hnm_final", head, "pgi_on" if args.isolate_classes else "pgi_off",
                "focal" if args.focal else "weighted_bce",
                "multi_target" if len(targets_used) > 1 else "single_target"]
            run = wbu.init_phase("6", extra_tags=tags, job_type="hnm",
                                 config={"lr": args.lr, "epochs": args.epochs,
                                         "oversample": args.oversample, "seed": args.seed,
                                         "head": head, "n_classes": n_classes,
                                         "isolate_classes": args.isolate_classes,
                                         "focal": args.focal,
                                         "source_checkpoint": str(args.checkpoint),
                                         "hard_negatives": list(args.hard_negatives),
                                         "n_hard_negs": len(fp_records),
                                         "n_hard_negs_per_class": per_t,
                                         "mining_targets": targets_used,
                                         "select_by": "macro_paper"})
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without logging.")
            run = None

    # ---- data (built with the model's native flag) ------------------
    train_anns = load_annotations(list(cfg.TRAIN_DATASETS))
    train_manifest = get_file_manifest(list(cfg.TRAIN_DATASETS))
    midx = train_manifest.set_index(["dataset", "filename"])
    pos_segs = [extend_segment_to_fixed_length(
                    s, cfg.TRAIN_SEGMENT_S,
                    float(midx.loc[(s.dataset, s.filename), "duration_s"]))
                for s in build_positive_segments(train_anns, train_manifest)
                if (s.dataset, s.filename) in midx.index]
    hard_segs, used = build_hard_negative_segments(fp_records, train_manifest, train_anns)
    print(f"  {len(pos_segs)} positives | {len(hard_segs)} hard-negs "
          f"x{args.oversample} = {len(hard_segs)*args.oversample}/epoch")

    hard_neg_class_map = (build_hard_neg_class_map(hard_segs, used, n_classes)
                          if args.isolate_classes else {})
    if args.isolate_classes:
        multi = sum(1 for v in hard_neg_class_map.values() if len(v) > 1)
        print(f"  PGI: {len(hard_neg_class_map)} locations ({multi} multi-class)")

    train_ds = HnmTrainingDataset(pos_segs, hard_segs, args.oversample,
                                  train_manifest, train_anns)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True)

    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_loader = DataLoader(WhaleDataset(build_val_segments(val_manifest, val_anns)),
                            batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                            pin_memory=True)
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)   # always 3-class GT

    # ---- model (built fresh w/ correct flag, then load base weights) -
    model, spec = build_model(n_classes, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    pos_weight = (torch.tensor(ckpt["pos_weight"], dtype=torch.float32, device=device)
                  if ckpt.get("pos_weight") is not None else None)
    if pos_weight is not None:
        print(f"  pos_weight (from ckpt): {pos_weight.tolist()}")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=cfg.WEIGHT_DECAY, betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=4, min_lr=1e-7)

    # ---- epoch 0 baseline -------------------------------------------
    print(f"\n{'='*60}\nInitial validation (epoch 0)\n{'='*60}")
    v0 = validate(model, spec, val_loader, device, gt_events, pos_weight,
                  n_classes, is_3class, args.val_workers, args.focal)
    print(f"  start paper-F1 = {v0['paper_f1']:.4f}  (loss {v0['loss']:.4f})")
    best_f1 = v0["paper_f1"]
    history = [{"epoch": 0, "train_loss": float("nan"), "paper_f1": best_f1,
                "loss": v0["loss"], "per_class": v0["per_class"]}]
    save_posteriors(v0["probs3"], run_dir / "best_posteriors.npz")
    no_improve = 0

    print(f"\nHNM fine-tune {args.epochs} ep @ lr={args.lr}, oversample="
          f"{args.oversample}, PGI={args.isolate_classes}, select=macro_paper")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch > 1 and epoch % HNM_RESAMPLE_EVERY == 0:
            train_ds.resample_negatives()
            print(f"  resampled randoms; train size={len(train_ds.segments)}")

        cfg.USE_3CLASS = is_3class      # workers fork with the model's flag
        train_loss = train_epoch(model, spec, train_loader, optimizer, pos_weight,
                                 device, hard_neg_class_map, n_classes, args.focal)
        v = validate(model, spec, val_loader, device, gt_events, pos_weight,
                     n_classes, is_3class, args.val_workers, args.focal)

        improved = v["paper_f1"] > best_f1
        print(f"\nEpoch {epoch:2d}/{args.epochs} ({time.time()-t0:.0f}s)"
              f"{'  *** new best' if improved else ''}")
        print(f"  train {train_loss:.4f} | val {v['loss']:.4f} | "
              f"paper-F1 {v['paper_f1']:.4f} (best {best_f1:.4f})")
        print(f"  thresholds {['%.2f' % t for t in v['thresholds']]}")
        for c in cfg.CALL_TYPES_3:
            m = v["per_class"].get(c, {})
            print(f"    {c.upper():6} P={m.get('precision',0):.3f} "
                  f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")

        scheduler.step(v["paper_f1"])
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "paper_f1": v["paper_f1"], "loss": v["loss"],
                        "per_class": v["per_class"]})
        if run is not None:
            try:
                run.log({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": v["loss"], "paper_f1": v["paper_f1"]})
            except Exception:
                pass

        save = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "paper_f1": v["paper_f1"], "thresholds": torch.tensor(v["thresholds"]),
                "n_classes": n_classes, "isolate_classes": args.isolate_classes,
                "source_checkpoint": str(args.checkpoint), "hnm_meta": hnm_meta,
                "pos_weight": (pos_weight.cpu().tolist() if pos_weight is not None else None)}
        torch.save(save, run_dir / "latest_model.pt")
        if improved:
            best_f1 = v["paper_f1"]
            torch.save(save, run_dir / "best_model.pt")
            save_posteriors(v["probs3"], run_dir / "best_posteriors.npz")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= HNM_EARLY_STOP:
            print(f"\n  Early stop after {no_improve} stale epochs")
            break

    delta = best_f1 - v0["paper_f1"]
    verdict = (f"HNM {head} PGI={'on' if args.isolate_classes else 'off'} "
               f"targets={targets_used}: paper-F1 {v0['paper_f1']:.4f} -> "
               f"{best_f1:.4f} ({delta:+.4f}).")
    print(f"\n{'='*60}\n{verdict}\nBest: {run_dir/'best_model.pt'}\n{'='*60}")
    # Per-class deltas: did fine-tuning move each 3-class group's F1?
    # is_target flags the groups this run actually mined FPs for (group or
    # subclass), so the runs table can filter "all d-target runs" cleanly.
    subs_of = {c3: {c7 for c7 in cfg.CALL_TYPES_7 if cfg.COLLAPSE_MAP[c7] == c3}
               for c3 in cfg.CALL_TYPES_3}
    per_class_delta = {}
    for c in cfg.CALL_TYPES_3:
        start_c = v0["per_class"].get(c, {}).get("f1", 0.0)
        best_c = max(h["per_class"].get(c, {}).get("f1", 0.0) for h in history)
        is_tgt = (c in targets_used) or bool(subs_of[c] & set(targets_used))
        per_class_delta[c] = {"start": start_c, "best": best_c,
                              "delta": best_c - start_c, "is_target": is_tgt}
        print(f"  {c.upper():6} F1 {start_c:.3f} -> {best_c:.3f} "
              f"({best_c - start_c:+.3f}){'  [target]' if is_tgt else ''}")
    if run is not None:
        try:
            run.summary["start_paper_f1"] = float(v0["paper_f1"])
            run.summary["best_paper_f1"] = float(best_f1)
            run.summary["delta_paper_f1"] = float(delta)
            for c, d in per_class_delta.items():
                run.summary[f"start_f1_{c}"] = float(d["start"])
                run.summary[f"best_f1_{c}"] = float(d["best"])
                run.summary[f"delta_f1_{c}"] = float(d["delta"])
                run.summary[f"is_target_{c}"] = bool(d["is_target"])
            run.summary["mining_targets"] = list(targets_used)
            run.summary["verdict"] = verdict
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
