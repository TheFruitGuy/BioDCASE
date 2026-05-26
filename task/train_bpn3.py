"""
WhaleVAD-BPN: faithful BPN-multi reproduction (Phase 9d)
=======================================================

Trains the full boundary proposal network as described in arXiv:2510.21280v2
(Fig. 2 + Table III), tracked in W&B as phase ``9d`` within the
``phase9_bpn_reproduction`` group. All three depthwise taps feed their own
projection heads; the proposal network expands each head into R ROI vectors via
two conv-transposes (R = 8 per head, derived from the architecture, not
configured); each of the H*R = 24 ROIs is scored over time by a shared BiLSTM
into one gate logit; the H*R sigmoid gates are combined by a learned global
weighted mean into a single-channel mask that gates every class equally.

Training recipe (paper Section V-B5): AdamW, focal loss on gated probabilities,
LR 1e-3 fixed, weight decay 0.01, batch 48, <=32 epochs, ~30 s segments, with
per-epoch negative resampling and a gate initialised to pass-through.

Memory: H*R = 24 ROI sequences per sample run through the BiLSTM, so a large
batch can OOM. Use ``--batch-size`` with ``--accum-steps`` to hold the effective
batch at 48 while shrinking the per-step batch.

Usage
-----
::

    python train_bpn3.py
    python train_bpn3.py --batch-size 24 --accum-steps 2 --wandb
    python train_bpn3.py --init-from runs/bpn0_<timestamp>/bpn0_best.pt
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
    collapse_probs_to_3class, stitch_segments, smooth_probabilities,
    threshold_to_detections, merge_and_filter,
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

def build_model(device: torch.device, init_from: str | None = None,
                use_bpn: bool = True):
    """Faithful BPN-multi model (3 taps, architecture-derived R) + spectrogram.

    With ``use_bpn=False`` the BPN branch is dropped entirely, leaving the
    dilated backbone + classifier. Returns logits in that case; the caller is
    responsible for sigmoid before computing focal loss / probs.
    """
    model = WhaleVADBPN(
        num_classes=7, use_bpn=use_bpn,
        bpn_taps=(0, 1, 2), bpn_use_bilstm=True,
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

def train_one_epoch(model, spec_extractor, loader, optimizer, device,
                    accum_steps=1):
    """One training pass with masked focal loss on gated probabilities.

    ``accum_steps`` accumulates gradients over that many batches before each
    optimizer step, so a reduced ``--batch-size`` can still reach the paper's
    effective batch of 48 (batch_size * accum_steps) - keeping the R-sweep a
    fair comparison even when a large R forces a smaller per-step batch.
    """
    model.train()
    losses, n = 0.0, 0
    n_batches = len(loader)
    optimizer.zero_grad()
    for i, (audio, targets, mask, _) in enumerate(
            tqdm(loader, desc="Train", leave=False)):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        out = model(spec_extractor(audio))
        probs = out if model.returns_probs else torch.sigmoid(out)
        T = min(probs.size(1), targets.size(1))
        probs, targets, mask = probs[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = focal_loss_with_probs(probs, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)

        (loss / accum_steps).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

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
        out = model(spec_extractor(audio))
        probs = (out if model.returns_probs else torch.sigmoid(out)).cpu().numpy()
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


@torch.no_grad()
def tune_thresholds(model, spec_extractor, loader, device,
                    val_annotations, file_start_dts,
                    grid=np.arange(0.05, 0.96, 0.02)):
    """Per-class threshold sweep on the development set.

    Per-class event-level F1 only depends on that class's threshold (event
    matching is per-class), so each class is optimised independently. Stitching
    and median smoothing are cached and reused across the sweep; only the
    thresholding + merge/filter step runs inside the loop.

    Returns
    -------
    dict
        ``{"thresholds": [tau_bmabz, tau_d, tau_bp],
           "per_class": {name: {"f1","precision","recall","threshold"}},
           "macro_f1": float, "micro_f1": float}``
    """
    model.eval()
    all_probs_7 = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Tune-inference", leave=False):
        audio = audio.to(device)
        probs = model(spec_extractor(audio)).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs_7[key] = probs[j, :n_frames, :]
    all_probs_3 = collapse_probs_to_3class(all_probs_7)

    cfg.USE_3CLASS = True
    try:
        file_probs = stitch_segments(all_probs_3)
        file_smoothed = {k: smooth_probabilities(v) for k, v in file_probs.items()}

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

        tuned = np.full(3, 0.5, dtype=np.float64)
        per_class_out = {}
        for c_idx, c_name in enumerate(cfg.CALL_TYPES_3):
            best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "tau": 0.5}
            for tau in grid:
                thr = np.full(3, 0.99, dtype=np.float64)  # silence other classes
                thr[c_idx] = float(tau)
                dets = []
                for (ds, fn), p in file_smoothed.items():
                    dets.extend(threshold_to_detections(p, thr, ds, fn))
                events = merge_and_filter(dets)
                m = compute_metrics(events, gt_events, iou_threshold=0.3)
                f1 = m.get(c_name, {}).get("f1", 0.0)
                if f1 > best["f1"]:
                    best = {
                        "f1": f1,
                        "precision": m[c_name]["precision"],
                        "recall": m[c_name]["recall"],
                        "tau": float(tau),
                    }
            tuned[c_idx] = best["tau"]
            per_class_out[c_name] = best

        # Joint score at the tuned per-class thresholds.
        joint_dets = []
        for (ds, fn), p in file_smoothed.items():
            joint_dets.extend(threshold_to_detections(p, tuned, ds, fn))
        joint_events = merge_and_filter(joint_dets)
        joint = compute_metrics(joint_events, gt_events, iou_threshold=0.3)
    finally:
        cfg.USE_3CLASS = False

    per_class_f1 = [per_class_out[n]["f1"] for n in cfg.CALL_TYPES_3]
    return {
        "thresholds": tuned.tolist(),
        "per_class": per_class_out,
        "macro_f1": float(np.mean(per_class_f1)),
        "micro_f1": float(joint.get("overall", {}).get("f1", 0.0)),
    }


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
        description="BPN ladder rung 3: full multi-ROI proposal network.")
    p.add_argument("--wandb", action="store_true",
                   help="Log to Weights & Biases (off by default).")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--batch-size", type=int, default=cfg.BPN_BATCH_SIZE,
                   help=f"Per-step batch size (default {cfg.BPN_BATCH_SIZE}). "
                        "Lower it if a large --R runs out of GPU memory.")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps; effective batch = "
                        "batch_size * accum_steps. Use to hold the effective "
                        "batch at 48 while shrinking --batch-size for large R.")
    p.add_argument("--init-from", default=None,
                   help="Optional rung-0 checkpoint to warm-start the backbone.")
    p.add_argument("--no-bpn", action="store_true",
                   help="Ablation: keep the dilated backbone but drop the BPN "
                        "gate. Trains the same classifier on raw logits with "
                        "the same focal-loss-on-probs (post-sigmoid).")
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
        run = wbu.init_phase("9d", config={
            "lr": cfg.BPN_LR, "weight_decay": cfg.BPN_WEIGHT_DECAY,
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "effective_batch": args.batch_size * args.accum_steps,
            "threshold": cfg.THRESHOLD,
            "seed": seed, "neg_ratio": cfg.NEG_RATIO,
            "segment_s": cfg.TRAIN_SEGMENT_S, "epochs": args.epochs,
            "loss": "focal", "focal_alpha": cfg.FOCAL_ALPHA,
            "focal_gamma": cfg.FOCAL_GAMMA,
            "bpn_taps": (0, 1, 2), "bpn_use_bilstm": True,
            "bpn_weighted_mean": "global", "bpn_mask": "single_channel",
            "bpn_gate_init_bias": cfg.BPN_GATE_INIT_BIAS,
            "init_from": args.init_from,
            "train_sites": train_sites, "val_sites": val_sites,
        }, group=wbu.WANDB_GROUP_BPN,
            extra_tags=["phase9", "reproduction_of_bpn", "faithful"],
            mode=args.wandb_mode)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = "bpn3_nogate" if args.no_bpn else "bpn3"
    run_dir = Path(cfg.OUTPUT_DIR) / f"{run_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration (BPN faithful reproduction - BPN-multi):")
    print(f"  Taps: (0, 1, 2)   ROI processor: BiLSTM   (R derived from arch)")
    print(f"  Weighted mean over ROIs: global   Mask: single-channel")
    print(f"  Gate init bias: {cfg.BPN_GATE_INIT_BIAS} (mask starts ~pass-through)")
    print(f"  Loss: focal (alpha={cfg.FOCAL_ALPHA}, gamma={cfg.FOCAL_GAMMA})")
    print(f"  LR={cfg.BPN_LR}, wd={cfg.BPN_WEIGHT_DECAY}, "
          f"batch={args.batch_size} x accum {args.accum_steps} "
          f"(effective {args.batch_size * args.accum_steps}), "
          f"epochs={args.epochs}")

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
        WhaleDataset(val_segments), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, {len(val_segments)} segments")

    # --- model + optimizer ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, spec_extractor = build_model(
        device, init_from=args.init_from, use_bpn=not args.no_bpn)
    n_params = sum(p.numel() for p in model.parameters())
    if model.use_bpn:
        n_roi = model.bpn_n_roi
        R = n_roi // len(model.bpn_taps)
        print(f"\nModel parameters: {n_params:,}")
        print(f"  Derived R={R} per head, {n_roi} ROIs total "
              f"({len(model.bpn_taps)} taps)")
    else:
        n_roi, R = 0, 0
        print(f"\nModel parameters: {n_params:,}")
        print("  Ablation: BPN gate disabled (dilated backbone + classifier only)")
    if run is not None:
        run.config.update({"n_params": n_params, "bpn_R": R,
                           "bpn_n_roi": n_roi}, allow_val_change=True)

    optimizer = AdamW(
        model.parameters(), lr=cfg.BPN_LR, weight_decay=cfg.BPN_WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2))

    # --- training loop ---
    history = []
    print(f"\n{'=' * 60}\nTraining {args.epochs} epochs\n{'=' * 60}")
    best_f1 = 0.0
    best_macro = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_segments = resample_negatives_for_epoch(
            pos_segs, train_annotations, train_manifest, n_neg,
            cfg.TRAIN_SEGMENT_S, epoch, sampling_rng, verbose=True)
        train_loader = DataLoader(
            WhaleDataset(train_segments), batch_size=args.batch_size,
            shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True, **seeded_dataloader_kwargs(seed))

        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, optimizer, device,
            accum_steps=args.accum_steps)
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
                "f1": val["f1"], "history": history, "bpn_R": R}
        torch.save(ckpt, run_dir / f"{run_prefix}_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / f"{run_prefix}_best.pt")
        if macro > best_macro:
            best_macro = macro
            torch.save(ckpt, run_dir / f"{run_prefix}_best_macro.pt")

    # --- summary ---
    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    print(f"\n{'=' * 60}\nBPN FAITHFUL SUMMARY (R={R}, {n_roi} ROIs)\n{'=' * 60}")
    print(f"Best micro F1 @ flat 0.3: {max(f1s):.3f} "
          f"(epoch {f1s.index(max(f1s)) + 1})")
    print(f"Best macro F1 @ flat 0.3: {max(macros):.3f}")
    for name in cfg.CALL_TYPES_3:
        print(f"  best {name}: {max(h['per_class'][name]['f1'] for h in history):.3f}")

    # --- final per-class threshold tuning on the best checkpoint ---
    # The paper's BPN gains live in the precision-recall curve, so flat-0.3
    # numbers understate the model. Tune one threshold per class on the dev set
    # and report what the model actually delivers at the tuned operating point.
    print(f"\n{'-' * 60}\nThreshold tuning on best-macro checkpoint\n{'-' * 60}")
    macro_ckpt = run_dir / f"{run_prefix}_best_macro.pt"
    best_ckpt = macro_ckpt if macro_ckpt.exists() else (run_dir / f"{run_prefix}_best.pt")
    print(f"Loading {best_ckpt.name}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
    tuned = tune_thresholds(model, spec_extractor, val_loader, device,
                            val_annotations, file_start_dts)
    print(f"Tuned thresholds: " + ", ".join(
        f"{n}={t:.2f}" for n, t in zip(cfg.CALL_TYPES_3, tuned["thresholds"])))
    for name in cfg.CALL_TYPES_3:
        pc = tuned["per_class"][name]
        print(f"  {name}: F1={pc['f1']:.3f}  P={pc['precision']:.3f}  "
              f"R={pc['recall']:.3f}  (tau={pc['tau']:.2f})")
    print(f"Macro F1 @ tuned: {tuned['macro_f1']:.3f}   "
          f"Micro F1 @ tuned: {tuned['micro_f1']:.3f}")
    import json
    (run_dir / "tuned_thresholds.json").write_text(json.dumps(tuned, indent=2))

    if run is not None:
        import wandb_utils as wbu
        run.summary["tuned/macro_f1"] = tuned["macro_f1"]
        run.summary["tuned/micro_f1"] = tuned["micro_f1"]
        for name in cfg.CALL_TYPES_3:
            pc = tuned["per_class"][name]
            run.summary[f"tuned/{name}/f1"] = pc["f1"]
            run.summary[f"tuned/{name}/precision"] = pc["precision"]
            run.summary[f"tuned/{name}/recall"] = pc["recall"]
            run.summary[f"tuned/{name}/threshold"] = pc["tau"]
        wbu.finalize_phase(
            history,
            verdict=(f"Faithful BPN-multi (R={R}): flat-0.3 macro "
                     f"{max(macros):.3f}; tuned-per-class macro "
                     f"{tuned['macro_f1']:.3f}."),
            best_ckpt=run_dir / f"{run_prefix}_best.pt")


if __name__ == "__main__":
    main()
