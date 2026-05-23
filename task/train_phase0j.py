"""
Phase 0j: Per-Class Threshold Tuning at Evaluation
==================================================

We have run-level F1 numbers for Phase 0g (plain BCE), Phase 0h
(weighted BCE), and Phase 0i (focal+weighted BCE). All three are
evaluated at a fixed threshold of 0.3 across all classes. But the
DCASE protocol explicitly tunes per-class thresholds on a held-out set
(Section 2.9: "thresholds θc are selected per class c, from the
precision-recall curve on a held out set").

We already saw the magnitude of this effect on the official Geldenhuys
checkpoint:
  - At threshold = 0.5 across all classes: F1 = 0.318
  - With per-class tuned thresholds:        F1 = 0.443
  - That's a +0.125 jump from a free post-hoc procedure.

Phase 0j applies the same procedure to *our* trained checkpoints. The
expected outcome is that all our reported numbers move up — which
gives us a fairer comparison against the paper's tuned-threshold F1.

What this script does
---------------------
1. Loads a checkpoint (default: phase0g_best.pt — our current best).
2. Runs inference on the official val split, caching probabilities.
3. Sweeps per-class thresholds with coordinate descent.
4. Reports per-class metrics at the tuned thresholds and the new
   overall F1.
5. Optionally reports per-validation-site breakdown (matches paper
   Table 3 layout) for cross-comparison.
6. Logs the whole thing to wandb so the threshold-tuning runs sit
   alongside their parent training runs in the project, with the
   tuned-thresholds.pt attached as a wandb artifact.

This is a pure evaluation script — no training. Runs in ~5 minutes.

Usage
-----
::

    # Tune the Phase 0g checkpoint (the current overall winner)
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0j.py \\
        --checkpoint runs/phase0g_*/phase0g_best.pt

    # Tune any other Phase 0 checkpoint
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0j.py \\
        --checkpoint runs/phase0i_*/phase0i_best.pt

    # Per-site Table-3-style breakdown
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0j.py \\
        --checkpoint runs/phase0g_*/phase0g_best.pt \\
        --per_site
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    load_annotations, get_file_manifest, build_val_segments,
    WhaleDataset, collate_fn,
)
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
)
from train_phase0 import PHASE0_LSTM_HIDDEN, PHASE0_LSTM_LAYERS
from train_phase0f import PHASE0F_VAL_SITES


# ======================================================================
# Threshold sweep configuration
# ======================================================================

#: Candidate thresholds for each class. Coarser at the high end (we
#: rarely operate above 0.6) and finer at the low end (rare classes
#: often peak at 0.1-0.3). 25+ candidates per class is plenty for a
#: clean optimum without being slow.
THRESHOLD_GRID = np.concatenate([
    np.arange(0.05, 0.50, 0.025),    # 0.05, 0.075, ..., 0.475
    np.arange(0.50, 0.90, 0.05),     # 0.50, 0.55, ..., 0.85
])

#: Maps from a phase id (extracted from checkpoint path) to the wandb
#: registry. Phases not listed here fall back to "0g" — the current
#: best baseline — when init_phase is called with parent_override.
KNOWN_PARENT_PHASES = {"0g", "0h", "0i"}


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a Phase 0 checkpoint (small-LSTM model).")
    p.add_argument("--num_classes", type=int, default=3,
                   help="Output classes in the checkpoint (default 3).")
    p.add_argument("--iou", type=float, default=0.3,
                   help="IoU threshold for event matching.")
    p.add_argument("--per_site", action="store_true",
                   help="Print per-validation-site breakdown.")
    p.add_argument("--coarse_grid", action="store_true",
                   help="Use a coarser 11-point grid (faster, less precise).")
    return p.parse_args()


def detect_parent_phase(ckpt_path: str) -> str:
    """
    Infer the source phase from a checkpoint path.

    Looks for ``phase0X`` (where X is a single letter or digit) in the
    path components. Returns the phase id if it matches one of the
    known training phases, else falls back to ``"0g"``.

    Examples
    --------
    >>> detect_parent_phase("runs/phase0g_20260430_213000/phase0g_best.pt")
    '0g'
    >>> detect_parent_phase("runs/phase0i_20260501_010000/phase0i_best.pt")
    '0i'
    >>> detect_parent_phase("/some/path/checkpoint.pt")
    '0g'
    """
    m = re.search(r"phase(0[a-z0-9])", str(ckpt_path))
    if m and m.group(1) in KNOWN_PARENT_PHASES:
        return m.group(1)
    print(f"WARNING: could not infer parent phase from {ckpt_path!r}; "
          f"defaulting to '0g'.")
    return "0g"


def build_model_and_load(ckpt_path: str, num_classes: int,
                         device: torch.device):
    """
    Build a Phase 0-sized WhaleVAD and load the given checkpoint.

    The Phase 0 checkpoints were saved with the small LSTM
    (hidden=32, layers=1). We monkey-patch ``cfg`` to reproduce that
    sizing, build the model, materialize the lazy projection layer,
    then restore the original cfg values so the rest of the script
    isn't affected.
    """
    orig_hidden = cfg.LSTM_HIDDEN
    orig_layers = cfg.LSTM_LAYERS
    cfg.LSTM_HIDDEN = PHASE0_LSTM_HIDDEN
    cfg.LSTM_LAYERS = PHASE0_LSTM_LAYERS
    try:
        model = WhaleVAD(num_classes=num_classes).to(device)
        spec = SpectrogramExtractor().to(device)
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            model(spec(dummy))
    finally:
        cfg.LSTM_HIDDEN = orig_hidden
        cfg.LSTM_LAYERS = orig_layers

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded {ckpt_path}")
    saved_f1 = ckpt.get("f1")
    if isinstance(saved_f1, float):
        print(f"  Epoch: {ckpt.get('epoch', '?')}  "
              f"Saved F1: {saved_f1:.3f}")
    else:
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
    return model, spec, ckpt


@torch.no_grad()
def collect_probabilities(model, spec_extractor, loader, device):
    """
    Run inference once, return probabilities keyed by stitching key.

    Probabilities are cached so the threshold sweep below doesn't have
    to re-run inference for every candidate threshold.
    """
    from tqdm import tqdm
    all_probs = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Inference"):
        audio = audio.to(device)
        logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]
    return all_probs


def build_gt_events(val_annotations, file_start_dts):
    """Construct GT Detection objects for the 3 coarse classes."""
    gt = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt


def sweep_thresholds_per_class(all_probs, gt_events, iou: float,
                               grid: np.ndarray):
    """
    Coordinate-descent threshold sweep, one class at a time.

    For each class c, fixes the other two thresholds at their current
    best values and sweeps c's threshold over ``grid``, picking the
    value that maximizes c's F1. One pass through all three classes
    is sufficient because the per-class F1 metrics are nearly
    independent (different GT events, different prediction events).
    """
    best_thresholds = np.full(3, 0.3, dtype=np.float64)
    history = []

    for c, name in enumerate(cfg.CALL_TYPES_3):
        f1s = []
        for t in grid:
            trial = best_thresholds.copy()
            trial[c] = t
            preds = postprocess_predictions(all_probs, trial)
            metrics = compute_metrics(preds, gt_events, iou_threshold=iou)
            f1 = metrics.get(name, {}).get("f1", 0.0)
            f1s.append(f1)
        best_idx = int(np.argmax(f1s))
        best_thresholds[c] = float(grid[best_idx])
        history.append({
            "class": name,
            "candidates": grid.tolist(),
            "f1s": f1s,
            "best_threshold": best_thresholds[c],
            "best_f1": f1s[best_idx],
        })
        print(f"  {name:6}: best threshold = {best_thresholds[c]:.3f}  "
              f"(F1 = {f1s[best_idx]:.3f}, "
              f"min in grid = {min(f1s):.3f}, max = {max(f1s):.3f})")

    return best_thresholds, history


def report_per_site(all_probs, gt_events, thresholds, iou: float):
    """
    Compute per-site metrics (and print Table-3-style breakdown).

    Returns a dict ``{site_name: {class_name: {tp,fp,fn,p,r,f1}, ...}}``
    so the same numbers can be sent to wandb.
    """
    print(f"\n=== PER-SITE BREAKDOWN ===")
    print(f"\nReference (paper Table 3):")
    print(f"  casey2017     bmabz=0.624  d=0.054  bp=0.025")
    print(f"  kerguelen2014 bmabz=0.672  d=0.141  bp=0.480")
    print(f"  kerguelen2015 bmabz=0.565  d=0.165  bp=0.581")
    print(f"  paper overall: 0.440\n")

    all_preds = postprocess_predictions(all_probs, thresholds)
    per_site = {}
    for site in PHASE0F_VAL_SITES:
        ds_preds = [d for d in all_preds if d.dataset == site]
        ds_gts = [d for d in gt_events if d.dataset == site]
        m = compute_metrics(ds_preds, ds_gts, iou_threshold=iou)
        per_site[site] = m
        print(f"  {site}:")
        for name in cfg.CALL_TYPES_3:
            r = m.get(name, {})
            if not r:
                continue
            print(f"    {name:6} TP={r['tp']:5} FP={r['fp']:6} "
                  f"FN={r['fn']:5}  P={r['precision']:.3f} "
                  f"R={r['recall']:.3f} F1={r['f1']:.3f}")
        print(f"    overall F1 = {m['overall']['f1']:.3f}\n")
    return per_site


def main():
    """Entry point."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    grid = THRESHOLD_GRID if not args.coarse_grid else np.linspace(0.05, 0.85, 11)
    print(f"Threshold grid: {len(grid)} candidates "
          f"from {grid.min():.2f} to {grid.max():.2f}")

    # ------------------------------------------------------------------
    # Wandb setup. Parent phase is detected from the checkpoint path so
    # the cumulative-intervention chain reflects which model is being
    # tuned (e.g., tuning 0i's checkpoint stamps focal_loss into the
    # tag list as expected, even though 0j itself doesn't introduce it).
    # ------------------------------------------------------------------
    parent_phase = detect_parent_phase(args.checkpoint)
    print(f"Detected parent phase from checkpoint: {parent_phase}")

    run = wbu.init_phase(
        "0j",
        parent_override=parent_phase,
        job_type="eval",
        config={
            "checkpoint":  str(args.checkpoint),
            "num_classes": args.num_classes,
            "iou":         args.iou,
            "grid_size":   len(grid),
            "grid_min":    float(grid.min()),
            "grid_max":    float(grid.max()),
            "val_sites":   PHASE0F_VAL_SITES,
            "per_site":    bool(args.per_site),
        },
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, spec_extractor, ckpt = build_model_and_load(
        args.checkpoint, args.num_classes, device,
    )
    run.config.update({
        "source_epoch":     ckpt.get("epoch"),
        "source_train_f1":  ckpt.get("f1") if isinstance(ckpt.get("f1"), float)
                            else None,
    }, allow_val_change=True)

    # ------------------------------------------------------------------
    # Validation data (official BioDCASE split)
    # ------------------------------------------------------------------
    print(f"\nLoading validation data: {PHASE0F_VAL_SITES}")
    val_manifest = get_file_manifest(PHASE0F_VAL_SITES)
    val_annotations = load_annotations(PHASE0F_VAL_SITES,
                                       manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=32, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  {len(val_manifest)} files, "
          f"{len(val_annotations)} annotations, "
          f"{len(val_segments)} segments")

    # ------------------------------------------------------------------
    # Inference (cached for the threshold sweep)
    # ------------------------------------------------------------------
    print(f"\nRunning inference (cached for threshold sweep)...")
    t0 = time.time()
    all_probs = collect_probabilities(model, spec_extractor, val_loader, device)
    gt_events = build_gt_events(val_annotations, file_start_dts)
    print(f"  {len(all_probs)} prediction windows, "
          f"{len(gt_events)} GT events  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------
    # Baseline: F1 at the default threshold of 0.3
    # ------------------------------------------------------------------
    print(f"\n=== BASELINE (threshold 0.3 across all classes) ===")
    baseline_thresholds = np.array([0.3, 0.3, 0.3])
    baseline_preds = postprocess_predictions(all_probs, baseline_thresholds)
    baseline_metrics = compute_metrics(
        baseline_preds, gt_events, iou_threshold=args.iou,
    )
    for name in cfg.CALL_TYPES_3:
        m = baseline_metrics.get(name, {})
        print(f"  {name:6} F1={m.get('f1', 0.0):.3f}  "
              f"(P={m.get('precision', 0.0):.3f} "
              f"R={m.get('recall', 0.0):.3f})")
    print(f"  OVERALL F1 = {baseline_metrics.get('overall', {}).get('f1', 0.0):.3f}")

    # ------------------------------------------------------------------
    # Per-class threshold sweep
    # ------------------------------------------------------------------
    print(f"\n=== TUNING PER-CLASS THRESHOLDS ===")
    t0 = time.time()
    tuned_thresholds, sweep_history = sweep_thresholds_per_class(
        all_probs, gt_events, args.iou, grid,
    )
    print(f"  Sweep completed in {time.time() - t0:.0f}s")

    # Log the per-class sweep curves to wandb so the prof can see exactly
    # how F1 responds to threshold for each class. One line plot per class.
    for entry in sweep_history:
        cname = entry["class"]
        for cand, f1 in zip(entry["candidates"], entry["f1s"]):
            import wandb
            wandb.log({
                f"sweep/{cname}/threshold": cand,
                f"sweep/{cname}/f1":        f1,
            })

    # ------------------------------------------------------------------
    # Final metrics with tuned thresholds
    # ------------------------------------------------------------------
    print(f"\n=== TUNED ===")
    print(f"  Thresholds: bmabz={tuned_thresholds[0]:.3f}  "
          f"d={tuned_thresholds[1]:.3f}  bp={tuned_thresholds[2]:.3f}")
    tuned_preds = postprocess_predictions(all_probs, tuned_thresholds)
    tuned_metrics = compute_metrics(
        tuned_preds, gt_events, iou_threshold=args.iou,
    )
    for name in cfg.CALL_TYPES_3:
        m = tuned_metrics.get(name, {})
        b = baseline_metrics.get(name, {})
        delta = m.get("f1", 0.0) - b.get("f1", 0.0)
        print(f"  {name:6} F1={m.get('f1', 0.0):.3f}  "
              f"(was {b.get('f1', 0.0):.3f}, Δ={delta:+.3f})  "
              f"P={m.get('precision', 0.0):.3f} R={m.get('recall', 0.0):.3f}")
    overall_tuned = tuned_metrics.get("overall", {}).get("f1", 0.0)
    overall_baseline = baseline_metrics.get("overall", {}).get("f1", 0.0)
    print(f"  OVERALL F1 = {overall_tuned:.3f}  "
          f"(was {overall_baseline:.3f}, Δ={overall_tuned - overall_baseline:+.3f})")

    print(f"\nReference points:")
    print(f"  Paper (full pipeline, tuned thresholds): F1 = 0.440")
    print(f"  Official checkpoint, untuned (0.5):      F1 = 0.318")
    print(f"  Official checkpoint, tuned:              F1 = 0.443")

    # Optional per-site breakdown (also gets logged to wandb summary).
    per_site_metrics = None
    if args.per_site:
        per_site_metrics = report_per_site(
            all_probs, gt_events, tuned_thresholds, args.iou,
        )

    # ------------------------------------------------------------------
    # Save tuned thresholds and ship to wandb as an artifact
    # ------------------------------------------------------------------
    out_path = Path(args.checkpoint).parent / "tuned_thresholds.pt"
    torch.save({
        "thresholds": tuned_thresholds.tolist(),
        "baseline_f1": overall_baseline,
        "tuned_f1": overall_tuned,
        "sweep_history": sweep_history,
        "checkpoint": str(args.checkpoint),
    }, out_path)
    print(f"\nSaved tuned thresholds to {out_path}")

    # ------------------------------------------------------------------
    # Build the wandb summary dict — the prof's headline view.
    # ------------------------------------------------------------------
    summary = {
        "baseline_f1":  overall_baseline,
        "tuned_f1":     overall_tuned,
        "delta_f1":     overall_tuned - overall_baseline,
        "thresholds": {
            cname: float(tuned_thresholds[i])
            for i, cname in enumerate(cfg.CALL_TYPES_3)
        },
        "baseline_per_class": {
            cname: {k: float(v) for k, v in
                    baseline_metrics.get(cname, {}).items()
                    if isinstance(v, (int, float))}
            for cname in cfg.CALL_TYPES_3
        },
        "tuned_per_class": {
            cname: {k: float(v) for k, v in
                    tuned_metrics.get(cname, {}).items()
                    if isinstance(v, (int, float))}
            for cname in cfg.CALL_TYPES_3
        },
    }
    if per_site_metrics is not None:
        summary["per_site"] = {
            site: {
                "overall_f1": float(m["overall"]["f1"]),
                **{cname: float(m.get(cname, {}).get("f1", 0.0))
                   for cname in cfg.CALL_TYPES_3},
            }
            for site, m in per_site_metrics.items()
        }

    delta = overall_tuned - overall_baseline
    if delta >= 0.05:
        verdict = (f"Threshold tuning lifted overall F1 by "
                   f"{delta:+.3f} ({overall_baseline:.3f} → {overall_tuned:.3f}).")
    elif delta >= 0.01:
        verdict = (f"Modest gain from threshold tuning: "
                   f"{overall_baseline:.3f} → {overall_tuned:.3f}.")
    else:
        verdict = (f"Threshold tuning ineffective: "
                   f"{overall_baseline:.3f} → {overall_tuned:.3f}.")

    wbu.finalize_eval_phase(
        summary,
        verdict=verdict,
        artifact_path=out_path,
        artifact_type="thresholds",
        artifact_metadata={
            "source_checkpoint": str(args.checkpoint),
            "source_phase":      parent_phase,
            "baseline_f1":       overall_baseline,
            "tuned_f1":          overall_tuned,
            "iou":               args.iou,
        },
    )


if __name__ == "__main__":
    main()
