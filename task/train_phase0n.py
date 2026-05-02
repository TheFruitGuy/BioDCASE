"""
Phase 0n: Threshold Tuning for the 7-Class Phase 0m Checkpoint
==============================================================

Phase 0m (8 sites, full LSTM, 7-class train → 3-class collapse at eval,
plain BCE) hit F1=0.442 — essentially matching the paper's 0.443. All
its evaluations used a single fixed threshold of 0.3 across all three
coarse classes. The paper's reported number uses per-class tuned
thresholds, so 0m's F1 is *under-reported* relative to the protocol
the paper used.

Phase 0n applies the same threshold-tuning machinery as Phase 0j but
adapted to a 7-class checkpoint:

  1. Build a 7-class WhaleVAD with the paper-config full LSTM.
  2. Load the 0m checkpoint.
  3. Run inference on the official BioDCASE val split, producing
     per-window 7-channel probabilities.
  4. Collapse 7-channel probs to 3-channel via max-over-subclasses
     (using the existing ``collapse_probs_to_3class`` helper).
  5. Sweep per-class thresholds on the collapsed 3-channel output via
     coordinate descent on a 26-point grid.
  6. Report baseline (threshold 0.3) vs tuned F1 with delta.

The single subtle thing here vs 0j is the same bug 0m had: when
``cfg.USE_3CLASS`` is False (which is true while 0m was training and
which the cached checkpoint state assumes), ``cfg.class_names()``
returns the 7 fine-grained names. ``postprocess_predictions`` reads
those names to label its output detections. After collapse the probs
are 3-channel, so postprocess would mislabel them. The fix is the
same: temporarily toggle ``cfg.USE_3CLASS = True`` around the
postprocess call so labels come out as ``[bmabz, d, bp]``.

Caveat about double-dipping
---------------------------
Thresholds are tuned on the same val split that we then report F1 on.
This is the upper bound of what tuning could produce — strictly the
"best-case threshold-tuned val F1," not a held-out test number. For
the report, note this clearly. The paper does the same
(self-references the val split for threshold selection per Section 2.9).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0n.py \\
        --checkpoint runs/phase0m_<timestamp>/phase0m_best.pt

    # Optional per-site breakdown matching paper Table 3:
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase0n.py \\
        --checkpoint runs/phase0m_<timestamp>/phase0m_best.pt \\
        --per_site
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
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
    collapse_probs_to_3class,
)
from train_phase0f import PHASE0F_VAL_SITES
from train_phase0j import (
    THRESHOLD_GRID,
    collect_probabilities,
    build_gt_events,
)


def parse_args():
    """Parse CLI arguments. Mirrors Phase 0j's CLI surface."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a Phase 0m checkpoint (7-class output).")
    p.add_argument("--iou", type=float, default=0.3,
                   help="IoU threshold for event matching.")
    p.add_argument("--per_site", action="store_true",
                   help="Print per-validation-site breakdown.")
    p.add_argument("--coarse_grid", action="store_true",
                   help="Use a coarser 11-point grid (faster, less precise).")
    return p.parse_args()


def build_7class_model_and_load(ckpt_path: str, device: torch.device):
    """
    Build a paper-config 7-class WhaleVAD and load a Phase 0m checkpoint.

    Three things differ from ``train_phase0j.build_model_and_load``:

      1. Uses the FULL LSTM (hidden=128, layers=2) — Phase 0m used the
         paper-config LSTM, not the small Phase 0 variant.
      2. Builds with ``num_classes=7``, not 3, so the classifier head
         shape matches the saved state_dict.
      3. Sets ``cfg.USE_3CLASS = False`` so ``cfg.class_names()`` and
         ``cfg.n_classes()`` agree with the model's output dimension
         during training-mode operations. We toggle it back to True
         *only* around postprocess calls (see below).
    """
    cfg.USE_3CLASS = False

    model = WhaleVAD(num_classes=7).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    saved_f1 = ckpt.get("f1")
    saved_f1_str = f"{saved_f1:.3f}" if isinstance(saved_f1, float) else "?"
    print(f"Loaded {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}  Saved F1: {saved_f1_str}")
    return model, spec, ckpt


def postprocess_with_3class_labels(all_probs_3, thresholds):
    """
    Run ``postprocess_predictions`` with ``cfg.USE_3CLASS`` set to True.

    The probabilities passed in are already 3-channel (collapsed from 7).
    But postprocess reads channel names from ``cfg.class_names()`` which
    returns the 7-class list when USE_3CLASS=False — that would label
    the 3 collapsed channels as ``[bma, bmb, bmz]`` and then merge them
    all to ``bmabz`` via the COLLAPSE_MAP, dropping all d and bp
    predictions.

    We temporarily flip the flag so class_names returns
    ``[bmabz, d, bp]``, then restore it in finally so subsequent code
    (e.g., another inference call on the model) still sees 7-class
    mode.
    """
    cfg.USE_3CLASS = True
    try:
        return postprocess_predictions(all_probs_3, thresholds)
    finally:
        cfg.USE_3CLASS = False


def sweep_thresholds_3class_with_collapse(all_probs_3, gt_events,
                                          iou: float, grid: np.ndarray):
    """
    Coordinate-descent threshold sweep on collapsed 3-channel
    probabilities.

    Identical algorithm to ``train_phase0j.sweep_thresholds_per_class``,
    but uses ``postprocess_with_3class_labels`` instead of calling
    ``postprocess_predictions`` directly so labels come out as the
    coarse class names.

    Returns
    -------
    np.ndarray of shape (3,)
        Per-class tuned thresholds in cfg.CALL_TYPES_3 order.
    list of dict
        Per-class sweep history.
    """
    best_thresholds = np.full(3, 0.3, dtype=np.float64)
    history = []

    for c, name in enumerate(cfg.CALL_TYPES_3):
        f1s = []
        for t in grid:
            trial = best_thresholds.copy()
            trial[c] = t
            preds = postprocess_with_3class_labels(all_probs_3, trial)
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


def report_per_site_3class(all_probs_3, gt_events, thresholds, iou: float):
    """Per-site Table-3-style breakdown of 3-class metrics."""
    print(f"\n=== PER-SITE BREAKDOWN ===")
    print(f"\nReference (paper Table 3):")
    print(f"  casey2017     bmabz=0.624  d=0.054  bp=0.025")
    print(f"  kerguelen2014 bmabz=0.672  d=0.141  bp=0.480")
    print(f"  kerguelen2015 bmabz=0.565  d=0.165  bp=0.581")
    print(f"  paper overall: 0.440\n")

    all_preds = postprocess_with_3class_labels(all_probs_3, thresholds)
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
    """Entry point — mirrors Phase 0j's structure."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    grid = THRESHOLD_GRID if not args.coarse_grid else np.linspace(0.05, 0.85, 11)
    print(f"Threshold grid: {len(grid)} candidates "
          f"from {grid.min():.2f} to {grid.max():.2f}")

    # ------------------------------------------------------------------
    # Wandb setup. 0n is a pure eval phase whose parent is hardcoded to
    # 0m (unlike 0j which auto-detects, because 0n only ever applies to
    # 7-class checkpoints — there's no ambiguity).
    # ------------------------------------------------------------------
    run = wbu.init_phase(
        "0n",
        job_type="eval",
        config={
            "checkpoint":    str(args.checkpoint),
            "iou":           args.iou,
            "grid_size":     len(grid),
            "grid_min":      float(grid.min()),
            "grid_max":      float(grid.max()),
            "val_sites":     PHASE0F_VAL_SITES,
            "per_site":      bool(args.per_site),
            "training_classes": 7,
            "eval_classes":  3,
        },
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, spec_extractor, ckpt = build_7class_model_and_load(
        args.checkpoint, device,
    )
    run.config.update({
        "source_epoch":    ckpt.get("epoch"),
        "source_train_f1": ckpt.get("f1") if isinstance(ckpt.get("f1"), float)
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
    # Inference (cached for the threshold sweep) — produces 7-channel
    # probabilities since the model is 7-class.
    # ------------------------------------------------------------------
    print(f"\nRunning inference (cached for threshold sweep)...")
    t0 = time.time()
    all_probs_7 = collect_probabilities(model, spec_extractor, val_loader, device)
    gt_events = build_gt_events(val_annotations, file_start_dts)
    print(f"  {len(all_probs_7)} prediction windows, "
          f"{len(gt_events)} GT events  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------
    # Collapse 7→3 once. The collapsed array is reused for every
    # candidate threshold during the sweep; no need to redo the
    # max-over-subclasses op repeatedly.
    # ------------------------------------------------------------------
    print(f"\nCollapsing 7-channel probabilities to 3-channel...")
    all_probs_3 = collapse_probs_to_3class(all_probs_7)
    print(f"  Collapse complete; {len(all_probs_3)} windows ready for tuning.")

    # ------------------------------------------------------------------
    # Baseline: F1 at the default threshold of 0.3
    # ------------------------------------------------------------------
    print(f"\n=== BASELINE (threshold 0.3 across all classes) ===")
    baseline_thresholds = np.array([0.3, 0.3, 0.3])
    baseline_preds = postprocess_with_3class_labels(
        all_probs_3, baseline_thresholds,
    )
    baseline_metrics = compute_metrics(
        baseline_preds, gt_events, iou_threshold=args.iou,
    )
    for name in cfg.CALL_TYPES_3:
        m = baseline_metrics.get(name, {})
        print(f"  {name:6} F1={m.get('f1', 0.0):.3f}  "
              f"(P={m.get('precision', 0.0):.3f} "
              f"R={m.get('recall', 0.0):.3f})")
    print(f"  OVERALL F1 = "
          f"{baseline_metrics.get('overall', {}).get('f1', 0.0):.3f}")

    # ------------------------------------------------------------------
    # Per-class threshold sweep
    # ------------------------------------------------------------------
    print(f"\n=== TUNING PER-CLASS THRESHOLDS ===")
    t0 = time.time()
    tuned_thresholds, sweep_history = sweep_thresholds_3class_with_collapse(
        all_probs_3, gt_events, args.iou, grid,
    )
    print(f"  Sweep completed in {time.time() - t0:.0f}s")

    # Log the per-class sweep curves to wandb. The wandb UI then shows
    # one line plot per class: F1 (y) vs threshold candidate (x).
    import wandb
    for entry in sweep_history:
        cname = entry["class"]
        for cand, f1 in zip(entry["candidates"], entry["f1s"]):
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
    tuned_preds = postprocess_with_3class_labels(all_probs_3, tuned_thresholds)
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
          f"(was {overall_baseline:.3f}, "
          f"Δ={overall_tuned - overall_baseline:+.3f})")

    print(f"\nReference points:")
    print(f"  Phase 0m untuned (val): F1 = 0.442 (epoch 19 best)")
    print(f"  Paper (val, tuned):     F1 = 0.443")
    print(f"  Geldenhuys checkpoint:  F1 = 0.318 untuned, 0.443 tuned")

    per_site_metrics = None
    if args.per_site:
        per_site_metrics = report_per_site_3class(
            all_probs_3, gt_events, tuned_thresholds, args.iou,
        )

    # Save tuned thresholds alongside the checkpoint for downstream use.
    out_path = Path(args.checkpoint).parent / "tuned_thresholds_phase0n.pt"
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
    if overall_tuned >= 0.443:
        verdict = (f"Tuned 7-class checkpoint matched/beat the paper: "
                   f"F1 {overall_baseline:.3f} → {overall_tuned:.3f} "
                   f"(paper 0.443).")
    elif delta >= 0.05:
        verdict = (f"Threshold tuning lifted overall F1 by "
                   f"{delta:+.3f} ({overall_baseline:.3f} → {overall_tuned:.3f}).")
    elif delta >= 0.01:
        verdict = (f"Modest gain from threshold tuning: "
                   f"{overall_baseline:.3f} → {overall_tuned:.3f}.")
    else:
        verdict = (f"Threshold tuning ineffective on 0m: "
                   f"{overall_baseline:.3f} → {overall_tuned:.3f}.")

    wbu.finalize_eval_phase(
        summary,
        verdict=verdict,
        artifact_path=out_path,
        artifact_type="thresholds",
        artifact_metadata={
            "source_checkpoint": str(args.checkpoint),
            "source_phase":      "0m",
            "baseline_f1":       overall_baseline,
            "tuned_f1":          overall_tuned,
            "iou":               args.iou,
            "training_classes":  7,
            "eval_classes":      3,
        },
    )


if __name__ == "__main__":
    main()
