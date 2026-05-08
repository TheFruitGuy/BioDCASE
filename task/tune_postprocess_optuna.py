"""
Per-Class Post-Processing Tuner (Optuna)
========================================

Optuna search over the per-class post-processing hyperparameter space
defined in Geldenhuys et al. (arXiv:2510.21280v2) Table IV. Computes the
validation probabilities once per checkpoint (auto-detecting baseline
``WhaleVAD`` vs ``WhaleVADBPN``), combines them, then runs N trials over
the cached probabilities — each trial is just CPU post-processing, no
GPU.

The objective is **macro F1** (mean of per-class event-level F1 at IoU
0.3), matching the metric you've been comparing against the 0.516 macro.
The script prints the per-class breakdown for the best trial and writes
the result to a JSON file that ``inference.py`` will auto-load on the
next run.

Three combination modes
-----------------------
1. **Single checkpoint**: ``--checkpoint X.pt`` — tune for one model.

2. **Plain ensemble**: ``--checkpoints A B C [...] [--weights w1 w2 w3 ...]``
   — weighted average across all classes.

3. **Hybrid ensemble** (matches ``hybrid_ensemble_predict.py``):
   ``--checkpoints A B C D --weights 1 1 1 2 --d-from 3``
   — BMABZ and BP averaged across all models with the given weights,
   D-class taken ONLY from the model at index ``--d-from``. This is the
   combination the user has been validating with at macro 0.516; tuning
   on top of it adds class-dependent post-processing.

Examples
--------
Hybrid ensemble (replicates the validated 4-model setup)::

    CUDA_VISIBLE_DEVICES=1 python tune_postprocess_optuna.py \\
        --checkpoints \\
            runs/whalevad_20260504_152450/best_model.pt \\
            runs/phase5_20260506_204358/best_model.pt \\
            runs/whalevad_20260507_191223/best_model.pt \\
            runs/phase5_20260507_211504/best_model.pt \\
        --weights 1 1 1 2 \\
        --d-from 3 \\
        --n-trials 500 \\
        --per-model-eval

Single checkpoint::

    python tune_postprocess_optuna.py \\
        --checkpoint runs/whalevad_XXXX/best_model.pt --n-trials 300

The output path defaults to ``cfg.POSTPROCESS_CONFIG_PATH`` declared in
``config.py``, so passing ``--output`` is only needed if you want to
keep multiple tuned configs around.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from dataset import build_dataloaders, load_annotations, get_file_manifest
from postprocess import (
    Detection, compute_metrics, postprocess_predictions,
    collapse_probs_to_3class,
)
from postprocess_per_class import (
    PerClassPostprocessConfig,
    postprocess_predictions_per_class,
    default_config_from_global_cfg,
    SEARCH_MEDIAN_KERNEL_MS,
    SEARCH_HANGOVER_KERNEL_MS,
    SEARCH_MERGE_GAP_S,
    SEARCH_MIN_DUR_S,
    SEARCH_MAX_DUR_S,
    SEARCH_THRESHOLD_RANGE,
)

# Reuse the BPN-aware helpers and the per-class hybrid combiner from the
# hybrid ensemble script, so this tuner sees exactly the same probability
# stream that the inference path produces.
from hybrid_ensemble_predict import (
    build_model_for_ckpt,
    predict_probabilities,
    hybrid_combine,
)

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna is required for this script. Install with:")
    print("    pip install optuna")
    sys.exit(1)


CLASS_NAMES = list(cfg.CALL_TYPES_3)


# ======================================================================
# Probability collection (BPN-aware, mirrors hybrid_ensemble_predict.py)
# ======================================================================

@torch.no_grad()
def collect_probs_for_ckpt(
    ckpt_path: str,
    spec_extractor: SpectrogramExtractor,
    val_loader,
    device: torch.device,
) -> dict[tuple, np.ndarray]:
    """
    Load any checkpoint type (WhaleVAD baseline or WhaleVADBPN) and run
    inference on the full val loader.

    Same logic as ``hybrid_ensemble_predict.main``: detect type, init
    lazy ``feat_proj`` via dummy forward, ``load_state_dict(strict=False)``
    so BPN/non-BPN keys can be tolerated, and return a dict keyed by
    ``(dataset, filename, start_sample)``.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"  type: {model_type}")

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))

    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False,
    )
    if unexpected:
        print(f"  WARNING ({Path(ckpt_path).name}): unexpected keys: "
              f"{len(unexpected)}")
    if missing:
        non_bpn_missing = [k for k in missing if "bpn" not in k]
        if non_bpn_missing:
            print(f"  WARNING ({Path(ckpt_path).name}): missing non-BPN "
                  f"keys: {len(non_bpn_missing)}")

    probs = predict_probabilities(
        model, model_type, spec_extractor, val_loader, device,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def average_prob_dicts(
    prob_dicts: list[dict[tuple, np.ndarray]],
    weights: list[float] | None = None,
) -> dict[tuple, np.ndarray]:
    """
    Plain weighted mean of per-segment probability arrays — used in the
    non-hybrid ensemble case. Tolerates minor frame-count mismatches by
    truncating to the shortest, same as ``ensemble_predict.average_prob_dicts``.
    """
    if not prob_dicts:
        return {}
    if weights is None:
        weights = [1.0 / len(prob_dicts)] * len(prob_dicts)
    else:
        total = float(sum(weights))
        weights = [w / total for w in weights]

    common_keys = set(prob_dicts[0].keys())
    for d in prob_dicts[1:]:
        common_keys &= set(d.keys())

    out: dict[tuple, np.ndarray] = {}
    for key in common_keys:
        arrs = [d[key] for d in prob_dicts]
        min_T = min(a.shape[0] for a in arrs)
        min_C = min(a.shape[1] for a in arrs)
        avg = np.zeros((min_T, min_C), dtype=np.float32)
        for w, a in zip(weights, arrs):
            avg += w * a[:min_T, :min_C].astype(np.float32)
        out[key] = avg
    return out


def build_gt_events(val_annotations, file_start_dts) -> list[Detection]:
    """Build the 3-class ground-truth event list for evaluation."""
    gt: list[Detection] = []
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


# ======================================================================
# Macro F1 utility
# ======================================================================

def macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    """
    Mean per-class F1 across the 3 evaluation classes. Missing classes
    contribute 0 (matches what compute_metrics returns when neither
    predictions nor GT exist for a class).
    """
    f1s = [metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES]
    return float(np.mean(f1s))


# ======================================================================
# Per-model individual scoring (parity with --per-model-eval in hybrid)
# ======================================================================

def quick_per_class_threshold_tune(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
) -> tuple[np.ndarray, dict]:
    """
    Coordinate-descent threshold sweep for individual-model F1 reference.
    Same grid as ``hybrid_ensemble_predict.tune_thresholds_on_probs``.
    """
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    grids = [
        np.arange(0.20, 0.85, 0.05),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
    ]
    for c, name in enumerate(CLASS_NAMES):
        best_f1, best_t = -1.0, thresholds[c]
        for t in grids[c]:
            trial = thresholds.copy()
            trial[c] = float(t)
            preds = postprocess_predictions(all_probs, trial)
            metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
            f1 = metrics.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
    final_metrics = compute_metrics(
        postprocess_predictions(all_probs, thresholds),
        gt_events, iou_threshold=0.3,
    )
    return thresholds, final_metrics


# ======================================================================
# Optuna objective
# ======================================================================

def _suggest_range(trial: "optuna.Trial", name: str,
                   lo: float, hi: float, step: float) -> float:
    """Float suggestion with a fixed step; rounded for clean JSON keys."""
    val = trial.suggest_float(name, lo, hi, step=step)
    decimals = max(0, -int(np.floor(np.log10(step))))
    return round(val, decimals)


def make_objective(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
):
    """
    Build a closure over the cached probabilities and ground truth so
    Optuna's `study.optimize` can call it with just the trial argument.
    """

    def objective(trial: "optuna.Trial") -> float:
        cfg_obj = PerClassPostprocessConfig(
            median_kernel_ms={}, on_thresholds={}, off_thresholds={},
            hangover_kernel_ms={}, merge_gap_s={},
            min_dur_s={}, max_dur_s={},
        )

        for cls in CLASS_NAMES:
            # ---- Frame-level (Table IVa) ----
            cfg_obj.median_kernel_ms[cls] = trial.suggest_categorical(
                f"{cls}_median_ms", SEARCH_MEDIAN_KERNEL_MS,
            )

            on_lo, on_hi, on_step = SEARCH_THRESHOLD_RANGE
            on_thr = _suggest_range(trial, f"{cls}_on_thr", on_lo, on_hi, on_step)
            cfg_obj.on_thresholds[cls] = on_thr

            # off_thr ≤ on_thr (paper: hysteresis terminates at lower thr).
            # Sample the gap and subtract — keeps off_thr bounded in
            # [0.0, on_thr] without needing constraint sampling.
            off_gap = trial.suggest_float(
                f"{cls}_off_gap", 0.0, on_thr, step=0.05,
            )
            off_thr = round(on_thr - off_gap, 2)
            cfg_obj.off_thresholds[cls] = off_thr

            cfg_obj.hangover_kernel_ms[cls] = trial.suggest_categorical(
                f"{cls}_hangover_ms", SEARCH_HANGOVER_KERNEL_MS,
            )

            # ---- Event-level (Table IVb) ----
            mg_lo, mg_hi, mg_step = SEARCH_MERGE_GAP_S[cls]
            cfg_obj.merge_gap_s[cls] = _suggest_range(
                trial, f"{cls}_merge_gap_s", mg_lo, mg_hi, mg_step,
            )
            mn_lo, mn_hi, mn_step = SEARCH_MIN_DUR_S[cls]
            cfg_obj.min_dur_s[cls] = _suggest_range(
                trial, f"{cls}_min_dur_s", mn_lo, mn_hi, mn_step,
            )
            mx_lo, mx_hi, mx_step = SEARCH_MAX_DUR_S[cls]
            cfg_obj.max_dur_s[cls] = _suggest_range(
                trial, f"{cls}_max_dur_s", mx_lo, mx_hi, mx_step,
            )

        preds = postprocess_predictions_per_class(all_probs, cfg_obj, CLASS_NAMES)
        metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
        return macro_f1(metrics)

    return objective


# ======================================================================
# Reporting helpers
# ======================================================================

def _print_metrics_table(metrics: dict, title: str) -> None:
    print(f"\n  {title}:")
    for cls in CLASS_NAMES:
        m = metrics.get(cls, {})
        print(f"    {cls:<6} TP={m.get('tp', 0):5d} FP={m.get('fp', 0):5d} "
              f"FN={m.get('fn', 0):5d}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    print(f"    overall (micro) F1 = {metrics.get('overall', {}).get('f1', 0):.3f}")
    print(f"    macro F1          = {macro_f1(metrics):.3f}")


def _build_config_from_best_params(params: dict[str, Any]) -> PerClassPostprocessConfig:
    """Reconstruct a config from Optuna's flat ``params`` dict."""
    cfg_obj = PerClassPostprocessConfig(
        median_kernel_ms={}, on_thresholds={}, off_thresholds={},
        hangover_kernel_ms={}, merge_gap_s={},
        min_dur_s={}, max_dur_s={},
    )
    for cls in CLASS_NAMES:
        cfg_obj.median_kernel_ms[cls] = int(params[f"{cls}_median_ms"])
        on_thr = float(params[f"{cls}_on_thr"])
        off_gap = float(params[f"{cls}_off_gap"])
        cfg_obj.on_thresholds[cls] = round(on_thr, 2)
        cfg_obj.off_thresholds[cls] = round(on_thr - off_gap, 2)
        cfg_obj.hangover_kernel_ms[cls] = int(params[f"{cls}_hangover_ms"])
        cfg_obj.merge_gap_s[cls] = float(params[f"{cls}_merge_gap_s"])
        cfg_obj.min_dur_s[cls] = float(params[f"{cls}_min_dur_s"])
        cfg_obj.max_dur_s[cls] = float(params[f"{cls}_max_dur_s"])
    return cfg_obj


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna search over per-class post-processing "
                    "hyperparameters. Operates on cached val "
                    "probabilities so each trial is cheap.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str,
                     help="Single checkpoint to tune for.")
    src.add_argument("--checkpoints", nargs="+",
                     help="Multiple checkpoints — combined into an "
                          "ensemble before tuning. Use --d-from for "
                          "the per-class hybrid combination.")

    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint ensemble weights "
                        "(only valid with --checkpoints). Must match "
                        "in length; normalised to sum to 1.")
    p.add_argument("--d-from", type=int, default=None,
                   help="0-indexed position in --checkpoints of the "
                        "model to use for D-class predictions. When "
                        "set, BMABZ/BP are weighted-averaged across "
                        "all models but D is taken from this single "
                        "model — matches hybrid_ensemble_predict.py.")
    p.add_argument("--per-model-eval", action="store_true",
                   help="Score each checkpoint individually (with its "
                        "own coordinate-descent thresholds) before the "
                        "ensemble step. Adds ~10 sec per model. "
                        "Mirrors --per-model-eval in "
                        "hybrid_ensemble_predict.py.")

    p.add_argument("--n-trials", type=int, default=300,
                   help="Number of Optuna trials. 200-500 typical.")
    p.add_argument("--timeout", type=int, default=None,
                   help="Optional wall-clock cap (seconds).")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help="Seed for the TPE sampler.")
    p.add_argument("--output", type=str,
                   default=str(cfg.POSTPROCESS_CONFIG_PATH),
                   help=f"Output JSON. Defaults to "
                        f"cfg.POSTPROCESS_CONFIG_PATH "
                        f"({cfg.POSTPROCESS_CONFIG_PATH}).")
    p.add_argument("--study-name", type=str, default=None,
                   help="Optional name for the Optuna study.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress Optuna's per-trial INFO logs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

    # --d-from is meaningful only with --checkpoints
    if args.d_from is not None and args.checkpoints is None:
        raise SystemExit("--d-from requires --checkpoints (multiple).")
    if args.checkpoints is not None and args.d_from is not None:
        if not (0 <= args.d_from < len(args.checkpoints)):
            raise SystemExit(
                f"--d-from {args.d_from} out of range for "
                f"{len(args.checkpoints)} checkpoints "
                f"(0..{len(args.checkpoints)-1})."
            )
    if args.weights is not None and args.checkpoints is not None:
        if len(args.weights) != len(args.checkpoints):
            raise SystemExit(
                f"len(weights)={len(args.weights)} must equal "
                f"len(checkpoints)={len(args.checkpoints)}."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load val data once
    # ------------------------------------------------------------------
    print("\nLoading validation data...")
    _, _, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }
    spec_extractor = SpectrogramExtractor().to(device)
    gt_events = build_gt_events(val_annotations, file_start_dts)
    print(f"  {len(gt_events)} ground-truth events")

    # ------------------------------------------------------------------
    # Compute per-checkpoint val probabilities (BPN-aware)
    # ------------------------------------------------------------------
    individual_metrics: list[dict] = []
    if args.checkpoint is not None:
        print(f"\nCollecting probabilities from {args.checkpoint}")
        probs = collect_probs_for_ckpt(
            args.checkpoint, spec_extractor, val_loader, device,
        )
        all_probs = collapse_probs_to_3class(probs)
        ckpt_descriptor = args.checkpoint
        combination_label = "single checkpoint"
    else:
        prob_dicts: list[dict[tuple, np.ndarray]] = []
        for i, ckpt in enumerate(args.checkpoints):
            is_d = (args.d_from is not None and i == args.d_from)
            marker = "  ← D source" if is_d else ""
            print(f"\n[{i+1}/{len(args.checkpoints)}] Loading {ckpt}{marker}")
            probs = collect_probs_for_ckpt(
                ckpt, spec_extractor, val_loader, device,
            )
            probs = collapse_probs_to_3class(probs)
            prob_dicts.append(probs)

            if args.per_model_eval:
                print(f"  individual threshold tune:")
                ind_thr, ind_metrics = quick_per_class_threshold_tune(
                    probs, gt_events,
                )
                individual_metrics.append({
                    "path": ckpt,
                    "thresholds": ind_thr,
                    "metrics": ind_metrics,
                    "f1": ind_metrics.get("overall", {}).get("f1", 0.0),
                    "macro_f1": macro_f1(ind_metrics),
                })
                print(f"    micro F1={individual_metrics[-1]['f1']:.3f}  "
                      f"macro F1={individual_metrics[-1]['macro_f1']:.3f}  "
                      f"thr={[f'{t:.2f}' for t in ind_thr]}")

        # Combine.
        if args.d_from is not None:
            weights = (args.weights if args.weights is not None
                       else [1.0] * len(prob_dicts))
            print(f"\nHYBRID combination:")
            print(f"  BMABZ + BP : weighted avg (weights {weights})")
            print(f"  D          : from model #{args.d_from + 1} only "
                  f"({Path(args.checkpoints[args.d_from]).parent.name})")
            all_probs = hybrid_combine(
                prob_dicts, d_from_idx=args.d_from, weights=weights,
            )
            combination_label = (
                f"hybrid (weights={weights}, d_from={args.d_from})"
            )
        else:
            print(f"\nPLAIN ensemble (weighted average across all classes):")
            print(f"  weights = {args.weights or 'equal'}")
            all_probs = average_prob_dicts(prob_dicts, weights=args.weights)
            combination_label = (
                f"plain ensemble (weights={args.weights or 'equal'})"
            )
        ckpt_descriptor = list(args.checkpoints)
    print(f"  combined probs for {len(all_probs)} segments")

    # ------------------------------------------------------------------
    # Baseline reference points
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("BASELINES (for reference)")
    print("=" * 64)

    # 1. Default global threshold = 0.5 with the existing pipeline.
    default_thresholds = np.array([0.5, 0.5, 0.5])
    base_preds = postprocess_predictions(all_probs, default_thresholds)
    base_metrics = compute_metrics(base_preds, gt_events, iou_threshold=0.3)
    _print_metrics_table(
        base_metrics, "default postprocessing @ thr=0.5 (baseline)",
    )
    base_macro = macro_f1(base_metrics)

    # 2. Per-class wrapper with default config — should match (1) exactly.
    sanity_cfg = default_config_from_global_cfg()
    sanity_preds = postprocess_predictions_per_class(
        all_probs, sanity_cfg, CLASS_NAMES,
    )
    sanity_metrics = compute_metrics(sanity_preds, gt_events, iou_threshold=0.3)
    sanity_macro = macro_f1(sanity_metrics)
    _print_metrics_table(
        sanity_metrics,
        "per-class wrapper @ default config (sanity — should ≈ above)",
    )
    if abs(sanity_macro - base_macro) > 1e-3:
        print(f"  NOTE: sanity macro {sanity_macro:.4f} differs from "
              f"baseline macro {base_macro:.4f} by "
              f"{abs(sanity_macro - base_macro):.4f}; expected to match.")

    # 3. Per-class coordinate-descent threshold tune — your current 0.516
    #    baseline corresponds to roughly this number on the hybrid probs.
    cd_thresholds, cd_metrics = quick_per_class_threshold_tune(
        all_probs, gt_events,
    )
    _print_metrics_table(
        cd_metrics,
        f"per-class threshold-only tune (thresholds="
        f"{[f'{t:.2f}' for t in cd_thresholds]})",
    )
    cd_macro = macro_f1(cd_metrics)

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"OPTUNA SEARCH ({args.n_trials} trials, "
          f"combination: {combination_label})")
    print("=" * 64)
    sampler = TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
    )
    study.optimize(
        make_objective(all_probs, gt_events),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=not args.quiet,
    )

    # ------------------------------------------------------------------
    # Reconstruct best config and re-evaluate
    # ------------------------------------------------------------------
    best_cfg = _build_config_from_best_params(study.best_params)
    best_preds = postprocess_predictions_per_class(
        all_probs, best_cfg, CLASS_NAMES,
    )
    best_metrics = compute_metrics(best_preds, gt_events, iou_threshold=0.3)
    best_macro = macro_f1(best_metrics)

    print("\n" + "=" * 64)
    print("BEST CONFIG")
    print("=" * 64)
    _print_metrics_table(best_metrics, "tuned per-class postprocessing")

    print(f"\n  macro F1 progression:")
    print(f"    default thr=0.5            : {base_macro:.4f}")
    print(f"    per-class threshold tune   : {cd_macro:.4f}  "
          f"(Δ vs default: {cd_macro - base_macro:+.4f})")
    print(f"    full per-class postprocess : {best_macro:.4f}  "
          f"(Δ vs default: {best_macro - base_macro:+.4f}, "
          f"Δ vs thr-only: {best_macro - cd_macro:+.4f})")

    print("\n  Per-class config:")
    for cls in CLASS_NAMES:
        print(f"    {cls}:")
        print(f"      median_ms   = {best_cfg.median_kernel_ms[cls]:5d}")
        print(f"      on_thr      = {best_cfg.on_thresholds[cls]:.2f}")
        print(f"      off_thr     = {best_cfg.off_thresholds[cls]:.2f}")
        print(f"      hangover_ms = {best_cfg.hangover_kernel_ms[cls]:5d}")
        print(f"      merge_gap_s = {best_cfg.merge_gap_s[cls]:.2f}")
        print(f"      min_dur_s   = {best_cfg.min_dur_s[cls]:.2f}")
        print(f"      max_dur_s   = {best_cfg.max_dur_s[cls]:.2f}")

    if args.per_model_eval and individual_metrics:
        print("\n" + "=" * 64)
        print("INDIVIDUAL MODEL F1 (FOR REFERENCE)")
        print("=" * 64)
        print(f"  {'#':<3}  {'micro':>6}  {'macro':>6}  path")
        for i, im in enumerate(individual_metrics):
            short = Path(im["path"]).parent.name
            marker = " ← D source" if (
                args.d_from is not None and i == args.d_from
            ) else ""
            print(f"  {i+1:<3}  {im['f1']:>6.3f}  {im['macro_f1']:>6.3f}  "
                  f"{short}{marker}")
        print(f"  {'TUN':<3}  "
              f"{best_metrics.get('overall', {}).get('f1', 0):>6.3f}  "
              f"{best_macro:>6.3f}  tuned ensemble")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    metadata = {
        "checkpoints": ckpt_descriptor,
        "weights": (list(args.weights) if args.weights is not None
                    else None),
        "d_from": args.d_from,
        "combination": combination_label,
        "n_trials": int(args.n_trials),
        "macro_f1_default": float(base_macro),
        "macro_f1_threshold_only": float(cd_macro),
        "macro_f1_tuned": float(best_macro),
        "delta_vs_default": float(best_macro - base_macro),
        "delta_vs_threshold_only": float(best_macro - cd_macro),
        "study_name": args.study_name,
        "per_class_f1": {
            cls: float(best_metrics.get(cls, {}).get("f1", 0.0))
            for cls in CLASS_NAMES
        },
        "threshold_only_thresholds": [float(t) for t in cd_thresholds],
    }
    out_path = Path(args.output)
    best_cfg.save(out_path, metadata=metadata)
    print(f"\nSaved: {out_path}")
    print(f"  inference.py will auto-load this on next run "
          f"(unless --postprocess-config overrides).")


if __name__ == "__main__":
    main()
