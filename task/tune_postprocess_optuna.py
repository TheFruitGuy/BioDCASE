"""
Per-Class Post-Processing Tuner (Optuna)
========================================

Optuna search over the per-class post-processing hyperparameter space
defined in Geldenhuys et al. (arXiv:2510.21280v2) Table IV. Computes the
validation probabilities once per checkpoint (or averaged across an
ensemble), then runs N trials over the cached probabilities — each trial
is just CPU post-processing, no GPU.

The objective is **macro F1** (mean of per-class event-level F1 at IoU
0.3), matching the metric you've been comparing against the 0.516 macro.
The script prints the per-class breakdown for the best trial and writes
the result to a JSON file that ``inference.py`` will auto-load on the
next run.

Usage
-----
Single checkpoint::

    python tune_postprocess_optuna.py \\
        --checkpoint runs/whalevad_XXXX/best_model.pt \\
        --n-trials 300 \\
        --output postprocess_config.json

Ensemble of multi-seed checkpoints::

    python tune_postprocess_optuna.py \\
        --checkpoints runs/whalevad_42/best_model.pt \\
                      runs/whalevad_1337/best_model.pt \\
                      runs/whalevad_9999/best_model.pt \\
        --n-trials 500 \\
        --output postprocess_config_ensemble.json

The output path defaults to ``cfg.POSTPROCESS_CONFIG_PATH`` (declared
in ``config.py``), so passing ``--output`` is only needed if you want
to keep multiple tuned configs around.
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
from model import WhaleVAD
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

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:
    print("ERROR: optuna is required for this script. Install with:")
    print("    pip install optuna")
    sys.exit(1)


# ======================================================================
# Probability caching (same idea as tune_thresholds.py)
# ======================================================================

@torch.no_grad()
def collect_probs_one_ckpt(
    ckpt_path: str,
    spec_extractor: SpectrogramExtractor,
    val_loader,
    device: torch.device,
) -> dict[tuple, np.ndarray]:
    """
    Load a single WhaleVAD checkpoint and run inference on the full
    val loader. Returns a dict keyed by ``(dataset, filename,
    start_sample)`` with shape ``(T_frames, n_classes)`` arrays.

    Lazy ``feat_proj`` is initialised by a dummy forward before
    ``load_state_dict`` so the saved weights bind cleanly.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = cfg.n_classes()
    model = WhaleVAD(num_classes=n_classes).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))

    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False,
    )
    if unexpected:
        print(f"  WARNING ({Path(ckpt_path).name}): unexpected state_dict "
              f"keys: {len(unexpected)} (showing 3): {unexpected[:3]}")
    if missing:
        print(f"  WARNING ({Path(ckpt_path).name}): missing state_dict "
              f"keys: {len(missing)} (showing 3): {missing[:3]}")
    model.eval()

    out_probs: dict[tuple, np.ndarray] = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(
        val_loader, desc=f"  inference ({Path(ckpt_path).name})", leave=False,
    ):
        audio = audio.to(device, non_blocking=True)
        spec = spec_extractor(audio)
        logits = model(spec)
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            out_probs[key] = probs[j, :n_frames, :]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_probs


def average_prob_dicts(
    prob_dicts: list[dict[tuple, np.ndarray]],
    weights: list[float] | None = None,
) -> dict[tuple, np.ndarray]:
    """
    Mean (or weighted mean) of per-segment probability arrays. Tolerates
    minor frame-count mismatches by truncating to the shortest, same as
    ``ensemble_predict.average_prob_dicts``.
    """
    if not prob_dicts:
        return {}
    if weights is None:
        weights = [1.0 / len(prob_dicts)] * len(prob_dicts)
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
# Macro F1 utilities
# ======================================================================

CLASS_NAMES = list(cfg.CALL_TYPES_3)


def macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    """
    Mean per-class F1 across the 3 evaluation classes. Missing classes
    contribute 0 (matches what compute_metrics returns when neither
    predictions nor GT exist for a class).
    """
    f1s = [metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES]
    return float(np.mean(f1s))


# ======================================================================
# Optuna objective
# ======================================================================

def _suggest_range(trial: "optuna.Trial", name: str,
                   lo: float, hi: float, step: float) -> float:
    """Float suggestion with a fixed step; rounded for deterministic config keys."""
    val = trial.suggest_float(name, lo, hi, step=step)
    # Round to the step's decimal precision so the saved JSON has clean numbers.
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
            # We sample the gap and subtract — keeps off_thr bounded
            # in [0.0, on_thr] without needing a constraint sampler.
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

        # Run pipeline and score.
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
    """
    Reconstruct a ``PerClassPostprocessConfig`` from Optuna's flat
    ``params`` dict — used for the post-search "evaluate the best
    config" step.
    """
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
                     help="Multiple checkpoints — averaged into an "
                          "ensemble before tuning.")

    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint ensemble weights "
                        "(only valid with --checkpoints). Must match in "
                        "length; normalised to sum to 1.")
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
                   help="Optional name for the Optuna study (handy for "
                        "RDB-backed runs; otherwise unused).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress Optuna's per-trial INFO logs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

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
    # Compute (and possibly ensemble) val probabilities
    # ------------------------------------------------------------------
    if args.checkpoint is not None:
        print(f"\nCollecting probabilities from {args.checkpoint}")
        all_probs = collect_probs_one_ckpt(
            args.checkpoint, spec_extractor, val_loader, device,
        )
        ckpt_descriptor = args.checkpoint
    else:
        if args.weights is not None:
            assert len(args.weights) == len(args.checkpoints), \
                "len(weights) must equal len(checkpoints)"
            total = sum(args.weights)
            weights = [w / total for w in args.weights]
        else:
            weights = None
        prob_dicts = []
        for i, ckpt in enumerate(args.checkpoints):
            print(f"\n[{i+1}/{len(args.checkpoints)}] Collecting from {ckpt}")
            prob_dicts.append(collect_probs_one_ckpt(
                ckpt, spec_extractor, val_loader, device,
            ))
        all_probs = average_prob_dicts(prob_dicts, weights=weights)
        print(f"\nAveraged {len(prob_dicts)} models, {len(all_probs)} segments")
        ckpt_descriptor = args.checkpoints

    # Collapse to 3-class once up front (no-op for 3-class checkpoints).
    # Doing it here means the cached prob arrays are already in the
    # space the search operates on — saves work on every trial.
    all_probs = collapse_probs_to_3class(all_probs)

    # ------------------------------------------------------------------
    # Baseline reference points
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("BASELINES (for reference)")
    print("=" * 64)

    # 1. Naive default thresholds with the existing (non-per-class) pipeline.
    default_thresholds = np.array([0.5, 0.5, 0.5])
    base_preds = postprocess_predictions(all_probs, default_thresholds)
    base_metrics = compute_metrics(base_preds, gt_events, iou_threshold=0.3)
    _print_metrics_table(base_metrics, "default (single global threshold = 0.5)")
    base_macro = macro_f1(base_metrics)

    # 2. Per-class wrapper with default config — should match (1).
    default_cfg = default_config_from_global_cfg()
    sanity_preds = postprocess_predictions_per_class(
        all_probs, default_cfg, CLASS_NAMES,
    )
    sanity_metrics = compute_metrics(sanity_preds, gt_events, iou_threshold=0.3)
    _print_metrics_table(
        sanity_metrics,
        "per-class wrapper @ default config (sanity check — should ≈ above)",
    )

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"OPTUNA SEARCH ({args.n_trials} trials)")
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
    # Reconstruct best config and re-evaluate (avoids storing the whole
    # config in trial.user_attrs, which would inflate the journal).
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
    delta = best_macro - base_macro
    print(f"\n  macro F1: default {base_macro:.4f}  →  tuned {best_macro:.4f}  "
          f"(Δ = {delta:+.4f})")

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

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    metadata = {
        "checkpoint": ckpt_descriptor,
        "n_trials": int(args.n_trials),
        "macro_f1_default": float(base_macro),
        "macro_f1_tuned": float(best_macro),
        "delta": float(delta),
        "study_name": args.study_name,
        "per_class_f1": {
            cls: float(best_metrics.get(cls, {}).get("f1", 0.0))
            for cls in CLASS_NAMES
        },
    }
    out_path = Path(args.output)
    best_cfg.save(out_path, metadata=metadata)
    print(f"\nSaved: {out_path}")
    print(f"  inference.py will auto-load this on next run "
          f"(unless --postprocess-config overrides).")


if __name__ == "__main__":
    main()
