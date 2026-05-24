"""
tune_postprocess_optuna_narrow.py
==================================

Narrow-search-space variant of ``tune_postprocess_optuna.py`` that
tunes only the SITE-DEPENDENT postprocess parameters per class:

    on_thr            — model-calibration drift between sites
    off_gap_ratio     — same (ratio-parameterised hysteresis)
    median_kernel_ms  — noise-floor / prob-stream smoothness varies by site
    hangover_kernel_ms — same

Total: 4 params × 3 classes = **12 dimensions** (vs 21 in the full
tuner). Duration parameters (``min_dur_s``, ``max_dur_s``,
``merge_gap_s``) are FIXED from a JSON priors file produced by
``analyze_call_durations.py``, on the principle that call durations
are a property of the species, not the recording site, and should
not be tuned per validation set.

Why this exists
---------------
The full 21-dim tuner converged to a config below its own
threshold-only baseline (0.478 vs 0.490 macro on a 500-trial run)
because TPE was free to pick pathological values for duration filters
— specifically ``min_dur_s = 2.60`` for D-class, which destroyed
recall on the typical 1-2s D-call. Constraining those parameters from
data analysis prevents the failure mode and dramatically shrinks the
search space, so TPE converges in ~50-100 trials instead of needing
500+.

LOSO support
------------
With ``--held-out-site SITE``, the tuner splits validation events into
"tune" (everything but SITE) and "eval" (SITE only). The Optuna
objective is computed on the tune split; the final reported macro is
on the eval split. This is the methodology for cross-site
generalisation testing — drive it three times via
``run_loso_postprocess.py``.

Most of the heavy machinery (cache-aware probability collection,
per-class hybrid combine, parallel per-model eval, JournalFileStorage,
seed-trial logic) is imported from ``tune_postprocess_optuna``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from spectrogram import SpectrogramExtractor
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, compute_metrics, postprocess_predictions,
    collapse_probs_to_3class,
)
from postprocess_per_class import (
    PerClassPostprocessConfig,
    postprocess_predictions_per_class,
    SEARCH_MEDIAN_KERNEL_MS,
    SEARCH_HANGOVER_KERNEL_MS,
    SEARCH_THRESHOLD_RANGE,
)

# Heavy machinery reused verbatim from the full tuner.
from tune_postprocess_optuna import (
    CLASS_NAMES,
    get_or_compute_probs,
    average_prob_dicts,
    average_prob_dicts_per_class,
    build_gt_events,
    macro_f1,
    quick_per_class_threshold_tune,
    parallel_per_model_threshold_tune,
    _print_metrics_table,
    _snap_to_categorical,
    _snap_to_range,
    _WORKER_PROB_DICTS,  # referenced for type only; we set our own
    _WORKER_GT_EVENTS,
)
from hybrid_ensemble_predict import hybrid_combine

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import (
            JournalFileBackend as _JournalFileBackend,
        )
    except ImportError:
        from optuna.storages import JournalFileStorage as _JournalFileBackend
except ImportError:
    print("ERROR: optuna is required. Install with: pip install optuna")
    sys.exit(1)


# ======================================================================
# Priors loading
# ======================================================================

def load_priors(path: str) -> dict:
    """
    Load and validate a duration-priors JSON produced by
    ``analyze_call_durations.py``. Returns the ``priors`` sub-dict
    with keys ``min_dur_s``, ``max_dur_s``, ``merge_gap_s``, each a
    dict mapping class name → float.
    """
    with open(path) as f:
        blob = json.load(f)
    priors = blob.get("priors")
    if priors is None:
        raise ValueError(f"{path} missing top-level 'priors' key.")
    for key in ("min_dur_s", "max_dur_s", "merge_gap_s"):
        if key not in priors:
            raise ValueError(f"{path} priors missing '{key}'.")
        for cls in CLASS_NAMES:
            if cls not in priors[key]:
                raise ValueError(
                    f"{path} priors[{key!r}] missing class {cls!r}."
                )
    return priors


def print_priors(priors: dict) -> None:
    """Tabular display of the fixed parameters."""
    print(f"\nDuration priors (FIXED, not tuned):")
    print(f"  {'class':<6} {'min_dur_s':>10} {'max_dur_s':>10} "
          f"{'merge_gap_s':>12}")
    for cls in CLASS_NAMES:
        print(f"  {cls:<6} "
              f"{priors['min_dur_s'][cls]:>10.2f} "
              f"{priors['max_dur_s'][cls]:>10.2f} "
              f"{priors['merge_gap_s'][cls]:>12.2f}")


# ======================================================================
# Site filtering for LOSO
# ======================================================================

def filter_by_site(
    probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
    sites_to_keep: set[str],
) -> tuple[dict, list[Detection]]:
    """
    Return ``(probs_filtered, events_filtered)`` containing only entries
    whose ``dataset`` (first element of the prob key, ``.dataset``
    attribute on events) is in ``sites_to_keep``.

    Filtering both is necessary: keeping probs from sites not in the
    GT set would inflate FP count (detections with no matching GT),
    distorting F1. The Optuna objective is only meaningful when probs
    and events are aligned to the same set of sites.
    """
    probs_filtered = {
        k: v for k, v in probs.items() if k[0] in sites_to_keep
    }
    events_filtered = [
        e for e in gt_events if e.dataset in sites_to_keep
    ]
    return probs_filtered, events_filtered


# ======================================================================
# Narrow objective: tune 12 params, fix the rest from priors
# ======================================================================

def _suggest_range_narrow(
    trial: "optuna.Trial", name: str,
    lo: float, hi: float, step: float,
) -> float:
    """Local copy — same as full tuner's _suggest_range."""
    return float(trial.suggest_float(name, lo, hi, step=step))


def make_narrow_objective(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
    priors: dict,
):
    """
    Build an Optuna objective that varies only the 4 site-dependent
    params per class. Duration parameters and merge_gap_s come from
    the priors dict and are baked into the config every trial.
    """

    def objective(trial: "optuna.Trial") -> float:
        cfg_obj = PerClassPostprocessConfig(
            median_kernel_ms={}, on_thresholds={}, off_thresholds={},
            hangover_kernel_ms={}, merge_gap_s={},
            min_dur_s={}, max_dur_s={},
        )
        for cls in CLASS_NAMES:
            # ---- Site-dependent (TUNED) ------------------------------
            cfg_obj.median_kernel_ms[cls] = trial.suggest_categorical(
                f"{cls}_median_ms", SEARCH_MEDIAN_KERNEL_MS,
            )
            on_lo, on_hi, on_step = SEARCH_THRESHOLD_RANGE
            on_thr = _suggest_range_narrow(
                trial, f"{cls}_on_thr", on_lo, on_hi, on_step,
            )
            cfg_obj.on_thresholds[cls] = on_thr

            off_gap_ratio = trial.suggest_float(
                f"{cls}_off_gap_ratio", 0.0, 1.0, step=0.05,
            )
            cfg_obj.off_thresholds[cls] = round(
                on_thr * (1.0 - off_gap_ratio), 2,
            )

            cfg_obj.hangover_kernel_ms[cls] = trial.suggest_categorical(
                f"{cls}_hangover_ms", SEARCH_HANGOVER_KERNEL_MS,
            )

            # ---- Site-invariant (FIXED from priors) ------------------
            cfg_obj.merge_gap_s[cls] = float(priors["merge_gap_s"][cls])
            cfg_obj.min_dur_s[cls] = float(priors["min_dur_s"][cls])
            cfg_obj.max_dur_s[cls] = float(priors["max_dur_s"][cls])

        preds = postprocess_predictions_per_class(
            all_probs, cfg_obj, CLASS_NAMES,
        )
        metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
        return macro_f1(metrics)

    return objective


def _build_narrow_config_from_best_params(
    params: dict[str, Any], priors: dict,
) -> PerClassPostprocessConfig:
    """Reconstruct a full config from the 12 tuned params + priors."""
    cfg_obj = PerClassPostprocessConfig(
        median_kernel_ms={}, on_thresholds={}, off_thresholds={},
        hangover_kernel_ms={}, merge_gap_s={},
        min_dur_s={}, max_dur_s={},
    )
    for cls in CLASS_NAMES:
        cfg_obj.median_kernel_ms[cls] = int(params[f"{cls}_median_ms"])
        on_thr = float(params[f"{cls}_on_thr"])
        cfg_obj.on_thresholds[cls] = round(on_thr, 2)
        off_gap_ratio = float(params[f"{cls}_off_gap_ratio"])
        cfg_obj.off_thresholds[cls] = round(
            on_thr * (1.0 - off_gap_ratio), 2,
        )
        cfg_obj.hangover_kernel_ms[cls] = int(params[f"{cls}_hangover_ms"])
        cfg_obj.merge_gap_s[cls] = float(priors["merge_gap_s"][cls])
        cfg_obj.min_dur_s[cls] = float(priors["min_dur_s"][cls])
        cfg_obj.max_dur_s[cls] = float(priors["max_dur_s"][cls])
    return cfg_obj


def _build_narrow_seed_params(cd_thresholds: np.ndarray) -> dict[str, Any]:
    """
    Seed-trial params for the narrow search. Only the 12 tuned params;
    durations come from priors and aren't in the search space.

    The seed approximates threshold-only tuning: on_thr from cd, no
    hysteresis, no smoothing, no hangover. This guarantees Trial 1
    scores at or near the threshold-only baseline, so the final best
    is never worse than threshold-only.
    """
    seed: dict[str, Any] = {}
    for c, cls in enumerate(CLASS_NAMES):
        seed[f"{cls}_median_ms"] = _snap_to_categorical(
            cfg.SMOOTH_KERNEL_MS, SEARCH_MEDIAN_KERNEL_MS,
        )
        seed[f"{cls}_on_thr"] = _snap_to_range(
            float(cd_thresholds[c]), *SEARCH_THRESHOLD_RANGE,
        )
        seed[f"{cls}_off_gap_ratio"] = 0.0
        seed[f"{cls}_hangover_ms"] = _snap_to_categorical(
            0, SEARCH_HANGOVER_KERNEL_MS,
        )
    return seed


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Narrow-search Optuna postprocess tuner. Tunes 12 "
                    "site-dependent params; duration params come from "
                    "a priors JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str)
    src.add_argument("--checkpoints", nargs="+")

    p.add_argument("--priors", type=str, required=True,
                   help="Path to JSON priors file from "
                        "analyze_call_durations.py.")

    # Ensemble combination (same as full tuner).
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--d-from", type=int, default=None)
    p.add_argument("--weights-bmabz", nargs="+", type=float, default=None)
    p.add_argument("--weights-d", nargs="+", type=float, default=None)
    p.add_argument("--weights-bp", nargs="+", type=float, default=None)

    # LOSO support.
    p.add_argument("--held-out-site", type=str, default=None,
                   help="When set, Optuna tunes on all val sites EXCEPT "
                        "this one, and the final reported macro is "
                        "computed on this site only. Used by "
                        "run_loso_postprocess.py for cross-site "
                        "generalisation tests.")

    # Per-model eval + parallel.
    p.add_argument("--per-model-eval", action="store_true")
    p.add_argument("--parallel-eval-workers", type=int, default=1)

    # Cache + inference.
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S)
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--cache-dir", type=str, default="runs/prob_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--use-fp16", action="store_true")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)

    # Optuna.
    p.add_argument("--n-trials", type=int, default=100,
                   help="With the 12-dim narrow space TPE typically "
                        "converges in 50-100 trials. Default 100.")
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--output", type=str,
                   default="runs/postprocess_config_narrow.json")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--storage-path", type=str, default=None)
    p.add_argument("--no-storage", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-seed-trial", action="store_true")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    # Validate per-class weights.
    per_class = [args.weights_bmabz, args.weights_d, args.weights_bp]
    if any(w is not None for w in per_class):
        if not all(w is not None for w in per_class):
            p.error("All three --weights-<class> flags must be set together.")
        if args.checkpoints is None:
            p.error("--weights-<class> requires --checkpoints.")
        if args.weights is not None or args.d_from is not None:
            p.error("--weights-<class> excludes --weights / --d-from.")
        n = len(args.checkpoints)
        for cname, w in zip(("bmabz", "d", "bp"), per_class):
            if len(w) != n:
                p.error(f"--weights-{cname} has {len(w)} values, "
                        f"need {n}.")
    return args


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

    # Load priors first — fail fast if the file is broken.
    priors = load_priors(args.priors)
    print(f"Loaded duration priors from {args.priors}")
    print_priors(priors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir}")
    print(f"Eval tiles: {args.segment_s:.0f}s, overlap {args.overlap_s:.1f}s")

    # Load val data.
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }
    spec_extractor = SpectrogramExtractor().to(device)
    gt_events_all = build_gt_events(val_anns, file_start_dts)
    print(f"\nLoaded {len(gt_events_all)} val events across "
          f"{len(cfg.VAL_DATASETS)} sites: {list(cfg.VAL_DATASETS)}")

    _val_loader_cache: list = []

    def get_val_loader():
        if _val_loader_cache:
            return _val_loader_cache[0]
        val_segs = build_val_segments(
            val_manifest, val_anns,
            segment_s=args.segment_s, overlap_s=args.overlap_s,
        )
        loader = DataLoader(
            WhaleDataset(val_segs), batch_size=args.batch_size,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            collate_fn=collate_fn, pin_memory=True,
        )
        _val_loader_cache.append(loader)
        return loader

    # Per-checkpoint probs.
    if args.checkpoint is not None:
        all_probs = get_or_compute_probs(
            args.checkpoint, spec_extractor, get_val_loader, device,
            cache_dir, segment_s=args.segment_s,
            no_cache=args.no_cache, use_fp16=args.use_fp16,
        )
        ckpt_descriptor = args.checkpoint
        combination_label = "single checkpoint"
    else:
        prob_dicts = []
        for i, ckpt in enumerate(args.checkpoints):
            print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt}")
            probs = get_or_compute_probs(
                ckpt, spec_extractor, get_val_loader, device,
                cache_dir, segment_s=args.segment_s,
                no_cache=args.no_cache, use_fp16=args.use_fp16,
            )
            prob_dicts.append(probs)

        per_class_mode = args.weights_bmabz is not None
        if per_class_mode:
            raw = np.array(
                [args.weights_bmabz, args.weights_d, args.weights_bp],
                dtype=np.float64,
            )
            raw = raw / raw.sum(axis=1, keepdims=True)
            all_probs = average_prob_dicts_per_class(
                prob_dicts, raw.astype(np.float32),
            )
            combination_label = f"per-class hybrid ({len(prob_dicts)} ckpts)"
        elif args.d_from is not None:
            weights = (args.weights if args.weights is not None
                       else [1.0] * len(prob_dicts))
            all_probs = hybrid_combine(
                prob_dicts, d_from_idx=args.d_from, weights=weights,
            )
            combination_label = f"hybrid (d_from={args.d_from})"
        else:
            all_probs = average_prob_dicts(prob_dicts, weights=args.weights)
            combination_label = (
                f"plain ensemble ({len(prob_dicts)} ckpts)"
            )
        ckpt_descriptor = list(args.checkpoints)

    # ------------------------------------------------------------------
    # LOSO split: separate tune/eval event sets if --held-out-site set
    # ------------------------------------------------------------------
    if args.held_out_site:
        all_val_sites = set(cfg.VAL_DATASETS)
        if args.held_out_site not in all_val_sites:
            raise SystemExit(
                f"--held-out-site {args.held_out_site!r} not in "
                f"cfg.VAL_DATASETS = {sorted(all_val_sites)}."
            )
        tune_sites = all_val_sites - {args.held_out_site}
        eval_sites = {args.held_out_site}
        tune_probs, tune_events = filter_by_site(
            all_probs, gt_events_all, tune_sites,
        )
        eval_probs, eval_events = filter_by_site(
            all_probs, gt_events_all, eval_sites,
        )
        print(f"\nLOSO mode: held-out = {args.held_out_site}")
        print(f"  tune sites: {sorted(tune_sites)} → "
              f"{len(tune_probs)} segs, {len(tune_events)} events")
        print(f"  eval site:  {args.held_out_site} → "
              f"{len(eval_probs)} segs, {len(eval_events)} events")
    else:
        tune_probs, tune_events = all_probs, gt_events_all
        eval_probs, eval_events = all_probs, gt_events_all
        print(f"\nNon-LOSO: tune & eval on all {len(cfg.VAL_DATASETS)} "
              f"val sites ({len(gt_events_all)} events)")

    # ------------------------------------------------------------------
    # Reference baselines on the TUNE split (the Optuna objective set)
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"BASELINES on tune split  (combination: {combination_label})")
    print("=" * 64)

    # Threshold-only tune on the tune split.
    cd_thresholds, cd_metrics = quick_per_class_threshold_tune(
        tune_probs, tune_events,
    )
    _print_metrics_table(
        cd_metrics,
        f"per-class threshold-only tune (thr="
        f"{[f'{t:.2f}' for t in cd_thresholds]})",
    )
    cd_macro_tune = macro_f1(cd_metrics)

    # ------------------------------------------------------------------
    # Optuna study (narrow space)
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"OPTUNA NARROW SEARCH ({args.n_trials} trials, "
          f"12 dims)")
    print("=" * 64)

    storage = None
    storage_path: Path | None = None
    if not args.no_storage:
        if args.storage_path is not None:
            storage_path = Path(args.storage_path)
        elif args.study_name:
            storage_path = (
                Path("runs/optuna_studies") / f"{args.study_name}.log"
            )
        if storage_path is not None:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage = JournalStorage(_JournalFileBackend(str(storage_path)))
            print(f"Storage: JournalFileStorage at {storage_path}")

    study_name = args.study_name
    if storage is not None and study_name is None:
        study_name = storage_path.stem if storage_path else None

    sampler = TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=(storage is not None and not args.no_resume),
    )

    n_existing_trials = len(study.trials)
    if not args.no_seed_trial:
        seed_params = _build_narrow_seed_params(cd_thresholds)
        already_seeded = any(
            all(t.params.get(k) == v for k, v in seed_params.items())
            for t in study.trials
        )
        if not already_seeded:
            print(f"Seeding Optuna with threshold-only baseline as "
                  f"trial {n_existing_trials + 1}:")
            for k, v in seed_params.items():
                print(f"    {k:<22} = {v}")
            study.enqueue_trial(seed_params)

    if n_existing_trials > 0 and not args.no_resume:
        print(f"Resuming at {n_existing_trials} existing trials.")

    study.optimize(
        make_narrow_objective(tune_probs, tune_events, priors),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=not args.quiet,
    )

    # ------------------------------------------------------------------
    # Reconstruct best config; eval on tune split AND eval split
    # ------------------------------------------------------------------
    best_cfg = _build_narrow_config_from_best_params(
        study.best_params, priors,
    )

    tune_preds = postprocess_predictions_per_class(
        tune_probs, best_cfg, CLASS_NAMES,
    )
    tune_metrics = compute_metrics(
        tune_preds, tune_events, iou_threshold=0.3,
    )
    tune_macro = macro_f1(tune_metrics)

    eval_preds = postprocess_predictions_per_class(
        eval_probs, best_cfg, CLASS_NAMES,
    )
    eval_metrics = compute_metrics(
        eval_preds, eval_events, iou_threshold=0.3,
    )
    eval_macro = macro_f1(eval_metrics)

    print("\n" + "=" * 64)
    print("BEST CONFIG")
    print("=" * 64)
    _print_metrics_table(tune_metrics, "Tune-split metrics")
    if args.held_out_site:
        _print_metrics_table(
            eval_metrics,
            f"HELD-OUT site '{args.held_out_site}' metrics",
        )

    print(f"\n  Macro F1 progression on TUNE split:")
    print(f"    threshold-only             : {cd_macro_tune:.4f}")
    print(f"    narrow Optuna postprocess  : {tune_macro:.4f}  "
          f"(Δ = {tune_macro - cd_macro_tune:+.4f})")
    if args.held_out_site:
        print(f"\n  Macro F1 on HELD-OUT site '{args.held_out_site}':")
        print(f"    eval-split macro           : {eval_macro:.4f}")

    print(f"\n  Per-class config:")
    for cls in CLASS_NAMES:
        print(f"    {cls}:")
        print(f"      median_ms   = {best_cfg.median_kernel_ms[cls]:>5}")
        print(f"      on_thr      = {best_cfg.on_thresholds[cls]:>5.2f}")
        print(f"      off_thr     = {best_cfg.off_thresholds[cls]:>5.2f}")
        print(f"      hangover_ms = {best_cfg.hangover_kernel_ms[cls]:>5}  (tuned)")
        print(f"      merge_gap_s = {best_cfg.merge_gap_s[cls]:>5.2f}  (prior)")
        print(f"      min_dur_s   = {best_cfg.min_dur_s[cls]:>5.2f}  (prior)")
        print(f"      max_dur_s   = {best_cfg.max_dur_s[cls]:>5.2f}  (prior)")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    metadata = {
        "mode": "narrow",
        "priors_file": args.priors,
        "checkpoints": ckpt_descriptor,
        "combination": combination_label,
        "weights_per_class": (
            {"bmabz": list(args.weights_bmabz),
             "d": list(args.weights_d),
             "bp": list(args.weights_bp)}
            if args.weights_bmabz is not None else None
        ),
        "held_out_site": args.held_out_site,
        "segment_s": float(args.segment_s),
        "n_trials": int(args.n_trials),
        "macro_f1_threshold_only_tune": float(cd_macro_tune),
        "macro_f1_narrow_tune": float(tune_macro),
        "macro_f1_held_out": float(eval_macro)
                              if args.held_out_site else None,
        "per_class_f1_tune": {
            cls: float(tune_metrics.get(cls, {}).get("f1", 0.0))
            for cls in CLASS_NAMES
        },
        "per_class_f1_held_out": (
            {cls: float(eval_metrics.get(cls, {}).get("f1", 0.0))
             for cls in CLASS_NAMES}
            if args.held_out_site else None
        ),
        "best_params": dict(study.best_params),
        "best_config": {
            "median_kernel_ms": dict(best_cfg.median_kernel_ms),
            "on_thresholds": dict(best_cfg.on_thresholds),
            "off_thresholds": dict(best_cfg.off_thresholds),
            "hangover_kernel_ms": dict(best_cfg.hangover_kernel_ms),
            "merge_gap_s": dict(best_cfg.merge_gap_s),
            "min_dur_s": dict(best_cfg.min_dur_s),
            "max_dur_s": dict(best_cfg.max_dur_s),
        },
        "study_name": study_name,
        "storage_path": str(storage_path) if storage_path else None,
        "n_existing_trials_at_start": n_existing_trials,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
