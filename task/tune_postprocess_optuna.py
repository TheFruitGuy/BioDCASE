"""
Per-Class Post-Processing Tuner (Optuna)
========================================

Optuna search over the per-class post-processing hyperparameter space
defined in Geldenhuys et al. (arXiv:2510.21280v2) Table IV. Reuses the
per-checkpoint probability cache built by ``ensemble_predict_cached.py``
— same cache file naming (``<run_dir>{_segN}_probs.pkl``), same
already-3-class probs on disk. With caches hit, this script is fully
CPU-bound: each trial is just postprocess + scoring.

The objective is **macro F1** (mean of per-class event-level F1 at IoU
0.3) — the BioDCASE Task 2 official metric. The script prints the
per-class breakdown for the best trial and writes the result to a JSON
file that ``inference.py`` will auto-load on the next run.

Four combination modes
----------------------
1. **Single checkpoint**: ``--checkpoint X.pt`` — tune for one model.

2. **Plain ensemble**: ``--checkpoints A B C [...] [--weights w1 w2 w3 ...]``
   — weighted average across all classes.

3. **Hybrid (single-D-source)** — preserved from the BPN workflow:
   ``--checkpoints A B C D --weights 1 1 1 2 --d-from 3``
   BMABZ + BP averaged across all models, D taken ONLY from the model
   at index ``--d-from``. Matches ``hybrid_ensemble_predict.py``.

4. **Per-class hybrid (weight vectors)** — for the PGI/AMSE split that
   reached macro 0.490 in ``ensemble_predict_cached.py``:
   ``--weights-bmabz w1...wN --weights-d w1...wN --weights-bp w1...wN``
   One weight per checkpoint per class; ``0`` excludes a checkpoint
   from that class. Vectors are normalized per row. Mutually exclusive
   with ``--weights`` and ``--d-from``.

Per-model eval parallelism
--------------------------
``--per-model-eval`` runs ``quick_per_class_threshold_tune`` for each
checkpoint as a diagnostic. At 60s tiles this is ~40 grid trials ×
~15s = ~10 min/model — fully CPU-bound. ``--parallel-eval-workers N``
(default 1) switches to flat-phase scheduling: for each of the three
coordinate-descent phases (BMABZ → D → BP), all
``n_models × ~13 thresholds`` independent trials are submitted to one
worker pool. Because within-class trials are independent (they only
differ in one threshold slot), N can usefully exceed the number of
models. Across-class trials must remain sequential — phase k+1 needs
each model's best phase-k threshold.

Rule of thumb: set N to ``min(os.cpu_count(), n_checkpoints × 13)``.
Workers share the parent's probability dicts via fork (copy-on-write),
so the memory overhead is one address-space snapshot regardless of
worker count. Linux only.

Examples
--------
Per-class hybrid at 60s tiles (PGI for BMABZ+BP, AMSE MT for D)::

    python tune_postprocess_optuna.py --use-fp16 --segment-s 60 \\
        --checkpoints \\
            runs/hnm_pgi_whalevad_<s1>/best_model.pt \\
            ... 4 more PGI ckpts ... \\
            runs/mt_hnm_pgi_amse_whalevad_<s1>/best_model.pt \\
            ... 4 more AMSE ckpts ... \\
        --weights-bmabz 1 1 1 1 1  0 0 0 0 0 \\
        --weights-d     0 0 0 0 0  1 1 1 1 1 \\
        --weights-bp    1 1 1 1 1  0 0 0 0 0 \\
        --n-trials 500 --per-model-eval --parallel-eval-workers 10

BPN-style hybrid (single D source)::

    CUDA_VISIBLE_DEVICES=1 python tune_postprocess_optuna.py \\
        --checkpoints A.pt B.pt C.pt D.pt \\
        --weights 1 1 1 2 --d-from 3 \\
        --n-trials 500 --per-model-eval

Single checkpoint::

    python tune_postprocess_optuna.py \\
        --checkpoint runs/whalevad_XXXX/best_model.pt --n-trials 300

Cache reuse
-----------
``--segment-s`` (default ``cfg.EVAL_SEGMENT_S``) and ``--cache-dir``
(default ``runs/prob_cache``) must match what ``ensemble_predict_cached.py``
used to build the cache. With matching values, cache hits load the
already-3-class probs from disk in milliseconds; cache misses run
inference once and write to the same cache that the predictor will
reuse later. ``--no-cache`` forces full recomputation.

The output path defaults to ``cfg.POSTPROCESS_CONFIG_PATH`` declared in
``config.py``; pass ``--output`` only to keep multiple tuned configs.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    default_config_from_global_cfg,
    SEARCH_MEDIAN_KERNEL_MS,
    SEARCH_HANGOVER_KERNEL_MS,
    SEARCH_MERGE_GAP_S,
    SEARCH_MIN_DUR_S,
    SEARCH_MAX_DUR_S,
    SEARCH_THRESHOLD_RANGE,
)

# Reuse the BPN-aware helpers and the single-D-source combiner from the
# hybrid ensemble script, so this tuner sees exactly the same probability
# stream that the inference path produces.
from hybrid_ensemble_predict import (
    build_model_for_ckpt,
    predict_probabilities,
    hybrid_combine,
)

# Reuse the on-disk cache contract from ensemble_predict_cached.py.
# Same filename scheme (``<run_dir>{_segN}_probs.pkl``), same float16
# storage, same 3-class semantics on disk — so caches built by the
# predictor are readable here and vice versa.
from ensemble_predict_cached import (
    cache_path_for,
    load_cached_probs,
    save_probs_to_cache,
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
# Probability collection: cache-aware, BPN-aware
# ======================================================================

@torch.no_grad()
def get_or_compute_probs(
    ckpt_path: str,
    spec_extractor: SpectrogramExtractor,
    val_loader_fn,
    device: torch.device,
    cache_dir: Path,
    segment_s: float = 30.0,
    no_cache: bool = False,
    use_fp16: bool = False,
) -> dict[tuple, np.ndarray]:
    """
    Cache hit → load pickle. Cache miss → run inference, save, return.

    Cache contract mirrors ``ensemble_predict_cached.py``: pickles store
    already-3-class probs in float16, keyed by ``(checkpoint, segment_s)``
    via :func:`cache_path_for`. 30s segments keep the legacy filename
    (no suffix); other lengths get ``_segN``.

    ``val_loader_fn`` is a zero-arg callable that returns the DataLoader.
    Deferred construction means cache-only runs never pay the manifest
    scan cost.
    """
    cp = cache_path_for(Path(ckpt_path), cache_dir, segment_s=segment_s)
    if not no_cache and cp.exists():
        print(f"  cache hit: {cp.name}")
        return load_cached_probs(cp)

    print(f"  cache miss, running inference...")
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"    type: {model_type}")

    # Materialize lazy projection layer before load_state_dict — sized
    # to the eval tile length so the cached CNN+BiLSTM matches what the
    # predictor will produce.
    with torch.no_grad():
        dummy = torch.randn(
            1, int(cfg.SAMPLE_RATE * segment_s), device=device,
        )
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
    model.eval()

    val_loader = val_loader_fn()
    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            probs = predict_probabilities(
                model, model_type, spec_extractor, val_loader, device,
            )
    else:
        probs = predict_probabilities(
            model, model_type, spec_extractor, val_loader, device,
        )
    probs = collapse_probs_to_3class(probs)

    print(f"    computed in {time.time()-t0:.0f}s, "
          f"{len(probs)} segment prob arrays")
    if not no_cache:
        save_probs_to_cache(probs, cp)
        print(f"    cached → {cp}")

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


def average_prob_dicts_per_class(
    prob_dicts: list[dict[tuple, np.ndarray]],
    weights_per_class: np.ndarray,
) -> dict[tuple, np.ndarray]:
    """
    Per-class weighted average — the combiner behind the new
    ``--weights-bmabz/-d/-bp`` flags.

    Parameters
    ----------
    prob_dicts : list of dict[key -> ndarray (T, C)]
        One dict per checkpoint. Caches from ``ensemble_predict_cached``
        store already-3-class probs, so C = 3 in normal use.
    weights_per_class : ndarray of shape ``(n_classes, n_models)``
        Row c is the (already-normalized) weight vector across models
        for class c. Each row should sum to 1; rows that sum to zero
        are caught upstream.

    Returns
    -------
    dict mapping each common key to ``(T, C)`` ndarray with
    ``out[:, c] = sum_m weights_per_class[c, m] * prob_dicts[m][:, c]``,
    truncated to the shortest T across models.

    A zero weight excludes a model from that class entirely — this is
    the mechanism for "PGI ensemble for BMABZ + BP, AMSE MT ensemble
    for D" and similar per-class hybrids.
    """
    if not prob_dicts:
        return {}
    n_models = len(prob_dicts)
    n_classes = weights_per_class.shape[0]
    assert weights_per_class.shape == (n_classes, n_models), (
        f"weights_per_class shape {weights_per_class.shape} does not "
        f"match (n_classes={n_classes}, n_models={n_models})"
    )

    common_keys = set(prob_dicts[0].keys())
    for d in prob_dicts[1:]:
        common_keys &= set(d.keys())
    n_dropped = len(prob_dicts[0]) - len(common_keys)
    if n_dropped > 0:
        print(f"  WARNING: {n_dropped} keys missing from at least one "
              f"model; dropping them from the per-class ensemble.")

    # Broadcasting shape: (M, 1, C) against stacked (M, T, C) → (M, T, C).
    w_bc = weights_per_class.T[:, None, :].astype(np.float32)

    out: dict[tuple, np.ndarray] = {}
    for key in common_keys:
        arrs = [d[key] for d in prob_dicts]
        min_T = min(a.shape[0] for a in arrs)
        stacked = np.stack(
            [a[:min_T, :n_classes].astype(np.float32) for a in arrs],
            axis=0,
        )  # (M, T, C)
        out[key] = (stacked * w_bc).sum(axis=0)  # (T, C)
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
# Parallel per-model eval — flat-phase scheduling
# ======================================================================
#
# quick_per_class_threshold_tune is coordinate descent over three
# classes; within each class the ~13 grid trials are independent
# (all use the same other-class thresholds, varying only one). So the
# natural parallel structure is:
#
#   phase 1 (BMABZ): n_models × n_BMABZ_thresholds independent trials
#   phase 2 (D):     n_models × n_D_thresholds       independent trials
#                    (using each model's best BMABZ from phase 1)
#   phase 3 (BP):    n_models × n_BP_thresholds      independent trials
#                    (using each model's best BMABZ + best D)
#   final:           n_models                        full-metrics evals
#
# Flat-phase scheduling submits each phase as a single flat task list
# to ONE worker pool — so with N workers and 130 tasks/phase, each
# phase takes ceil(130/N) × trial_time. This is strictly better than
# per-model parallelism whenever N > n_models (otherwise equivalent).
#
# Workers access the prob dicts via module-level globals inherited
# from the parent at fork time (no pickling of the ~26 MB per-model
# probability dict). Only `fork` start method is supported.

_WORKER_PROB_DICTS: list = []
_WORKER_GT_EVENTS: list = []


def _grid_trial_task(
    payload: tuple[int, int, float, np.ndarray],
) -> tuple[int, int, float, float]:
    """
    Worker: evaluate one (model_idx, class_idx, threshold) grid trial.

    ``trial_thresholds`` is the full ``[t0, t1, t2]`` vector with the
    class_idx slot already set to ``threshold_value``; the other two
    slots are the model's current-best from earlier phases.
    Returns the class-c F1 only — that's all coordinate descent needs.
    """
    model_idx, class_idx, threshold_value, trial_thresholds = payload
    probs = _WORKER_PROB_DICTS[model_idx]
    preds = postprocess_predictions(probs, trial_thresholds)
    metrics = compute_metrics(preds, _WORKER_GT_EVENTS, iou_threshold=0.3)
    class_name = CLASS_NAMES[class_idx]
    f1 = float(metrics.get(class_name, {}).get("f1", 0.0))
    return model_idx, class_idx, float(threshold_value), f1


def _final_eval_task(
    payload: tuple[int, str, np.ndarray],
) -> tuple[int, str, np.ndarray, dict]:
    """
    Worker: final per-model eval with the tuned thresholds.

    Computes the full metrics dict (overall + per-class) so the
    individual-model summary table can report micro/macro F1.
    """
    model_idx, ckpt_path, final_thresholds = payload
    probs = _WORKER_PROB_DICTS[model_idx]
    preds = postprocess_predictions(probs, final_thresholds)
    metrics = compute_metrics(preds, _WORKER_GT_EVENTS, iou_threshold=0.3)
    return model_idx, ckpt_path, final_thresholds, metrics


def parallel_per_model_threshold_tune(
    n_models: int,
    ckpt_paths: list[str],
    n_workers: int,
    ctx,
) -> list[dict]:
    """
    Flat-phase parallel implementation of per-model threshold tuning.

    Same grids and same coordinate-descent semantics as
    ``quick_per_class_threshold_tune``, but with all (model, threshold)
    trials inside a class flattened into a single task list. Three
    sequential phases (one per class), then a final eval phase.

    Requires ``_WORKER_PROB_DICTS`` and ``_WORKER_GT_EVENTS`` to be
    populated in the parent before this is called (workers inherit
    them via fork).

    Returns a list of dicts, one per model, in input order, with the
    same shape as the sequential path's ``individual_metrics`` entries.
    """
    grids = [
        np.arange(0.20, 0.85, 0.05),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
    ]
    # Initial thresholds: 0.5 for every class, every model.
    model_thresholds = [
        np.array([0.5, 0.5, 0.5], dtype=np.float64) for _ in range(n_models)
    ]

    total_grid = sum(len(g) for g in grids) * n_models
    print(f"  Plan: {len(grids)} sequential phases × "
          f"{n_models} models × ~{len(grids[0])} thresholds = "
          f"{total_grid} grid trials + {n_models} final evals, "
          f"on {n_workers} workers")

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        # ------------------ three coordinate-descent phases ------------
        for class_idx, class_name in enumerate(CLASS_NAMES):
            t0 = time.time()
            tasks: list = []
            for m in range(n_models):
                for t in grids[class_idx]:
                    trial = model_thresholds[m].copy()
                    trial[class_idx] = float(t)
                    tasks.append((m, class_idx, float(t), trial))

            # Track best (f1, threshold) per model for this class.
            best_per_model: list[tuple[float, float]] = [
                (-1.0, 0.5) for _ in range(n_models)
            ]
            futures = [pool.submit(_grid_trial_task, task)
                       for task in tasks]
            for fut in as_completed(futures):
                m, _, t_val, f1 = fut.result()
                if f1 > best_per_model[m][0]:
                    best_per_model[m] = (f1, t_val)

            for m in range(n_models):
                model_thresholds[m][class_idx] = best_per_model[m][1]

            chosen = [f"{best_per_model[m][1]:.2f}"
                      for m in range(n_models)]
            print(f"  phase {class_idx+1}/3 [{class_name:6}]: "
                  f"{len(tasks)} trials in {time.time()-t0:.0f}s   "
                  f"chosen={chosen}")

        # ------------------ final per-model full-metrics eval ----------
        t0 = time.time()
        final_tasks = [
            (m, ckpt_paths[m], model_thresholds[m].copy())
            for m in range(n_models)
        ]
        futures = [pool.submit(_final_eval_task, task)
                   for task in final_tasks]
        results: list = [None] * n_models
        for fut in as_completed(futures):
            m, ckpt_path, final_thr, metrics = fut.result()
            results[m] = {
                "path": ckpt_path,
                "thresholds": final_thr,
                "metrics": metrics,
                "f1": metrics.get("overall", {}).get("f1", 0.0),
                "macro_f1": float(np.mean(
                    [metrics.get(c, {}).get("f1", 0.0)
                     for c in CLASS_NAMES]
                )),
            }
        print(f"  final eval: {n_models} per-model metrics in "
              f"{time.time()-t0:.0f}s")

    print(f"  total parallel tune wall time: "
          f"{time.time()-t_start:.0f}s")
    return results


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

            
            off_gap_ratio = trial.suggest_float(f"{cls}_off_gap_ratio", 0.0, 1.0, step=0.05)
            off_thr = round(on_thr * (1.0 - off_gap_ratio), 2)
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
                    "hyperparameters. Reuses the per-checkpoint cache "
                    "built by ensemble_predict_cached.py so each trial "
                    "is CPU-only when caches exist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str,
                     help="Single checkpoint to tune for.")
    src.add_argument("--checkpoints", nargs="+",
                     help="Multiple checkpoints — combined into an "
                          "ensemble before tuning. Use --d-from for "
                          "single-D-source hybrid, or "
                          "--weights-bmabz/-d/-bp for per-class hybrid.")

    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint ensemble weights "
                        "(only valid with --checkpoints). Must match "
                        "in length; normalised to sum to 1. Mutually "
                        "exclusive with --weights-<class> flags.")
    p.add_argument("--d-from", type=int, default=None,
                   help="0-indexed position in --checkpoints of the "
                        "model to use for D-class predictions. When "
                        "set, BMABZ/BP are weighted-averaged across "
                        "all models but D is taken from this single "
                        "model — matches hybrid_ensemble_predict.py. "
                        "Mutually exclusive with --weights-<class>.")

    # Per-class weight vectors. If any is set, all three must be set,
    # and --weights / --d-from must not be set.
    p.add_argument("--weights-bmabz", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for the BMABZ class. "
                        "Use 0 to exclude. Length must equal "
                        "--checkpoints. Vectors are normalized per row.")
    p.add_argument("--weights-d", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for the D class.")
    p.add_argument("--weights-bp", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for the BP class.")

    p.add_argument("--per-model-eval", action="store_true",
                   help="Score each checkpoint individually (with its "
                        "own coordinate-descent thresholds) before the "
                        "ensemble step. Pure CPU; ~10 min/model at 60s "
                        "tiles. Use --parallel-eval-workers to fan out "
                        "across processes.")
    p.add_argument("--parallel-eval-workers", type=int, default=1,
                   help="Number of worker processes for parallel "
                        "per-model threshold tuning (only used with "
                        "--per-model-eval). Uses flat-phase scheduling: "
                        "each of three sequential coordinate-descent "
                        "phases (BMABZ, D, BP) submits "
                        "n_models × ~13 independent trials to one pool, "
                        "so N can usefully exceed the number of "
                        "checkpoints. Set N to "
                        "min(cpu_count, n_ckpts × 13) for max speedup. "
                        "Workers share probs via fork (no pickling). "
                        "Default 1 = sequential. Linux only.")

    # ---- Cache + inference config (matches ensemble_predict_cached) ----
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Eval tile length in seconds. Cache files are "
                        "keyed by this — must match what "
                        "ensemble_predict_cached.py used to build the "
                        "cache (default: cfg.EVAL_SEGMENT_S). 30s keeps "
                        "the legacy filename for backward compat.")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S,
                   help="Overlap between consecutive eval tiles in "
                        "seconds. Only used on cache miss.")
    p.add_argument("--cache-dir", type=str, default="runs/prob_cache",
                   help="Directory of per-checkpoint cached probs. "
                        "Reuses caches built by "
                        "ensemble_predict_cached.py.")
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-inference even if cache exists.")
    p.add_argument("--use-fp16", action="store_true",
                   help="FP16 autocast during inference (cache-miss "
                        "path only). ~1.5-2x speedup, <0.001 F1 hit.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                   help="Batch size for inference (cache-miss path "
                        "only).")

    # ---- Optuna config ----
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

    # ------------------------------------------------------------------
    # CLI validation
    # ------------------------------------------------------------------
    per_class_w = [args.weights_bmabz, args.weights_d, args.weights_bp]
    any_pc = any(w is not None for w in per_class_w)
    all_pc = all(w is not None for w in per_class_w)
    per_class_mode = False

    if any_pc:
        if not all_pc:
            raise SystemExit(
                "If any --weights-<class> flag is set, all three "
                "(--weights-bmabz, --weights-d, --weights-bp) must be "
                "set together."
            )
        if args.checkpoints is None:
            raise SystemExit(
                "--weights-<class> requires --checkpoints (multiple)."
            )
        if args.weights is not None:
            raise SystemExit(
                "--weights-<class> flags are mutually exclusive with "
                "--weights."
            )
        if args.d_from is not None:
            raise SystemExit(
                "--weights-<class> flags are mutually exclusive with "
                "--d-from."
            )
        n = len(args.checkpoints)
        for cname, w in zip(("bmabz", "d", "bp"), per_class_w):
            if len(w) != n:
                raise SystemExit(
                    f"--weights-{cname} has {len(w)} values, "
                    f"need {n} (one per checkpoint)."
                )
        per_class_mode = True

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
    if args.parallel_eval_workers < 1:
        raise SystemExit("--parallel-eval-workers must be >= 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir} (no_cache={args.no_cache}, "
          f"use_fp16={args.use_fp16})")
    print(f"Eval tiles: {args.segment_s:.0f}s segments, "
          f"{args.overlap_s:.1f}s overlap")

    # ------------------------------------------------------------------
    # Load val data once. The val_loader is built lazily — if every
    # cache hits we never pay for the DataLoader construction.
    # ------------------------------------------------------------------
    print("\nLoading validation data...")
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }
    spec_extractor = SpectrogramExtractor().to(device)
    gt_events = build_gt_events(val_annotations, file_start_dts)
    print(f"  {len(gt_events)} ground-truth events")

    _val_loader_cache: list = []  # closure-friendly memo for the loader

    def get_val_loader():
        """Build the segment-aware val loader on first call; reuse after."""
        if _val_loader_cache:
            return _val_loader_cache[0]
        val_segs = build_val_segments(
            val_manifest, val_annotations,
            segment_s=args.segment_s,
            overlap_s=args.overlap_s,
        )
        loader = DataLoader(
            WhaleDataset(val_segs), batch_size=args.batch_size,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            collate_fn=collate_fn, pin_memory=True,
        )
        print(f"  built val loader: {len(val_segs)} "
              f"{args.segment_s:.0f}s tiles "
              f"({args.overlap_s:.1f}s overlap)")
        _val_loader_cache.append(loader)
        return loader

    # ------------------------------------------------------------------
    # Compute per-checkpoint val probabilities (cache-aware, BPN-aware).
    # When --parallel-eval-workers > 1, defer the per-model tune until
    # all probs are loaded, then fan out across worker processes.
    # ------------------------------------------------------------------
    inline_per_model_eval = (
        args.per_model_eval and args.parallel_eval_workers <= 1
    )
    individual_metrics: list[dict] = []

    if args.checkpoint is not None:
        print(f"\nCollecting probabilities from {args.checkpoint}")
        all_probs = get_or_compute_probs(
            args.checkpoint, spec_extractor, get_val_loader, device,
            cache_dir, segment_s=args.segment_s,
            no_cache=args.no_cache, use_fp16=args.use_fp16,
        )
        ckpt_descriptor = args.checkpoint
        combination_label = "single checkpoint"
    else:
        prob_dicts: list[dict[tuple, np.ndarray]] = []
        for i, ckpt in enumerate(args.checkpoints):
            if per_class_mode:
                marker = ""
            elif args.d_from is not None and i == args.d_from:
                marker = "  ← D source"
            else:
                marker = ""
            print(f"\n[{i+1}/{len(args.checkpoints)}] Loading {ckpt}{marker}")
            probs = get_or_compute_probs(
                ckpt, spec_extractor, get_val_loader, device,
                cache_dir, segment_s=args.segment_s,
                no_cache=args.no_cache, use_fp16=args.use_fp16,
            )
            prob_dicts.append(probs)

            if inline_per_model_eval:
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

        # Parallel per-model tune (only if requested and not inlined).
        if args.per_model_eval and args.parallel_eval_workers > 1:
            try:
                ctx = mp.get_context("fork")
            except (ValueError, RuntimeError) as exc:
                raise SystemExit(
                    f"--parallel-eval-workers > 1 requires the 'fork' "
                    f"multiprocessing start method (Linux). Got: {exc}. "
                    f"Use --parallel-eval-workers 1 on this platform."
                )
            n_workers = args.parallel_eval_workers
            print(f"\nParallel per-model threshold tune "
                  f"(flat-phase scheduling):")

            # Populate module globals BEFORE pool creation so forked
            # workers inherit them (copy-on-write, no pickling).
            global _WORKER_PROB_DICTS, _WORKER_GT_EVENTS
            _WORKER_PROB_DICTS = prob_dicts
            _WORKER_GT_EVENTS = gt_events

            individual_metrics = parallel_per_model_threshold_tune(
                n_models=len(prob_dicts),
                ckpt_paths=list(args.checkpoints),
                n_workers=n_workers,
                ctx=ctx,
            )

            # Echo per-model summary in input order — matches the
            # sequential path's printout.
            print()
            for i, im in enumerate(individual_metrics):
                short = Path(im["path"]).parent.name
                print(f"  #{i+1} {short}: "
                      f"micro={im['f1']:.3f} "
                      f"macro={im['macro_f1']:.3f} "
                      f"thr={[f'{t:.2f}' for t in im['thresholds']]}")

            # Drop the global references — Optuna doesn't need them and
            # we don't want them keeping ~260 MB alive longer than
            # necessary. (The actual probs are still alive via the
            # local prob_dicts; this just unbinds the global names.)
            _WORKER_PROB_DICTS = []
            _WORKER_GT_EVENTS = []

        # Combine.
        if per_class_mode:
            raw = np.array(
                [args.weights_bmabz, args.weights_d, args.weights_bp],
                dtype=np.float64,
            )
            row_sums = raw.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                bad = [c for c, s in zip(CLASS_NAMES, row_sums.ravel())
                       if s <= 0]
                raise SystemExit(
                    f"Per-class weights for {bad} sum to zero — no "
                    f"models would contribute to that class."
                )
            weights_per_class = (raw / row_sums).astype(np.float32)
            print(f"\nPER-CLASS HYBRID combination "
                  f"(rows = classes, cols = checkpoints in input order):")
            for c, row in zip(CLASS_NAMES, weights_per_class):
                pretty = ", ".join("%.2f" % w for w in row)
                print(f"  {c:6}: [{pretty}]")
            all_probs = average_prob_dicts_per_class(
                prob_dicts, weights_per_class,
            )
            combination_label = (
                f"per-class hybrid "
                f"(weights_bmabz/d/bp, normalized; "
                f"{len(args.checkpoints)} ckpts)"
            )
        elif args.d_from is not None:
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
            if per_class_mode:
                contribs = [
                    c for c, row in zip(CLASS_NAMES, weights_per_class)
                    if row[i] > 0
                ]
                marker = (f"  [{','.join(contribs)}]" if contribs
                          else "  [excluded from all classes]")
            elif args.d_from is not None and i == args.d_from:
                marker = "  ← D source"
            else:
                marker = ""
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
        "weights_per_class": (
            {
                "bmabz": list(args.weights_bmabz),
                "d": list(args.weights_d),
                "bp": list(args.weights_bp),
            } if per_class_mode else None
        ),
        "segment_s": float(args.segment_s),
        "overlap_s": float(args.overlap_s),
        "parallel_eval_workers": int(args.parallel_eval_workers),
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
