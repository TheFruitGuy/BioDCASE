"""
validation_core
===============

Single source of truth for the per-class coordinate-descent threshold
sweep used at validation time. Previously this logic lived in four
places — ``train.py`` (inline inside ``validate``), ``train_phase_hnm.py``
(via ``tune_thresholds_on_probs``), ``ensemble_predict.py``
(``tune_thresholds_on_probs``), and ``tune_postprocess_optuna.py``
(``quick_per_class_threshold_tune``) — with the same grids hard-coded in
each. Centralising it here removes the duplication and gives every
caller the optional within-class parallel grid search "for free".

What's tuned
------------
Three per-class probability thresholds in ``cfg.CALL_TYPES_3`` order
(``bmabz``, ``d``, ``bp``). The tune is **coordinate descent**: BMABZ
first (held thresholds = [t, 0.5, 0.5]), then D using the chosen BMABZ
([best_bmabz, t, 0.5]), then BP using both previous picks. Within a
class, every grid point is independent and can run in parallel; across
classes, phase k+1 needs phase k's pick, so the three phases are
sequential.

The grids are the same as the originals — kept as module-level
constants so they're cited rather than copy-pasted:

- BMABZ : step 0.05 from 0.20 to 0.80  (13 points)
- D     : step 0.05 from 0.05 to 0.45, step 0.10 from 0.50 to 0.80
- BP    : same shape as D

Why parallelism matters here
----------------------------
Each grid trial is ``postprocess_predictions`` + ``compute_metrics`` on
the whole val set — pure CPU, ~10-15s at 60s eval tiles. Sequential
cost is ~40 × 15s ≈ 10 min per validate() call. With ``parallel_workers
= 13`` (one per BMABZ/D/BP grid point), wall time drops to
~4 phases × 15s ≈ 1 min. Run inside a training loop with 32 epochs,
that's ~5 hours saved per training run.

Workers share the parent's probability dict via fork (copy-on-write)
through a module-level global, so there's no pickling cost — fine for
single-model use during training. Linux only (fork requirement); pass
``parallel_workers=1`` on macOS/Windows.

Quick integration examples
--------------------------

In ``train.py`` / ``train_phase_hnm.py``::

    from validation_core import tune_thresholds_per_class, \\
        evaluate_with_thresholds, macro_f1

    # Inside validate():
    if tune_thresholds:
        used_thresholds = tune_thresholds_per_class(
            all_probs, gt_events,
            start=used_thresholds,
            parallel_workers=args.val_workers,  # e.g. 13
        )
    metrics = evaluate_with_thresholds(all_probs, gt_events, used_thresholds)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
    macro = macro_f1(metrics)

In ``ensemble_predict.py``::

    from validation_core import tune_thresholds_per_class
    # ... replace tune_thresholds_on_probs body with a one-liner call.

In ``tune_postprocess_optuna.py``::

    from validation_core import (
        THRESHOLD_GRIDS, tune_thresholds_per_class,
        evaluate_with_thresholds, macro_f1,
    )
    # quick_per_class_threshold_tune becomes a thin wrapper;
    # the flat-phase multi-model orchestrator can use THRESHOLD_GRIDS
    # as its single source of truth.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

import numpy as np

import config as cfg
from postprocess import (
    Detection, compute_metrics, postprocess_predictions,
)

# ----------------------------------------------------------------------
# Public constants — single source of truth for the per-class grids.
# ----------------------------------------------------------------------

CLASS_NAMES: list[str] = list(cfg.CALL_TYPES_3)

BMABZ_THRESHOLD_GRID: np.ndarray = np.arange(0.20, 0.85, 0.05)
"""Coarse step-0.05 grid for BMABZ. Common class, broad F1 peak."""

D_THRESHOLD_GRID: np.ndarray = np.concatenate([
    np.arange(0.05, 0.5, 0.05),
    np.arange(0.5, 0.85, 0.10),
])
"""Fine low-end, coarse high-end grid for D. Rare class, optimum often well below 0.5."""

BP_THRESHOLD_GRID: np.ndarray = np.concatenate([
    np.arange(0.05, 0.5, 0.05),
    np.arange(0.5, 0.85, 0.10),
])
"""Same shape as D — rare class with low-end optimum."""

THRESHOLD_GRIDS: list[np.ndarray] = [
    BMABZ_THRESHOLD_GRID,
    D_THRESHOLD_GRID,
    BP_THRESHOLD_GRID,
]
"""Indexed in cfg.CALL_TYPES_3 order — match indices against CLASS_NAMES."""

IOU_THRESHOLD: float = 0.3
"""Event-level matching IoU. Same as the BioDCASE Task 2 metric."""


# ----------------------------------------------------------------------
# Worker-side state for parallel mode (fork copy-on-write).
# Populated by the parent inside ``tune_thresholds_per_class`` before
# the pool spawns; workers read these via inherited globals.
# ----------------------------------------------------------------------

_WORKER_PROBS: dict | None = None
_WORKER_GT: list[Detection] | None = None


def _grid_eval_task(
    payload: tuple[int, float, np.ndarray],
) -> tuple[int, float, float]:
    """
    Worker: evaluate one (class_idx, threshold_value, trial_vec) trial.

    Returns the class-c F1 only — that's the single number coordinate
    descent uses to pick the per-class best.
    """
    class_idx, threshold_value, trial_vec = payload
    preds = postprocess_predictions(_WORKER_PROBS, trial_vec)
    metrics = compute_metrics(preds, _WORKER_GT, iou_threshold=IOU_THRESHOLD)
    class_name = CLASS_NAMES[class_idx]
    f1 = float(metrics.get(class_name, {}).get("f1", 0.0))
    return class_idx, float(threshold_value), f1


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def tune_thresholds_per_class(
    probs: dict[tuple, np.ndarray],
    gt_events: Sequence[Detection],
    start: np.ndarray | Sequence[float] | None = None,
    parallel_workers: int = 1,
) -> np.ndarray:
    """
    Per-class coordinate-descent threshold sweep.

    Parameters
    ----------
    probs : dict[(dataset, filename, start_sample) -> ndarray (T, 3)]
        Per-segment probability arrays, already collapsed to 3 classes.
    gt_events : sequence of Detection
        Ground-truth events for event-level matching.
    start : ndarray or sequence of 3 floats, optional
        Starting thresholds. Default ``[0.5, 0.5, 0.5]``. Pass the
        previous epoch's tuned values for a small warm-start gain.
    parallel_workers : int, default 1
        Number of worker processes for within-class grid evaluation.
        1 = sequential. >1 = fan the class's ~13 trials across N
        processes via fork. ``min(os.cpu_count(), 13)`` is the sweet
        spot — extra workers beyond 13 sit idle within a phase.

    Returns
    -------
    thresholds : ndarray of shape (3,)
        Tuned thresholds in ``cfg.CALL_TYPES_3`` order.

    Notes
    -----
    Three sequential phases (BMABZ → D → BP), each phase parallel
    across grid points. Identical pick semantics to the legacy inline
    code in ``train.py`` and ``ensemble_predict.tune_thresholds_on_probs``
    — verified against the same grids and the same coordinate-descent
    order, so per-epoch F1 numbers stay comparable.

    Worker access to ``probs`` and ``gt_events`` is via fork-inherited
    module globals (no pickling). Linux only when
    ``parallel_workers > 1``; on other platforms pass 1.
    """
    if start is None:
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    else:
        thresholds = np.asarray(start, dtype=np.float64).copy()
        if thresholds.shape != (3,):
            raise ValueError(
                f"start must have shape (3,), got {thresholds.shape}"
            )

    if parallel_workers <= 1:
        return _tune_sequential(thresholds, probs, gt_events)
    return _tune_parallel(thresholds, probs, gt_events, parallel_workers)


def _tune_sequential(
    thresholds: np.ndarray,
    probs: dict[tuple, np.ndarray],
    gt_events: Sequence[Detection],
) -> np.ndarray:
    """Sequential coordinate descent — same as the legacy inline code."""
    for c, name in enumerate(CLASS_NAMES):
        best_f1, best_t = -1.0, thresholds[c]
        for t in THRESHOLD_GRIDS[c]:
            trial = thresholds.copy()
            trial[c] = float(t)
            preds = postprocess_predictions(probs, trial)
            metrics = compute_metrics(
                preds, gt_events, iou_threshold=IOU_THRESHOLD,
            )
            f1 = metrics.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
    return thresholds


def _tune_parallel(
    thresholds: np.ndarray,
    probs: dict[tuple, np.ndarray],
    gt_events: Sequence[Detection],
    parallel_workers: int,
) -> np.ndarray:
    """
    Parallel coordinate descent — three sequential phases, within-class
    trials submitted as a flat task list to a fork-based worker pool.
    """
    try:
        ctx = mp.get_context("fork")
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"parallel_workers > 1 requires the 'fork' multiprocessing "
            f"start method (Linux). Got: {exc}. Pass parallel_workers=1 "
            f"on macOS/Windows."
        ) from exc

    global _WORKER_PROBS, _WORKER_GT
    _WORKER_PROBS = probs
    _WORKER_GT = list(gt_events)  # ensure list (Sequence may be tuple)

    try:
        with ProcessPoolExecutor(
            max_workers=parallel_workers, mp_context=ctx,
        ) as pool:
            for c, name in enumerate(CLASS_NAMES):
                # Build tasks for this class phase (using current best
                # of earlier classes, 0.5 for later classes).
                tasks = []
                for t in THRESHOLD_GRIDS[c]:
                    trial = thresholds.copy()
                    trial[c] = float(t)
                    tasks.append((c, float(t), trial))

                best_f1, best_t = -1.0, thresholds[c]
                futures = [pool.submit(_grid_eval_task, task)
                           for task in tasks]
                for fut in as_completed(futures):
                    _, t_val, f1 = fut.result()
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t_val
                thresholds[c] = best_t
    finally:
        # Unbind so the probs dict can be garbage-collected when the
        # caller releases its reference. Important if validate() is
        # called many times during training.
        _WORKER_PROBS = None
        _WORKER_GT = None

    return thresholds


def evaluate_with_thresholds(
    probs: dict[tuple, np.ndarray],
    gt_events: Sequence[Detection],
    thresholds: np.ndarray | Sequence[float],
) -> dict:
    """
    Postprocess at the given thresholds and compute event-level metrics.

    Thin wrapper around ``postprocess_predictions`` + ``compute_metrics``
    so callers don't have to import both modules just to do a one-line
    final eval. Same IoU=0.3 matching as the rest of the pipeline.

    Returns the standard metrics dict (overall + per-class breakdown).
    """
    thr = np.asarray(thresholds, dtype=np.float64)
    preds = postprocess_predictions(probs, thr)
    return compute_metrics(preds, gt_events, iou_threshold=IOU_THRESHOLD)


def macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    """
    Mean per-class F1 across ``cfg.CALL_TYPES_3``.

    Missing classes contribute 0 (matches ``compute_metrics`` behaviour
    when neither predictions nor GT exist for a class). This is the
    official BioDCASE Task 2 metric.
    """
    f1s = [metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES]
    return float(np.mean(f1s))


def macro_f1_paper(metrics: dict[str, dict[str, float]]) -> float:
    """
    Paper-convention macro-F1: F1 computed from mean-precision and
    mean-recall across ``cfg.CALL_TYPES_3``.

    This is the headline metric reported in Geldenhuys et al. (DCASE 2025)
    Whale-VAD (the 0.440 reference number). Differs from ``macro_f1`` —
    that one averages per-class F1s, this one averages P and R separately
    then takes F1 of the means. Missing classes contribute 0 to both.
    """
    precisions = [metrics.get(c, {}).get("precision", 0.0) for c in CLASS_NAMES]
    recalls = [metrics.get(c, {}).get("recall", 0.0) for c in CLASS_NAMES]
    p_bar = float(np.mean(precisions))
    r_bar = float(np.mean(recalls))
    return 2.0 * p_bar * r_bar / (p_bar + r_bar + 1e-8)


def tune_and_evaluate(
    probs: dict[tuple, np.ndarray],
    gt_events: Sequence[Detection],
    start: np.ndarray | Sequence[float] | None = None,
    parallel_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Convenience wrapper: tune thresholds, then evaluate at the tuned
    point in one call. Returns ``(thresholds, metrics)``.

    Most validation paths want both, and computing the metrics dict
    requires one extra ``postprocess_predictions`` + ``compute_metrics``
    pass at the end (the tuner only needs per-class F1 internally).
    """
    thresholds = tune_thresholds_per_class(
        probs, gt_events, start=start, parallel_workers=parallel_workers,
    )
    metrics = evaluate_with_thresholds(probs, gt_events, thresholds)
    return thresholds, metrics


__all__ = [
    # Constants
    "CLASS_NAMES",
    "BMABZ_THRESHOLD_GRID",
    "D_THRESHOLD_GRID",
    "BP_THRESHOLD_GRID",
    "THRESHOLD_GRIDS",
    "IOU_THRESHOLD",
    # Functions
    "tune_thresholds_per_class",
    "evaluate_with_thresholds",
    "macro_f1",
    "macro_f1_paper",
    "tune_and_evaluate",
]
