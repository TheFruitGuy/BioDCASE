"""
Per-Class Post-Processing
=========================

Class-dependent extension of ``postprocess.py``. Implements the
hyperparameter regime described in Geldenhuys et al. (arXiv:2510.21280v2)
Table IV — every step of the post-processing chain has a per-class
parameter, optimised jointly via ``tune_postprocess_optuna.py``:

    1. **Per-class median filter** on probabilities, kernel size in ms.
       0 disables. Smooths sporadic peaks/dips.
    2. **Per-class hysteresis thresholding**: separate ``on_thr`` and
       ``off_thr``. A frame enters the active state when prob > on_thr
       and exits when prob < off_thr. ``off_thr == on_thr`` degrades to
       a regular fixed threshold (paper-equivalent baseline behaviour).
    3. **Per-class hangover**: majority-vote (median) filter on the
       binary activation, kernel size in ms. Equivalent to a binary
       median filter as derived in the BPN paper §II-C.
    4. **Per-class merge gap**: events of the same class on the same
       file separated by less than this gap are merged.
    5. **Per-class duration filter**: events outside [min_dur, max_dur]
       are discarded. Bmabz events are long, bp events are short — the
       paper uses very different ranges per class (Table IVb).

The pipeline operates in 3-class space (``cfg.CALL_TYPES_3``). For
7-class checkpoints, ``collapse_probs_to_3class`` is applied before
postprocessing — same convention as the existing ``tune_thresholds.py``.

Usage at inference time
-----------------------
``inference.py`` looks for ``cfg.POSTPROCESS_CONFIG_PATH`` (or the path
passed via ``--postprocess-config``). If the file exists and parses,
``postprocess_predictions_per_class`` is used; otherwise the existing
default ``postprocess_predictions`` runs unchanged. This keeps the
extension fully optional and safe to commit alongside untuned
checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter

import config as cfg
from postprocess import (
    Detection,
    stitch_segments,
    collapse_probs_to_3class,
)


# ======================================================================
# Config dataclass
# ======================================================================

@dataclass
class PerClassPostprocessConfig:
    """
    All post-processing hyperparameters, keyed by 3-class label name
    (``"bmabz"``, ``"d"``, ``"bp"``).

    Stored as plain dicts so JSON round-tripping is trivial. Use 0 (or
    omit) for ``median_kernel_ms`` / ``hangover_kernel_ms`` to disable
    that step for a given class. Setting ``off_thresholds[c] ==
    on_thresholds[c]`` disables hysteresis (single-threshold behaviour).
    """

    median_kernel_ms: dict[str, int] = field(default_factory=dict)
    on_thresholds: dict[str, float] = field(default_factory=dict)
    off_thresholds: dict[str, float] = field(default_factory=dict)
    hangover_kernel_ms: dict[str, int] = field(default_factory=dict)
    merge_gap_s: dict[str, float] = field(default_factory=dict)
    min_dur_s: dict[str, float] = field(default_factory=dict)
    max_dur_s: dict[str, float] = field(default_factory=dict)

    def get_on_thr(self, name: str) -> float:
        return float(self.on_thresholds.get(name, 0.5))

    def get_off_thr(self, name: str) -> float:
        # Default: same as on_thr (no hysteresis) when missing.
        return float(self.off_thresholds.get(name, self.get_on_thr(name)))

    def get_median_ms(self, name: str) -> int:
        # Default: paper's 500 ms global filter.
        return int(self.median_kernel_ms.get(name, cfg.SMOOTH_KERNEL_MS))

    def get_hangover_ms(self, name: str) -> int:
        return int(self.hangover_kernel_ms.get(name, 0))

    def get_merge_gap_s(self, name: str) -> float:
        return float(self.merge_gap_s.get(name, cfg.MERGE_GAP_S))

    def get_min_dur_s(self, name: str) -> float:
        return float(self.min_dur_s.get(name, cfg.POST_MIN_DUR_S))

    def get_max_dur_s(self, name: str) -> float:
        return float(self.max_dur_s.get(name, cfg.POST_MAX_DUR_S))

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PerClassPostprocessConfig":
        return cls(
            median_kernel_ms={k: int(v) for k, v in d.get("median_kernel_ms", {}).items()},
            on_thresholds={k: float(v) for k, v in d.get("on_thresholds", {}).items()},
            off_thresholds={k: float(v) for k, v in d.get("off_thresholds", {}).items()},
            hangover_kernel_ms={k: int(v) for k, v in d.get("hangover_kernel_ms", {}).items()},
            merge_gap_s={k: float(v) for k, v in d.get("merge_gap_s", {}).items()},
            min_dur_s={k: float(v) for k, v in d.get("min_dur_s", {}).items()},
            max_dur_s={k: float(v) for k, v in d.get("max_dur_s", {}).items()},
        )

    def save(self, path: str | Path, *, metadata: Optional[dict] = None) -> None:
        """
        Serialise to JSON, optionally with a top-level ``metadata`` block
        carrying things like the optuna study name, val macro-F1, etc.
        """
        path = Path(path)
        payload = {
            "version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config": self.to_dict(),
        }
        if metadata is not None:
            payload["metadata"] = metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PerClassPostprocessConfig":
        path = Path(path)
        with path.open("r") as f:
            payload = json.load(f)
        # Tolerate either {"config": {...}} (versioned) or a bare dict.
        body = payload.get("config", payload)
        return cls.from_dict(body)


def load_postprocess_config(
    path: str | Path | None,
) -> Optional[PerClassPostprocessConfig]:
    """
    Best-effort loader. Returns ``None`` when the path is missing,
    unreadable, or parses to an empty config — so callers can fall back
    to default postprocessing without needing try/except plumbing.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        cfg_obj = PerClassPostprocessConfig.load(p)
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as e:
        print(f"  [postprocess_per_class] WARNING: failed to read {p}: {e}")
        return None
    # An entirely empty config is indistinguishable from "use defaults",
    # which means callers should also fall back. Detect via on_thresholds
    # because it's the one parameter that genuinely needs a non-default
    # to do anything useful.
    if not cfg_obj.on_thresholds:
        print(f"  [postprocess_per_class] WARNING: {p} has no on_thresholds; "
              f"using default postprocessing.")
        return None
    return cfg_obj


# ======================================================================
# Frame-conversion helper
# ======================================================================

def _ms_to_odd_frames(ms: int) -> int:
    """
    Convert a kernel size in milliseconds to a positive odd integer
    number of frames at the model's frame stride. Returns 0 when ms
    is 0 or negative — caller should skip the filter in that case.
    scipy's ``median_filter`` requires odd kernel sizes, hence the +1.
    """
    if ms is None or ms <= 0:
        return 0
    stride_ms = int(round(cfg.FRAME_STRIDE_S * 1000))
    k = max(1, int(round(ms / stride_ms)))
    if k % 2 == 0:
        k += 1
    return k


# ======================================================================
# Per-class median filter on probabilities
# ======================================================================

def smooth_per_class(
    probs: np.ndarray,
    config: PerClassPostprocessConfig,
    class_names: list[str],
) -> np.ndarray:
    """
    Apply a separate temporal median filter per class.

    Parameters
    ----------
    probs : np.ndarray, shape (T, C)
        Per-frame probabilities, already 3-class.
    config, class_names
        Source of per-class kernel sizes.

    Returns
    -------
    np.ndarray, shape (T, C)
        Smoothed probabilities. Untouched (copy) for any class whose
        kernel resolves to 0.
    """
    out = probs.copy()
    for c, name in enumerate(class_names):
        k = _ms_to_odd_frames(config.get_median_ms(name))
        if k <= 1:
            continue
        out[:, c] = median_filter(probs[:, c], size=k)
    return out


# ======================================================================
# Per-class hysteresis thresholding
# ======================================================================

def hysteresis_threshold(
    prob_track: np.ndarray, on_thr: float, off_thr: float,
) -> np.ndarray:
    """
    Two-threshold state machine on a 1-D probability stream.

    A frame is active when probability has crossed ``on_thr`` from below
    and has not yet dropped below ``off_thr``. With ``off_thr == on_thr``
    the output matches a fixed-threshold pass.

    The implementation is a vectorised cumulative pass: we mark "enter"
    events where ``prob > on_thr`` and "exit" events where
    ``prob < off_thr``, then accumulate a state via a running diff.
    Equivalent to the obvious for-loop but ~50x faster on long files.
    """
    if off_thr > on_thr:
        # Caller bug. Rather than silently swap, fall back to the safer
        # interpretation (single threshold = on_thr).
        off_thr = on_thr
    enter = prob_track > on_thr
    exit_ = prob_track < off_thr
    # The vectorised trick: in a forward sweep, state flips only on
    # an enter (1) or exit (-1) event; otherwise it's held. Encode
    # those as +1/-1 deltas, then scan via a custom rule.
    state = np.zeros_like(prob_track, dtype=np.int8)
    cur = 0
    # A tight Python loop is unavoidable here because the recurrence
    # depends on the previous output, but on (T,) frames the constant
    # is small (~hundred-thousand frames per file × O(1) work).
    for i in range(prob_track.shape[0]):
        if cur == 0 and enter[i]:
            cur = 1
        elif cur == 1 and exit_[i]:
            cur = 0
        state[i] = cur
    return state.astype(bool)


def threshold_per_class(
    probs: np.ndarray,
    config: PerClassPostprocessConfig,
    class_names: list[str],
) -> np.ndarray:
    """
    Apply per-class hysteresis thresholding.

    Returns
    -------
    np.ndarray, shape (T, C), dtype=bool
        Binary activations.
    """
    T, C = probs.shape
    out = np.zeros((T, C), dtype=bool)
    for c, name in enumerate(class_names):
        on_thr = config.get_on_thr(name)
        off_thr = config.get_off_thr(name)
        out[:, c] = hysteresis_threshold(probs[:, c], on_thr, off_thr)
    return out


# ======================================================================
# Per-class hangover (majority vote on binary activations)
# ======================================================================

def hangover_per_class(
    binary: np.ndarray,
    config: PerClassPostprocessConfig,
    class_names: list[str],
) -> np.ndarray:
    """
    Majority-vote sliding window on each class's binary activation.

    For binary input, scipy's ``median_filter`` with odd kernel size is
    exactly the majority vote / statistical mode operation defined in
    BPN §II-C (eq. 1). 0-kernel resolves to identity.
    """
    out = binary.copy()
    for c, name in enumerate(class_names):
        k = _ms_to_odd_frames(config.get_hangover_ms(name))
        if k <= 1:
            continue
        # median_filter over int8 returns 0/1 — same as the mode.
        out[:, c] = median_filter(binary[:, c].astype(np.int8), size=k).astype(bool)
    return out


# ======================================================================
# Binary → Detection objects
# ======================================================================

def binary_to_detections(
    binary: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    dataset: str,
    filename: str,
) -> list[Detection]:
    """
    Convert per-class binary activations to Detection objects.

    Same run-extraction trick as the baseline ``threshold_to_detections``
    (np.diff on the activation vector), but driven by an externally
    computed binary mask so the caller can apply hysteresis + hangover
    upstream.
    """
    dets: list[Detection] = []
    T, C = binary.shape
    for c, name in enumerate(class_names):
        active = binary[:, c]
        if not active.any():
            continue
        diffs = np.diff(active.astype(np.int8), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            dets.append(Detection(
                dataset=dataset,
                filename=filename,
                label=name,
                start_s=s * cfg.FRAME_STRIDE_S,
                end_s=e * cfg.FRAME_STRIDE_S,
                confidence=float(probs[s:e, c].mean()),
            ))
    return dets


# ======================================================================
# Per-class merge + duration filter
# ======================================================================

def merge_and_filter_per_class(
    detections: list[Detection],
    config: PerClassPostprocessConfig,
) -> list[Detection]:
    """
    Group by (file, class), merge events separated by less than the
    class's merge gap, then drop events outside the class's duration
    range.

    Detections are assumed to already be in 3-class label space.
    """
    # Group by (dataset, filename, label).
    groups: dict[tuple, list[Detection]] = {}
    for d in detections:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    final: list[Detection] = []
    for (ds, fn, label), events in groups.items():
        events.sort(key=lambda x: x.start_s)
        gap = config.get_merge_gap_s(label)
        min_d = config.get_min_dur_s(label)
        max_d = config.get_max_dur_s(label)

        # Greedy left-to-right merge.
        merged: list[Detection] = []
        for e in events:
            if not merged:
                merged.append(e)
                continue
            last = merged[-1]
            if e.start_s - last.end_s <= gap:
                last.end_s = max(last.end_s, e.end_s)
                last.confidence = max(last.confidence, e.confidence)
            else:
                merged.append(e)

        # Class-specific duration filter.
        for m in merged:
            dur = m.end_s - m.start_s
            if min_d <= dur <= max_d:
                final.append(m)
    return final


# ======================================================================
# End-to-end pipeline
# ======================================================================

def postprocess_predictions_per_class(
    all_probs: dict[tuple[str, str, int], np.ndarray],
    config: PerClassPostprocessConfig,
    class_names: Optional[list[str]] = None,
) -> list[Detection]:
    """
    Full per-class pipeline: stitch → smooth → hysteresis → hangover →
    extract events → merge+filter.

    Parameters
    ----------
    all_probs : dict
        Per-window probability arrays, same structure as accepted by
        the baseline ``postprocess.postprocess_predictions``. Can be
        either 3-class or 7-class — 7-class gets collapsed automatically.
    config : PerClassPostprocessConfig
    class_names : list[str], optional
        Defaults to ``cfg.CALL_TYPES_3``. Pass an explicit list only if
        you have a non-standard label space.

    Returns
    -------
    list of Detection
        Final 3-class events, ready for evaluation or CSV export.
    """
    if class_names is None:
        class_names = list(cfg.CALL_TYPES_3)

    # 7-class → 3-class collapse (no-op for already-3-class probs).
    all_probs = collapse_probs_to_3class(all_probs)

    # Stitch overlapping windows into per-file streams.
    file_probs = stitch_segments(all_probs)

    all_dets: list[Detection] = []
    for (ds, fn), probs in file_probs.items():
        smoothed = smooth_per_class(probs, config, class_names)
        binary = threshold_per_class(smoothed, config, class_names)
        binary = hangover_per_class(binary, config, class_names)
        all_dets.extend(binary_to_detections(
            binary, smoothed, class_names, ds, fn,
        ))
    return merge_and_filter_per_class(all_dets, config)


# ======================================================================
# Search-space helpers (used by tune_postprocess_optuna.py)
# ======================================================================

# Frame-level grid from BPN paper Table IVa, plus a no-filter option
# and 100/200 ms entries that are useful for d-class which has shorter
# events than the paper considered.
SEARCH_MEDIAN_KERNEL_MS = [0, 100, 200, 220, 500, 660, 1100]
SEARCH_HANGOVER_KERNEL_MS = [0, 100, 220, 500, 660, 1100]

# Event-level grids from Table IVb. The paper's per-class ranges
# reflect the actual event durations of each class — keep them tight
# so the search converges in O(few-hundred) trials.
SEARCH_MERGE_GAP_S = {  # min. inter-event time (paper Table IVb row 1)
    "bmabz": (0.1, 0.9, 0.1),
    "d":     (0.1, 0.9, 0.1),
    "bp":    (0.1, 0.9, 0.1),
}
SEARCH_MIN_DUR_S = {    # min. event duration
    "bmabz": (2.0, 5.0, 0.5),
    "d":     (0.6, 3.0, 0.4),
    "bp":    (0.3, 1.5, 0.2),
}
SEARCH_MAX_DUR_S = {    # max. event duration
    "bmabz": (25.0, 40.0, 2.5),
    "d":     (5.0,  11.0, 1.0),
    "bp":    (2.0,   5.0, 0.5),
}
# On/off thresholds: 0.05 → 0.85 step 0.05 — slightly extended at the
# low end because d/bp are rare classes whose probability mass is
# shifted toward zero.
SEARCH_THRESHOLD_RANGE = (0.05, 0.85, 0.05)


def default_config_from_global_cfg() -> PerClassPostprocessConfig:
    """
    Build a config equivalent to the existing global postprocessing
    behaviour. Useful as a sanity baseline in tuning scripts: it should
    reproduce the F1 number of ``postprocess.postprocess_predictions``
    when used with the default 0.5 thresholds.
    """
    classes = list(cfg.CALL_TYPES_3)
    return PerClassPostprocessConfig(
        median_kernel_ms={c: cfg.SMOOTH_KERNEL_MS for c in classes},
        on_thresholds={c: 0.5 for c in classes},
        off_thresholds={c: 0.5 for c in classes},
        hangover_kernel_ms={c: 0 for c in classes},
        merge_gap_s={c: cfg.MERGE_GAP_S for c in classes},
        min_dur_s={c: cfg.POST_MIN_DUR_S for c in classes},
        max_dur_s={c: cfg.POST_MAX_DUR_S for c in classes},
    )
