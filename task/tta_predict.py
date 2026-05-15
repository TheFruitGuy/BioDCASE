"""
tta_predict.py
==============

Test-time augmentation wrapper around the cached ensemble inference pipeline.
Two TTA axes are supported.

**Multi-stride TTA** -- re-tile the validation set at multiple stride values
(e.g. 60s, 30s, 15s strides over 60s windows). Each frame ends up covered by
multiple overlapping windows with different temporal contexts. The existing
``stitch_segments`` function natively averages over overlapping windows, so
multi-stride plugs in cleanly: we just add windows at additional start
positions and let stitching do the work. This is the novel angle -- most
SED systems do single-stride inference and a call sitting at the edge of one
window sits in the middle of another in a different tiling.

**Audio gain TTA** -- scale the waveform by multiple constant factors before
the spectrogram. Because the project's spectrogram extractor uses complex
demeaning (``stft - stft.mean(dim=time, keepdim=True)``), real-positive gain
scales magnitude linearly but leaves the cos/sin phase channels unchanged
(``arg(g*x) == arg(x)`` for ``g > 0``). Only one of the three input
channels is perturbed, so the expected gain-TTA win is modest. Kept in as a
free additional axis; the headline is multi-stride.

Pipeline
--------
For each checkpoint, we run inference once per (stride, gain) variant and
cache the result on disk keyed by ``(ckpt, stride, gain)``. Subsequent runs
that include the same variant skip inference entirely.

Combining the variants:

1. Per checkpoint, collect a list of per-variant probability dicts.
2. Merge into a single dict with **disambiguated 4-tuple keys**:
   ``(dataset, filename, start_sample, variant_id)``. This is necessary
   because two strides may produce a window starting at the same sample
   (e.g. both stride=60 and stride=30 produce a window at t=0).
3. Pass the merged dict through a *local* stitcher that strips the variant
   id when grouping by file. The accumulator/counter logic inside the
   stitcher then does multi-variant averaging in a single sweep, exactly
   the same way it averages overlapping single-stride windows.
4. Across checkpoints, the per-checkpoint dicts share the same variant ids
   (everyone ran the same TTA spec) so ``average_prob_dicts`` works
   unchanged on the 4-tuple keys.

Threshold tuning and event-level evaluation use local wrappers around the
project's existing post-processing that accept 4-tuple keys.

Usage
-----
::

    # Multi-stride TTA only (the headline; ~Nstride * baseline cost):
    python tta_predict.py \\
        --checkpoints runs/hnm_D_*/best_model.pt \\
        --strides 60 30 15 \\
        --gains 1.0 \\
        --per_model_eval

    # Full TTA (multi-stride + gain; ~Nstride * Ngain * baseline):
    python tta_predict.py \\
        --checkpoints runs/hnm_D_*/best_model.pt \\
        --strides 60 30 15 \\
        --gains 1.0 0.7 1.4 \\
        --per_model_eval

    # Same as above but no caching (always recompute):
    python tta_predict.py \\
        --checkpoints runs/hnm_D_*/best_model.pt \\
        --strides 60 30 \\
        --gains 1.0 \\
        --no_cache

Notes
-----
- Stride values are in seconds and refer to the *step* between consecutive
  window start positions, not the overlap. With a 60s window, stride=60
  means non-overlapping windows, stride=30 means 50% overlap, stride=15
  means 75% overlap. The smallest practical stride is governed by the cost
  budget; stride=15 produces ~4x the windows of stride=60.
- The eval segment length is read from ``cfg.EVAL_SEGMENT_S`` (60s in your
  current config). Strides are converted to ``overlap_s = segment_s - stride``
  before being passed to ``build_val_segments``.
- Gain TTA with the identity factor (1.0) is the "no-perturbation"
  variant. Including it in the gains list is recommended -- it gives the
  averaging step a clean reference prediction. If you only want pure
  multi-stride, use ``--gains 1.0`` (single element).
- Caching uses fp16 on disk (~50% size reduction); the per-variant cost
  is roughly ``(60 / stride)`` times the baseline inference cost.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, collapse_probs_to_3class, compute_metrics,
    smooth_probabilities, threshold_to_detections, merge_and_filter,
)
from spectrogram import SpectrogramExtractor

from ensemble_predict import build_model_for_ckpt, average_prob_dicts


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to .pt checkpoint files. Same auto-detect as "
                        "ensemble_predict.py (baseline vs BPN).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint weights. Same semantics as "
                        "ensemble_predict.py.")
    p.add_argument("--strides", nargs="+", type=float, default=None,
                   help="Stride values in seconds (default: a single stride "
                        "matching the existing pipeline, i.e. "
                        "EVAL_SEGMENT_S - EVAL_OVERLAP_S = no TTA). "
                        "For real TTA, the cleanest single recipe is a fine "
                        "stride like 15 (covers every 15s, ~4 views/frame). "
                        "Multi-value strides are useful when the values are "
                        "coprime (e.g. 29 15) to avoid redundant tiles; "
                        "non-coprime mixes like [58 30 15] have ~30%% "
                        "redundant compute (stride=15 already covers "
                        "stride=30 starts).")
    p.add_argument("--gains", nargs="+", type=float, default=[1.0],
                   help="Gain factors to apply to audio (default: [1.0], i.e. "
                        "no gain TTA). Common useful values: 1.0 0.7 1.4 "
                        "(approximately +/- 3 dB amplitude).")
    p.add_argument("--per_model_eval", action="store_true",
                   help="Also score each checkpoint individually for reference. "
                        "Costs ~5 s per model.")
    p.add_argument("--cache_dir", type=str, default="runs/prob_cache_tta",
                   help="Directory for per-(checkpoint,variant) cached probs.")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable caching (always recompute).")
    p.add_argument("--use_fp16", action="store_true",
                   help="FP16 autocast during inference. ~1.5-2x speedup.")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    return p.parse_args()


# ======================================================================
# Variant id and cache paths
# ======================================================================

# Module-level registry mapping (stride, gain) -> integer id. Stable
# across the run so the disambiguated keys are consistent.
_VARIANT_REGISTRY: dict[tuple[float, float], int] = {}


def variant_id(stride_s: float, gain: float) -> int:
    """Return a stable integer id for a (stride, gain) variant."""
    key = (round(float(stride_s), 4), round(float(gain), 4))
    if key not in _VARIANT_REGISTRY:
        _VARIANT_REGISTRY[key] = len(_VARIANT_REGISTRY)
    return _VARIANT_REGISTRY[key]


def variant_tag(stride_s: float, gain: float) -> str:
    """Human-readable tag for logging and cache filenames."""
    return f"stride{stride_s:g}_gain{gain:.2f}"


def cache_path_for(
    ckpt_path: Path, stride_s: float, gain: float, cache_dir: Path,
) -> Path:
    tag = variant_tag(stride_s, gain)
    return cache_dir / f"{ckpt_path.parent.name}__{tag}.pkl"


def load_cached_probs(path: Path) -> dict[tuple, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {k: v.astype(np.float32) for k, v in d.items()}


def save_probs_to_cache(probs: dict[tuple, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: v.astype(np.float16) for k, v in probs.items()}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


# ======================================================================
# Inference with gain perturbation
# ======================================================================

@torch.no_grad()
def predict_with_gain(
    model: torch.nn.Module, model_type: str,
    spec_extractor: SpectrogramExtractor,
    val_loader: DataLoader, device: torch.device,
    gain: float,
    use_fp16: bool = False,
) -> dict[tuple, np.ndarray]:
    """
    Single-pass inference over the val loader with an audio gain factor.

    Identical to ``ensemble_predict.predict_probabilities`` except the audio
    is multiplied by ``gain`` before the spectrogram extractor. With the
    project's complex-demean spectrogram this perturbs only the magnitude
    channel; phase channels are unchanged.
    """
    model.eval()
    out_probs: dict[tuple, np.ndarray] = {}
    hop = spec_extractor.hop_length

    # FP16 autocast on CUDA when enabled; identity context otherwise.
    if use_fp16 and device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = _NullCtx()

    for audio, _, _, metas in tqdm(val_loader, desc=f"    gain={gain:.2f}",
                                    leave=False):
        audio = audio.to(device, non_blocking=True)
        if gain != 1.0:
            audio = audio * gain

        with autocast_ctx:
            spec = spec_extractor(audio)
            out = model(spec)

        if model_type == "bpn":
            probs = out["probs"]
        else:
            probs = torch.sigmoid(out)

        probs_np = probs.detach().float().cpu().numpy()

        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs_np[j].shape[0])
            out_probs[key] = probs_np[j, :n_frames, :]

    return out_probs


class _NullCtx:
    """Trivial context manager for the CPU path."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ======================================================================
# Per-checkpoint, per-variant runner with caching
# ======================================================================

def build_val_loader_for_stride(
    stride_s: float,
    val_manifest,
    val_anns,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, int]:
    """
    Build a fresh val loader tiled at ``stride_s`` seconds.

    Returns (loader, n_segments). ``stride_s`` is the step between
    consecutive window starts; ``overlap_s`` for ``build_val_segments`` is
    derived as ``EVAL_SEGMENT_S - stride_s`` and is clamped at 0 below.
    """
    segment_s = float(cfg.EVAL_SEGMENT_S)
    overlap_s = max(0.0, segment_s - stride_s)
    val_segs = build_val_segments(
        val_manifest, val_anns,
        segment_s=segment_s, overlap_s=overlap_s,
    )
    loader = DataLoader(
        WhaleDataset(val_segs), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    return loader, len(val_segs)


@torch.no_grad()
def get_or_compute_variant(
    ckpt_path: str, stride_s: float, gain: float,
    val_manifest, val_anns,
    spec_extractor: SpectrogramExtractor, device: torch.device,
    cache_dir: Path, no_cache: bool, use_fp16: bool,
    batch_size: int, num_workers: int,
    cached_model_state: dict | None = None,
) -> tuple[dict[tuple, np.ndarray], dict | None]:
    """
    Get (compute or load from cache) the probs_dict for a single
    (checkpoint, stride, gain) variant.

    To avoid reloading the model state from disk for every variant of the
    same checkpoint, callers may pass in a ``cached_model_state`` dict
    that this function fills on first use and reuses on subsequent calls.
    """
    cp = cache_path_for(Path(ckpt_path), stride_s, gain, cache_dir)
    if not no_cache and cp.exists():
        print(f"    cache hit: {cp.name}")
        probs = load_cached_probs(cp)
        return probs, cached_model_state

    print(f"    cache miss: running inference for stride={stride_s:g}, "
          f"gain={gain:.2f}")
    t0 = time.time()

    # Lazy model setup: build once per checkpoint, reuse across variants.
    if cached_model_state is None or cached_model_state.get("path") != ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_type = build_model_for_ckpt(ckpt, device)
        print(f"    loaded checkpoint (type={model_type})")
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            _ = model(spec_extractor(dummy))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        cached_model_state = {"path": ckpt_path, "model": model, "type": model_type}
    else:
        model = cached_model_state["model"]
        model_type = cached_model_state["type"]

    # Build a stride-specific loader. The loader is cheap to construct;
    # the heavy work is in the dataset workers doing soundfile reads.
    val_loader, n_segs = build_val_loader_for_stride(
        stride_s, val_manifest, val_anns, batch_size, num_workers,
    )
    print(f"    val tiling at stride={stride_s:g}: {n_segs} windows")

    probs = predict_with_gain(
        model, model_type, spec_extractor, val_loader, device,
        gain=gain, use_fp16=use_fp16,
    )
    probs = collapse_probs_to_3class(probs)

    elapsed = time.time() - t0
    print(f"    computed in {elapsed:.0f}s, {len(probs)} segment arrays")

    if not no_cache:
        save_probs_to_cache(probs, cp)
        print(f"    cached -> {cp}")

    return probs, cached_model_state


# ======================================================================
# TTA merging with disambiguated keys
# ======================================================================

def merge_tta_variants(
    variant_probs: list[tuple[float, float, dict[tuple, np.ndarray]]],
) -> dict[tuple, np.ndarray]:
    """
    Merge per-variant probability dicts into a single dict keyed by
    4-tuples ``(dataset, filename, start_sample, variant_id)``.

    The variant id ensures that windows starting at the same sample in
    different stride or gain variants don't collide. The downstream local
    stitcher strips the variant id when grouping by file.

    Parameters
    ----------
    variant_probs : list of (stride_s, gain, probs_dict)
        One entry per TTA variant.

    Returns
    -------
    dict
        Keys are 4-tuples; values are the same arrays as the inputs.
    """
    merged: dict[tuple, np.ndarray] = {}
    for stride_s, gain, pd in variant_probs:
        vid = variant_id(stride_s, gain)
        for (ds, fn, start_samp), probs in pd.items():
            merged[(ds, fn, start_samp, vid)] = probs
    return merged


# ======================================================================
# Local post-processing that handles 4-tuple keys
# ======================================================================

def stitch_segments_tta(
    all_probs: dict[tuple, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """
    Same accumulator-based stitching as ``postprocess.stitch_segments`` but
    accepts both 3-tuple ``(ds, fn, start)`` and 4-tuple ``(ds, fn, start,
    variant)`` keys. The variant component is stripped during grouping so
    duplicates across TTA variants all contribute to the same per-frame
    accumulator. The net effect is per-frame averaging across all
    overlapping windows of all variants in a single sweep.
    """
    stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)

    file_segs: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = \
        defaultdict(list)
    for key, probs in all_probs.items():
        if len(key) == 3:
            ds, fn, start_samp = key
        elif len(key) == 4:
            ds, fn, start_samp, _ = key
        else:
            raise ValueError(f"Unexpected key length: {key}")
        file_segs[(ds, fn)].append((int(start_samp), probs))

    result: dict[tuple[str, str], np.ndarray] = {}
    for key, segs in file_segs.items():
        segs.sort(key=lambda x: x[0])
        max_end = max(s + p.shape[0] * stride_samp for s, p in segs)
        total_frames = max_end // stride_samp + 1
        nc = segs[0][1].shape[1]

        accum = np.zeros((total_frames, nc), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)
        for start_samp, probs in segs:
            f0 = start_samp // stride_samp
            T = min(probs.shape[0], total_frames - f0)
            accum[f0:f0 + T] += probs[:T]
            counts[f0:f0 + T] += 1

        counts = np.maximum(counts, 1)
        result[key] = (accum / counts[:, None]).astype(np.float32)

    return result


def postprocess_tta(
    all_probs: dict[tuple, np.ndarray],
    thresholds: np.ndarray,
) -> list[Detection]:
    """
    Drop-in replacement for ``postprocess.postprocess_predictions`` that
    routes through ``stitch_segments_tta``.
    """
    file_probs = stitch_segments_tta(all_probs)
    all_dets: list[Detection] = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter(all_dets)


def tune_thresholds_tta(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
) -> np.ndarray:
    """
    Per-class coordinate-descent sweep against the TTA-stitched predictions.
    Same grids as ``ensemble_predict.tune_thresholds_on_probs``.
    """
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    grids = [
        np.arange(0.20, 0.85, 0.05),                                # bmabz
        np.concatenate([np.arange(0.05, 0.50, 0.05),
                        np.arange(0.50, 0.85, 0.10)]),               # d
        np.concatenate([np.arange(0.05, 0.50, 0.05),
                        np.arange(0.50, 0.85, 0.10)]),               # bp
    ]
    for c, name in enumerate(cfg.CALL_TYPES_3):
        best_f1, best_t = -1.0, float(thresholds[c])
        for t in grids[c]:
            trial = thresholds.copy()
            trial[c] = float(t)
            preds = postprocess_tta(all_probs, trial)
            m = compute_metrics(preds, gt_events, iou_threshold=0.3)
            f1 = m.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
        print(f"    {name:<6}  best t={best_t:.2f}  f1={best_f1:.3f}")
    return thresholds


def evaluate_with_thresholds_tta(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
    thresholds: np.ndarray,
) -> dict[str, dict[str, float]]:
    preds = postprocess_tta(all_probs, thresholds)
    return compute_metrics(preds, gt_events, iou_threshold=0.3)


# ======================================================================
# Reporting
# ======================================================================

def per_class_print(metrics, thresholds, label="ENSEMBLE"):
    print(f"\n  {label} RESULT:")
    for c in cfg.CALL_TYPES_3:
        m = metrics.get(c, {})
        idx = cfg.CALL_TYPES_3.index(c)
        print(f"    {c.upper():6} t={thresholds[idx]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):5} "
              f"FN={m.get('fn', 0):5}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    overall = metrics.get("overall", {})
    macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                           for c in cfg.CALL_TYPES_3]))
    print(f"    OVERALL F1={overall.get('f1', 0):.3f}  "
          f"MACRO F1={macro:.3f}")
    print(f"    Tuned thresholds: "
          f"{['%.2f' % t for t in thresholds]}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir} (no_cache={args.no_cache}, "
          f"use_fp16={args.use_fp16})")

    strides = (list(args.strides) if args.strides is not None
               else [float(cfg.EVAL_SEGMENT_S) - float(cfg.EVAL_OVERLAP_S)])
    gains = list(args.gains)
    print(f"\nTTA spec:")
    print(f"  strides (s): {strides}")
    print(f"  gains:       {gains}")
    print(f"  total variants per checkpoint: {len(strides) * len(gains)}")
    print(f"  total inference passes:       "
          f"{len(strides) * len(gains) * len(args.checkpoints)}")

    if args.weights is not None:
        assert len(args.weights) == len(args.checkpoints), \
            "len(weights) must equal len(checkpoints)"
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
        print(f"\nPer-checkpoint weights (normalized): "
              f"{['%.3f' % w for w in weights]}")
    else:
        weights = None

    # ----------------------------------------------------------------------
    # Shared val manifest + annotations (loaders are built per-stride lazily)
    # ----------------------------------------------------------------------
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    print(f"  {len(val_manifest)} files, {len(val_anns)} annotations")

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    gt_events: list[Detection] = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"  {len(gt_events)} ground-truth events")

    spec_extractor = SpectrogramExtractor().to(device)

    # ----------------------------------------------------------------------
    # Pre-register all (stride, gain) variants in deterministic order so
    # variant ids match across checkpoints.
    # ----------------------------------------------------------------------
    for s in strides:
        for g in gains:
            variant_id(s, g)
    print(f"\nVariant registry ({len(_VARIANT_REGISTRY)} variants):")
    for (s, g), vid in sorted(_VARIANT_REGISTRY.items(),
                              key=lambda kv: kv[1]):
        print(f"  vid={vid}: stride={s:g}, gain={g:.2f}")

    # ----------------------------------------------------------------------
    # Per checkpoint: compute all (stride, gain) variants, merge with
    # disambiguated keys, optionally score the per-ckpt TTA result.
    # ----------------------------------------------------------------------
    per_ckpt_merged: list[dict[tuple, np.ndarray]] = []

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        variant_probs: list[tuple[float, float, dict]] = []
        model_state: dict | None = None

        for s in strides:
            for g in gains:
                probs, model_state = get_or_compute_variant(
                    ckpt_path, s, g, val_manifest, val_anns,
                    spec_extractor, device,
                    cache_dir, args.no_cache, args.use_fp16,
                    args.batch_size, args.num_workers,
                    cached_model_state=model_state,
                )
                variant_probs.append((s, g, probs))

        # Free the model right after this checkpoint's variants are done.
        if model_state is not None and "model" in model_state:
            del model_state["model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        merged = merge_tta_variants(variant_probs)
        print(f"  merged probs across {len(variant_probs)} variants -> "
              f"{len(merged)} disambiguated entries")
        per_ckpt_merged.append(merged)

        if args.per_model_eval:
            print(f"  per-ckpt TTA threshold tune:")
            thr = tune_thresholds_tta(merged, gt_events)
            m = evaluate_with_thresholds_tta(merged, gt_events, thr)
            macro = float(np.mean([m.get(c, {}).get("f1", 0.0)
                                   for c in cfg.CALL_TYPES_3]))
            print(f"  ckpt TTA: overall F1="
                  f"{m.get('overall', {}).get('f1', 0):.3f}, "
                  f"macro={macro:.3f}, thr={['%.2f' % t for t in thr]}")
            for c in cfg.CALL_TYPES_3:
                cm = m.get(c, {})
                print(f"    {c.upper():6} F1={cm.get('f1', 0):.3f} "
                      f"P={cm.get('precision', 0):.3f} "
                      f"R={cm.get('recall', 0):.3f}")

    # ----------------------------------------------------------------------
    # Ensemble across checkpoints (keys already disambiguated and identical
    # across ckpts because the same TTA spec ran for all). The standard
    # average_prob_dicts works unchanged on 4-tuple keys.
    # ----------------------------------------------------------------------
    print(f"\n{'='*64}\nTTA ENSEMBLE ({len(per_ckpt_merged)} checkpoints)\n"
          f"{'='*64}")
    ens_probs = (per_ckpt_merged[0] if len(per_ckpt_merged) == 1
                 else average_prob_dicts(per_ckpt_merged, weights=weights))
    print(f"Averaged across checkpoints: {len(ens_probs)} entries")

    print("\nTuning thresholds on TTA ensemble probabilities...")
    ens_thr = tune_thresholds_tta(ens_probs, gt_events)
    metrics = evaluate_with_thresholds_tta(ens_probs, gt_events, ens_thr)
    per_class_print(metrics, ens_thr, label="TTA ENSEMBLE")


if __name__ == "__main__":
    main()
