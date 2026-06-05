"""
ensemble_with_d_detector.py
===========================

Per-class hybrid ensemble that takes the D class from one or more
**dedicated single-class D detectors** (``train_d_detector.py`` /
``WhaleVADBPNV2`` with ``num_classes=1``), while BMABZ and BP come from
the usual 3-class ensemble.

Why a separate script
---------------------
The dedicated D detector is incompatible with ``ensemble_predict.py`` and
``hybrid_ensemble_predict.py`` for three reasons:

1. It emits a width-1 ``(T, 1)`` probability stream, not ``(T, 3)``. The
   existing averaging / hybrid_combine code assumes 3-class arrays and
   reads the D column at index 1.
2. It is a ``bpn_v2`` checkpoint (``bpn_cfg_v2`` key). The shared
   ``build_model_for_ckpt`` only recognises v1 ``bpn_cfg`` and baseline,
   so a v2 checkpoint silently degrades to a plain ``WhaleVAD`` loaded
   with ``strict=False`` (i.e. garbage weights).
3. If it was trained with a non-zero ``--highpass-hz``, the filter is
   applied to the audio inside the detector's own dataset. The ensemble
   inference path feeds raw audio to the spectrogram, so the same filter
   must be re-applied here or the detector sees out-of-distribution input.

This script handles all three, runs every model on the **same** eval
tiles (so per-segment keys line up for merging), and reuses the proven
threshold-tuning / scoring / CSV helpers from the existing scripts.

D-source modes
--------------
- ``--d-blend-frac 0.0`` (default): D is taken purely from the dedicated
  detector(s). This *replaces* whatever D source you used before.
- ``--d-blend-frac f`` with ``--d-blend-checkpoints``: D becomes
  ``(1 - f) * detector_D + f * blend_D``, where ``blend_D`` is the D
  column of a 3-class subset average (e.g. your 4 mtfb checkpoints).
  Use this to *augment* rather than replace the current D source.

Usage
-----
Verify on val (replace D with the detector, tune thresholds, score):

    CUDA_VISIBLE_DEVICES=1 python ensemble_with_d_detector.py \
        --bp-checkpoints \
            runs/hnm_3class_s2024_ens_pgi/best_model.pt \
            runs/hnm_3class_s7777_ens_pgi/best_model.pt \
            ... (your 12 hnm models) ... \
        --d-detector-checkpoints \
            runs/phase_d_detector_<ts>/final_model.pt \
        --per-model-eval

Blend the detector with your existing mtfb D source (50/50):

    CUDA_VISIBLE_DEVICES=1 python ensemble_with_d_detector.py \
        --bp-checkpoints  ... (12 hnm) ... \
        --d-detector-checkpoints runs/phase_d_detector_<ts>/final_model.pt \
        --d-blend-checkpoints \
            runs/mtfb_<a>/final_best.pt runs/mtfb_<b>/final_best.pt \
            runs/mtfb_<c>/final_best.pt runs/mtfb_<d>/final_best.pt \
        --d-blend-frac 0.5 \
        --per-model-eval

Apply to test (frozen thresholds from val, no scoring):

    CUDA_VISIBLE_DEVICES=1 python ensemble_with_d_detector.py \
        --bp-checkpoints ... --d-detector-checkpoints ... \
        --thresholds 0.27 0.20 0.37 \
        --test-datasets kerguelen2020 ddu2021 \
        --output-csv test_predictions.csv

Notes
-----
- ``--thresholds`` order is BMABZ D BP (matches ``cfg.CALL_TYPES_3``).
- ``--segment-s`` / ``--overlap-s`` set the eval tile length for *all*
  models. Default is ``cfg.EVAL_SEGMENT_S`` / ``cfg.EVAL_OVERLAP_S``.
  Your notes say 60s tiles beat 30s for the ensemble; pass
  ``--segment-s 60`` to match that operating point.
- This script does not use the prob cache; it always recomputes. Caching
  is keyed by checkpoint dir name only in your snapshot and does not
  track segment length, which would silently mix tile lengths here.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import Detection, collapse_probs_to_3class, postprocess_predictions
from spectrogram import SpectrogramExtractor
from model_bpn_v2 import BPNConfigV2, WhaleVADBPNV2

# Reuse the proven helpers rather than re-implementing them.
from ensemble_predict import (
    build_model_for_ckpt,            # baseline + v1 BPN (for BMABZ/BP + blend)
    predict_probabilities,           # 3-class inference
    average_prob_dicts,
    tune_thresholds_on_probs,
    evaluate_with_thresholds,
)
from hybrid_ensemble_predict import save_predictions_csv


D_IDX = cfg.CALL_TYPES_3.index("d")  # 1


# ----------------------------------------------------------------------
# Per-checkpoint prob cache (keyed by checkpoint dir name + tile length)
# ----------------------------------------------------------------------
# The cache key includes the segment/overlap tile length, unlike
# ensemble_predict_cached.py whose key is the dir name only. That bug
# silently mixes 30s and 60s probs; this keeps them in separate files.

def _cache_path(ckpt_path: str, cache_dir: Path, segment_s: float,
                overlap_s: float, suffix: str) -> Path:
    name = Path(ckpt_path).parent.name
    tag = f"s{int(round(segment_s))}_o{int(round(overlap_s))}"
    return cache_dir / f"{name}__{tag}__{suffix}.pkl"


def _load_cache(path: Path) -> dict[tuple, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {k: v.astype(np.float32) for k, v in d.items()}


def _save_cache(probs: dict[tuple, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: v.astype(np.float16) for k, v in probs.items()}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------
# D-detector construction + inference (the part the shared code can't do)
# ----------------------------------------------------------------------

def build_d_detector(ckpt: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    """
    Build the dedicated single-class D detector from its checkpoint.

    Supports both architectures train_d_detector.py can save:
      - ``bpn_v2`` (default): WhaleVADBPNV2, output dict with "probs".
      - ``whalevad``        : plain WhaleVAD, output logits (sigmoid here).

    Returns (model, kind) where kind is "bpn_v2" or "whalevad".
    """
    arch = ckpt.get("arch")
    if arch == "bpn_v2" or "bpn_cfg_v2" in ckpt:
        if "bpn_cfg_v2" not in ckpt:
            raise KeyError(
                "Checkpoint looks like bpn_v2 but has no 'bpn_cfg_v2' key; "
                "cannot reconstruct the BPN configuration."
            )
        n_out = int(ckpt.get("num_classes", len(ckpt.get("target_labels", ["d"]))))
        bpn_cfg = BPNConfigV2(**dict(ckpt["bpn_cfg_v2"]))
        model = WhaleVADBPNV2(num_classes=n_out, bpn_cfg=bpn_cfg).to(device)
        return model, "bpn_v2"

    if arch == "whalevad":
        from model import WhaleVAD
        n_out = int(ckpt.get("num_classes", len(ckpt.get("target_labels", ["d"]))))
        model = WhaleVAD(num_classes=n_out).to(device)
        return model, "whalevad"

    raise ValueError(
        f"Unrecognised D-detector arch={arch!r} (no 'bpn_cfg_v2' either). "
        "Expected a checkpoint from train_d_detector.py."
    )


def _highpass_sos(highpass_hz: float) -> np.ndarray | None:
    if highpass_hz <= 0:
        return None
    return butter(N=4, Wn=float(highpass_hz), btype="highpass",
                  fs=cfg.SAMPLE_RATE, output="sos").astype(np.float32)


def _apply_highpass_batch(audio_np: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    """Zero-phase high-pass over a (B, T) batch, matching train_d_detector."""
    if sos is None:
        return audio_np
    try:
        out = sosfiltfilt(sos, audio_np, axis=1)
    except ValueError:  # clips shorter than scipy's padding requirement
        out = sosfilt(sos, audio_np, axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


@torch.no_grad()
def predict_d_stream(
    model: torch.nn.Module,
    kind: str,
    spec_extractor: SpectrogramExtractor,
    val_loader: DataLoader,
    device: torch.device,
    highpass_hz: float,
    target_mode: str,
) -> dict[tuple, np.ndarray]:
    """
    Run the D detector and return a width-1 D stream per segment,
    keyed identically to ``predict_probabilities`` so it merges cleanly.

    For ``target_mode == "subtype"`` (bmd/bpd outputs) the two subtype
    columns are collapsed to a single coarse-D column by max.
    """
    model.eval()
    sos = _highpass_sos(highpass_hz)
    hop = spec_extractor.hop_length
    out_probs: dict[tuple, np.ndarray] = {}

    for audio, _, _, metas in tqdm(val_loader, desc="  d-detector", leave=False):
        if sos is not None:
            audio = torch.from_numpy(
                _apply_highpass_batch(audio.cpu().numpy(), sos)
            )
        audio = audio.to(device, non_blocking=True)
        spec = spec_extractor(audio)
        out = model(spec)

        if kind == "bpn_v2":
            probs = out["probs"]            # (B, T, n_out), already sigmoid+gated
        else:
            probs = torch.sigmoid(out)      # plain WhaleVAD logits

        probs_np = probs.detach().float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs_np[j].shape[0])
            arr = probs_np[j, :n_frames, :]
            if target_mode == "subtype" and arr.shape[1] > 1:
                d_col = arr.max(axis=1, keepdims=True)     # collapse bmd/bpd -> d
            else:
                d_col = arr[:, :1]                          # merged: column 0 is d
            out_probs[key] = d_col.astype(np.float32)       # (T, 1)
    return out_probs


# ----------------------------------------------------------------------
# 3-class checkpoint inference (BMABZ/BP source and optional D-blend)
# ----------------------------------------------------------------------

@torch.no_grad()
def predict_3class(
    ckpt_path: str,
    spec_extractor: SpectrogramExtractor,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[tuple, np.ndarray]:
    """Load a 3-class (baseline or v1-BPN) checkpoint and return its probs."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    non_bpn_missing = [k for k in missing if "bpn" not in k]
    if non_bpn_missing:
        print(f"    WARNING: {len(non_bpn_missing)} missing non-BPN keys "
              f"(showing 3): {non_bpn_missing[:3]}")
    if unexpected:
        print(f"    WARNING: {len(unexpected)} unexpected keys "
              f"(showing 3): {unexpected[:3]}")
    probs = predict_probabilities(model, model_type, spec_extractor, val_loader, device)
    probs = collapse_probs_to_3class(probs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def cached_3class(
    ckpt_path: str,
    spec_extractor: SpectrogramExtractor,
    val_loader: DataLoader,
    device: torch.device,
    cache_dir: Path,
    no_cache: bool,
    segment_s: float,
    overlap_s: float,
) -> dict[tuple, np.ndarray]:
    """predict_3class with a segment-aware on-disk cache."""
    cp = _cache_path(ckpt_path, cache_dir, segment_s, overlap_s, "probs3")
    if not no_cache and cp.exists():
        print(f"      cache hit: {cp.name}")
        return _load_cache(cp)
    probs = predict_3class(ckpt_path, spec_extractor, val_loader, device)
    if not no_cache:
        _save_cache(probs, cp)
    return probs


# ----------------------------------------------------------------------
# Combination
# ----------------------------------------------------------------------

def combine(
    bp_avg: dict[tuple, np.ndarray],
    d_avg: dict[tuple, np.ndarray] | None,
    blend_avg: dict[tuple, np.ndarray] | None,
    d_blend_frac: float,
) -> dict[tuple, np.ndarray]:
    """
    Final per-class assembly:
      col 0 (bmabz) <- bp_avg[:, 0]
      col 2 (bp)    <- bp_avg[:, 2]
      col 1 (d)     <- one of:
                         detector + blend : (1-f)*d_avg[:,0] + f*blend_avg[:,D_IDX]
                         detector only    : d_avg[:,0]
                         blend only       : blend_avg[:,D_IDX]
    At least one of d_avg / blend_avg must be provided.
    Truncates to the shortest T per key across the sources used.
    """
    if d_avg is None and blend_avg is None:
        raise ValueError("combine needs a D source: detector and/or blend.")

    keys = set(bp_avg)
    if d_avg is not None:
        keys &= set(d_avg)
    if blend_avg is not None:
        keys &= set(blend_avg)
    dropped = (len(bp_avg) - len(keys))
    if dropped > 0:
        print(f"  NOTE: {dropped} segments dropped (key mismatch across sources).")

    out: dict[tuple, np.ndarray] = {}
    for key in keys:
        bp = bp_avg[key]
        Ts = [bp.shape[0]]
        if d_avg is not None:
            Ts.append(d_avg[key].shape[0])
        if blend_avg is not None:
            Ts.append(blend_avg[key].shape[0])
        T = min(Ts)

        combined = np.zeros((T, 3), dtype=np.float32)
        combined[:, 0] = bp[:T, 0]
        combined[:, 2] = bp[:T, 2]

        if d_avg is not None and blend_avg is not None:
            combined[:, 1] = ((1.0 - d_blend_frac) * d_avg[key][:T, 0]
                              + d_blend_frac * blend_avg[key][:T, D_IDX])
        elif d_avg is not None:
            combined[:, 1] = d_avg[key][:T, 0]
        else:  # blend only
            combined[:, 1] = blend_avg[key][:T, D_IDX]
        out[key] = combined
    return out


def _normalize(weights: list[float] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0 / n] * n
    assert len(weights) == n, f"expected {n} weights, got {len(weights)}"
    s = float(sum(weights))
    return [w / s for w in weights]


def d_only_d_f1(d_stream: dict[tuple, np.ndarray], gt_events: list[Detection]) -> float:
    """Standalone D-F1 of a width-1 D stream (placed into a 3-class array)."""
    as3 = {k: np.concatenate(
        [np.zeros_like(v), v, np.zeros_like(v)], axis=1).astype(np.float32)
        for k, v in d_stream.items()}
    thr = tune_thresholds_on_probs(as3, gt_events)
    m = evaluate_with_thresholds(as3, gt_events, thr)
    return float(m.get("d", {}).get("f1", 0.0))


def print_metrics(metrics, thresholds, title):
    print(f"\n  {title}:")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():<6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5d} FP={m.get('fp', 0):5d} "
              f"FN={m.get('fn', 0):5d}  "
              f"P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                           for c in cfg.CALL_TYPES_3]))
    print(f"    OVERALL (micro) F1={metrics.get('overall', {}).get('f1', 0):.3f}  "
          f"MACRO F1={macro:.3f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bp-checkpoints", nargs="+", required=True,
                   help="3-class checkpoints for BMABZ + BP (e.g. your hnm models).")
    p.add_argument("--bp-weights", nargs="+", type=float, default=None,
                   help="Optional weights for --bp-checkpoints. Default: equal.")
    p.add_argument("--d-detector-checkpoints", nargs="+", default=None,
                   help="Optional single-class D-detector checkpoint(s) "
                        "(train_d_detector.py / WhaleVADBPNV2). Averaged if >1. "
                        "If omitted, D comes from --d-blend-checkpoints alone "
                        "(your no-detector baseline).")
    p.add_argument("--d-detector-weights", nargs="+", type=float, default=None,
                   help="Optional weights for the D detectors. Default: equal.")
    p.add_argument("--d-blend-checkpoints", nargs="+", default=None,
                   help="Optional 3-class checkpoints whose D column is blended "
                        "with the detector D (e.g. your 4 mtfb models).")
    p.add_argument("--d-blend-weights", nargs="+", type=float, default=None,
                   help="Optional weights for --d-blend-checkpoints. Default: equal.")
    p.add_argument("--d-blend-frac", type=float, default=0.0,
                   help="Blend fraction f in [0,1]: final D = (1-f)*detector + f*blend. "
                        "0.0 (default) = detector replaces the old D source.")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Eval tile length (s) for ALL models. Notes say 60 beats 30.")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S,
                   help="Eval tile overlap (s) for ALL models.")
    p.add_argument("--thresholds", nargs=3, type=float, default=None,
                   metavar=("BMABZ", "D", "BP"),
                   help="Frozen thresholds [BMABZ D BP]. If omitted, tuned on val GT.")
    p.add_argument("--test-datasets", nargs="+", default=None,
                   help="Run on these datasets instead of cfg.VAL_DATASETS. "
                        "No scoring; requires --thresholds.")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Write predictions CSV here.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/prob_cache"),
                   help="Per-checkpoint prob cache dir (keyed by dir name + tile length).")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute all probs and skip reading/writing the cache.")
    p.add_argument("--per-model-eval", action="store_true",
                   help="Also report each D detector's standalone D-F1.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.d_blend_frac <= 1.0):
        raise SystemExit("--d-blend-frac must be in [0, 1].")
    use_detector = bool(args.d_detector_checkpoints)
    use_blend = bool(args.d_blend_checkpoints)
    if not use_detector and not use_blend:
        raise SystemExit("Need a D source: pass --d-detector-checkpoints "
                         "and/or --d-blend-checkpoints.")
    if use_detector and not use_blend and args.d_blend_frac > 0:
        print("  NOTE: no --d-blend-checkpoints, so --d-blend-frac is ignored "
              "(D from detector alone).")

    test_mode = args.test_datasets is not None
    if test_mode:
        if args.thresholds is None:
            raise SystemExit("--test-datasets requires --thresholds (no GT to tune on).")
        print(f"TEST MODE: pointing val_datasets at {args.test_datasets}")
        cfg.VAL_DATASETS = list(args.test_datasets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Eval tiles: {args.segment_s:.0f}s windows, {args.overlap_s:.0f}s overlap")

    # Shared val loader — every model runs on the SAME tiles so keys align.
    print("\nLoading data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(
        val_manifest, val_anns,
        segment_s=args.segment_s, overlap_s=args.overlap_s,
    )
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  {len(val_segs)} tiles")

    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}
    gt_events: list[Detection] = []
    if not test_mode:
        for _, row in val_anns.iterrows():
            fsd = file_start_dts.get((row["dataset"], row["filename"]))
            if fsd is None:
                continue
            gt_events.append(Detection(
                dataset=row["dataset"], filename=row["filename"],
                label=row["label_3class"],
                start_s=(row["start_datetime"] - fsd).total_seconds(),
                end_s=(row["end_datetime"] - fsd).total_seconds(),
            ))
        print(f"  {len(gt_events)} ground-truth events")

    spec_extractor = SpectrogramExtractor().to(device)

    # ------------------------------------------------------------------
    # BMABZ/BP source (3-class ensemble)
    # ------------------------------------------------------------------
    print(f"\nBMABZ/BP source: {len(args.bp_checkpoints)} checkpoint(s)")
    bp_dicts = []
    for i, ckpt in enumerate(args.bp_checkpoints):
        print(f"  [{i+1}/{len(args.bp_checkpoints)}] {ckpt}")
        bp_dicts.append(cached_3class(
            ckpt, spec_extractor, val_loader, device,
            args.cache_dir, args.no_cache, args.segment_s, args.overlap_s))
    bp_w = _normalize(args.bp_weights, len(bp_dicts))
    bp_avg = bp_dicts[0] if len(bp_dicts) == 1 else average_prob_dicts(bp_dicts, weights=bp_w)

    # ------------------------------------------------------------------
    # D source (optional dedicated detector(s))
    # ------------------------------------------------------------------
    d_avg = None
    if use_detector:
        print(f"\nD source: {len(args.d_detector_checkpoints)} dedicated detector(s)")
        d_dicts = []
        for i, ckpt_path in enumerate(args.d_detector_checkpoints):
            print(f"  [{i+1}/{len(args.d_detector_checkpoints)}] {ckpt_path}")
            dp = _cache_path(ckpt_path, args.cache_dir,
                             args.segment_s, args.overlap_s, "dstream")
            if not args.no_cache and dp.exists():
                print(f"      cache hit: {dp.name}")
                d_stream = _load_cache(dp)
            else:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model, kind = build_d_detector(ckpt, device)
                highpass_hz = float(ckpt.get("highpass_hz", 0.0))
                target_mode = ckpt.get("target_mode", "merged")
                print(f"    kind={kind}  target_mode={target_mode}  "
                      f"highpass_hz={highpass_hz:.1f}  "
                      f"trained_segment_s={ckpt.get('segment_s')}")
                if highpass_hz > 0:
                    print(f"    -> re-applying {highpass_hz:.1f} Hz high-pass to match training")

                with torch.no_grad():
                    dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
                    _ = model(spec_extractor(dummy))   # materialise lazy feat_proj
                missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if missing:
                    print(f"    WARNING: {len(missing)} missing keys (showing 3): {missing[:3]}")
                if unexpected:
                    print(f"    WARNING: {len(unexpected)} unexpected keys (showing 3): {unexpected[:3]}")

                d_stream = predict_d_stream(
                    model, kind, spec_extractor, val_loader, device,
                    highpass_hz=highpass_hz, target_mode=target_mode,
                )
                if not args.no_cache:
                    _save_cache(d_stream, dp)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            d_dicts.append(d_stream)
            if args.per_model_eval and not test_mode:
                f1 = d_only_d_f1(d_stream, gt_events)
                print(f"    standalone D-F1: {f1:.3f}")

        d_w = _normalize(args.d_detector_weights, len(d_dicts))
        d_avg = d_dicts[0] if len(d_dicts) == 1 else average_prob_dicts(d_dicts, weights=d_w)

    # ------------------------------------------------------------------
    # Optional D-blend source (3-class subset, e.g. mtfb).
    # Loaded whenever --d-blend-checkpoints is given: it is the D source
    # when no detector is present, and the blend partner when one is.
    # ------------------------------------------------------------------
    blend_avg = None
    if use_blend:
        tag = (f"(frac={args.d_blend_frac})" if use_detector
               else "(sole D source — no detector)")
        print(f"\nD-blend source: {len(args.d_blend_checkpoints)} checkpoint(s)  {tag}")
        blend_dicts = []
        for i, ckpt in enumerate(args.d_blend_checkpoints):
            print(f"  [{i+1}/{len(args.d_blend_checkpoints)}] {ckpt}")
            blend_dicts.append(cached_3class(
                ckpt, spec_extractor, val_loader, device,
                args.cache_dir, args.no_cache, args.segment_s, args.overlap_s))
        blend_w = _normalize(args.d_blend_weights, len(blend_dicts))
        blend_avg = (blend_dicts[0] if len(blend_dicts) == 1
                     else average_prob_dicts(blend_dicts, weights=blend_w))

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("HYBRID COMBINATION")
    print(f"  BMABZ + BP : weighted avg of {len(bp_dicts)} model(s)")
    if use_detector and use_blend:
        print(f"  D          : {1-args.d_blend_frac:.2f}*detector + "
              f"{args.d_blend_frac:.2f}*blend")
        d_label = "D from detector+blend"
    elif use_detector:
        print(f"  D          : dedicated detector(s) only")
        d_label = "D from dedicated detector"
    else:
        print(f"  D          : blend source only (NO detector — baseline)")
        d_label = "D from blend only (no detector)"
    print("=" * 64)
    hybrid = combine(bp_avg, d_avg, blend_avg, args.d_blend_frac)
    print(f"  combined probs for {len(hybrid)} segments")

    # ------------------------------------------------------------------
    # Thresholds + score / CSV
    # ------------------------------------------------------------------
    if args.thresholds is not None:
        thresholds = np.array(args.thresholds, dtype=np.float64)
        print(f"\nUsing frozen thresholds: {list(thresholds)}")
    else:
        print("\nThreshold tuning on hybrid probabilities:")
        thresholds = tune_thresholds_on_probs(hybrid, gt_events)

    if not test_mode:
        metrics = evaluate_with_thresholds(hybrid, gt_events, thresholds)
        print_metrics(metrics, thresholds, f"HYBRID RESULT ({d_label})")

    if args.output_csv is not None:
        print("\nWriting predictions CSV...")
        preds = postprocess_predictions(hybrid, thresholds)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_predictions_csv(preds, args.output_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
