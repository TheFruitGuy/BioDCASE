"""
Eval with bout + frequency filters
==================================

Drop-in wrapper around `ensemble_predict_cached`. Loads (or computes)
cached per-checkpoint probabilities, averages them, then:

  1. Reports the un-filtered baseline (current operating point: tuned
     thresholds, no filters).
  2. Applies the temporal bout filter and/or the frequency guardrail
     to the resulting Detection events.
  3. Optionally re-tunes thresholds WITH the filters inside the inner
     loop (--retune; default ON in 'primary' mode), to let the threshold
     search see the new FP/FN landscape.

The original `ensemble_predict_cached.py` is never touched — this script
imports the same functions and re-uses the same cache directory.

Usage
-----
    # Primary mode: filter-then-retune (both filters on)
    python eval_with_bout_filter.py \
        --checkpoints runs/ens/seed42/best.pt runs/ens/seed1337/best.pt ... \
        --retune

    # Ablation: temporal only
    python eval_with_bout_filter.py --checkpoints ... --disable_freq

    # Ablation: frequency only
    python eval_with_bout_filter.py --checkpoints ... --disable_temporal

    # Ablation: tune-then-filter (don't retune)
    python eval_with_bout_filter.py --checkpoints ... --no_retune
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, collapse_probs_to_3class, postprocess_predictions,
)
from spectrogram import SpectrogramExtractor

from ensemble_predict import (
    average_prob_dicts, build_model_for_ckpt, predict_probabilities,
    compute_metrics, tune_thresholds_on_probs, evaluate_with_thresholds,
)
from ensemble_predict_cached import (
    cache_path_for, load_cached_probs, save_probs_to_cache,
    get_or_compute_probs,
)

from postprocess_bout_filter import (
    apply_temporal_bout_filter, analyze_ici,
    FrequencyGuardrail, make_default_audio_resolver,
    BIOLOGICAL_BANDS_HZ, NOISE_BANDS_HZ,
)


# ---------------------------------------------------------------------
# Filter composition. Order matters: temporal first (cheap, cuts down
# the set the freq guardrail has to load STFTs for), then frequency.
# ---------------------------------------------------------------------

def apply_filters(
    detections: list[Detection],
    guard: FrequencyGuardrail | None,
    use_temporal: bool,
    use_freq: bool,
    bout_kwargs: dict | None = None,
    freq_kwargs: dict | None = None,
    verbose: bool = True,
) -> list[Detection]:
    bout_kwargs = bout_kwargs or {}
    freq_kwargs = freq_kwargs or {}
    out = detections
    if use_temporal:
        out = apply_temporal_bout_filter(out, **bout_kwargs)
    if use_freq and guard is not None:
        out = guard.filter(out, verbose=verbose, **freq_kwargs)
    return out


# ---------------------------------------------------------------------
# Threshold tuning with filters inside the inner loop (filter-then-retune)
# Mirrors the structure of `tune_thresholds_on_probs` so the grids and
# coordinate-descent order are identical — only the scoring step differs.
# ---------------------------------------------------------------------

def tune_thresholds_with_filters(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
    guard: FrequencyGuardrail | None,
    use_temporal: bool,
    use_freq: bool,
    bout_kwargs: dict | None = None,
    freq_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Coordinate-descent threshold sweep with both filters applied before
    `compute_metrics` at each grid point. The freq-guardrail STFT cache
    is reused across all grid points (same files, different event sets).
    """
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    grids = [
        np.arange(0.20, 0.85, 0.05),                                     # bmabz
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),                    # d
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),                    # bp
    ]
    bout_kwargs = bout_kwargs or {}
    freq_kwargs = freq_kwargs or {}

    print("  [retune-with-filters] grids:")
    for name, g in zip(cfg.CALL_TYPES_3, grids):
        print(f"    {name:<6}  {len(g)} candidates "
              f"[{g.min():.2f} .. {g.max():.2f}]")

    for c, name in enumerate(cfg.CALL_TYPES_3):
        best_f1, best_t = -1.0, float(thresholds[c])
        for t in grids[c]:
            trial = thresholds.copy()
            trial[c] = float(t)
            events = postprocess_predictions(all_probs, trial)
            events = apply_filters(
                events, guard, use_temporal, use_freq,
                bout_kwargs, freq_kwargs, verbose=False,
            )
            metrics = compute_metrics(events, gt_events, iou_threshold=0.3)
            f1 = metrics.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
        print(f"    {name:<6}  best t={best_t:.2f}  f1={best_f1:.3f}")
    return thresholds


# ---------------------------------------------------------------------
# Pretty printing — match ensemble_predict_cached's format so the numbers
# are directly comparable side-by-side in tmux.
# ---------------------------------------------------------------------

def print_class_table(metrics, thresholds, label, baseline_macro=None):
    print(f"\n  {label}:")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):5} "
              f"FN={m.get('fn', 0):5}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    overall = metrics.get("overall", {})
    macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                           for c in cfg.CALL_TYPES_3]))
    line = (f"    OVERALL F1={overall.get('f1', 0):.3f}  "
            f"MACRO F1={macro:.3f}")
    if baseline_macro is not None:
        delta = macro - baseline_macro
        sign = "+" if delta >= 0 else ""
        line += f"  (Δmacro={sign}{delta:.4f})"
    print(line)
    return macro


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--cache_dir", type=str, default="runs/prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)

    # Filter knobs
    p.add_argument("--disable_temporal", action="store_true",
                   help="Skip the bout filter (frequency-only ablation).")
    p.add_argument("--disable_freq", action="store_true",
                   help="Skip the frequency guardrail (temporal-only ablation).")
    p.add_argument("--bout_window_s", type=float, default=3600.0,
                   help="Total bout-filter window in seconds (default: 3600 = ±0.5h).")
    p.add_argument("--bout_min_calls", type=int, default=3)
    p.add_argument("--bout_high_conf", type=float, default=0.7)
    p.add_argument("--bout_classes", nargs="+", default=["d", "bmabz"],
                   help="Classes the bout filter targets. bp is excluded by default.")
    p.add_argument("--freq_classes", nargs="+", default=["d", "bmabz"],
                   help="Classes the frequency guardrail targets.")
    p.add_argument("--freq_noise_ratio", type=float, default=1.5,
                   help="Drop a detection if bio_energy/noise_energy < this ratio.")

    # Tuning mode
    p.add_argument("--no_retune", action="store_true",
                   help="Keep the baseline-tuned thresholds (ablation). "
                        "Default is to re-tune with filters in the loop.")

    # Diagnostics
    p.add_argument("--ici_plot_path", type=str, default=None,
                   help="If set, save the d-class ICI histogram here.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir} (no_cache={args.no_cache}, "
          f"use_fp16={args.use_fp16})")

    # --- Validation loader + ground truth (same as cached script) ---
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  {len(val_segs)} 30s tiles")

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    gt_events = []
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

    # --- Get/compute per-checkpoint probabilities ---
    all_prob_dicts = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        probs = get_or_compute_probs(
            ckpt_path, spec_extractor, val_loader, device,
            cache_dir, no_cache=args.no_cache, use_fp16=args.use_fp16)
        all_prob_dicts.append(probs)

    # --- Ensemble average ---
    weights = None
    if args.weights:
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
        print(f"\nPer-model weights (normalized): {weights}")

    ens_probs = (all_prob_dicts[0] if len(all_prob_dicts) == 1
                 else average_prob_dicts(all_prob_dicts, weights=weights))

    # --- 1. Baseline: tune + score with NO filters (current op point) ---
    print(f"\n{'='*70}\n[1/3] BASELINE — no filters, threshold-tuned on raw events\n{'='*70}")
    base_thr = tune_thresholds_on_probs(ens_probs, gt_events)
    base_metrics = evaluate_with_thresholds(ens_probs, gt_events, base_thr)
    base_macro = print_class_table(base_metrics, base_thr, "BASELINE")

    # --- 2. Build the frequency guardrail (lazy STFT cache) ---
    guard = None
    if not args.disable_freq:
        resolver = make_default_audio_resolver(cfg.DATA_ROOT)
        guard = FrequencyGuardrail(
            audio_path_resolver=resolver,
            sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
            biological_bands=BIOLOGICAL_BANDS_HZ,
            noise_bands=NOISE_BANDS_HZ,
            target_classes=tuple(args.freq_classes),
            max_cached_files=32,
        )

    bout_kwargs = dict(
        target_classes=tuple(args.bout_classes),
        window_size_sec=args.bout_window_s,
        min_calls=args.bout_min_calls,
        high_conf_threshold=args.bout_high_conf,
    )
    freq_kwargs = dict(noise_threshold_ratio=args.freq_noise_ratio)

    # --- Diagnostic: ICI on the baseline detection set ---
    print(f"\n{'='*70}\n[ICI DIAGNOSTIC] consecutive 'd' intervals on baseline events\n{'='*70}")
    base_events = postprocess_predictions(ens_probs, base_thr)
    analyze_ici(base_events, target_class="d",
                score_threshold=args.bout_high_conf * 0.3,  # ~0.2 default
                show=False, save_path=args.ici_plot_path)
    if "bmabz" in args.bout_classes:
        analyze_ici(base_events, target_class="bmabz",
                    score_threshold=args.bout_high_conf * 0.3,
                    show=False)

    # --- 3a. Filters at baseline thresholds (tune→filter, no retune) ---
    print(f"\n{'='*70}\n[2/3] FILTERED @ baseline thresholds (no retune)\n{'='*70}")
    t0 = time.time()
    filt_events = apply_filters(
        base_events, guard,
        use_temporal=not args.disable_temporal,
        use_freq=not args.disable_freq,
        bout_kwargs=bout_kwargs, freq_kwargs=freq_kwargs,
        verbose=True,
    )
    filt_metrics = compute_metrics(filt_events, gt_events, iou_threshold=0.3)
    print(f"  (filter time: {time.time()-t0:.1f}s)")
    print_class_table(filt_metrics, base_thr,
                      "FILTERED @ baseline thr",
                      baseline_macro=base_macro)

    if args.no_retune:
        print("\n--no_retune set: skipping the filter-then-retune step.")
        return

    # --- 3b. Re-tune thresholds WITH filters in the loop (primary mode) ---
    print(f"\n{'='*70}\n[3/3] RE-TUNE thresholds with filters in the loop\n{'='*70}")
    t0 = time.time()
    retuned_thr = tune_thresholds_with_filters(
        ens_probs, gt_events, guard,
        use_temporal=not args.disable_temporal,
        use_freq=not args.disable_freq,
        bout_kwargs=bout_kwargs, freq_kwargs=freq_kwargs,
    )
    print(f"  (retune time: {time.time()-t0:.1f}s)")

    # Final scoring with retuned thresholds + filters.
    final_events = postprocess_predictions(ens_probs, retuned_thr)
    final_events = apply_filters(
        final_events, guard,
        use_temporal=not args.disable_temporal,
        use_freq=not args.disable_freq,
        bout_kwargs=bout_kwargs, freq_kwargs=freq_kwargs,
        verbose=True,
    )
    final_metrics = compute_metrics(final_events, gt_events, iou_threshold=0.3)
    print_class_table(final_metrics, retuned_thr,
                      "FILTERED + RETUNED",
                      baseline_macro=base_macro)

    # --- Summary delta table for tmux scrollback ---
    print(f"\n{'='*70}\nSUMMARY (macro F1)\n{'='*70}")
    macro_base = base_macro
    macro_filt = float(np.mean([filt_metrics.get(c, {}).get("f1", 0.0)
                                for c in cfg.CALL_TYPES_3]))
    macro_final = float(np.mean([final_metrics.get(c, {}).get("f1", 0.0)
                                 for c in cfg.CALL_TYPES_3]))
    print(f"  baseline           : {macro_base:.4f}")
    print(f"  filtered (no retune): {macro_filt:.4f}  "
          f"(Δ={macro_filt-macro_base:+.4f})")
    print(f"  filtered + retuned : {macro_final:.4f}  "
          f"(Δ={macro_final-macro_base:+.4f})")
    print(f"\n  active filters: "
          f"temporal={'OFF' if args.disable_temporal else 'ON'}, "
          f"freq={'OFF' if args.disable_freq else 'ON'}")

    if guard is not None:
        guard.clear_cache()


if __name__ == "__main__":
    main()
