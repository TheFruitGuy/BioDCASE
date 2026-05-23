"""
diagnose_freq_bands.py
======================

Per-event frequency-band diagnostic for ensemble predictions.

Purpose
-------
We want to know: of the false positives our model produces, what
fraction have their acoustic energy in the WRONG frequency band for
the predicted class? If most FPs are out-of-band, a post-hoc per-class
band filter will help. If FPs are already in-band, the model has
internalised the band constraint and the remaining errors are
temporal/contextual — and a band filter won't move the needle.

Pipeline
--------
1. Load predictions from a CSV produced by ``hybrid_ensemble_predict.py``
   (columns: dataset, filename, onset, offset, event_label, confidence).
2. Load ground-truth events for the same datasets.
3. Match predictions → GT with greedy 1-D IoU (mirrors
   ``postprocess.compute_metrics``). Each prediction becomes TP or FP;
   each unmatched GT becomes FN.
4. For every TP / FP / FN event, re-read the audio segment from disk,
   compute the spectrogram via the SAME ``SpectrogramExtractor`` the
   model uses, and measure mean magnitude inside vs outside the
   class-specific frequency band.
5. Emit a per-event CSV and a console summary, including a simulated
   "what-if" sweep showing how P/R/F1 would change if events with
   ``band_score < tau_c`` were rejected.

This is a diagnostic ONLY — it doesn't train, modify, or save any
model. By default it runs on ``cfg.VAL_DATASETS`` (Casey2017,
Kerguelen2014, Kerguelen2015). The BioDCASE test sites should remain
held-out until final reporting.

Frequency bands per class (Hz)
------------------------------
- ``bmabz``  : 15–30  (Z-call/A/B fundamentals)
- ``d``      : 20–120 (downsweep across most of the audible band)
- ``bp``     : 15–30 + 80–120 (fundamental + optional overtone)

Out-of-band region is everything in 5–120 Hz NOT covered by the class
band. The 0–5 Hz region is excluded because demeaning makes DC-adjacent
bins noisy.

Usage
-----
Two modes — pick whichever fits.

**Mode 1: ensemble in-process** (recommended). Mirrors
``ensemble_predict_cached.py`` and reuses its prob cache::

    python diagnose_freq_bands.py \\
        --checkpoints \\
            runs/hnm_D_whalevad_20260504_152450/best_model.pt \\
            runs/hnm_D_phase5_20260506_204358/best_model.pt \\
            runs/hnm_D_whalevad_20260507_191223/best_model.pt \\
            runs/hnm_D_phase5_20260507_211504/best_model.pt \\
        --out-csv val_band_diagnostic.csv

If you've already run ``ensemble_predict_cached.py`` with the same
checkpoints, ``runs/prob_cache/`` is already populated, so this just
hits the cache and finishes in seconds. Add ``--weights w1 w2 ...``
to bias the average, ``--save-preds-csv val_preds.csv`` to also dump
the predictions for later re-use.

**Mode 2: from a predictions CSV**. Use this if you've already produced
a CSV from elsewhere (e.g. ``hybrid_ensemble_predict.py``)::

    python diagnose_freq_bands.py \\
        --pred-csv val_hybrid_predictions.csv \\
        --out-csv val_band_diagnostic.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, compute_iou_1d, postprocess_predictions,
)
from spectrogram import SpectrogramExtractor

# Lazily imported (only needed in --checkpoints mode) to avoid hard
# dependencies on the ensemble pipeline at import time.
try:
    from ensemble_predict import (
        average_prob_dicts, tune_thresholds_on_probs,
    )
    from ensemble_predict_cached import get_or_compute_probs
    _ENSEMBLE_AVAILABLE = True
except ImportError as _e:
    _ENSEMBLE_AVAILABLE = False
    _ENSEMBLE_IMPORT_ERR = str(_e)


# ======================================================================
# Class frequency bands (Hz)
# ======================================================================

CALL_BANDS_HZ: dict[str, list[tuple[float, float]]] = {
    "bmabz": [(15.0, 30.0)],
    "d":     [(20.0, 120.0)],
    "bp":    [(15.0, 30.0), (80.0, 120.0)],
}

#: Out-of-band region is taken to be everything outside the class's
#: bands but still within this informative range. The very bottom
#: (0–5 Hz) is excluded because of demeaning / DC artefacts.
INFO_BAND_HZ: tuple[float, float] = (5.0, 120.0)


def hz_to_bin(hz: float, nyq: float, n_freq: int) -> int:
    """Closest STFT bin index for ``hz`` given ``nyq`` and ``n_freq``."""
    return int(round(hz / nyq * (n_freq - 1)))


def class_band_mask(class_name: str, n_freq: int, nyq: float) -> np.ndarray:
    """Boolean mask (length ``n_freq``) selecting the class's expected bins."""
    mask = np.zeros(n_freq, dtype=bool)
    for lo, hi in CALL_BANDS_HZ[class_name]:
        b_lo = hz_to_bin(lo, nyq, n_freq)
        b_hi = hz_to_bin(hi, nyq, n_freq)
        mask[b_lo:b_hi + 1] = True
    return mask


def info_band_mask(n_freq: int, nyq: float) -> np.ndarray:
    """Boolean mask of the informative frequency region (5–120 Hz)."""
    lo, hi = INFO_BAND_HZ
    mask = np.zeros(n_freq, dtype=bool)
    b_lo = hz_to_bin(lo, nyq, n_freq)
    b_hi = hz_to_bin(hi, nyq, n_freq)
    mask[b_lo:b_hi + 1] = True
    return mask


# ======================================================================
# I/O: predictions and ground-truth
# ======================================================================

def load_predictions_csv(path: Path) -> list[Detection]:
    """
    Load a predictions CSV in the ``hybrid_ensemble_predict.py`` format
    (``dataset, filename, onset, offset, event_label, confidence``).
    """
    df = pd.read_csv(path)
    required = {"dataset", "filename", "onset", "offset", "event_label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Predictions CSV missing columns: {sorted(missing)}. "
            f"Expected: dataset, filename, onset, offset, event_label, "
            f"confidence (optional)."
        )
    preds: list[Detection] = []
    for _, row in df.iterrows():
        d = Detection(
            dataset=str(row["dataset"]),
            filename=str(row["filename"]),
            label=str(row["event_label"]),
            start_s=float(row["onset"]),
            end_s=float(row["offset"]),
        )
        if "confidence" in df.columns:
            try:
                d.confidence = float(row["confidence"])
            except (TypeError, ValueError):
                pass
        preds.append(d)
    return preds


def build_gt_events(
    datasets: list[str],
) -> tuple[list[Detection], pd.DataFrame]:
    """Build the GT Detection list and return it along with the manifest."""
    manifest = get_file_manifest(datasets)
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }
    ann = load_annotations(datasets, manifest=manifest)
    gt: list[Detection] = []
    for _, row in ann.iterrows():
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
    return gt, manifest


# ======================================================================
# In-process ensemble inference (mirrors ensemble_predict_cached.py)
# ======================================================================

def run_ensemble_inference(
    checkpoint_paths: list[str],
    weights: list[float] | None,
    cache_dir: Path,
    use_fp16: bool,
    batch_size: int,
    device: torch.device,
    no_cache: bool = False,
) -> tuple[list[Detection], list[Detection], pd.DataFrame, np.ndarray]:
    """
    Replicate ``ensemble_predict_cached.py`` end-to-end and return the
    final predictions in memory, so the diagnostic can run from
    checkpoint paths in a single command.

    Behaviour matches the user's 0.518-macro pipeline exactly:
      1. Build the val loader for ``cfg.VAL_DATASETS``.
      2. For each checkpoint, fetch or compute per-segment probs
         (hits the shared ``runs/prob_cache/`` directory).
      3. Combine via plain weighted average across all classes.
      4. Tune per-class thresholds on the ensemble probs against GT.
      5. Run ``postprocess_predictions`` to produce final Detections.

    Returns
    -------
    preds : list[Detection]
        Final predictions after thresholding + post-processing.
    gt_events : list[Detection]
        Ground-truth events on the val set.
    manifest : pd.DataFrame
        File manifest (for audio loading downstream).
    thresholds : np.ndarray, shape (3,)
        Tuned per-class thresholds in CALL_TYPES_3 order.
    """
    if not _ENSEMBLE_AVAILABLE:
        raise SystemExit(
            f"--checkpoints mode requires ensemble_predict.py and "
            f"ensemble_predict_cached.py to be importable, but: "
            f"{_ENSEMBLE_IMPORT_ERR}"
        )

    # ------------------------------------------------------------------
    # Val loader + GT
    # ------------------------------------------------------------------
    print("Loading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  {len(val_segs)} 30s tiles")

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }
    gt_events: list[Detection] = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"  {len(gt_events)} GT events")

    # ------------------------------------------------------------------
    # Per-checkpoint probabilities (cached)
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    all_probs = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_path}")
        probs = get_or_compute_probs(
            ckpt_path, spec_extractor, val_loader, device,
            cache_dir, no_cache=no_cache, use_fp16=use_fp16,
        )
        all_probs.append(probs)

    # ------------------------------------------------------------------
    # Combine + tune thresholds + post-process
    # ------------------------------------------------------------------
    norm_weights = None
    if weights:
        if len(weights) != len(checkpoint_paths):
            raise SystemExit(
                f"--weights has {len(weights)} entries but "
                f"--checkpoints has {len(checkpoint_paths)}."
            )
        total = sum(weights)
        norm_weights = [w / total for w in weights]
        print(f"\nPer-model weights (normalised): {norm_weights}")

    ens_probs = (
        all_probs[0] if len(all_probs) == 1
        else average_prob_dicts(all_probs, weights=norm_weights)
    )

    print("\nTuning per-class thresholds on ensemble probabilities...")
    thresholds = tune_thresholds_on_probs(ens_probs, gt_events)
    print(f"  Tuned thresholds: "
          f"{ {c: f'{thresholds[i]:.2f}' for i, c in enumerate(cfg.CALL_TYPES_3)} }")

    print("Post-processing → Detections...")
    preds = postprocess_predictions(ens_probs, thresholds)
    print(f"  {len(preds)} predictions after post-processing")

    return preds, gt_events, manifest, thresholds


# ======================================================================
# Matching: preds → TP / FP, GTs → FN
# ======================================================================

def match_predictions(
    preds: list[Detection],
    gt_events: list[Detection],
    iou_threshold: float,
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """
    Greedy per-(class, file) 1-D IoU matching, mirroring
    ``postprocess.compute_metrics``:

    - For each (class, file), each GT is matched to its best-IoU unmatched
      prediction. If best IoU >= ``iou_threshold``, that prediction is a TP.
      Otherwise the GT is an FN.
    - Predictions never matched to any GT become FPs.

    Returns
    -------
    (tps, fps, fns) : (list, list, list) of Detection
    """
    classes = sorted({d.label for d in list(preds) + list(gt_events)})

    tp_preds: list[Detection] = []
    fp_preds: list[Detection] = []
    fn_gts:   list[Detection] = []
    matched_pred_ids: set[int] = set()

    for cls in classes:
        cls_preds = [p for p in preds if p.label == cls]
        cls_gts = [g for g in gt_events if g.label == cls]
        files = {(d.dataset, d.filename) for d in cls_preds + cls_gts}

        for fk in files:
            file_preds = sorted(
                [p for p in cls_preds if (p.dataset, p.filename) == fk],
                key=lambda x: x.start_s,
            )
            file_gts = sorted(
                [g for g in cls_gts if (g.dataset, g.filename) == fk],
                key=lambda x: x.start_s,
            )
            matched_local: set[int] = set()
            for gt in file_gts:
                best_iou, best_i = 0.0, -1
                for i, pr in enumerate(file_preds):
                    if i in matched_local:
                        continue
                    iou = compute_iou_1d(
                        pr.start_s, pr.end_s, gt.start_s, gt.end_s,
                    )
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_iou >= iou_threshold and best_i >= 0:
                    tp_preds.append(file_preds[best_i])
                    matched_local.add(best_i)
                    matched_pred_ids.add(id(file_preds[best_i]))
                else:
                    fn_gts.append(gt)

    for p in preds:
        if id(p) not in matched_pred_ids:
            fp_preds.append(p)

    return tp_preds, fp_preds, fn_gts


# ======================================================================
# Audio → spectrogram → band score
# ======================================================================

@dataclass
class FileEntry:
    path: str
    duration_s: float


def build_file_index(manifest: pd.DataFrame) -> dict:
    return {
        (r["dataset"], r["filename"]): FileEntry(
            path=r["path"], duration_s=float(r["duration_s"]),
        )
        for _, r in manifest.iterrows()
    }


@torch.no_grad()
def compute_event_profile(
    event: Detection,
    file_index: dict,
    spec_extractor: SpectrogramExtractor,
    device: torch.device,
    margin_s: float = 0.5,
    min_dur_s: float = 1.0,
) -> np.ndarray | None:
    """
    Read the audio segment for ``event`` (±``margin_s`` of context),
    compute the spectrogram via the same pipeline the model uses, and
    return the per-frequency-bin mean magnitude over time.

    Returns None if the file can't be located or the segment is too
    short to STFT.

    Shape: ``(n_freq,)``, typically ``(129,)``.
    """
    fe = file_index.get((event.dataset, event.filename))
    if fe is None:
        return None

    start_s = max(0.0, event.start_s - margin_s)
    end_s = min(fe.duration_s, event.end_s + margin_s)

    # Pad to a minimum duration if necessary (rare for FNs / short events).
    if end_s - start_s < min_dur_s:
        mid = 0.5 * (start_s + end_s)
        half = 0.5 * min_dur_s
        start_s = max(0.0, mid - half)
        end_s = min(fe.duration_s, mid + half)

    start_samp = int(start_s * cfg.SAMPLE_RATE)
    end_samp = int(end_s * cfg.SAMPLE_RATE)
    if end_samp - start_samp < cfg.N_FFT:
        return None

    try:
        audio, sr = sf.read(
            fe.path, start=start_samp, stop=end_samp, dtype="float32",
        )
    except Exception:
        return None
    if sr != cfg.SAMPLE_RATE:
        return None

    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, n)
    feat = spec_extractor(audio_t)            # (1, 3, n_freq, n_frames)
    mag = feat[0, 0].detach().cpu().numpy()   # channel 0 = magnitude
    return mag.mean(axis=1)                   # (n_freq,)


def compute_band_score(
    profile: np.ndarray,
    class_name: str,
    n_freq: int,
    nyq: float,
) -> dict:
    """
    Reduce a per-frequency-bin profile to a band-consistency score.

    The score is ``mean(profile[in_band]) − mean(profile[out_of_band])``.
    Higher = more energy is concentrated in the class's expected band,
    which is what we want to see for TPs.
    """
    in_mask = class_band_mask(class_name, n_freq, nyq)
    inf_mask = info_band_mask(n_freq, nyq)
    out_mask = inf_mask & ~in_mask

    e_in = float(profile[in_mask].mean()) if in_mask.any() else float("nan")
    e_out = float(profile[out_mask].mean()) if out_mask.any() else float("nan")
    band_score = e_in - e_out

    # Peak frequency in the informative range (useful for inspecting FPs).
    masked = profile.copy()
    masked[~inf_mask] = -np.inf
    peak_bin = int(np.argmax(masked))
    peak_freq = peak_bin / (n_freq - 1) * nyq

    eps = 1e-9
    log_ratio = float(np.log(
        (max(e_in, 0.0) + eps) / (max(e_out, 0.0) + eps)
    ))

    return dict(
        E_in=e_in, E_out=e_out, band_score=band_score,
        log_band_ratio=log_ratio, peak_freq_hz=peak_freq,
    )


# ======================================================================
# Console summaries
# ======================================================================

def print_distribution_summary(df: pd.DataFrame) -> None:
    """Per-class distribution of band_score, split by TP / FP / FN."""
    print("\n" + "=" * 78)
    print("  Band-score distribution by class × kind")
    print("=" * 78)
    print(
        f"\n{'class':<8} {'kind':<5} {'n':>6} "
        f"{'P5':>9} {'P25':>9} {'median':>9} {'P75':>9} {'P95':>9}"
    )
    print("-" * 78)
    for cls in cfg.CALL_TYPES_3:
        for kind in ("TP", "FP", "FN"):
            sub = df[(df["class"] == cls) & (df["kind"] == kind)]
            if len(sub) == 0:
                continue
            s = sub["band_score"].dropna().values
            if len(s) == 0:
                continue
            print(
                f"{cls:<8} {kind:<5} {len(sub):>6} "
                f"{np.percentile(s, 5):>9.4f} "
                f"{np.percentile(s, 25):>9.4f} "
                f"{np.median(s):>9.4f} "
                f"{np.percentile(s, 75):>9.4f} "
                f"{np.percentile(s, 95):>9.4f}"
            )
        print()


def print_threshold_sweep(df: pd.DataFrame) -> None:
    """
    Simulated what-if: reject events with ``band_score < tau_c`` and
    recompute P/R/F1 per class. Tells you whether a band filter would
    help and where to set the threshold if it does.
    """
    print("\n" + "=" * 78)
    print("  Simulated band-filter threshold sweep (per class)")
    print("=" * 78)
    print(
        "Reject predicted events with band_score < tau_c, then recompute "
        "P / R / F1.\nKilled TPs become additional FNs. "
        "Pre-existing FNs are unaffected by the filter."
    )

    for cls in cfg.CALL_TYPES_3:
        tp_sub = df[(df["class"] == cls) & (df["kind"] == "TP")]
        fp_sub = df[(df["class"] == cls) & (df["kind"] == "FP")]
        fn_sub = df[(df["class"] == cls) & (df["kind"] == "FN")]

        n_tp = int(len(tp_sub))
        n_fp = int(len(fp_sub))
        n_fn = int(len(fn_sub))
        if n_tp + n_fp + n_fn == 0:
            continue

        prec0 = n_tp / max(n_tp + n_fp, 1)
        rec0 = n_tp / max(n_tp + n_fn, 1)
        f1_0 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-9)

        print(f"\n--- {cls.upper()} ---")
        print(
            f"  {'tau_c':>10}   "
            f"{'TP':>5} {'FP':>5} {'FN':>5}   "
            f"{'P':>6} {'R':>6} {'F1':>6}"
        )
        print(
            f"  {'(no filter)':>10}   "
            f"{n_tp:>5} {n_fp:>5} {n_fn:>5}   "
            f"{prec0:>6.3f} {rec0:>6.3f} {f1_0:>6.3f}"
        )

        tp_scores = tp_sub["band_score"].dropna().values
        fp_scores = fp_sub["band_score"].dropna().values
        all_scores = np.concatenate([tp_scores, fp_scores])
        if len(all_scores) == 0:
            continue

        # Sweep across percentiles of the observed score distribution —
        # gives a wide spread without manual scale-picking.
        candidate_taus = np.percentile(
            all_scores, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        )
        best_f1, best_tau = f1_0, None
        for tau in candidate_taus:
            n_tp_kept = int((tp_scores >= tau).sum())
            n_fp_kept = int((fp_scores >= tau).sum())
            n_fn_new = n_fn + (n_tp - n_tp_kept)
            p = n_tp_kept / max(n_tp_kept + n_fp_kept, 1)
            r = n_tp_kept / max(n_tp_kept + n_fn_new, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            marker = "  <-- new best" if f1 > best_f1 else ""
            print(
                f"  {tau:>10.4f}   "
                f"{n_tp_kept:>5} {n_fp_kept:>5} {n_fn_new:>5}   "
                f"{p:>6.3f} {r:>6.3f} {f1:>6.3f}{marker}"
            )
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)

        if best_tau is not None:
            delta = best_f1 - f1_0
            print(
                f"  >> Best tau_c = {best_tau:.4f}   "
                f"F1: {f1_0:.3f} -> {best_f1:.3f}   "
                f"(delta = {delta:+.3f})"
            )
        else:
            print("  >> No threshold improves F1; band filter wouldn't help.")


def print_overlap_summary(df: pd.DataFrame) -> None:
    """
    Quick rank-style separation metric: of all (TP, FP) pairs in a class,
    in what fraction does TP have a higher band_score than FP? This is
    the AUC of band_score as a TP-vs-FP classifier, and it's the cleanest
    one-number signal of whether the filter has any traction.
    """
    print("\n" + "=" * 78)
    print("  band_score AUC (TP-vs-FP separability per class)")
    print("=" * 78)
    print("AUC = P(band_score_TP > band_score_FP). 0.5 = no signal, "
          "1.0 = perfect.")
    print()
    for cls in cfg.CALL_TYPES_3:
        tp_scores = df.loc[
            (df["class"] == cls) & (df["kind"] == "TP"), "band_score"
        ].dropna().values
        fp_scores = df.loc[
            (df["class"] == cls) & (df["kind"] == "FP"), "band_score"
        ].dropna().values
        if len(tp_scores) == 0 or len(fp_scores) == 0:
            print(f"  {cls:<8}  (insufficient data: TP={len(tp_scores)} "
                  f"FP={len(fp_scores)})")
            continue
        # Standard Mann-Whitney U-based AUC, vectorised.
        tp_r = tp_scores[:, None]
        fp_r = fp_scores[None, :]
        wins = (tp_r > fp_r).sum() + 0.5 * (tp_r == fp_r).sum()
        auc = wins / (len(tp_scores) * len(fp_scores))
        verdict = (
            "strong signal" if auc > 0.70
            else "moderate signal" if auc > 0.60
            else "weak signal" if auc > 0.55
            else "essentially none"
        )
        print(f"  {cls:<8}  AUC = {auc:.3f}   "
              f"(TP n={len(tp_scores):4d}, FP n={len(fp_scores):4d})   "
              f"-> {verdict}")


# ======================================================================
# Main
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Two modes — exactly one must be supplied.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred-csv", type=Path,
                     help="Predictions CSV in hybrid_ensemble_predict "
                          "format. Use this if you already have a "
                          "predictions CSV from elsewhere.")
    src.add_argument("--checkpoints", nargs="+",
                     help="Checkpoint paths to ensemble. Replicates the "
                          "ensemble_predict_cached.py pipeline in-process "
                          "(cache hits in --cache-dir are reused, so this "
                          "is fast if you've already run that script).")

    # Ensemble-mode-only args
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for averaging. "
                        "Default: equal. (--checkpoints mode only.)")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("runs/prob_cache"),
                   help="Per-checkpoint prob cache (shared with "
                        "ensemble_predict_cached.py).")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable cache reads/writes (force recompute).")
    p.add_argument("--use-fp16", action="store_true",
                   help="FP16 autocast during inference.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                   help="Inference batch size.")

    # Common args
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Datasets to evaluate on. Default: "
                        "cfg.VAL_DATASETS. (Ignored in --checkpoints "
                        "mode, which always uses cfg.VAL_DATASETS to "
                        "match the cached pipeline.)")
    p.add_argument("--iou", type=float, default=0.3,
                   help="IoU threshold for TP matching (default 0.3, "
                        "matches BioDCASE).")
    p.add_argument("--margin-s", type=float, default=0.5,
                   help="Margin (s) around each event for spec extraction.")
    p.add_argument("--min-dur-s", type=float, default=1.0,
                   help="Min duration (s) of audio loaded per event.")
    p.add_argument("--out-csv", type=Path,
                   default=Path("band_diagnostic.csv"),
                   help="Per-event diagnostic CSV output.")
    p.add_argument("--save-preds-csv", type=Path, default=None,
                   help="If set, also write the ensemble's predictions "
                        "to this CSV (hybrid_ensemble_predict format). "
                        "Useful for re-running the diagnostic later "
                        "without redoing inference. (--checkpoints "
                        "mode only.)")
    p.add_argument("--device", type=str, default=None,
                   help="Override torch device (default: cuda if available).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Two source modes: pre-built CSV, or ensemble in-process
    # ------------------------------------------------------------------
    if args.checkpoints is not None:
        # In-process ensemble — must use cfg.VAL_DATASETS so the cached
        # probs (keyed by val segments) line up.
        if args.datasets is not None:
            print("Note: --datasets ignored in --checkpoints mode "
                  "(pipeline uses cfg.VAL_DATASETS).")
        datasets = list(cfg.VAL_DATASETS)

        print(f"\n[ensemble mode] {len(args.checkpoints)} checkpoint(s), "
              f"cache_dir={args.cache_dir}, fp16={args.use_fp16}")
        preds, gt_events, manifest, thresholds = run_ensemble_inference(
            checkpoint_paths=list(args.checkpoints),
            weights=args.weights,
            cache_dir=args.cache_dir,
            use_fp16=args.use_fp16,
            batch_size=args.batch_size,
            device=device,
            no_cache=args.no_cache,
        )

        # Optionally save the predictions CSV so the user can re-run the
        # diagnostic later without redoing inference.
        if args.save_preds_csv is not None:
            _save_predictions_csv(preds, args.save_preds_csv)

    else:
        # Pre-built CSV path.
        datasets = args.datasets or list(cfg.VAL_DATASETS)
        # Refuse to silently run on the held-out test set.
        test_sites = set(getattr(cfg, "EVAL_DATASETS", []))
        if test_sites and set(datasets) & test_sites:
            print(
                f"\n*** WARNING: requested datasets include held-out "
                f"test sites: {sorted(set(datasets) & test_sites)}. "
                f"This burns the hold-out. Continue only if "
                f"intentional.\n"
            )

        print(f"\n[csv mode] Loading predictions from {args.pred_csv}")
        preds = load_predictions_csv(args.pred_csv)
        print(f"  {len(preds)} predictions")

        print(f"\nBuilding GT for: {datasets}")
        gt_events, manifest = build_gt_events(datasets)
        print(f"  {len(gt_events)} GT events")

        ds_set = set(datasets)
        n_before = len(preds)
        preds = [p for p in preds if p.dataset in ds_set]
        if len(preds) != n_before:
            print(f"  filtered to {len(preds)} predictions "
                  f"inside target datasets")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    print(f"\nMatching predictions to GT at IoU >= {args.iou}")
    tps, fps, fns = match_predictions(preds, gt_events, args.iou)
    print(f"  TP = {len(tps)}")
    print(f"  FP = {len(fps)}")
    print(f"  FN = {len(fns)}")

    # Sanity-check P/R/F1 against the user's known ensemble numbers.
    print("\n  Per-class sanity check vs ensemble metrics:")
    for cls in cfg.CALL_TYPES_3:
        n_tp = sum(1 for d in tps if d.label == cls)
        n_fp = sum(1 for d in fps if d.label == cls)
        n_fn = sum(1 for d in fns if d.label == cls)
        p = n_tp / max(n_tp + n_fp, 1)
        r = n_tp / max(n_tp + n_fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"    {cls:<6} TP={n_tp:5} FP={n_fp:5} FN={n_fn:5}   "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # ------------------------------------------------------------------
    # Spectrogram extraction + band scoring
    # ------------------------------------------------------------------
    file_index = build_file_index(manifest)
    spec_extractor = SpectrogramExtractor().to(device)
    spec_extractor.eval()

    n_freq = cfg.N_FFT // 2 + 1
    nyq = cfg.SAMPLE_RATE / 2.0
    print(f"\nSpectrogram: n_freq={n_freq}, nyquist={nyq:.1f} Hz, "
          f"bin width={nyq / (n_freq - 1):.3f} Hz")

    print("\nScoring events...")
    all_rows: list[dict] = []
    n_missing = 0
    for kind, events in (("TP", tps), ("FP", fps), ("FN", fns)):
        for ev in tqdm(events, desc=f"{kind:<2} ({len(events):5d})"):
            profile = compute_event_profile(
                ev, file_index, spec_extractor, device,
                margin_s=args.margin_s, min_dur_s=args.min_dur_s,
            )
            if profile is None:
                n_missing += 1
                continue
            scores = compute_band_score(profile, ev.label, n_freq, nyq)
            row = {
                "kind": kind,
                "class": ev.label,
                "dataset": ev.dataset,
                "filename": ev.filename,
                "start_s": float(ev.start_s),
                "end_s": float(ev.end_s),
                "duration_s": float(ev.end_s - ev.start_s),
                "confidence": (
                    float(getattr(ev, "confidence", float("nan")))
                    if kind != "FN" else float("nan")
                ),
                **scores,
            }
            all_rows.append(row)

    if n_missing:
        print(f"  (skipped {n_missing} events: file not in manifest / "
              f"too short / read error)")

    # ------------------------------------------------------------------
    # Save and summarise
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote per-event diagnostic: {args.out_csv}  ({len(df)} rows)")

    if len(df) == 0:
        print("No events scored; nothing more to report.")
        return

    print_distribution_summary(df)
    print_overlap_summary(df)
    print_threshold_sweep(df)

    print("\n" + "=" * 78)
    print("  How to read this")
    print("=" * 78)
    print("""
- The AUC table is the single number to look at first. AUC > 0.65 means
  TPs and FPs are reliably separated by band_score, and a per-class
  band filter will help. AUC near 0.5 means the model is firing at
  acoustically band-consistent locations even when it's wrong — band
  filtering won't help and the bottleneck is elsewhere.

- In the distribution table, compare TP-median to FP-median per class:
  a clear gap (FP-median measurably below TP-median) is the same signal
  in a different form.

- The threshold sweep tells you the operating point. If the best-F1
  row shows F1 > 'no filter', you have a free precision boost; the tau_c
  values are absolute on the same band_score scale (E_in - E_out).

- For FNs the band_score should be HIGH (these are real calls the model
  missed). If FN band_scores look like FPs, your GT events are
  acoustically subtle and the recall problem isn't band-related.
""")


def _save_predictions_csv(preds: list[Detection], path: Path) -> None:
    """Mirror hybrid_ensemble_predict.save_predictions_csv format."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        preds, key=lambda d: (d.dataset, d.filename, d.start_s, d.label),
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "filename", "onset", "offset",
                    "event_label", "confidence"])
        for d in rows:
            w.writerow([
                d.dataset, d.filename,
                f"{d.start_s:.3f}", f"{d.end_s:.3f}",
                d.label, f"{getattr(d, 'confidence', 1.0):.4f}",
            ])
    print(f"  saved {len(rows)} predictions to {path}")


if __name__ == "__main__":
    main()
