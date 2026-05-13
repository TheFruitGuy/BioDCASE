"""
Stage 1: Build the Call Bank
=============================

Extracts every annotated call from the training sites and packs them
into a single ``call_bank.pt`` file consumed by the call-splice
augmentation. Each entry stores the call's audio waveform (resampled
to ``cfg.SAMPLE_RATE``) plus its time-frequency metadata so the
augmentation can bandpass-filter and SNR-mix at training time.

Pipeline position
-----------------
This is the first script to run. It only reads from ``cfg.TRAIN_DATASETS``;
the validation and evaluation sites are not touched.

Why store raw (not bandpassed) audio
------------------------------------
The bandpass margin and Hann taper length are augmentation
hyperparameters that we may want to tune without rebuilding the bank.
A 5 s call at 250 Hz = 1250 samples ≈ 5 kB; storing ~10 k calls is
under 50 MB. The on-the-fly bandpass inside the augmentation is
negligible compute per batch.

Output schema
-------------
``call_bank.pt`` is a torch-pickled dict::

    {
        "calls": [
            {
                "audio":          torch.float32, shape (n_samples,)
                "f_low_hz":       float
                "f_high_hz":      float
                "label_7class":   str   (e.g. "bma", "bmz", "bp20plus")
                "label_3class":   str   (e.g. "bmabz", "d", "bp")
                "source_site":    str
                "source_file":    str
                "duration_s":     float
                "sample_rate":    int   (always cfg.SAMPLE_RATE)
            },
            ...
        ],
        "by_label_3class": {"bmabz": [idx, ...], "d": [...], "bp": [...]},
        "by_label_7class": {"bma": [...], ...},
        "by_site":         {"ballenyislands2015": [...], ...},
        "config": {
            "sample_rate":      int,
            "pre_pad_s":        float,
            "post_pad_s":       float,
            "min_duration_s":   float,
            "max_duration_s":   float,
            "n_calls_total":    int,
        },
    }

The three flat lookup dicts let the augmentation sample by class
(balanced or weighted), by site, or by composite key without scanning
the full list.

Usage
-----
::

    python 01_build_call_bank.py --out call_bank.pt

Use ``--datasets`` to override which sites contribute. Default is
``cfg.TRAIN_DATASETS``.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

# Make the parent project importable when running this script from
# inside ``pipeline_call_splice/``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from dataset import get_file_manifest, load_annotations  # noqa: E402


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Resample 1-D audio from ``src_sr`` to ``dst_sr``.

    Uses ``scipy.signal.resample_poly`` (polyphase filter); preserves
    the call's spectral content well at low SR ratios and avoids
    introducing zero-crossings that a naive FFT resample would.
    """
    if src_sr == dst_sr:
        return audio
    from math import gcd
    g = gcd(src_sr, dst_sr)
    from scipy.signal import resample_poly
    return resample_poly(audio, up=dst_sr // g, down=src_sr // g).astype(np.float32)


def _read_call_audio(
    file_path: Path,
    call_start_s: float,
    call_end_s: float,
    pre_pad_s: float,
    post_pad_s: float,
    target_sr: int,
) -> tuple[np.ndarray | None, int]:
    """
    Read a single call segment from a WAV file with padding.

    The padding margin gives the runtime augmentation room to apply
    the Hann taper without eating into the call's signal. If the call
    sits near a file boundary the pad is clamped.

    Returns
    -------
    audio : np.ndarray or None
        Float32 waveform resampled to ``target_sr``. ``None`` if the
        file is missing or the read fails.
    sample_rate : int
        Returned for logging; always equals ``target_sr`` on success.
    """
    try:
        info = sf.info(str(file_path))
    except RuntimeError:
        return None, 0
    src_sr = info.samplerate
    file_duration_s = info.frames / src_sr

    start_s = max(0.0, call_start_s - pre_pad_s)
    end_s = min(file_duration_s, call_end_s + post_pad_s)
    start_frame = int(start_s * src_sr)
    end_frame = int(end_s * src_sr)
    if end_frame <= start_frame:
        return None, 0

    try:
        audio, _ = sf.read(
            str(file_path), start=start_frame, stop=end_frame, dtype="float32",
        )
    except RuntimeError:
        return None, 0
    if audio.ndim > 1:                 # stereo → mono mixdown
        audio = audio.mean(axis=1)
    audio = _resample(audio, src_sr, target_sr)
    return audio.astype(np.float32), target_sr


def build_call_bank(
    datasets: list[str],
    pre_pad_s: float,
    post_pad_s: float,
    out_path: Path,
) -> None:
    """
    Walk all annotations from ``datasets`` and extract each call.

    Annotations whose duration falls outside
    ``[cfg.MIN_CALL_DURATION_S, cfg.MAX_CALL_DURATION_S]`` are skipped
    silently; these are the same bounds the main training pipeline uses
    in ``dataset.build_positive_segments``.

    Parameters
    ----------
    datasets : list of str
        Source sites. Should be a subset of ``cfg.TRAIN_DATASETS``.
    pre_pad_s, post_pad_s : float
        Extra audio to include before/after the annotated call. Gives
        the runtime Hann taper room to ramp without clipping the call.
    out_path : Path
        Destination ``.pt`` file.
    """
    # Quarantine assertion: never extract from evaluation sites.
    forbidden = set(cfg.EVAL_DATASETS) & set(datasets)
    if forbidden:
        raise RuntimeError(
            f"Refusing to extract calls from test sites: {sorted(forbidden)}. "
            f"These are BioDCASE evaluation sites and must remain held out."
        )

    print(f"Building manifest for {len(datasets)} site(s)...")
    manifest = get_file_manifest(datasets)
    print(f"  {len(manifest)} files in manifest")

    print(f"Loading annotations for {len(datasets)} site(s)...")
    annotations = load_annotations(datasets, manifest=manifest)
    print(f"  {len(annotations)} annotations before duration filter")

    manifest_idx = manifest.set_index(["dataset", "filename"])

    calls: list[dict] = []
    by_label_3class: dict[str, list[int]] = defaultdict(list)
    by_label_7class: dict[str, list[int]] = defaultdict(list)
    by_site: dict[str, list[int]] = defaultdict(list)

    skipped_duration = 0
    skipped_io = 0
    skipped_no_file = 0

    for _, row in tqdm(
        annotations.iterrows(), total=len(annotations), desc="Extracting calls",
    ):
        key = (row["dataset"], row["filename"])
        if key not in manifest_idx.index:
            skipped_no_file += 1
            continue
        file_row = manifest_idx.loc[key]
        file_start_dt = file_row["start_dt"]
        if file_start_dt is None or pd.isna(file_start_dt):
            skipped_no_file += 1
            continue

        call_start_s = (row["start_datetime"] - file_start_dt).total_seconds()
        call_end_s = (row["end_datetime"] - file_start_dt).total_seconds()
        duration_s = call_end_s - call_start_s

        # Same duration filters as the main training pipeline.
        if (duration_s <= 0
                or duration_s < cfg.MIN_CALL_DURATION_S
                or duration_s > cfg.MAX_CALL_DURATION_S):
            skipped_duration += 1
            continue

        # Track where the true call (annotation bbox) sits within the
        # padded extract so the augmentation can mark targets at the
        # actual call extent rather than the padded-clip extent.
        clip_start_s = max(0.0, call_start_s - pre_pad_s)
        true_start_offset_s = call_start_s - clip_start_s

        audio, sr = _read_call_audio(
            Path(file_row["path"]),
            call_start_s=call_start_s,
            call_end_s=call_end_s,
            pre_pad_s=pre_pad_s,
            post_pad_s=post_pad_s,
            target_sr=cfg.SAMPLE_RATE,
        )
        if audio is None or len(audio) < int(duration_s * cfg.SAMPLE_RATE * 0.5):
            skipped_io += 1
            continue

        # Sanity-check frequency bounds. Annotations occasionally have
        # zero or inverted frequencies; clip to Nyquist.
        f_low = float(row["low_frequency"])
        f_high = float(row["high_frequency"])
        if f_high <= f_low or f_high > cfg.SAMPLE_RATE / 2:
            skipped_duration += 1
            continue
        f_low = max(1.0, f_low)
        f_high = min(cfg.SAMPLE_RATE / 2 - 1.0, f_high)

        entry = {
            "audio":               torch.from_numpy(audio),
            "f_low_hz":            f_low,
            "f_high_hz":           f_high,
            "label_7class":        str(row["annotation"]),
            "label_3class":        str(row["label_3class"]),
            "source_site":         str(row["dataset"]),
            "source_file":         str(row["filename"]),
            "duration_s":          float(duration_s),
            "true_start_offset_s": float(true_start_offset_s),
            "sample_rate":         cfg.SAMPLE_RATE,
        }
        idx = len(calls)
        calls.append(entry)
        by_label_3class[entry["label_3class"]].append(idx)
        by_label_7class[entry["label_7class"]].append(idx)
        by_site[entry["source_site"]].append(idx)

    print()
    print(f"Extracted {len(calls)} calls")
    print(f"  skipped (duration/freq out of range):   {skipped_duration}")
    print(f"  skipped (file I/O failure):             {skipped_io}")
    print(f"  skipped (no matching file in manifest): {skipped_no_file}")
    print()
    print("Per 3-class label:")
    for k in sorted(by_label_3class):
        print(f"  {k:<8} {len(by_label_3class[k]):5d}")
    print("Per 7-class label:")
    for k in sorted(by_label_7class):
        print(f"  {k:<10} {len(by_label_7class[k]):5d}")
    print("Per source site:")
    for k in sorted(by_site):
        print(f"  {k:<22} {len(by_site[k]):5d}")

    bank = {
        "calls": calls,
        "by_label_3class": dict(by_label_3class),
        "by_label_7class": dict(by_label_7class),
        "by_site": dict(by_site),
        "config": {
            "sample_rate":     cfg.SAMPLE_RATE,
            "pre_pad_s":       pre_pad_s,
            "post_pad_s":      post_pad_s,
            "min_duration_s":  cfg.MIN_CALL_DURATION_S,
            "max_duration_s":  cfg.MAX_CALL_DURATION_S,
            "n_calls_total":   len(calls),
        },
    }
    torch.save(bank, out_path)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, default=Path("call_bank.pt"),
        help="Destination .pt file.",
    )
    p.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Source sites (default: cfg.TRAIN_DATASETS).",
    )
    p.add_argument(
        "--pre_pad_s", type=float, default=0.3,
        help="Audio padding before the annotated call onset, for Hann taper.",
    )
    p.add_argument(
        "--post_pad_s", type=float, default=0.3,
        help="Audio padding after the annotated call offset, for Hann taper.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = args.datasets if args.datasets else cfg.TRAIN_DATASETS
    build_call_bank(
        datasets=datasets,
        pre_pad_s=args.pre_pad_s,
        post_pad_s=args.post_pad_s,
        out_path=args.out,
    )
