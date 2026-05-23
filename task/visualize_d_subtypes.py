"""
visualize_d_subtypes.py  (v2 — fixed for actual schema)
========================================================

Generates spectrogram comparison PNGs of D-class calls so you can see
with your eyes whether they look acoustically distinct enough to warrant
specialised training.

If the diagnostic showed a bimodal frequency distribution, this script
splits D-class annotations into low-band and high-band groups by
low_frequency and plots them as separate "subtypes". Otherwise, it
plots all D-class calls as one group.

Run from BioDCASE/task/:

    python visualize_d_subtypes.py

Output: PNGs in d_diagnostics/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram

import config as cfg
from dataset import load_annotations, get_file_manifest


# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------

OUT_DIR = Path("d_diagnostics")
N_EXAMPLES_PER_SUBTYPE = 8
WINDOW_SECONDS = 8.0
N_FFT = 256
HOP = 32
RANDOM_SEED = 0

# If only one fine-label exists, we synthesize subtypes by binning on
# low_frequency. Anything below this threshold goes to "low_band",
# anything above to "high_band". 30 Hz is a reasonable split between
# fin-whale 20-Hz pulses and blue-whale D-calls.
SYNTHETIC_FREQ_SPLIT_HZ = 30.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def find_fine_col(df: pd.DataFrame) -> str | None:
    for c in ["annotation", "label", "label_7class", "label_orig", "label_full"]:
        if c in df.columns and c != "label_3class":
            return c
    return None


def attach_offsets(ann: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Annotations come with absolute datetimes. Audio is read by sample
    offset. Compute (start_s, end_s) relative to each file's start time.
    """
    if "start_dt" not in manifest.columns:
        raise RuntimeError(
            "manifest has no 'start_dt' column — cannot map annotation "
            "datetimes to file offsets. Inspect get_file_manifest output."
        )

    file_start = {
        (row["dataset"], row["filename"]): pd.to_datetime(row["start_dt"], utc=True)
        for _, row in manifest.iterrows()
    }

    ann = ann.copy()
    ann["start_dt"] = pd.to_datetime(ann["start_datetime"], utc=True, errors="coerce")
    ann["end_dt"] = pd.to_datetime(ann["end_datetime"], utc=True, errors="coerce")

    keys = list(zip(ann["dataset"], ann["filename"]))
    ann["file_start_dt"] = [file_start.get(k) for k in keys]

    ann["start_s"] = (ann["start_dt"] - ann["file_start_dt"]).dt.total_seconds()
    ann["end_s"] = (ann["end_dt"] - ann["file_start_dt"]).dt.total_seconds()
    return ann


def load_segment(
    path: str, start_s: float, end_s: float, sample_rate: int, window_s: float,
) -> tuple[np.ndarray | None, float, float]:
    info = sf.info(path)
    file_dur = info.frames / info.samplerate
    call_mid = 0.5 * (start_s + end_s)
    win_start = max(0.0, call_mid - 0.5 * window_s)
    win_end = min(file_dur, win_start + window_s)
    win_start = max(0.0, win_end - window_s)

    s_off = int(win_start * sample_rate)
    n_read = int((win_end - win_start) * sample_rate)
    if n_read <= 0:
        return None, 0.0, 0.0
    audio, sr = sf.read(path, start=s_off, frames=n_read,
                        dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != sample_rate:
        return None, 0.0, 0.0
    return audio, start_s - win_start, end_s - win_start


def plot_call(
    ax, audio: np.ndarray, sr: int, call_t0: float, call_t1: float,
    title: str, low_hz: float | None = None, high_hz: float | None = None,
) -> None:
    f, t, S = spectrogram(audio, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          mode="magnitude")
    S_db = 20.0 * np.log10(S + 1e-8)
    vmin, vmax = np.percentile(S_db, [5, 99])

    ax.imshow(S_db, origin="lower", aspect="auto",
              extent=[t.min(), t.max(), f.min(), f.max()],
              vmin=vmin, vmax=vmax, cmap="viridis")
    # Vertical: call start/end
    ax.axvline(call_t0, color="red", linewidth=1.2, alpha=0.8)
    ax.axvline(call_t1, color="red", linewidth=1.2, alpha=0.8)
    # Horizontal: annotated low/high frequency band, if available
    if low_hz is not None and not pd.isna(low_hz):
        ax.axhline(low_hz, color="white", linewidth=0.8, alpha=0.6,
                   linestyle="--")
    if high_hz is not None and not pd.isna(high_hz):
        ax.axhline(high_hz, color="white", linewidth=0.8, alpha=0.6,
                   linestyle="--")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("freq (Hz)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)


def make_combined_grid(
    by_subtype: dict[str, list[dict]], split_name: str,
    out_path: Path, sample_rate: int, n_per_row: int,
) -> None:
    subtypes = list(by_subtype.keys())
    if not subtypes:
        return

    cols = n_per_row
    rows = len(subtypes)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows),
                             squeeze=False)

    for r, subtype in enumerate(subtypes):
        examples = by_subtype[subtype][:cols]
        for c in range(cols):
            ax = axes[r][c]
            if c < len(examples):
                ex = examples[c]
                plot_call(ax, ex["audio"], sample_rate,
                          ex["call_t0"], ex["call_t1"], ex["caption"],
                          low_hz=ex.get("low_hz"), high_hz=ex.get("high_hz"))
                if c == 0:
                    ax.set_ylabel(f"{subtype}\nfreq (Hz)", fontsize=8)
            else:
                ax.axis("off")

    legend_elements = [
        mpatches.Patch(color="red", label="annotation start/end (vertical)"),
        mpatches.Patch(color="white", label="annotated low/high frequency (horizontal dashed)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=8, ncol=2)
    fig.suptitle(f"{split_name}: D-class subtype comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-split driver
# ----------------------------------------------------------------------

def process_split(split_name: str, datasets: list[str], sample_rate: int) -> None:
    print(f"\n=== {split_name} ===")
    ann = load_annotations(datasets)
    manifest = get_file_manifest(datasets)

    if "path" not in manifest.columns:
        print("  manifest has no 'path' column — cannot resolve audio paths.")
        return

    fine_col = find_fine_col(ann)
    if fine_col is None:
        print("  no fine-label column found — cannot subtype by label")
        return

    d = ann[ann["label_3class"] == "d"].copy()
    if d.empty:
        print("  no D-class annotations")
        return

    try:
        d = attach_offsets(d, manifest)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    d = d.dropna(subset=["start_s", "end_s"])

    path_lookup = {
        (row["dataset"], row["filename"]): row["path"]
        for _, row in manifest.iterrows()
    }

    # Decide grouping. If only one fine label, fall back to synthetic
    # frequency-band groups.
    fine_labels = d[fine_col].value_counts()
    print(f"  fine labels in D-class: {dict(fine_labels)}")

    if len(fine_labels) > 1:
        groups = [(str(lab), d[d[fine_col] == lab]) for lab, _ in fine_labels.items()]
        grouping_desc = f"by '{fine_col}'"
    else:
        if "low_frequency" not in d.columns:
            print("  single fine label and no low_frequency column — "
                  "no way to subtype")
            return
        d["low_frequency"] = pd.to_numeric(d["low_frequency"], errors="coerce")
        d_lo = d[d["low_frequency"] < SYNTHETIC_FREQ_SPLIT_HZ]
        d_hi = d[d["low_frequency"] >= SYNTHETIC_FREQ_SPLIT_HZ]
        groups = [
            (f"low_band (<{SYNTHETIC_FREQ_SPLIT_HZ:.0f} Hz)", d_lo),
            (f"high_band (≥{SYNTHETIC_FREQ_SPLIT_HZ:.0f} Hz)", d_hi),
        ]
        grouping_desc = (f"synthetic split on low_frequency at "
                         f"{SYNTHETIC_FREQ_SPLIT_HZ:.0f} Hz "
                         f"(only one fine label present)")
    print(f"  grouping: {grouping_desc}")

    rng = np.random.default_rng(RANDOM_SEED)
    by_subtype: dict[str, list[dict]] = {}

    for group_name, group_df in groups:
        n_avail = len(group_df)
        if n_avail == 0:
            print(f"  {group_name}: 0 annotations — skipping")
            continue
        take = min(N_EXAMPLES_PER_SUBTYPE, n_avail)
        idx = rng.choice(n_avail, size=take, replace=False)
        picked = group_df.iloc[idx]

        examples = []
        for _, row in picked.iterrows():
            key = (row["dataset"], row["filename"])
            if key not in path_lookup:
                continue
            audio, ct0, ct1 = load_segment(
                path_lookup[key],
                float(row["start_s"]), float(row["end_s"]),
                sample_rate, WINDOW_SECONDS,
            )
            if audio is None or len(audio) < int(0.5 * WINDOW_SECONDS * sample_rate):
                continue
            lo_hz = row.get("low_frequency", None)
            hi_hz = row.get("high_frequency", None)
            try:
                lo_hz = float(lo_hz) if lo_hz is not None else None
            except (TypeError, ValueError):
                lo_hz = None
            try:
                hi_hz = float(hi_hz) if hi_hz is not None else None
            except (TypeError, ValueError):
                hi_hz = None
            cap = (f"{row['dataset']}\n"
                   f"{row['filename'][:24]}\n"
                   f"{lo_hz:.0f}-{hi_hz:.0f}Hz, {row['end_s'] - row['start_s']:.1f}s"
                   if lo_hz is not None and hi_hz is not None else
                   f"{row['dataset']}\n"
                   f"{row['filename'][:24]}\n"
                   f"call {row['start_s']:.0f}-{row['end_s']:.0f}s")
            examples.append({
                "audio": audio,
                "call_t0": ct0,
                "call_t1": ct1,
                "caption": cap,
                "low_hz": lo_hz,
                "high_hz": hi_hz,
            })
        if examples:
            by_subtype[group_name] = examples
            print(f"  {group_name}: {len(examples)} examples loaded")

    if not by_subtype:
        print("  no plottable examples — skipping PNG generation")
        return

    out_combined = OUT_DIR / f"{split_name}_subtype_comparison.png"
    make_combined_grid(by_subtype, split_name, out_combined, sample_rate,
                       n_per_row=N_EXAMPLES_PER_SUBTYPE)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Sample rate (cfg): {cfg.SAMPLE_RATE}")
    print(f"Window per call: {WINDOW_SECONDS}s")
    print(f"Examples per subtype: {N_EXAMPLES_PER_SUBTYPE}")
    print(f"Synthetic split threshold: {SYNTHETIC_FREQ_SPLIT_HZ} Hz")

    process_split("train", cfg.TRAIN_DATASETS, cfg.SAMPLE_RATE)
    process_split("val", cfg.VAL_DATASETS, cfg.SAMPLE_RATE)

    print(f"\nDone. PNGs in {OUT_DIR.resolve()}/")
    print(f"\nDownload from your laptop with:")
    print(f"  scp matthias-nagl@<server>.cp.jku.at:'~/BioDCASE/task/{OUT_DIR}/*.png' .")


if __name__ == "__main__":
    main()
