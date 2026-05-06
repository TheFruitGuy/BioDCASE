"""
visualize_d_subtypes.py
=======================

Generates spectrogram comparison PNGs of D-class call subtypes so you can
see (with your eyes, not just numbers) whether they're acoustically
distinct enough to warrant specialised training.

For each subtype found in the training annotations, samples N random
example calls, loads a window of audio centred on each, and plots its
spectrogram. Saves one PNG per (split, subtype) pair plus a combined
grid PNG showing all subtypes side by side.

Run from BioDCASE/task/:

    python visualize_d_subtypes.py

Output: PNGs in d_diagnostics/ subfolder.

Notes
-----
- Uses scipy.signal.spectrogram (not the project's SpectrogramExtractor)
  for visualisation: simpler, single-channel, easier to read on screen.
  The frequency axis matches what the model sees because it's the same
  STFT parameters (n_fft=256, hop=128 ≈ 0.51s @ 250 Hz).
- Spectrograms are plotted in dB with a per-call min/max normalisation
  so individual calls are visually comparable regardless of recorded SNR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display, save-to-file only
import matplotlib.pyplot as plt
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
N_EXAMPLES_PER_SUBTYPE = 6      # examples per subtype shown
WINDOW_SECONDS = 8.0            # audio window centred on each call
N_FFT = 256                     # matches project's STFT
HOP = 32                        # ~0.13 s — finer than training, easier to read by eye
RANDOM_SEED = 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def find_fine_label_column(df: pd.DataFrame) -> str | None:
    """Return name of the fine-grained label column if present."""
    for c in ["label", "label_7class", "label_orig", "label_full"]:
        if c in df.columns and c != "label_3class":
            return c
    return None


def load_segment(
    path: str, start_s: float, end_s: float, sample_rate: int,
    window_s: float,
) -> tuple[np.ndarray, float, float]:
    """
    Load a `window_s`-second window centred on the call. Returns
    (audio, call_start_local_s, call_end_local_s) where the local
    timestamps mark where the call is inside the returned audio.
    """
    info = sf.info(path)
    file_dur = info.frames / info.samplerate
    call_dur = end_s - start_s

    # Centre window on the call midpoint, clamp to file boundaries.
    call_mid = 0.5 * (start_s + end_s)
    win_start = max(0.0, call_mid - 0.5 * window_s)
    win_end = min(file_dur, win_start + window_s)
    win_start = max(0.0, win_end - window_s)  # readjust if clamp ate the end

    s_off = int(win_start * sample_rate)
    n_read = int((win_end - win_start) * sample_rate)
    audio, sr = sf.read(path, start=s_off, frames=n_read,
                        dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != sample_rate:
        # Defensive: skip files that aren't the expected SR.
        return None, 0.0, 0.0

    return audio, start_s - win_start, end_s - win_start


def plot_call(
    ax, audio: np.ndarray, sr: int, call_t0: float, call_t1: float,
    title: str,
) -> None:
    """Plot a single spectrogram with a box marking the annotated call."""
    f, t, S = spectrogram(audio, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          mode="magnitude")
    S_db = 20.0 * np.log10(S + 1e-8)
    # Per-call normalisation so faint calls aren't visually dominated by loud ones.
    vmin, vmax = np.percentile(S_db, [5, 99])

    ax.imshow(S_db, origin="lower", aspect="auto",
              extent=[t.min(), t.max(), f.min(), f.max()],
              vmin=vmin, vmax=vmax, cmap="viridis")
    # Mark the actual call boundaries
    ax.axvline(call_t0, color="red", linewidth=1.0, alpha=0.7)
    ax.axvline(call_t1, color="red", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("freq (Hz)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)


def make_subtype_grid(
    examples: list[dict],
    title: str,
    out_path: Path,
    sample_rate: int,
) -> None:
    """Plot N example calls of a single subtype in a 2 x ceil(N/2) grid."""
    n = len(examples)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows),
                             squeeze=False)

    for i, ex in enumerate(examples):
        ax = axes[i // cols][i % cols]
        plot_call(ax, ex["audio"], sample_rate, ex["call_t0"], ex["call_t1"],
                  ex["caption"])
    # Hide any leftover empty axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_combined_grid(
    by_subtype: dict[str, list[dict]],
    split_name: str,
    out_path: Path,
    sample_rate: int,
    n_per_row: int = 4,
) -> None:
    """One PNG with rows=subtypes, cols=examples — best for direct comparison."""
    subtypes = list(by_subtype.keys())
    if not subtypes:
        return

    cols = n_per_row
    rows = len(subtypes)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows),
                             squeeze=False)

    for r, subtype in enumerate(subtypes):
        examples = by_subtype[subtype][:cols]
        for c in range(cols):
            ax = axes[r][c]
            if c < len(examples):
                ex = examples[c]
                plot_call(ax, ex["audio"], sample_rate,
                          ex["call_t0"], ex["call_t1"],
                          ex["caption"])
                if c == 0:
                    ax.set_ylabel(f"{subtype}\nfreq (Hz)", fontsize=8)
            else:
                ax.axis("off")

    fig.suptitle(f"{split_name}: D-class subtype comparison "
                 f"(red lines = call boundaries)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-split driver
# ----------------------------------------------------------------------

def process_split(
    split_name: str, datasets: list[str], sample_rate: int,
) -> None:
    print(f"\n=== {split_name} ===")
    ann = load_annotations(datasets)
    manifest = get_file_manifest(datasets)

    fine_col = find_fine_label_column(ann)
    if fine_col is None:
        print(f"  no fine-grained label column — skipping")
        return

    d = ann[ann["label_3class"] == "d"].copy()
    if d.empty:
        print(f"  no D-class annotations")
        return

    # Build a (dataset, filename) -> path lookup from the manifest.
    if "path" not in manifest.columns:
        print("  manifest has no 'path' column — cannot load audio. Aborting "
              "this split.")
        return
    path_lookup = {
        (row["dataset"], row["filename"]): row["path"]
        for _, row in manifest.iterrows()
    }

    rng = np.random.default_rng(RANDOM_SEED)
    by_subtype: dict[str, list[dict]] = {}

    for subtype, group in d.groupby(fine_col):
        n_avail = len(group)
        if n_avail == 0:
            continue
        # Random sample without replacement, capped at requested count
        take = min(N_EXAMPLES_PER_SUBTYPE, n_avail)
        idx = rng.choice(n_avail, size=take, replace=False)
        picked = group.iloc[idx]

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
            cap = (f"{row['dataset']}\n"
                   f"{row['filename'][:24]}…\n"
                   f"call {row['start_s']:.1f}-{row['end_s']:.1f}s")
            examples.append({
                "audio": audio,
                "call_t0": ct0,
                "call_t1": ct1,
                "caption": cap,
            })
        if examples:
            by_subtype[str(subtype)] = examples
            print(f"  {subtype}: {len(examples)} examples loaded")

    if not by_subtype:
        print(f"  no plottable examples for {split_name}")
        return

    # Per-subtype individual grids
    for subtype, examples in by_subtype.items():
        safe_name = "".join(c if c.isalnum() else "_" for c in str(subtype))
        out = OUT_DIR / f"{split_name}_{safe_name}.png"
        make_subtype_grid(
            examples,
            title=f"{split_name}: subtype = '{subtype}'  (N={len(examples)})",
            out_path=out,
            sample_rate=sample_rate,
        )

    # Combined comparison grid
    out_combined = OUT_DIR / f"{split_name}_subtype_comparison.png"
    make_combined_grid(by_subtype, split_name, out_combined, sample_rate)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Sample rate (from cfg): {cfg.SAMPLE_RATE}")
    print(f"Window per call: {WINDOW_SECONDS}s")
    print(f"Examples per subtype: {N_EXAMPLES_PER_SUBTYPE}")

    process_split("train", cfg.TRAIN_DATASETS, cfg.SAMPLE_RATE)
    process_split("val", cfg.VAL_DATASETS, cfg.SAMPLE_RATE)

    print(f"\nDone. PNGs are in {OUT_DIR.resolve()}/")
    print(f"Download with e.g.:")
    print(f"  scp 'matthias-nagl@<server>:~/BioDCASE/task/{OUT_DIR}/*.png' .")


if __name__ == "__main__":
    main()
