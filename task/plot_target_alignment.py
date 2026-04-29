"""
Target-Alignment Diagnostic Plot
================================

Picks a positive segment from the training set and saves a two-panel
PNG showing:

    (top)    The magnitude spectrogram of the segment.
    (bottom) The per-frame target tensor at 20 ms resolution, one row
             per class.

If the red bars in the bottom panel sit directly under the call energy
in the top panel, your targets are aligned correctly. If they're shifted,
truncated, or in the wrong class row — that's a data-pipeline bug and
no amount of training tweaks will fix the F1 gap.

By default this plots one D-class segment (the failing class) plus one
BMABZ segment (the working class) for comparison. Override the focal
class with --class.

Usage
-----
::

    python plot_target_alignment.py
    python plot_target_alignment.py --class d --n_samples 3
    python plot_target_alignment.py --output debug_plots/

Notes
-----
Runs entirely independently of training — safe to invoke in a separate
terminal while ``train.py`` is running. Uses the same cached manifest
and annotations, so startup is fast after the first training run.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
from dataset import (
    load_annotations, get_file_manifest,
    build_positive_segments, WhaleDataset,
)
from spectrogram import SpectrogramExtractor


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--class", dest="cls", type=str, default=None,
        help="Class to plot ('bmabz', 'd', or 'bp'). If omitted, plots one "
             "of each class in turn.",
    )
    p.add_argument(
        "--n_samples", type=int, default=2,
        help="Number of segments per class to plot (default: 2).",
    )
    p.add_argument(
        "--output", type=str, default="./debug_plots",
        help="Output directory for PNG files (default: ./debug_plots).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for picking which segments to plot. Change this "
             "to see different segments.",
    )
    return p.parse_args()


def find_segments_with_class(segments, target_class, n_wanted, rng):
    """
    Return up to ``n_wanted`` segments (with their indices) whose
    annotations include the target class.

    Parameters
    ----------
    segments : list of Segment
    target_class : str
        Class name to filter for, e.g. ``"d"``.
    n_wanted : int
    rng : numpy.random.Generator

    Returns
    -------
    list of (idx, Segment)
    """
    label_key = "label_3class" if cfg.USE_3CLASS else "label"
    candidates = [
        (i, s) for i, s in enumerate(segments)
        if any(a[label_key] == target_class for a in s.annotations)
    ]
    if not candidates:
        return []

    if len(candidates) <= n_wanted:
        return candidates

    # Reproducible random pick.
    chosen = rng.choice(len(candidates), size=n_wanted, replace=False)
    return [candidates[i] for i in sorted(chosen)]


def plot_segment(idx, seg, ds, spec_ext, out_path):
    """
    Render and save the two-panel diagnostic plot for one segment.

    Parameters
    ----------
    idx : int
        Index in the segments list (used in the title only).
    seg : Segment
        The segment being plotted.
    ds : WhaleDataset
        Dataset wrapper used to load audio + targets.
    spec_ext : SpectrogramExtractor
        Used to compute the magnitude spectrogram.
    out_path : Path
        Destination PNG path.
    """
    # Pull the same (audio, targets) pair the model would see during training.
    audio, targets, _, meta = ds[idx]

    # Magnitude channel only — the cos/sin phase channels would just look
    # like coloured noise here.
    spec = spec_ext(audio.unsqueeze(0))[0, 0].numpy()

    # Segment duration in seconds — used as the x-axis extent for both
    # panels so they share a consistent time axis without per-frame
    # coordinate arrays.
    seg_dur_s = (seg.end_sample - seg.start_sample) / cfg.SAMPLE_RATE

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- Spectrogram (top) ----------------------------------------------
    # imshow with extent= places the image in data coordinates so it lines
    # up correctly with the time axis we share with the targets panel.
    # Avoids the dimension-mismatch quirk older matplotlib has with
    # pcolormesh and exact-size coordinate arrays.
    ax1.imshow(
        spec, aspect="auto", origin="lower", cmap="viridis",
        extent=(0.0, seg_dur_s, 0.0, cfg.SAMPLE_RATE / 2),
    )
    ax1.set_ylabel("Frequency (Hz)")
    labels = [a[("label_3class" if cfg.USE_3CLASS else "label")]
              for a in seg.annotations]
    ax1.set_title(
        f"Segment {idx}  •  {meta['dataset']} / {meta['filename']}\n"
        f"Annotations in segment: {labels}"
    )

    # Overlay the original annotation boundaries as faint vertical lines
    # for visual cross-check. We use file-relative seconds; convert to
    # segment-relative.
    seg_start_s = seg.start_sample / cfg.SAMPLE_RATE
    for a in seg.annotations:
        local_start = a["start_s"] - seg_start_s
        local_end = a["end_s"] - seg_start_s
        ax1.axvline(local_start, color="white", linewidth=0.8, alpha=0.6,
                    linestyle="--")
        ax1.axvline(local_end, color="white", linewidth=0.8, alpha=0.6,
                    linestyle="--")

    # --- Target tensor (bottom) -----------------------------------------
    # Each row is one class; brighter = positive frame. The class names go
    # on the y-axis so it's instantly clear which class the model would
    # be trained against.
    class_names = cfg.class_names()
    ax2.imshow(
        targets.T.numpy(), aspect="auto", origin="lower", cmap="Reds",
        vmin=0, vmax=1,
        extent=(0.0, seg_dur_s, 0.0, len(class_names)),
    )
    # Tick at the centre of each row.
    ax2.set_yticks(np.arange(len(class_names)) + 0.5)
    ax2.set_yticklabels(class_names)
    ax2.set_ylabel("Class")
    ax2.set_xlabel("Time within segment (s)")
    ax2.set_xlim(0, seg_dur_s)

    # Same annotation boundary overlays in the target panel for direct
    # comparison.
    for a in seg.annotations:
        local_start = a["start_s"] - seg_start_s
        local_end = a["end_s"] - seg_start_s
        ax2.axvline(local_start, color="black", linewidth=0.8, alpha=0.6,
                    linestyle="--")
        ax2.axvline(local_end, color="black", linewidth=0.8, alpha=0.6,
                    linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    """Entry point."""
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training annotations and manifest (cached)...")
    annotations = load_annotations(cfg.TRAIN_DATASETS)
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)

    print("Building positive segments...")
    segments = build_positive_segments(annotations, manifest)
    print(f"  {len(segments)} positive segments available")

    ds = WhaleDataset(segments)
    spec_ext = SpectrogramExtractor()
    rng = np.random.default_rng(args.seed)

    classes_to_plot = [args.cls] if args.cls else cfg.class_names()

    for cls in classes_to_plot:
        print(f"\nLooking for {args.n_samples} {cls!r}-class segments...")
        picks = find_segments_with_class(segments, cls, args.n_samples, rng)
        if not picks:
            print(f"  ⚠️  No segments found containing class {cls!r}.")
            continue

        for idx, seg in picks:
            out_path = out_dir / f"target_align_{cls}_seg{idx}.png"
            plot_segment(idx, seg, ds, spec_ext, out_path)

    print("\nDone. Open the PNGs and check that:")
    print("  - The bright energy bursts in the top panel correspond to the")
    print("    times where there are red bars in the bottom panel.")
    print("  - The red bars appear in the correct class row (e.g. d-class")
    print("    annotations should fill the 'd' row, not 'bmabz' or 'bp').")
    print("  - The dashed vertical lines (annotation boundaries) line up")
    print("    between the two panels.")


if __name__ == "__main__":
    main()
