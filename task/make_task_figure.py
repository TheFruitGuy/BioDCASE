#!/usr/bin/env python
"""Task-illustration figure for the paper: a real ATBFL spectrogram with the
strongly-labelled call boundaries drawn as per-class boxes (BMABZ / D / BP).

Run from the NAVE repo root (needs nave_config.py, dataset.py, and the
development data at cfg.DATA_ROOT). It reuses your own annotation loader, so the
datetime->file mapping and the 7->3 class collapse match training exactly.

    python make_task_figure.py                          # auto-pick a good window
    python make_task_figure.py --dataset kerguelen2015  # try another site
    python make_task_figure.py --file 2014-... --t0 120 # force a specific window
    python make_task_figure.py --window 60 --fmax 100 --out fig_task.pdf

Boxes span each call's annotated TIME extent (full height); the annotations have
no frequency bounds, so vertical position is not annotated -- the spectrogram
shows the call's frequency content.
"""
from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import nave_config as cfg
import dataset as ds

# Okabe-Ito, colour-blind safe. Match your paper palette if it differs.
CLASS_COLOR = {"bmabz": "#0072B2", "d": "#D55E00", "bp": "#009E73"}
CLASS_NAME = {"bmabz": "BMABZ", "d": "D", "bp": "BP"}
ORDER = ["bmabz", "d", "bp"]


def _col(df, *candidates):
    """Return the first present column name from candidates (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _resolve_path(file_row, manifest):
    """Best-effort absolute path to the wav for a manifest row."""
    pcol = _col(manifest, "path", "filepath", "file_path", "wav_path", "abspath")
    if pcol is not None and isinstance(file_row[pcol], str):
        return file_row[pcol]
    raise SystemExit(
        "Could not find a path column in the manifest. Columns are:\n  "
        + ", ".join(manifest.columns)
        + "\nEdit _resolve_path() to point at the right one."
    )


def pick_window(events, window_s):
    """events: list of (start_s, end_s, cls). Return t0 maximising class
    coverage then event count within [t0, t0+window_s]."""
    best = (-1, -1, 0.0)  # (n_classes, n_events, t0)
    starts = sorted(e[0] for e in events)
    for t0 in starts:
        win = [e for e in events if e[0] < t0 + window_s and e[1] > t0]
        n_cls = len({e[2] for e in win})
        score = (n_cls, len(win), t0)
        if score[:2] > best[:2]:
            best = score
    return best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="casey2017")
    ap.add_argument("--window", type=float, default=90.0, help="window length [s]")
    ap.add_argument("--fmax", type=float, default=None, help="max frequency [Hz]")
    ap.add_argument("--db-range", type=float, default=70.0, help="dynamic range [dB]")
    ap.add_argument("--file", default=None, help="force a wav filename")
    ap.add_argument("--t0", type=float, default=None, help="force window start [s]")
    ap.add_argument("--out", default="fig_task.pdf")
    args = ap.parse_args()

    sr = cfg.SAMPLE_RATE

    manifest = ds.get_file_manifest([args.dataset])
    ann = ds.load_annotations([args.dataset], manifest=manifest)
    print("manifest columns:", list(manifest.columns))
    print("annotation columns:", list(ann.columns))

    lab_col = _col(ann, "label_3class")
    if lab_col is None:
        ann = ann.copy()
        ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
        lab_col = "label_3class"

    fn_col = _col(ann, "filename", "file", "wav")
    m_fn_col = _col(manifest, "filename", "file", "wav")
    m_start_col = _col(manifest, "start_dt", "start_datetime", "start")
    m_dur_col = _col(manifest, "duration_s", "duration")

    # choose the file (unless forced): the one whose annotations span the most classes
    if args.file:
        target_fn = args.file
    else:
        cov = (
            ann.groupby(fn_col)[lab_col]
            .agg(lambda s: len(set(s)))
            .sort_values(ascending=False)
        )
        target_fn = cov.index[0]
        print(f"auto-selected file {target_fn} (covers {cov.iloc[0]} classes)")

    file_row = manifest[manifest[m_fn_col] == target_fn].iloc[0]
    file_start = file_row[m_start_col]
    path = _resolve_path(file_row, manifest)

    lo_col = _col(ann, "low_frequency", "low_freq", "freq_low", "f_low")
    hi_col = _col(ann, "high_frequency", "high_freq", "freq_high", "f_high")
    a = ann[ann[fn_col] == target_fn].copy()
    a["start_s"] = (a["start_datetime"] - file_start).dt.total_seconds()
    a["end_s"] = (a["end_datetime"] - file_start).dt.total_seconds()
    events = list(zip(a["start_s"], a["end_s"], a[lo_col], a[hi_col], a[lab_col]))

    # pick_window only needs (start, end, cls)
    t0 = args.t0 if args.t0 is not None else pick_window(
        [(s, e, c) for (s, e, lo, hi, c) in events], args.window)
    t1 = t0 + args.window
    print(f"window: {t0:.1f}-{t1:.1f} s of {target_fn}")

    win_hi = [hi for (s, e, lo, hi, c) in events if e > t0 and s < t1]
    fmax = args.fmax if args.fmax is not None else min(
        sr / 2.0, (max(win_hi) if win_hi else sr / 2.0) * 1.15)

    # load just the window
    x, file_sr = sf.read(path, start=int(t0 * sr), stop=int(t1 * sr))
    if x.ndim > 1:
        x = x[:, 0]
    assert file_sr == sr, f"file sr {file_sr} != cfg.SAMPLE_RATE {sr}"

    f, t, Z = stft(
        x, fs=sr, window="hann",
        nperseg=cfg.WIN_LENGTH, noverlap=cfg.WIN_LENGTH - cfg.HOP_LENGTH,
        nfft=cfg.N_FFT, boundary=None,
    )
    S = 20.0 * np.log10(np.abs(Z) + 1e-10)
    vmax = float(S.max())
    vmin = vmax - args.db_range

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.pcolormesh(t, f, S, vmin=vmin, vmax=vmax, cmap="magma", rasterized=True, shading="auto")
    ax.set_ylim(0, fmax)
    ax.set_xlim(0, args.window)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    for s, e, lo, hi, cls in events:
        rs, re = s - t0, e - t0
        if re <= 0 or rs >= args.window:
            continue
        rs, re = max(rs, 0), min(re, args.window)
        c = CLASS_COLOR.get(cls, "#999999")
        ax.add_patch(Rectangle(
            (rs, lo), re - rs, hi - lo,
            fill=False, edgecolor=c, linewidth=1.3, zorder=3,
        ))

    handles = [Patch(facecolor=CLASS_COLOR[c], alpha=0.5, edgecolor=CLASS_COLOR[c],
                     label=CLASS_NAME[c]) for c in ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=6, frameon=True,
              framealpha=0.85, handlelength=1.0, borderpad=0.3, labelspacing=0.2)

    fig.tight_layout(pad=0.2)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
