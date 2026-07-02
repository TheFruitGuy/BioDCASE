#!/usr/bin/env python
"""Three-panel call gallery for the paper: the clearest example of each call
type (BMABZ, D, BP), auto-selected by how far the annotated box stands above the
local background, so each call's structure is actually visible.

Run from the NAVE repo root (needs nave_config.py, dataset.py, dev data).

    python make_call_gallery.py                          # search casey2017
    python make_call_gallery.py --dataset kerguelen2015  # try another site
    python make_call_gallery.py --topk 40 --pad 4 --out fig_gallery.pdf

Display uses the same per-frequency baseline removal as make_task_figure.py
(the analogue of the front-end's mean subtraction). Candidates are ranked by the
median enhanced energy inside the annotated time-frequency box minus the median
at the same frequencies just outside it.
"""
from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import nave_config as cfg
import dataset as ds

CLASS_COLOR = {"bmabz": "#0072B2", "d": "#D55E00", "bp": "#009E73"}
CLASS_NAME = {"bmabz": "BMABZ", "d": "D", "bp": "BP"}
ORDER = ["bmabz", "d", "bp"]


def _col(df, *candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def enh_spec(x, sr):
    """Enhanced (per-frequency baseline-removed) log-spectrogram of a clip."""
    f, t, Z = stft(
        x, fs=sr, window="hann",
        nperseg=cfg.WIN_LENGTH, noverlap=cfg.WIN_LENGTH - cfg.HOP_LENGTH,
        nfft=cfg.N_FFT, boundary=None,
    )
    S = 20.0 * np.log10(np.abs(Z) + 1e-10)
    S = S - np.median(S, axis=1, keepdims=True)
    return f, t, S


def box_snr(f, t, S, s0, e0, lo, hi):
    """Median enhanced energy inside the box minus median at the same
    frequencies outside the box's time span. Higher = call stands out more."""
    fi = (f >= lo) & (f <= hi)
    ti = (t >= s0) & (t <= e0)
    if fi.sum() < 1 or ti.sum() < 1 or (~ti).sum() < 1:
        return -1e9
    inb = S[np.ix_(fi, ti)]
    outb = S[np.ix_(fi, ~ti)]
    return float(np.median(inb) - np.median(outb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="casey2017")
    ap.add_argument("--file", default=None, help="restrict search to one wav")
    ap.add_argument("--topk", type=int, default=30,
                    help="candidates scored per class (bounds runtime)")
    ap.add_argument("--pad", type=float, default=3.0,
                    help="seconds of context on each side of the call")
    ap.add_argument("--pctl", type=float, default=99.5)
    ap.add_argument("--out", default="fig_gallery.pdf")
    args = ap.parse_args()

    sr = cfg.SAMPLE_RATE
    manifest = ds.get_file_manifest([args.dataset])
    ann = ds.load_annotations([args.dataset], manifest=manifest)

    lab = _col(ann, "label_3class") or "annotation"
    if lab == "annotation":
        ann = ann.copy()
        ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
        lab = "label_3class"
    fn = _col(ann, "filename", "file", "wav")
    lo_c = _col(ann, "low_frequency", "low_freq", "f_low")
    hi_c = _col(ann, "high_frequency", "high_freq", "f_high")

    mfn = _col(manifest, "filename", "file", "wav")
    mstart = _col(manifest, "start_dt", "start_datetime", "start")
    mpath = _col(manifest, "path", "filepath", "wav_path")
    if mpath is None:
        raise SystemExit("No path column in manifest; columns: " + ", ".join(manifest.columns))
    finfo = {r[mfn]: (r[mpath], r[mstart]) for _, r in manifest.iterrows()}

    if args.file:
        ann = ann[ann[fn] == args.file]

    picks = {}
    for cls in ORDER:
        cand = ann[ann[lab] == cls]
        if len(cand) == 0:
            print(f"warning: no {cls} annotations in {args.dataset}")
            continue
        cand = cand.head(args.topk)
        best = None
        for _, r in cand.iterrows():
            if r[fn] not in finfo:
                continue
            path, fstart = finfo[r[fn]]
            s0 = (r["start_datetime"] - fstart).total_seconds()
            e0 = (r["end_datetime"] - fstart).total_seconds()
            lo, hi = float(r[lo_c]), float(r[hi_c])
            a = max(0.0, s0 - args.pad)
            x, fsr = sf.read(path, start=int(a * sr), stop=int((e0 + args.pad) * sr))
            if len(x) < cfg.WIN_LENGTH:
                continue
            if x.ndim > 1:
                x = x[:, 0]
            f, t, S = enh_spec(x, sr)
            snr = box_snr(f, t, S, s0 - a, e0 - a, lo, hi)
            item = dict(snr=snr, x=x, s=s0 - a, e=e0 - a, lo=lo, hi=hi,
                        file=r[fn], t0=s0)
            if best is None or snr > best["snr"]:
                best = item
        if best is not None:
            picks[cls] = best
            print(f"{cls}: {best['file']} @ {best['t0']:.1f}s  SNR={best['snr']:.1f} dB")

    ncol = len(picks)
    fig, axes = plt.subplots(1, ncol, figsize=(2.4 * ncol, 2.3), squeeze=False)
    for ax, cls in zip(axes[0], [c for c in ORDER if c in picks]):
        p = picks[cls]
        f, t, S = enh_spec(p["x"], sr)
        vmax = float(np.percentile(S, args.pctl))
        ax.pcolormesh(t, f, S, vmin=0.0, vmax=vmax, cmap="magma",
                      rasterized=True, shading="auto")
        c = CLASS_COLOR[cls]
        ax.add_patch(Rectangle((p["s"], p["lo"]), p["e"] - p["s"], p["hi"] - p["lo"],
                               fill=False, edgecolor=c, linewidth=1.4, zorder=3))
        ax.set_ylim(max(0.0, p["lo"] - 10), min(sr / 2.0, p["hi"] + 10))
        ax.set_xlim(0, t[-1])
        ax.set_title(CLASS_NAME[cls], color=c, fontsize=9)
        ax.set_xlabel("Time [s]")
    axes[0][0].set_ylabel("Frequency [Hz]")

    fig.tight_layout(pad=0.3)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
