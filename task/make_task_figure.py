#!/usr/bin/env python
"""Task-illustration figure: a real ATBFL spectrogram with the strongly-labelled
call boundaries drawn as per-class time-frequency boxes (BMABZ / D / BP).

The window is chosen to contain the *clearest* multi-class cluster: candidate
windows are scored by the summed SNR of the calls they contain (median enhanced
energy inside each annotated box minus the median just outside it), with a bonus
for class diversity. Display uses per-frequency baseline removal by default (the
analogue of the front-end's mean subtraction), so faint calls stand out.

Run from the NAVE repo root (needs nave_config.py, dataset.py, dev data).

    python make_task_figure.py                          # auto: clearest window in casey2017
    python make_task_figure.py --dataset kerguelen2015 --window 40
    python make_task_figure.py --file <wav> --t0 120    # force a specific window
    python make_task_figure.py --raw                    # disable enhancement
    python make_task_figure.py --label-size 6 --tick-size 5   # even smaller axis text
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
    """Per-frequency baseline-removed log-spectrogram of a clip."""
    f, t, Z = stft(
        x, fs=sr, window="hann",
        nperseg=cfg.WIN_LENGTH, noverlap=cfg.WIN_LENGTH - cfg.HOP_LENGTH,
        nfft=cfg.N_FFT, boundary=None,
    )
    S = 20.0 * np.log10(np.abs(Z) + 1e-10)
    return f, t, S - np.median(S, axis=1, keepdims=True)


def box_snr(f, t, S, s0, e0, lo, hi):
    """Median enhanced energy in the box minus median at the same frequencies
    outside the box's time span. Higher = the call stands out more."""
    fi = (f >= lo) & (f <= hi)
    ti = (t >= s0) & (t <= e0)
    if fi.sum() < 1 or ti.sum() < 1 or (~ti).sum() < 1:
        return -1e9
    return float(np.median(S[np.ix_(fi, ti)]) - np.median(S[np.ix_(fi, ~ti)]))


def score_events_snr(events, path, sr, pad, cap):
    """SNR per event (clips loaded on demand, capped for runtime)."""
    snrs = []
    for i, (s, e, lo, hi, c) in enumerate(events):
        if i >= cap:
            snrs.append(-1e9)
            continue
        a = max(0.0, s - pad)
        try:
            x, _ = sf.read(path, start=int(a * sr), stop=int((e + pad) * sr))
        except Exception:
            snrs.append(-1e9)
            continue
        if x.ndim > 1:
            x = x[:, 0]
        if len(x) < cfg.WIN_LENGTH:
            snrs.append(-1e9)
            continue
        f, t, S = enh_spec(x, sr)
        snrs.append(box_snr(f, t, S, s - a, e - a, lo, hi))
    return snrs


def window_indices(events, t0, window_s):
    return [j for j, (s, e, *_ ) in enumerate(events) if s < t0 + window_s and e > t0]


def mesh_edges(v, lo, hi):
    """Cell edges for pcolormesh(shading='flat') from bin centres `v`, with the
    outer edges clamped to `lo`/`hi` so the mesh fills the axis exactly. STFT
    with boundary=None starts half a window in and stops early, which otherwise
    leaves a blank strip at each end of the time axis."""
    mid = 0.5 * (v[:-1] + v[1:])
    return np.concatenate(([lo], mid, [hi]))


def qualifies(events, idx, min_classes, min_per_class):
    """Window (given by event indices) meets the class constraints."""
    cnt = {}
    for j in idx:
        c = events[j][4]
        cnt[c] = cnt.get(c, 0) + 1
    if len(cnt) < min_classes:
        return False
    if min_per_class > 0 and any(cnt.get(c, 0) < min_per_class for c in ORDER):
        return False
    return True


def pick_window(events, snrs, window_s, lam, min_classes, min_per_class, margin, file_dur):
    """Return (t0, score, met): best window by summed call SNR + lam * #classes,
    among windows meeting the class constraints and keeping every call at least
    `margin` seconds inside both borders. Falls back to the best window if none
    satisfy the constraints."""
    best = (-1e18, 0.0)
    best_any = (-1e18, 0.0)
    for ev in events:
        # anchor so this event sits at the left margin, not on the edge
        t0 = max(0.0, ev[0] - margin)
        if t0 + window_s > file_dur:
            continue
        idx = window_indices(events, t0, window_s)
        if not idx:
            continue
        tot = sum(max(snrs[j], 0.0) for j in idx)
        ncls = len({events[j][4] for j in idx})
        score = tot + lam * ncls
        if score > best_any[0]:
            best_any = (score, t0)
        first_start = min(events[j][0] for j in idx)
        last_end = max(events[j][1] for j in idx)
        margin_ok = (first_start >= t0 + margin - 1e-6 and
                     last_end <= t0 + window_s - margin + 1e-6)
        if (margin_ok and qualifies(events, idx, min_classes, min_per_class)
                and score > best[0]):
            best = (score, t0)
    if best[0] == -1e18:
        return best_any[1], best_any[0], False
    return best[1], best[0], True


def events_for_file(fn, ann, cols, finfo):
    fn_c, lab_c, lo_c, hi_c = cols
    path, fstart, dur = finfo[fn]
    a = ann[ann[fn_c] == fn].copy()
    a["s"] = (a["start_datetime"] - fstart).dt.total_seconds()
    a["e"] = (a["end_datetime"] - fstart).dt.total_seconds()
    events = list(zip(a["s"], a["e"], a[lo_c].astype(float), a[hi_c].astype(float), a[lab_c]))
    return events, path, fstart, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="casey2017")
    ap.add_argument("--window", type=float, default=45.0, help="window length [s]")
    ap.add_argument("--nfiles", type=int, default=4, help="most call-dense files to search")
    ap.add_argument("--scan-cap", type=int, default=200, help="events scored per file")
    ap.add_argument("--pad", type=float, default=3.0, help="context for SNR scoring [s]")
    ap.add_argument("--lam", type=float, default=8.0, help="class-diversity weight")
    ap.add_argument("--min-classes", type=int, default=1,
                    help="require at least this many classes in the window")
    ap.add_argument("--min-per-class", type=int, default=0,
                    help="require at least this many calls of EACH class in the window")
    ap.add_argument("--margin", type=float, default=7.0,
                    help="keep every call at least this many seconds inside both borders")
    ap.add_argument("--fmax", type=float, default=None, help="max frequency [Hz]")
    ap.add_argument("--raw", action="store_true", help="disable per-frequency baseline removal")
    ap.add_argument("--db-range", type=float, default=70.0, help="dynamic range if --raw")
    ap.add_argument("--pctl", type=float, default=99.5, help="upper colour percentile")
    ap.add_argument("--label-size", type=float, default=7.0, help="axis label font size [pt]")
    ap.add_argument("--tick-size", type=float, default=6.0, help="tick label font size [pt]")
    ap.add_argument("--pad-inches", type=float, default=0.02,
                    help="white border kept around the figure [in]")
    ap.add_argument("--fig-w", type=float, default=3.4, help="figure width [in]")
    ap.add_argument("--fig-h", type=float, default=2.0, help="figure height [in]")
    ap.add_argument("--file", default=None, help="force a wav filename")
    ap.add_argument("--t0", type=float, default=None, help="force window start [s]")
    ap.add_argument("--out", default="fig_task.pdf")
    args = ap.parse_args()

    sr = cfg.SAMPLE_RATE
    manifest = ds.get_file_manifest([args.dataset])
    ann = ds.load_annotations([args.dataset], manifest=manifest)

    lab_c = _col(ann, "label_3class")
    if lab_c is None:
        ann = ann.copy()
        ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
        lab_c = "label_3class"
    fn_c = _col(ann, "filename", "file", "wav")
    lo_c = _col(ann, "low_frequency", "low_freq", "f_low")
    hi_c = _col(ann, "high_frequency", "high_freq", "f_high")
    cols = (fn_c, lab_c, lo_c, hi_c)

    m_fn = _col(manifest, "filename", "file", "wav")
    m_start = _col(manifest, "start_dt", "start_datetime", "start")
    m_path = _col(manifest, "path", "filepath", "wav_path")
    m_dur = _col(manifest, "duration_s", "duration")
    if m_path is None:
        raise SystemExit("No path column in manifest; columns: " + ", ".join(manifest.columns))
    finfo = {r[m_fn]: (r[m_path], r[m_start], float(r[m_dur])) for _, r in manifest.iterrows()}

    # candidate files: forced, else the most class-diverse few
    if args.file:
        candidates = [args.file]
    else:
        cov = ann.groupby(fn_c)[lab_c].agg(lambda s: len(set(s))).sort_values(ascending=False)
        candidates = list(cov.index[: args.nfiles])

    # find the best (file, t0) across candidates
    best = None  # (score, fn, t0, events, path, met)
    for fn in candidates:
        events, path, _, file_dur = events_for_file(fn, ann, cols, finfo)
        if args.t0 is not None and fn == candidates[0]:
            t0, score, met = args.t0, 0.0, True
        else:
            snrs = score_events_snr(events, path, sr, args.pad, args.scan_cap)
            t0, score, met = pick_window(events, snrs, args.window, args.lam,
                                         args.min_classes, args.min_per_class,
                                         args.margin, file_dur)
        rank = score - (0.0 if met else 1e6)  # prefer files that satisfy the constraints
        if best is None or rank > best[0]:
            best = (rank, fn, t0, events, path, met)
    _, target_fn, t0, events, path, met = best
    if not met:
        print("warning: no window satisfied the class/margin constraints; showing the best available")
    t1 = t0 + args.window
    idx = window_indices(events, t0, args.window)
    counts = {}
    for j in idx:
        counts[events[j][4]] = counts.get(events[j][4], 0) + 1
    cstr = ", ".join(f"{CLASS_NAME[c]}:{counts.get(c, 0)}" for c in ORDER)
    print(f"window: {t0:.1f}-{t1:.1f} s of {target_fn}  ({cstr})")

    win_hi = [events[j][3] for j in idx]
    fmax = args.fmax if args.fmax is not None else min(
        sr / 2.0, (max(win_hi) if win_hi else sr / 2.0) * 1.15)

    x, file_sr = sf.read(path, start=int(t0 * sr), stop=int(t1 * sr))
    if x.ndim > 1:
        x = x[:, 0]
    assert file_sr == sr, f"file sr {file_sr} != cfg.SAMPLE_RATE {sr}"

    if args.raw:
        f, t, Z = stft(x, fs=sr, window="hann", nperseg=cfg.WIN_LENGTH,
                       noverlap=cfg.WIN_LENGTH - cfg.HOP_LENGTH, nfft=cfg.N_FFT, boundary=None)
        S = 20.0 * np.log10(np.abs(Z) + 1e-10)
        vmax, vmin = float(S.max()), float(S.max()) - args.db_range
    else:
        f, t, S = enh_spec(x, sr)
        vmax, vmin = float(np.percentile(S, args.pctl)), 0.0

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))
    df = float(f[1] - f[0])
    t_edges = mesh_edges(t, 0.0, args.window)
    f_edges = mesh_edges(f, 0.0, float(f[-1]) + 0.5 * df)
    ax.pcolormesh(t_edges, f_edges, S, vmin=vmin, vmax=vmax, cmap="magma",
                  rasterized=True, shading="flat")
    ax.set_ylim(0, fmax)
    ax.set_xlim(0, args.window)
    ax.set_xlabel("Time [s]", fontsize=args.label_size, labelpad=2)
    ax.set_ylabel("Frequency [Hz]", fontsize=args.label_size, labelpad=2)
    ax.tick_params(labelsize=args.tick_size, pad=2)

    for s, e, lo, hi, cls in events:
        rs, re = s - t0, e - t0
        if re <= 0 or rs >= args.window:
            continue
        rs, re = max(rs, 0), min(re, args.window)
        c = CLASS_COLOR.get(cls, "#999999")
        ax.add_patch(Rectangle((rs, lo), re - rs, hi - lo,
                               fill=False, edgecolor=c, linewidth=1.3, zorder=3))

    handles = [Patch(facecolor=CLASS_COLOR[c], alpha=0.5, edgecolor=CLASS_COLOR[c],
                     label=CLASS_NAME[c]) for c in ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=6, frameon=True,
              framealpha=0.85, handlelength=1.0, borderpad=0.3, labelspacing=0.2)

    fig.tight_layout(pad=0.2)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=args.pad_inches, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
