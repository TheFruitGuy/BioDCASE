"""
postproc_search.py — post-processing hyperparameter search for the ensemble
===========================================================================

A faithful re-implementation of the WhaleVAD-BPN (Geldenhuys et al., 2025)
post-processing optimisation, applied to *your* ensemble's combined per-class
posteriors, and evaluated under the paper's leave-one-site-out (3-fold) scheme.

What it does
------------
Given the ensemble members (the same ``--checkpoints`` + per-class routing
weights as your hybrid command), it:

  1. Loads each member's saved ``_final`` val posteriors (``best_posteriors.npz``
     or ``paper_best_posteriors.npz``) and combines them per class with the
     routing weights — exactly the combined probabilities your deployed ensemble
     scores, at 30 s tiling. No GPU, no inference.
  2. Runs the paper's **backward-search**, per class, in two stages
     (Fig. 3c / Table IV):
        Stage 1 (event-level): fix the threshold at the equal-precision/recall
            point on the dev folds, then grid-search min-time-between-events,
            min- and max-event-duration (per-class ranges, Table IVb).
        Stage 2 (frame-level): fix the event-level params, then grid-search the
            median-filter kernel, the on-threshold (and optionally the
            hysteresis off-threshold and hangover kernel — Table IVa).
  3. Does this under **leave-one-site-out CV** over the 3 val sites: params are
     selected on 2 sites, scored on the held-out site, rotated over all 3 turns.
     The headline is the paper macro-F1 recomputed from precision/recall averaged
     across classes and test folds — i.e. a *selection-unbiased* number.
  4. Reports the searched result against the **threshold-only baseline** (your
     current pipeline: fixed 500 ms median, uniform 0.5 s merge, [0.5, 30] s
     duration, only the per-class threshold tuned) under the *same* LOSO, so you
     can see whether the search squeezes anything out, and from which knob.

Why LOSO matters here: your 0.502 hybrid number is val-*selected* (routing,
window, thresholds all chosen on the 3 val sites), so it is optimistic. This
script selects on 2 sites and scores on the 3rd, which is the honest
generalisation estimate and what the paper reports.

Scoring is done with ``postprocess_final.compute_metrics`` (greedy 1-D IoU at
0.3) so every number is directly comparable to your other ``_final`` results.

Usage
-----
::

    # the 0.502 hybrid ensemble (16 ckpts), default search (threshold + median + per-class event-level)
    python postproc_search.py \
        --checkpoints runs/hnm_3class_s2024_ens_nopgi/best_model.pt ... \
        --weights-bmabz 1 1 ... --weights-d 0 0 ... --weights-bp 1 1 ...

    # the full paper search (adds hysteresis off-threshold + hangover; slower — use a coarser --thr-step)
    python postproc_search.py --checkpoints ... --hysteresis --hangover --thr-step 0.1

    # event-level only (matches the paper's "event-level alone" row), no frame stage
    python postproc_search.py --checkpoints ... --no-frame-stage

    # quick sanity check that the pure search logic works (no data / server modules needed)
    python postproc_search.py --selftest
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

import config_final as cfg
from postprocess_final import Detection, compute_metrics

CLASSES = ["bmabz", "d", "bp"]
STRIDE = float(cfg.FRAME_STRIDE_S)          # 0.02 s / frame

# ---- Table IVb: per-class event-level search ranges -------------------------
EVENT_GRID = {
    "bmabz": dict(merge=(0.1, 0.9, 0.1), min_dur=(2.0, 5.0, 0.5), max_dur=(25.0, 40.0, 2.5)),
    "d":     dict(merge=(0.1, 0.9, 0.1), min_dur=(0.6, 3.0, 0.4), max_dur=(5.0, 11.0, 1.0)),
    "bp":    dict(merge=(0.1, 0.9, 0.1), min_dur=(0.3, 1.5, 0.2), max_dur=(2.0, 5.0, 0.5)),
}
# ---- Table IVa: frame-level kernels (frames) --------------------------------
FILTER_KERNELS = [None, 11, 33, 55]
HANGOVER_KERNELS = [None, 11, 33, 55]


def _grid(a, b, s):
    return [round(float(x), 4) for x in np.arange(a, b + 1e-9, s)]


# =============================================================================
# Post-processing primitives  (operate on one class's 1-D probability track)
# =============================================================================
def median_smooth(p: np.ndarray, k):
    """Per-class temporal median filter; ``k`` in frames (None / <=1 = no-op).

    Matches ``postprocess_final.smooth_probabilities`` (scipy median_filter,
    default 'reflect' edges); that function uses k = SMOOTH_KERNEL_MS // 20."""
    if k is None or k <= 1:
        return p
    return median_filter(p, size=int(k))


def hysteresis_binary(p: np.ndarray, on: float, off: float) -> np.ndarray:
    """Two-threshold hysteresis -> binary activity (vectorised, no Python loop).

    Enter an event when ``p > on``; stay active until ``p < off`` (off <= on).
    For ``on == off`` this reduces to the strict ``p > on`` used by
    ``threshold_to_detections`` (exact baseline match)."""
    if off >= on:                                   # degenerate -> plain threshold
        return p > on
    state = np.full(p.shape, -1, dtype=np.int8)      # -1 = hold previous
    state[p > on] = 1
    state[p < off] = 0
    decided = state != -1
    idx = np.where(decided, np.arange(p.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = state[idx]
    filled[~decided[idx]] = 0                        # before the first decision -> off
    return filled.astype(bool)


def hangover_binary(b: np.ndarray, k) -> np.ndarray:
    """Causal majority vote over the last ``k`` frames (paper Eq. 1; None = no-op).

    y~_t = 1 iff sum_{i=0}^{k-1} y_{t-i} > k/2. Vectorised via cumsum; the window
    shrinks at the start. (The paper calls this a binary-domain median filter.)"""
    if k is None or k <= 1:
        return b
    w = int(k)
    bi = b.astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(bi)])
    t = np.arange(bi.shape[0])
    lo = np.maximum(0, t + 1 - w)
    s = cs[t + 1] - cs[lo]
    win = (t + 1) - lo
    return s > (win / 2.0)


def runs_to_events(binary, prob, ds, fn, label) -> list:
    """Contiguous True runs -> Detection list (mirrors threshold_to_detections).

    Per-run mean confidence is computed vectorised (cumsum) rather than per-run."""
    d = np.diff(binary.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if starts.size == 0:
        return []
    cs = np.concatenate(([0.0], np.cumsum(prob, dtype=np.float64)))
    lens = np.maximum(ends - starts, 1)
    means = (cs[ends] - cs[starts]) / lens
    return [Detection(ds, fn, label, float(s * STRIDE), float(e * STRIDE), float(m))
            for s, e, m in zip(starts, ends, means)]


def event_filter(events, merge_gap, min_dur, max_dur) -> list:
    """Per-file merge (gap <= merge_gap) then keep duration in [min_dur, max_dur].

    Mirrors ``merge_and_filter`` but with per-class (merge_gap, min_dur, max_dur)
    instead of the global MERGE_GAP_S / POST_MIN_DUR_S / POST_MAX_DUR_S."""
    by_file = defaultdict(list)
    for e in events:
        by_file[(e.dataset, e.filename)].append(e)
    out = []
    for evs in by_file.values():
        evs = sorted(evs, key=lambda x: x.start_s)
        merged = []
        for e in evs:
            if merged and e.start_s - merged[-1].end_s <= merge_gap:
                merged[-1].end_s = max(merged[-1].end_s, e.end_s)
                merged[-1].confidence = max(merged[-1].confidence, e.confidence)
            else:
                merged.append(copy.copy(e))
        for m in merged:
            if min_dur <= (m.end_s - m.start_s) <= max_dur:
                out.append(m)
    return out


# =============================================================================
# Raw-event production with a per-(file, median-kernel) cache
# =============================================================================
class RawEventCache:
    """Caches median-smoothed tracks per (site, file, kernel) so the frame-level
    search doesn't recompute the (expensive) median for every threshold."""

    def __init__(self, tracks_by_site):
        # tracks_by_site[site] = {fn: 1-D float32 prob track for ONE class}
        self.tracks = tracks_by_site
        self._med = {}                                # (site, fn, k) -> smoothed track

    def _smoothed(self, site, fn, k):
        key = (site, fn, k)
        s = self._med.get(key)
        if s is None:
            s = median_smooth(self.tracks[site][fn], k)
            self._med[key] = s
        return s

    def raw_events(self, label, sites, median_k, on, off, hangover_k):
        out = []
        for site in sites:
            for fn, _ in self.tracks[site].items():
                s = self._smoothed(site, fn, median_k)
                b = hysteresis_binary(s, on, off)
                b = hangover_binary(b, hangover_k)
                out.extend(runs_to_events(b, s, site, fn, label))
        return out


# =============================================================================
# Scoring (one class, on a chosen set of sites)
# =============================================================================
from postprocess_final import compute_iou_1d


def to_gt_by_file(gt_events):
    """{(ds, fn): [Detection sorted by start]} for a single class's gt list."""
    g = defaultdict(list)
    for e in gt_events:
        g[(e.dataset, e.filename)].append(e)
    for k in g:
        g[k].sort(key=lambda x: x.start_s)
    return dict(g)


def score_events(pred_events, gt_by_file, iou):
    """Greedy 1-D IoU P/R/F1 for ONE class. Identical rule + tie-breaking to
    ``postprocess_final.compute_metrics`` (gt in start order, earliest pred wins
    ties via strict ``>``), but predictions are grouped once (O(E)) instead of
    re-filtered per file (O(F*E)) — ~20x faster in the search loop."""
    pred_by_file = defaultdict(list)
    for e in pred_events:
        pred_by_file[(e.dataset, e.filename)].append(e)
    tp = fp = fn = 0
    for fk in set(pred_by_file) | set(gt_by_file):
        preds = sorted(pred_by_file.get(fk, []), key=lambda x: x.start_s)
        gts = gt_by_file.get(fk, [])
        matched = set()
        for g in gts:
            best_iou, best_i = 0.0, -1
            for i, p in enumerate(preds):
                if i in matched:
                    continue
                v = compute_iou_1d(p.start_s, p.end_s, g.start_s, g.end_s)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_iou >= iou and best_i >= 0:
                tp += 1
                matched.add(best_i)
            else:
                fn += 1
        fp += len(preds) - len(matched)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r + 1e-8), tp=tp, fp=fp, fn=fn)


def class_pr_f1(pred_events, gt_events, iou):
    """Reference scorer via compute_metrics (used only for the self-test cross-check)."""
    if not pred_events and not gt_events:
        return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=0)
    label = (pred_events or gt_events)[0].label
    m = compute_metrics(pred_events, gt_events, iou_threshold=iou)
    return m.get(label, dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=0))


def f1_of(p, r):
    return 2 * p * r / (p + r + 1e-8)


# ---- fast numpy event-level scoring (hot selection loops) -------------------
# Equivalent to score_events(event_filter(raw, merge, min, max), gt) but works on
# pre-grouped (start, end) arrays and avoids per-combo Detection churn.
def raw_to_by_file(events):
    """{(ds, fn): (n,2) float array [start, end] sorted by start} from a raw list."""
    bf = defaultdict(list)
    for e in events:
        bf[(e.dataset, e.filename)].append((e.start_s, e.end_s))
    return {k: np.array(sorted(v), dtype=np.float64) for k, v in bf.items()}


def _merge_filter_np(arr, merge_gap, min_dur, max_dur):
    """Per-file merge (gap <= merge_gap) + duration filter; matches event_filter.

    Vectorised: a new merged group starts where start_i - cummax(ends_<i) > gap;
    merged span = (first start, max end) per group (arr sorted by start)."""
    n = arr.shape[0]
    if n == 0:
        return arr[:, 0], arr[:, 1]
    starts = arr[:, 0]
    ends = arr[:, 1]
    cummax = np.maximum.accumulate(ends)
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = (starts[1:] - cummax[:-1]) > merge_gap
    gid = np.cumsum(new_group) - 1
    ms = starts[new_group]
    me = np.zeros(int(gid[-1]) + 1, dtype=np.float64)
    np.maximum.at(me, gid, ends)
    dur = me - ms
    keep = (dur >= min_dur) & (dur <= max_dur)
    return ms[keep], me[keep]


def score_event_level(raw_by_file, gt_by_file_arr, merge_gap, min_dur, max_dur, iou):
    """Greedy 1-D IoU P/R/F1 on (start,end) arrays; identical rule to score_events
    (gt in start order; masked argmax => earliest pred wins ties, as compute_metrics).
    gt_by_file_arr[(ds,fn)] = (n,2) array sorted by start."""
    tp = fp = fn = 0
    empty = np.zeros((0, 2))
    for fk in set(raw_by_file) | set(gt_by_file_arr):
        ps, pe = _merge_filter_np(raw_by_file.get(fk, empty), merge_gap, min_dur, max_dur)
        g = gt_by_file_arr.get(fk, empty)
        P, G = ps.shape[0], g.shape[0]
        if G == 0:
            fp += P
            continue
        if P == 0:
            fn += G
            continue
        gs, ge = g[:, 0], g[:, 1]
        inter = np.minimum(pe[:, None], ge[None, :]) - np.maximum(ps[:, None], gs[None, :])
        np.clip(inter, 0.0, None, out=inter)
        union = (pe - ps)[:, None] + (ge - gs)[None, :] - inter
        iou_mat = inter / np.maximum(union, 1e-12)        # (P, G), gt in start order
        matched = np.zeros(P, dtype=bool)
        for gj in range(G):
            cand = np.where(matched, -1.0, iou_mat[:, gj])
            i = int(np.argmax(cand))                      # first max = earliest-start pred
            if cand[i] >= iou:
                matched[i] = True
                tp += 1
            else:
                fn += 1
        fp += P - int(matched.sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return p, r, f1_of(p, r)


def gt_arrays(gt_by_file):
    """{(ds,fn): [Detection]} -> {(ds,fn): (n,2) array sorted by start}."""
    return {k: np.array([[d.start_s, d.end_s] for d in v], dtype=np.float64)
            for k, v in gt_by_file.items()}


# =============================================================================
# Stage 1 init: equal precision/recall threshold on the dev folds
# =============================================================================
def equal_pr_threshold(cache, label, dev_sites, gt_dev_bf, thr_grid, iou):
    """Threshold minimising |P - R| on the pooled dev folds (single threshold,
    no median / hysteresis / hangover) — the paper's backward-search Stage-1 init."""
    best_t, best_gap = thr_grid[len(thr_grid) // 2], 1e9
    for t in thr_grid:
        ev = cache.raw_events(label, dev_sites, None, t, t, None)
        m = score_events(ev, gt_dev_bf, iou)
        gap = abs(m["precision"] - m["recall"])
        if gap < best_gap:
            best_gap, best_t = gap, t
    return best_t


# =============================================================================
# Backward-search for ONE class, with LOSO over the sites
# =============================================================================
def search_class(label, tracks_by_site, gt_by_site_label, sites, opts):
    cache = RawEventCache(tracks_by_site)
    thr_grid = opts["thr_grid"]
    eg = EVENT_GRID[label]
    merge_g = _grid(*eg["merge"])
    min_g = _grid(*eg["min_dur"])
    max_g = _grid(*eg["max_dur"])
    iou = opts["iou"]

    folds = []
    for test_site in sites:
        dev_sites = [s for s in sites if s != test_site]
        gt_dev_bf = to_gt_by_file([g for s in dev_sites for g in gt_by_site_label[(s, label)]])
        gt_test_bf = to_gt_by_file(list(gt_by_site_label[(test_site, label)]))
        gt_dev_arr = gt_arrays(gt_dev_bf)            # numpy gt for the fast scorer

        # ---- Stage 1: event-level (threshold fixed at equal P/R) -------------
        # raw events are fixed across the 441 event combos -> group once, reuse.
        t0 = equal_pr_threshold(cache, label, dev_sites, gt_dev_bf, thr_grid, iou)
        raw0_bf = raw_to_by_file(cache.raw_events(label, dev_sites, None, t0, t0, None))
        best_ev, best_f1 = (cfg.MERGE_GAP_S, cfg.POST_MIN_DUR_S, cfg.POST_MAX_DUR_S), -1.0
        for mg in merge_g:
            for mnd in min_g:
                for mxd in max_g:
                    f1 = score_event_level(raw0_bf, gt_dev_arr, mg, mnd, mxd, iou)[2]
                    if f1 > best_f1:
                        best_f1, best_ev = f1, (mg, mnd, mxd)

        # ---- Stage 2: frame-level (event-level fixed) ------------------------
        best_fr = (None, t0, t0, None)               # (median_k, on, off, hangover_k)
        if opts["frame_stage"]:
            best_f1b = -1.0
            offs_for = (lambda on: [o for o in thr_grid if o <= on]) if opts["hysteresis"] \
                else (lambda on: [on])
            hangs = HANGOVER_KERNELS if opts["hangover"] else [None]
            for med in FILTER_KERNELS:
                for on in thr_grid:
                    for off in offs_for(on):
                        for hg in hangs:
                            raw_bf = raw_to_by_file(cache.raw_events(label, dev_sites, med, on, off, hg))
                            f1 = score_event_level(raw_bf, gt_dev_arr, *best_ev, iou)[2]
                            if f1 > best_f1b:
                                best_f1b, best_fr = f1, (med, on, off, hg)
        else:
            best_f1b = best_f1

        # ---- evaluate the chosen config on the held-out site -----------------
        # (Detection path here: byte-faithful to merge_and_filter / compute_metrics)
        med, on, off, hg = best_fr
        raw_t = cache.raw_events(label, [test_site], med, on, off, hg)
        m = score_events(event_filter(raw_t, *best_ev), gt_test_bf, iou)
        folds.append(dict(test=test_site, dev_f1=best_f1b,
                          params=dict(median_k=med, on=on, off=off, hangover_k=hg,
                                      merge_gap=best_ev[0], min_dur=best_ev[1], max_dur=best_ev[2]),
                          P=m["precision"], R=m["recall"], f1=m["f1"],
                          tp=m["tp"], fp=m["fp"], fn=m["fn"]))
    return folds


# =============================================================================
# Threshold-only baseline (your current pipeline) under the SAME LOSO
# =============================================================================
def baseline_class(label, tracks_by_site, gt_by_site_label, sites, opts):
    """Fixed 500 ms median, uniform merge 0.5 s, duration [0.5, 30] s; tune only
    the per-class threshold on dev, score the held-out site."""
    cache = RawEventCache(tracks_by_site)
    thr_grid = opts["thr_grid"]
    iou = opts["iou"]
    med_k = max(1, cfg.SMOOTH_KERNEL_MS // int(cfg.FRAME_STRIDE_S * 1000))
    if med_k % 2 == 0:
        med_k += 1
    ev_fixed = (cfg.MERGE_GAP_S, cfg.POST_MIN_DUR_S, cfg.POST_MAX_DUR_S)

    folds = []
    for test_site in sites:
        dev_sites = [s for s in sites if s != test_site]
        gt_dev_bf = to_gt_by_file([g for s in dev_sites for g in gt_by_site_label[(s, label)]])
        gt_test_bf = to_gt_by_file(list(gt_by_site_label[(test_site, label)]))
        best_t, best_f1 = thr_grid[0], -1.0
        for t in thr_grid:
            raw = cache.raw_events(label, dev_sites, med_k, t, t, None)
            f1 = score_events(event_filter(raw, *ev_fixed), gt_dev_bf, iou)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        raw_t = cache.raw_events(label, [test_site], med_k, best_t, best_t, None)
        m = score_events(event_filter(raw_t, *ev_fixed), gt_test_bf, iou)
        folds.append(dict(test=test_site, threshold=best_t,
                          P=m["precision"], R=m["recall"], f1=m["f1"]))
    return folds


# =============================================================================
# LOSO aggregation -> paper macro-F1 (P/R averaged over classes AND test folds)
# =============================================================================
def aggregate(per_class_folds):
    """per_class_folds[label] = list of fold dicts with P, R. Returns
    (paper_macro_f1, per_class) where per_class[label] = (P_bar, R_bar, F1)."""
    per_class = {}
    Ps, Rs = [], []
    for label in CLASSES:
        folds = per_class_folds[label]
        pbar = float(np.mean([f["P"] for f in folds]))
        rbar = float(np.mean([f["R"] for f in folds]))
        per_class[label] = (pbar, rbar, f1_of(pbar, rbar))
        Ps.append(pbar)
        Rs.append(rbar)
    return f1_of(float(np.mean(Ps)), float(np.mean(Rs))), per_class


# =============================================================================
# Data loading  (server-only deps imported lazily inside these two functions)
# =============================================================================
def _find_posteriors(run_dir: Path, override: str | None):
    names = [override] if override else ["best_posteriors.npz", "paper_best_posteriors.npz"]
    for n in names:
        if n and (run_dir / n).exists():
            return run_dir / n
    return None


def load_combined(checkpoints, weights, posterior_name):
    """Combine members' saved per-class posteriors with the routing weights.

    weights: dict class -> array (n_members,). Returns combined[(ds, fn)] = (T, 3)."""
    members = []
    for ck in checkpoints:
        rd = Path(ck).parent
        npz_p = _find_posteriors(rd, posterior_name)
        if npz_p is None:
            raise FileNotFoundError(f"no posteriors npz in {rd} "
                                    f"(looked for best_posteriors.npz / paper_best_posteriors.npz)")
        npz = np.load(npz_p)
        members.append({k: npz[k].astype(np.float32) for k in npz.files})
        print(f"  [load] {rd.name:42s} {len(npz.files):4d} files  <- {npz_p.name}")

    keys = set(members[0])
    for m in members[1:]:
        keys &= set(m)
    w = {c: np.asarray(weights[c], dtype=np.float64) for c in CLASSES}
    for c in CLASSES:
        if w[c].shape[0] != len(members):
            raise ValueError(f"--weights-{c} has {w[c].shape[0]} entries, expected {len(members)}")
        if w[c].sum() <= 0:
            raise ValueError(f"--weights-{c} sums to 0")

    combined = {}
    for k in keys:
        arrs = [m[k] for m in members]
        L = min(a.shape[0] for a in arrs)
        out = np.zeros((L, 3), dtype=np.float32)
        for ci in range(3):
            num = np.zeros(L, dtype=np.float64)
            for i, a in enumerate(arrs):
                if w[CLASSES[ci]][i] != 0:
                    num += w[CLASSES[ci]][i] * a[:L, ci]
            out[:, ci] = (num / w[CLASSES[ci]].sum()).astype(np.float32)
        ds, fn = k.split("|", 1)
        combined[(ds, fn)] = out
    print(f"  [combine] {len(combined)} files across {len(members)} members")
    return combined


def load_val_gt():
    """Build val ground-truth Detections exactly like ensemble_blockE."""
    cfg.USE_3CLASS = True
    from dataset_final import load_annotations, get_file_manifest
    from rescore_base_epochs import build_gt_events
    anns = load_annotations(list(cfg.VAL_DATASETS))
    manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    fsd = {(r["dataset"], r["filename"]): r["start_dt"] for _, r in manifest.iterrows()}
    return build_gt_events(anns, fsd)


def split_tracks_by_site(combined):
    """combined[(ds,fn)]=(T,3)  ->  {label: {site: {fn: track_c}}}."""
    out = {lab: defaultdict(dict) for lab in CLASSES}
    for (ds, fn), arr in combined.items():
        for ci, lab in enumerate(CLASSES):
            out[lab][ds][fn] = arr[:, ci]
    return out


# =============================================================================
# Reporting
# =============================================================================
def print_report(base_pc_folds, srch_pc_folds, opts):
    base_macro, base_pc = aggregate(base_pc_folds)
    srch_macro, srch_pc = aggregate(srch_pc_folds)
    bar = "=" * 78

    print(f"\n{bar}\nLEAVE-ONE-SITE-OUT RESULT  (selection on 2 sites, score on the held-out site)\n{bar}")
    print(f"  sites: {', '.join(cfg.VAL_DATASETS)}   |   IoU {opts['iou']}   |   30 s posteriors")
    print(f"\n  {'class':6}  {'threshold-only (current)':>26}     {'searched post-processing':>26}")
    print(f"  {'':6}  {'P':>7}{'R':>8}{'F1':>8}     {'P':>7}{'R':>8}{'F1':>8}   {'ΔF1':>8}")
    for lab in CLASSES:
        bp, br, bf = base_pc[lab]
        sp, sr, sf = srch_pc[lab]
        print(f"  {lab:6}  {bp:7.3f}{br:8.3f}{bf:8.3f}     {sp:7.3f}{sr:8.3f}{sf:8.3f}   {sf-bf:+8.3f}")
    print(f"  {'-'*72}")
    print(f"  {'PAPER':6}  {'':7}{'':8}{base_macro:8.3f}     {'':7}{'':8}{srch_macro:8.3f}   "
          f"{srch_macro-base_macro:+8.3f}   (macro-F1 of mean P/R)")

    print(f"\n{bar}\nPER-FOLD DETAIL (searched)\n{bar}")
    for lab in CLASSES:
        print(f"  {lab}:")
        for f in srch_pc_folds[lab]:
            p = f["params"]
            hh = "" if p["off"] >= p["on"] else f" off={p['off']:.2f}"
            hg = "" if p["hangover_k"] is None else f" hang={p['hangover_k']}"
            md = "med=none" if p["median_k"] is None else f"med={p['median_k']}"
            print(f"    hold-out {f['test']:14s}  F1={f['f1']:.3f} "
                  f"(P {f['P']:.3f} R {f['R']:.3f})  | on={p['on']:.2f}{hh} {md}{hg} "
                  f"merge={p['merge_gap']:.2f} dur[{p['min_dur']:.2f},{p['max_dur']:.2f}]")

    print(f"\n{bar}\nVERDICT\n{bar}")
    d = srch_macro - base_macro
    if d <= 0.002:
        print(f"  Searched post-processing does NOT beat threshold-only ({d:+.3f} paper-F1, LOSO).")
        print("  -> confirms the simple per-class-threshold pipeline; report it as the negative")
        print("     result, now cross-checked with the paper's own backward-search protocol.")
    elif d < 0.01:
        print(f"  Marginal gain of {d:+.3f} paper-F1 (LOSO) — within run-to-run noise; treat as a")
        print("     tie unless it is stable across folds and survives on the challenge test set.")
    else:
        print(f"  Searched post-processing gains {d:+.3f} paper-F1 (LOSO). Check which knob carries")
        print("     it in the per-fold detail above; per-class event-level (merge/min/max) and the")
        print("     median kernel are easy to deploy in postprocess_final, hysteresis/hangover are not.")
    print("  NOTE: this is selection-unbiased (LOSO); it is the honest generalisation estimate and")
    print("        is the number to report, not a single val-tuned figure.")


# =============================================================================
# Self-test on synthetic data (no server modules / no data needed)
# =============================================================================
def selftest():
    print("[selftest] primitives + search on synthetic tracks ...")
    rng = np.random.default_rng(0)
    # hysteresis: a clean bump 0.6 with a dip to 0.45 -> on=0.5,off=0.4 keeps it whole
    p = np.array([0.0, 0.6, 0.45, 0.6, 0.0])
    assert hysteresis_binary(p, 0.5, 0.4).tolist() == [False, True, True, True, False]
    assert hysteresis_binary(p, 0.5, 0.5).tolist() == [False, True, False, True, False]  # off>=on -> strict
    # hangover: lone spike removed by majority over 3
    b = np.array([0, 1, 0, 1, 1, 1, 0], dtype=bool)
    # causal majority over the last 3 frames: lone spike at t=1 suppressed; t>=3 hold (>=2 of 3)
    assert hangover_binary(b, 3).tolist() == [False, False, False, True, True, True, True]
    # runs + event filter: two 0.1 s events 0.2 s apart -> merge@0.5 then drop if <0.5 s min-dur
    track = np.zeros(60, dtype=np.float32); track[5:10] = 0.8; track[20:25] = 0.8
    evs = runs_to_events(track > 0.5, track, "S", "f", "d")
    assert len(evs) == 2
    assert len(event_filter(evs, 0.5, 0.0, 30.0)) == 1          # merged (gap 0.2 s <= 0.5)
    assert len(event_filter(evs, 0.05, 0.0, 30.0)) == 2         # not merged (gap 0.2 s > 0.05)
    assert len(event_filter(evs, 0.5, 0.5, 30.0)) == 0          # merged 0.3 s span < 0.5 s min

    # tiny end-to-end search: build synthetic combined + gt with a clear per-class optimum
    sites = ["A", "B", "C"]
    combined, gt = {}, []
    for si, s in enumerate(sites):
        for fi in range(4):
            fn = f"f{fi}"
            T = 500
            arr = rng.uniform(0, 0.2, size=(T, 3)).astype(np.float32)
            # plant 3 d-calls of ~1.0 s (50 frames) per file, model fires 0.7 over them
            for j in range(3):
                st = 60 + j * 120
                arr[st:st + 50, 1] = 0.7
                gt.append(Detection(s, fn, "d", st * STRIDE, (st + 50) * STRIDE))
            # plant 2 bmabz of ~4 s + a 0.4 s spurious blip (should be filtered by min-dur)
            arr[20:220, 0] = 0.75; gt.append(Detection(s, fn, "bmabz", 20 * STRIDE, 220 * STRIDE))
            arr[260:460, 0] = 0.75; gt.append(Detection(s, fn, "bmabz", 260 * STRIDE, 460 * STRIDE))
            arr[470:490, 0] = 0.75                              # 0.4 s false blip, no gt
            combined[(s, fn)] = arr
    tbs = split_tracks_by_site(combined)
    gbsl = defaultdict(list)
    for g in gt:
        gbsl[(g.dataset, g.label)].append(g)

    # fast scorer must match compute_metrics exactly (same greedy rule + tie-breaking)
    some = runs_to_events(combined[("A", "f0")][:, 1] > 0.5, combined[("A", "f0")][:, 1], "A", "f0", "d")
    a = score_events(some, to_gt_by_file(gbsl[("A", "d")]), 0.3)
    b = class_pr_f1(some, list(gbsl[("A", "d")]), 0.3)
    assert (a["tp"], a["fp"], a["fn"]) == (b["tp"], b["fp"], b["fn"]) and abs(a["f1"] - b["f1"]) < 1e-9, \
        f"score_events != compute_metrics: {a} vs {b}"

    # vectorised IoU must equal compute_iou_1d on partial overlaps
    for (a, b, c, d) in [(0.0, 2.0, 1.0, 3.0), (0.0, 1.0, 2.0, 3.0), (0.0, 4.0, 1.0, 2.0)]:
        inter = max(0.0, min(b, d) - max(a, c))
        mine = inter / max((b - a) + (d - c) - inter, 1e-12)
        assert abs(mine - compute_iou_1d(a, b, c, d)) < 1e-9, "IoU formula mismatch"

    # the numpy fused scorer must equal score_events(event_filter(...)) for every combo
    raw_all = []
    for s in sites:
        for fi in range(4):
            fn = f"f{fi}"
            tr = combined[(s, fn)][:, 0]
            raw_all += runs_to_events(tr > 0.4, tr, s, fn, "bmabz")
    gbf = to_gt_by_file([g for s in sites for g in gbsl[(s, "bmabz")]])
    gbf_arr = gt_arrays(gbf)
    raw_bf = raw_to_by_file(raw_all)
    for mg, mnd, mxd in [(0.5, 2.0, 30.0), (0.2, 3.0, 25.0), (0.9, 0.5, 40.0), (0.05, 1.0, 3.5)]:
        ref = score_events(event_filter(raw_all, mg, mnd, mxd), gbf, 0.3)
        fast = score_event_level(raw_bf, gbf_arr, mg, mnd, mxd, 0.3)
        assert (abs(fast[0] - ref["precision"]) < 1e-9 and abs(fast[1] - ref["recall"]) < 1e-9
                and abs(fast[2] - ref["f1"]) < 1e-9), f"score_event_level != event_filter path @ {mg,mnd,mxd}"

    opts = dict(thr_grid=_grid(0.1, 0.9, 0.1), iou=0.3,
                frame_stage=True, hysteresis=False, hangover=False)
    pc_folds = {lab: search_class(lab, tbs[lab], gbsl, sites, opts) for lab in CLASSES}
    macro, pc = aggregate(pc_folds)
    print(f"  synthetic d   F1={pc['d'][2]:.3f}  bmabz F1={pc['bmabz'][2]:.3f}  paper-macro={macro:.3f}")
    assert pc["d"][2] > 0.9 and pc["bmabz"][2] > 0.9, "search failed to recover the planted optima"
    print("[selftest] OK")


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", help="ensemble member best_model.pt paths")
    p.add_argument("--weights-bmabz", nargs="+", type=float)
    p.add_argument("--weights-d", nargs="+", type=float)
    p.add_argument("--weights-bp", nargs="+", type=float)
    p.add_argument("--posterior-name", default=None,
                   help="override npz filename (default: auto best_/paper_best_posteriors.npz)")
    p.add_argument("--thr-step", type=float, default=0.05,
                   help="threshold grid step (paper used 0.1; use 0.1 with --hysteresis to bound cost)")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--no-frame-stage", dest="frame_stage", action="store_false",
                   help="event-level only (Stage 1); matches the paper's 'event-level alone' row")
    p.add_argument("--hysteresis", action="store_true",
                   help="also search the hysteresis off-threshold (Table IVa); slower")
    p.add_argument("--hangover", action="store_true",
                   help="also search the hangover kernel (Table IVa); slower")
    p.add_argument("--out", default="postproc_search_result.json")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.checkpoints:
        raise SystemExit("provide --checkpoints (or --selftest)")

    n = len(args.checkpoints)
    weights = {
        "bmabz": args.weights_bmabz if args.weights_bmabz else [1.0] * n,
        "d":     args.weights_d if args.weights_d else [1.0] * n,
        "bp":    args.weights_bp if args.weights_bp else [1.0] * n,
    }
    thr_grid = _grid(args.thr_step, 1.0 - 1e-9, args.thr_step)
    opts = dict(thr_grid=thr_grid, iou=args.iou, frame_stage=args.frame_stage,
                hysteresis=args.hysteresis, hangover=args.hangover)

    # search-size estimate (per class, summed over folds)
    eg_combos = lambda lab: len(_grid(*EVENT_GRID[lab]["merge"])) * len(_grid(*EVENT_GRID[lab]["min_dur"])) \
        * len(_grid(*EVENT_GRID[lab]["max_dur"]))
    on_off = sum(1 for on in thr_grid for off in thr_grid if off <= on) if args.hysteresis else len(thr_grid)
    fr_combos = (len(FILTER_KERNELS) * on_off * (len(HANGOVER_KERNELS) if args.hangover else 1)) \
        if args.frame_stage else 0
    print(f"{'='*78}\nPOST-PROCESSING SEARCH (backward-search, LOSO over {len(cfg.VAL_DATASETS)} sites)\n{'='*78}")
    print(f"  threshold grid: {len(thr_grid)} pts (step {args.thr_step})   "
          f"frame stage: {'on' if args.frame_stage else 'off'}   "
          f"hysteresis: {args.hysteresis}   hangover: {args.hangover}")
    print(f"  configs / class / fold  ~ event {eg_combos('d')} + frame {fr_combos}")
    # rough ETA from benchmarked per-config costs (~0.025 s/event-combo, ~0.8 s/frame-config)
    eta_min = (sum(eg_combos(c) for c in CLASSES) * 0.025 + fr_combos * len(CLASSES) * 0.8) * 3 / 60 + 1.5
    print(f"  estimated runtime ~ {eta_min:.0f} min"
          + ("   (tip: --hysteresis/--hangover at --thr-step 0.1 to keep this bounded)"
             if (args.hysteresis or args.hangover) and args.thr_step < 0.1 else ""))

    print("\nLoading members + combining posteriors ...")
    combined = load_combined(args.checkpoints, weights, args.posterior_name)
    print("Building val ground truth ...")
    gt = load_val_gt()
    gbsl = defaultdict(list)
    for g in gt:
        if g.label in CLASSES:
            gbsl[(g.dataset, g.label)].append(g)
    tbs = split_tracks_by_site(combined)
    sites = list(cfg.VAL_DATASETS)

    print("\nSearching (this is the slow part) ...")
    srch_pc, base_pc = {}, {}
    for lab in CLASSES:
        t0 = time.time()
        base_pc[lab] = baseline_class(lab, tbs[lab], gbsl, sites, opts)
        srch_pc[lab] = search_class(lab, tbs[lab], gbsl, sites, opts)
        print(f"  [{lab:6}] done in {time.time()-t0:6.1f}s")

    print_report(base_pc, srch_pc, opts)

    base_macro, _ = aggregate(base_pc)
    srch_macro, _ = aggregate(srch_pc)
    Path(args.out).write_text(json.dumps(dict(
        loso_paper_f1_baseline=base_macro, loso_paper_f1_searched=srch_macro,
        delta=srch_macro - base_macro, opts={k: v for k, v in opts.items() if k != "thr_grid"},
        baseline=base_pc, searched=srch_pc), indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
