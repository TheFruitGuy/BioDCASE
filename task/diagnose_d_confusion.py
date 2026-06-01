#!/usr/bin/env python
"""
diagnose_d_confusion.py
=======================

Decision tool for: "would a softmax (multi-class) head help D-class recall?"

The softmax-vs-sigmoid lever is *inter-class competition*. Softmax can only
recover a missed D frame if D is being out-ranked by another CALL class at
that frame. It does nothing for D frames the model simply doesn't see (low
energy everywhere -> would lose to a no-call class under softmax too), and it
actively HURTS frames where D legitimately co-occurs with another class
(single-winner assumption).

So this decomposes the GT-D-active frames the current sigmoid ensemble
produces into:

  - polyphonic : D co-occurs with bmabz/bp in the GT  (softmax would hurt)
  and, on the remaining D-ONLY frames, at an activity threshold tau_act:
  - background : max class prob < tau_act     -> sensitivity miss, softmax won't help
  - D_leads    : argmax == d                  -> threshold/calibration, not confusion
  - confused   : argmax == bmabz or bp        -> inter-class confusion, softmax CAN help

Verdict heuristic (printed at the end):
  high background + D_leads  -> limiter is detection sensitivity (CNN frontend)
                               + threshold calibration; softmax is the wrong tool.
  high confused              -> softmax has a real mechanism; run the A/B.
  high polyphonic            -> single-winner assumption violated; strike
                               against a pure softmax head.

This is a frame-level proxy: it tests the *mechanism*. Because the official
metric is event-level, confirm any positive read with the actual
sigmoid-vs-softmax A/B (same CNN-BiLSTM, your 5 seeds, macro F1) before
committing to the refactor.

Reuses the ensemble prob cache from ensemble_predict_cached.py, so if you've
already evaluated these checkpoints the inference pass is a cache hit; the
only real cost is one pass over the val loader to read GT targets (no model).

VAL_DATASETS only (Casey2017 / Kerguelen2014 / Kerguelen2015). The held-out
BioDCASE test sites are never loaded. Drop this next to
ensemble_predict_cached.py and run from the task dir.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from spectrogram import SpectrogramExtractor
from ensemble_predict import average_prob_dicts
from ensemble_predict_cached import get_or_compute_probs


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Same checkpoints you ensemble at inference.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights (same convention as "
                        "ensemble_predict_cached). e.g. 1 1 1 2.")
    p.add_argument("--cache_dir", type=str, default="runs/prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--tau_act", nargs="+", type=float, default=[0.2, 0.3, 0.5],
                   help="Activity thresholds: a D frame counts as 'background' "
                        "if max class prob < tau_act. Swept for robustness.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise SystemExit(
            f"len(weights)={len(args.weights)} must equal "
            f"len(checkpoints)={len(args.checkpoints)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Guard: never touch held-out test sites.
    test_sites = {"kerguelen2020", "ddu2021"}
    bad = test_sites.intersection(set(cfg.VAL_DATASETS))
    if bad:
        raise SystemExit(
            f"REFUSING TO RUN: held-out test site(s) {bad} in cfg.VAL_DATASETS")
    print(f"Datasets (val only): {cfg.VAL_DATASETS}")

    # ------------------------------------------------------------------
    # Val loader (same construction as eval_with_bout_filter / cached).
    # ------------------------------------------------------------------
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  {len(val_segs)} val tiles")
    spec_extractor = SpectrogramExtractor().to(device)

    # ------------------------------------------------------------------
    # Ensemble probs (cache-aware). Cache hit -> seconds.
    # ------------------------------------------------------------------
    all_prob_dicts = []
    for i, ckpt in enumerate(args.checkpoints):
        print(f"\n[{i + 1}/{len(args.checkpoints)}] {ckpt}")
        all_prob_dicts.append(get_or_compute_probs(
            ckpt, spec_extractor, val_loader, device,
            args.cache_dir, no_cache=args.no_cache, use_fp16=args.use_fp16))

    weights = None
    if args.weights:
        tot = sum(args.weights)
        weights = [w / tot for w in args.weights]
        print(f"\nWeights (normalized): {weights}")
    ens_probs = (all_prob_dicts[0] if len(all_prob_dicts) == 1
                 else average_prob_dicts(all_prob_dicts, weights=weights))
    print(f"\nEnsemble: {len(ens_probs)} segment prob arrays")

    # ------------------------------------------------------------------
    # GT targets straight off the loader (frame-aligned to the probs).
    # WhaleDataset yields (audio, targets, mask, metas); targets are the
    # multi-hot frame labels in cfg.CALL_TYPES_3 order. Matching probs and
    # targets by (dataset, filename, start_sample) avoids any datetime /
    # manifest-column reconstruction.
    # ------------------------------------------------------------------
    print("\nReading GT targets off the val loader (no inference; loads audio)...")
    gt = {}
    for _, targets, mask, metas in val_loader:
        t = targets.numpy()
        m = mask.numpy().astype(bool)
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n = int(m[j].sum())            # valid (unpadded) frames
            gt[key] = t[j, :n, :]

    # ------------------------------------------------------------------
    # Decompose GT-D frames.
    # ------------------------------------------------------------------
    names = list(cfg.CALL_TYPES_3)
    BM, D, BP = names.index("bmabz"), names.index("d"), names.index("bp")
    hop = spec_extractor.hop_length

    seen = set()        # (ds, fn, abs_frame) dedup across overlapping tiles
    n_d = 0             # all GT-D frames (deduped)
    n_poly = 0          # D co-occurs with another class
    pure_probs = []     # prob vectors at D-only frames

    for key in (set(ens_probs) & set(gt)):
        p = np.asarray(ens_probs[key], dtype=np.float32)
        g = np.asarray(gt[key], dtype=np.float32)
        n = min(len(p), len(g))
        f_off = key[2] // hop
        for f in range(n):
            if g[f, D] < 0.5:
                continue
            absf = (key[0], key[1], f_off + f)
            if absf in seen:               # keep-first dedup for overlap frames
                continue
            seen.add(absf)
            n_d += 1
            if g[f, BM] >= 0.5 or g[f, BP] >= 0.5:
                n_poly += 1
            else:
                pure_probs.append(p[f])

    if n_d == 0:
        raise SystemExit("No GT-D frames found - check class order / USE_3CLASS.")

    arr = np.asarray(pure_probs)           # (N_pure, 3), cols = [bmabz, d, bp]
    n_pure = len(arr)

    print("\n" + "=" * 70)
    print("D-CLASS MISS DECOMPOSITION  (frame-level, GT-D-active frames)")
    print("=" * 70)
    print(f"  GT-D frames (deduped):      {n_d:>8}")
    print(f"  polyphonic (D + other):     {n_poly:>8}  ({100 * n_poly / n_d:5.1f}%)"
          f"   <- a pure softmax head would suppress these")
    print(f"  D-only (clean test set):    {n_pure:>8}  ({100 * n_pure / n_d:5.1f}%)")

    if n_pure == 0:
        raise SystemExit("All GT-D frames are polyphonic - softmax single-winner "
                         "assumption is badly violated; keep sigmoid.")

    maxp = arr.max(1)
    amax = arr.argmax(1)
    print(f"\n  On the {n_pure} D-only frames (argmax is threshold-free; it is "
          f"exactly\n  what a softmax/argmax decision would pick among the calls):")
    print(f"    {'tau_act':>8}  {'background':>11}  {'D_leads':>9}  "
          f"{'conf->bmabz':>12}  {'conf->bp':>9}  {'CONFUSION':>10}")

    verdicts = {}
    for tau in args.tau_act:
        bg = maxp < tau
        nonbg = ~bg
        d_leads = nonbg & (amax == D)
        c_bm = nonbg & (amax == BM)
        c_bp = nonbg & (amax == BP)
        conf = c_bm | c_bp
        pc = lambda mk: 100.0 * mk.sum() / n_pure
        print(f"    {tau:>8.2f}  {pc(bg):>10.1f}%  {pc(d_leads):>8.1f}%  "
              f"{pc(c_bm):>11.1f}%  {pc(c_bp):>8.1f}%  {pc(conf):>9.1f}%")
        verdicts[tau] = (pc(bg), pc(d_leads), pc(conf))

    # ------------------------------------------------------------------
    # Plain-English verdict at the middle tau.
    # ------------------------------------------------------------------
    tau_mid = sorted(args.tau_act)[len(args.tau_act) // 2]
    bg_pct, dlead_pct, conf_pct = verdicts[tau_mid]
    poly_pct = 100.0 * n_poly / n_d

    print("\n" + "-" * 70)
    print(f"VERDICT (at tau_act={tau_mid:.2f}):")
    print(f"  confusion (softmax-addressable):  {conf_pct:5.1f}%  of D-only frames")
    print(f"  background (sensitivity miss):    {bg_pct:5.1f}%")
    print(f"  D leads but sub-threshold:        {dlead_pct:5.1f}%")
    print(f"  polyphonic D (softmax penalty):   {poly_pct:5.1f}%  of all GT-D")
    print()
    if conf_pct >= 30 and poly_pct < 25:
        print("  => Real inter-class confusion. Softmax has a mechanism here.")
        print("     Worth the clean A/B: softmax+no-call vs sigmoid on the SAME")
        print("     CNN-BiLSTM, your 5 seeds, report macro F1.")
    elif bg_pct + dlead_pct >= 65:
        print("  => D is mostly MISSED, not confused. The limiter is detection")
        print("     sensitivity (CNN frontend) + threshold calibration - which")
        print("     matches your own bottleneck finding. Softmax will not move D")
        print("     recall. Spend the slot on the frontend (PCEN/LEAF) instead.")
    else:
        print("  => Mixed. Confusion mass is non-trivial but so is background.")
        print("     Softmax might help a little; lower-EV than the frontend. If")
        print("     you run it, keep it as an ensemble member, not a wholesale swap.")
    if poly_pct >= 25:
        print(f"\n  NOTE: {poly_pct:.0f}% of GT-D is polyphonic - a pure softmax head")
        print("        would suppress the overlapped D. Prefer softmax over")
        print("        {bmabz,d,bp,no-call} PLUS a residual sigmoid for overlap,")
        print("        or just keep sigmoid.")
    print("-" * 70)


if __name__ == "__main__":
    main()
