"""
Probability-averaging ensemble of Conformer checkpoints (the 13r recipe seeds).
==============================================================================

Loads N checkpoints, reuses ``eval_conformer.get_probs`` (the per-checkpoint val
probability cache), **averages the per-frame probabilities across members**, then
runs the *same* per-class coordinate-descent threshold tuner + event-level
scoring as the single-checkpoint eval. Reports each member's own tuned macro and
the ensemble's tuned macro / per-class, and writes the chosen thresholds.

Because get_probs caches per checkpoint, this is incremental: once a new seed
finishes, re-running only does the forward pass for that one member (~minutes);
every prior member and the tuning are ~seconds.

    # ensemble seed 42 (0.495) + new seeds, on a free A40:
    CUDA_VISIBLE_DEVICES=0 python ensemble_conformer.py \
        runs/phase13r_3c_s42_*/phase13r_best.pt \
        runs/phase13r_3c_s2024_*/phase13r_best.pt \
        runs/phase13r_3c_s7777_*/phase13r_best.pt \
        --workers 13

Point each path at that seed's EMA-best checkpoint (the ``*_best.pt`` in its run
dir). Needs >= 2 checkpoints. Probabilities are the canonical ensemble object
(not logits, not post-tuned detections) — averaging happens before thresholding.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from postprocess_final import Detection
from eval_conformer import (
    get_probs, tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, _fmt, CLASS_NAMES,
)


def average_probs(member_probs):
    """Mean per-frame probabilities across members.

    `member_probs` is a list of dicts {segment_key: (n_frames, C) float array}.
    Aligns on the intersection of keys and the min frame count per key (the
    recipe seeds are the same architecture so keys/frames match exactly; the
    intersection + truncation are defensive)."""
    keys = set(member_probs[0])
    for m in member_probs[1:]:
        keys &= set(m)
    dropped = max(len(m) for m in member_probs) - len(keys)
    if dropped:
        print(f"  [warn] {dropped} segment key(s) not shared by all members; "
              f"averaging over the {len(keys)} shared.")
    avg = {}
    for k in keys:
        arrs = [m[k] for m in member_probs]
        n = min(a.shape[0] for a in arrs)
        avg[k] = np.mean(np.stack([a[:n].astype(np.float32) for a in arrs]),
                         axis=0)
    return avg


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+",
                   help="Two or more Conformer checkpoints (each seed's *_best.pt).")
    p.add_argument("--workers", type=int, default=min((os.cpu_count() or 1), 13),
                   help="Parallel grid workers for tuning (default min(cpu,13)).")
    p.add_argument("--cache-dir", type=str, default="runs/conformer_eval_cache")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute probs (use when multiple ckpts share a dir).")
    p.add_argument("--use-fp16", action="store_true",
                   help="FP16 autocast on the forward pass (~1.5-2x faster).")
    p.add_argument("--no-members", action="store_true",
                   help="Skip per-member tuned scoring (faster; ensemble only).")
    p.add_argument("--out", type=str, default="runs/conformer_ensemble_thr.json")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("ensemble needs at least 2 checkpoints.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  workers={args.workers}  members={len(args.checkpoints)}")

    # --- collect each member's cached val probabilities ---
    member_probs, gt_tuples = [], None
    for cp in args.checkpoints:
        print(f"\n--- member: {cp} ---")
        probs3, gt, _ = get_probs(Path(cp), args, device)
        member_probs.append(probs3)
        if gt_tuples is None:                              # gt is identical across members
            gt_tuples = gt
    gt = [Detection(dataset=d, filename=f, label=lab, start_s=s, end_s=e)
          for (d, f, lab, s, e) in gt_tuples]

    # --- per-member tuned macro (diagnostic: shows ensemble lift over best member) ---
    member_macros = []
    if not args.no_members:
        print("\n=== members (tuned) ===")
        for cp, probs3 in zip(args.checkpoints, member_probs):
            thr_m = tune_thresholds_per_class(probs3, gt, start=[0.5, 0.5, 0.5],
                                              workers=args.workers)
            mm = evaluate_with_thresholds(probs3, gt, thr_m)
            member_macros.append((cp, macro_f1(mm)))
            print(f"  macro={macro_f1(mm):.3f}  macro_paper={macro_f1_paper(mm):.3f}"
                  f"  {Path(cp).parent.name}")

    # --- ensemble: average probabilities, then tune ---
    print(f"\n=== ensemble of {len(member_probs)} (probability average) ===")
    avg = average_probs(member_probs)
    t0 = time.time()
    thr = tune_thresholds_per_class(avg, gt, start=[0.5, 0.5, 0.5],
                                    workers=args.workers)
    tuned = evaluate_with_thresholds(avg, gt, thr)
    fixed = evaluate_with_thresholds(avg, gt, [0.3, 0.3, 0.3])
    print(_fmt("fixed@0.3", fixed))
    print(_fmt("tuned", tuned))
    print(f"  thresholds  bmabz={thr[0]:.2f}  d={thr[1]:.2f}  bp={thr[2]:.2f}"
          f"   (tune {time.time() - t0:.0f}s, {args.workers} workers)")
    if member_macros:
        best_member = max(m for _, m in member_macros)
        print(f"  ENSEMBLE macro {macro_f1(tuned):.3f}  vs best member "
              f"{best_member:.3f}  ({macro_f1(tuned) - best_member:+.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "members": list(args.checkpoints),
            "thresholds": {n: float(t) for n, t in zip(CLASS_NAMES, thr)},
            "tuned": {"macro": macro_f1(tuned), "macro_paper": macro_f1_paper(tuned),
                      "per_class": {n: tuned.get(n, {}) for n in CLASS_NAMES}},
            "member_macros": {cp: mm for cp, mm in member_macros},
        }, f, indent=2, default=float)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
