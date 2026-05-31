"""
ensemble_blockE.py — does pooling the Block E MT models help the ensemble?
==========================================================================

Posterior-averaging ensemble built from each run's saved ``best_posteriors.npz``
(the _final-stack stitched 3-class val posteriors at the selected epoch). For a
combo of arms, we average those posteriors across all member runs, tune the
per-class thresholds with the same _final sweep, and score. No inference, no
GPU, and no old/_final stack mismatch (we never re-run a model — we reuse the
exact posteriors it produced).

Why this is a faithful ensemble: averaging commutes with stitching (stitching is
per-frame averaging of overlaps + concat, linear), and every run shares the val
tiling, so average-stitched == stitch-averaged.

What to read
------------
* Equal-size combos (A0 vs A2 vs A3, all 5 members) isolate whether any single
  arm ensembles better than the baseline.
* Pooled combos (A0+A3, A0+A1+A2+A3) test whether adding the Block E models
  helps — but more members alone can shave variance, so a pooled lift over A0 is
  only "Block E value" if it beats what extra A0 seeds would give (it usually
  won't if the members are near-duplicates of A0, which the single-model
  analysis suggested).

Usage
-----
::

    python ensemble_blockE.py
    python ensemble_blockE.py --combos A0 A2 A3 A0+A3 A0+A1+A2+A3
    python ensemble_blockE.py --a0-pattern "whatever_s{seed}_..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config_final as cfg
from dataset_final import load_annotations, get_file_manifest
from rescore_base_epochs import (
    build_gt_events, evaluate, paper_f1, per_class_block, tune_thresholds,
)

SEEDS = [42, 2024, 7777, 9999, 1337]
ARM_PATTERN = {
    "A0": "mtfb_3class_s{seed}_ens_asymmetric_mse",
    "A1": "mtE_A1_3class_s{seed}_ens",
    "A2": "mtE_A2_3class_s{seed}_ens",
    "A3": "mtE_A3_3class_s{seed}_ens",
    "A4": "mtE_A4_3class_s{seed}_ens",
}


def load_member(run_dir: Path):
    npz_p = run_dir / "best_posteriors.npz"
    if not npz_p.exists():
        return None
    npz = np.load(npz_p)
    return {k: npz[k].astype(np.float32) for k in npz.files}   # {"ds|fn": (T,3)}


def average_members(members: list[dict]) -> dict:
    """Per-file mean across members (intersect files, align to min length)."""
    keys = set(members[0])
    for m in members[1:]:
        keys &= set(m)
    ens = {}
    for k in keys:
        arrs = [m[k] for m in members]
        L = min(a.shape[0] for a in arrs)
        ds, fn = k.split("|", 1)
        ens[(ds, fn, 0)] = np.mean([a[:L] for a in arrs], axis=0).astype(np.float32)
    return ens


def score(ens_probs, gt_events, val_workers):
    cfg.USE_3CLASS = True
    thr = tune_thresholds(ens_probs, gt_events, val_workers)
    metrics = evaluate(ens_probs, gt_events, thr)
    return paper_f1(metrics), per_class_block(metrics), np.asarray(thr, dtype=np.float64)


def gather(combo: str, seeds, runs_root: Path):
    """Collect member prob-dicts for a '+'-joined arm combo; return (members, labels)."""
    members, labels = [], []
    for arm in combo.split("+"):
        for seed in seeds:
            d = runs_root / ARM_PATTERN[arm].format(seed=seed)
            m = load_member(d)
            if m is None:
                print(f"  [missing] {arm} s{seed}: {d}")
                continue
            members.append(m)
            labels.append(f"{arm}.s{seed}")
    return members, labels


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--combos", nargs="+",
                   default=["A0", "A2", "A3", "A0+A3", "A0+A1+A2+A3"],
                   help="'+'-joined arm combos to score (e.g. A0+A3).")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--a0-pattern", default=None, help="Override A0 dir pattern ({seed}).")
    p.add_argument("--val-workers", type=int, default=13)
    return p.parse_args()


def main():
    args = parse_args()
    if args.a0_pattern:
        ARM_PATTERN["A0"] = args.a0_pattern
    runs_root = Path(args.runs_root)

    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)

    rows = []
    for combo in args.combos:
        members, labels = gather(combo, args.seeds, runs_root)
        if not members:
            print(f"  [skip] {combo}: no members found")
            continue
        ens = average_members(members)
        mp, pc, thr = score(ens, gt_events, args.val_workers)
        rows.append((combo, len(members), mp, pc, thr))
        print(f"  scored {combo:16} ({len(members)} members)")

    if not rows:
        print("No combos scored — check --runs-root / --a0-pattern.")
        return

    base = next((r for r in rows if r[0] == "A0"), None)

    print(f"\n{'='*86}\nENSEMBLE (posterior-average, _final posteriors)\n{'='*86}")
    print(f"  {'combo':16} {'mem':>4} {'paperF1':>9} {'D P':>7} {'D R':>7} "
          f"{'D F1':>7} {'BMABZ':>7} {'BP':>7} {'ΔpaperF1 vs A0':>15}")
    for combo, n, mp, pc, thr in rows:
        dM = f"{mp - base[2]:+.4f}" if base else "   —"
        print(f"  {combo:16} {n:>4} {mp:>9.4f} {pc['d']['precision']:>7.3f} "
              f"{pc['d']['recall']:>7.3f} {pc['d']['f1']:>7.3f} "
              f"{pc['bmabz']['f1']:>7.3f} {pc['bp']['f1']:>7.3f} {dM:>15}")

    print(f"\n  tuned thresholds (bmabz/d/bp):")
    for combo, n, mp, pc, thr in rows:
        print(f"    {combo:16} {['%.2f' % t for t in thr]}")

    # ---- verdict ----------------------------------------------------
    print(f"\n{'='*86}")
    if base:
        best = max(rows, key=lambda r: r[2])
        gain = best[2] - base[2]
        print(f"Baseline A0 ensemble (5): paper-F1 {base[2]:.4f}, D-F1 {base[3]['d']['f1']:.3f}")
        if best[0] == "A0":
            print("No combo beats the A0-only ensemble — the Block E models add no "
                  "ensemble value (expected: their single-model posteriors ≈ A0).")
        elif gain < 0.005:
            print(f"Best combo {best[0]} is +{gain:.4f} paper-F1 over A0 — within noise; "
                  "any lift is likely 'more members', not Block E decorrelation. "
                  "Check it against extra A0 seeds before claiming a gain.")
        else:
            print(f"Best combo {best[0]} is +{gain:.4f} paper-F1 over A0 ({best[1]} members). "
                  "If equal-size arm combos (A2/A3 at 5) don't show it, the lift is "
                  "from pooling more models, not the Block E arms specifically.")
    print("Scored from saved _final posteriors; no re-inference, no stack mismatch.")


if __name__ == "__main__":
    main()
