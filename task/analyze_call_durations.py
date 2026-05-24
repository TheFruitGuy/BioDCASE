"""
analyze_call_durations.py
=========================

Compute per-3-class call duration statistics from training + validation
annotations and write a JSON priors file consumable by
``tune_postprocess_optuna_narrow.py``.

The output fixes data-driven, site-invariant defaults for:

    min_dur_s   = max(0.1, P5_duration  * min_buffer)   per class
    max_dur_s   = P95_duration * max_buffer             per class
    merge_gap_s = fixed conservative default            per class

These priors freeze the parameters that should NOT be tuned per-site
(physical call properties of the species) and leave the genuinely
site-dependent parameters (thresholds, smoothing, hangover) for Optuna
to search. The motivation: the previous 500-trial full-space Optuna run
ended at macro 0.478, below its own 0.490 threshold-only baseline,
because TPE was free to pick ``min_dur_s = 2.60`` for D — a value that
filtered out genuine ~1-2s D-calls. Fixing duration from data prevents
that failure mode.

Computed from the union of train and val annotations: we are measuring
what calls *look like* across the corpus, not what is optimal for
detection on any one site. Test sites (Kerguelen2020, DDU2021) are
explicitly excluded — they are held-out for the BioDCASE evaluation.

Usage
-----
    python analyze_call_durations.py
    python analyze_call_durations.py --output runs/duration_priors.json
    python analyze_call_durations.py --datasets Casey2018 Scott2019 Prydz2013
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from dataset import load_annotations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", type=str,
                   default="runs/duration_priors.json",
                   help="Where to write the priors JSON.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Sites to include. Default: cfg.TRAIN_DATASETS + "
                        "cfg.VAL_DATASETS (test sites are NOT used).")
    p.add_argument("--p-low", type=float, default=0.05,
                   help="Lower percentile for min_dur_s (default 0.05). "
                        "5%% of real events are shorter than this — the "
                        "buffer below allows for slight test-set drift.")
    p.add_argument("--p-high", type=float, default=0.95,
                   help="Upper percentile for max_dur_s (default 0.95).")
    p.add_argument("--min-buffer", type=float, default=0.8,
                   help="min_dur_s = max(0.1, P_low * min_buffer). "
                        "Default 0.8 = 20%% slack below the empirical "
                        "P5, so test-set calls that are slightly "
                        "shorter than train-set P5 still pass.")
    p.add_argument("--max-buffer", type=float, default=1.25,
                   help="max_dur_s = P_high * max_buffer. Default 1.25 "
                        "= 25%% slack above the empirical P95.")
    p.add_argument("--merge-gap-s", type=float, default=0.1,
                   help="Fixed merge_gap_s value used for all classes. "
                        "Default 0.1s — conservatively short so genuine "
                        "neighbouring events stay separate. Override "
                        "with a data-driven value if you analyse "
                        "inter-event gaps separately.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    datasets = args.datasets or (
        list(cfg.TRAIN_DATASETS) + list(cfg.VAL_DATASETS)
    )
    print(f"Analyzing call durations from {len(datasets)} sites:")
    for d in datasets:
        print(f"  - {d}")

    ann = load_annotations(datasets)
    ann["duration_s"] = (
        pd.to_datetime(ann["end_datetime"])
        - pd.to_datetime(ann["start_datetime"])
    ).dt.total_seconds()

    print(f"\nTotal annotations: {len(ann)}")
    print(f"  per dataset:")
    for ds, n in ann["dataset"].value_counts().items():
        print(f"    {ds:<30}: {n:>6}")
    print(f"  per 3-class label:")
    for lbl, n in ann["label_3class"].value_counts().items():
        print(f"    {lbl:<30}: {n:>6}")

    classes = list(cfg.CALL_TYPES_3)
    stats: dict = {}
    priors: dict = {"min_dur_s": {}, "max_dur_s": {}, "merge_gap_s": {}}

    print(f"\nPer-class duration statistics (seconds):")
    print(f"  {'class':<6} {'n':>6}  "
          f"{'P5':>6} {'P25':>6} {'med':>6} {'P75':>6} {'P95':>6}  "
          f"→  {'min_dur':>7}  {'max_dur':>7}")
    for cls in classes:
        sub = ann.loc[ann["label_3class"] == cls, "duration_s"]
        if len(sub) == 0:
            print(f"  WARNING: no events for class {cls}; skipping")
            continue
        p_low = float(sub.quantile(args.p_low))
        p25 = float(sub.quantile(0.25))
        med = float(sub.median())
        p75 = float(sub.quantile(0.75))
        p_high = float(sub.quantile(args.p_high))

        min_dur = round(max(0.1, p_low * args.min_buffer), 2)
        max_dur = round(p_high * args.max_buffer, 2)

        stats[cls] = {
            "n": int(len(sub)),
            f"p{int(args.p_low*100)}": round(p_low, 4),
            "p25": round(p25, 4),
            "median": round(med, 4),
            "p75": round(p75, 4),
            f"p{int(args.p_high*100)}": round(p_high, 4),
            "mean": round(float(sub.mean()), 4),
            "std": round(float(sub.std()), 4),
        }
        priors["min_dur_s"][cls] = min_dur
        priors["max_dur_s"][cls] = max_dur
        priors["merge_gap_s"][cls] = args.merge_gap_s

        print(f"  {cls:<6} {len(sub):>6}  "
              f"{p_low:>6.2f} {p25:>6.2f} {med:>6.2f} {p75:>6.2f} "
              f"{p_high:>6.2f}  →  {min_dur:>7.2f}  {max_dur:>7.2f}")

    print(f"\nFixed parameters (not tuned):")
    print(f"  merge_gap_s = {args.merge_gap_s} for all classes")

    print(f"\nFreed parameters (tuned by narrow Optuna search):")
    print(f"  on_thr, off_gap_ratio, median_kernel_ms, hangover_kernel_ms")

    output = {
        "computed_from": datasets,
        "n_total": int(len(ann)),
        "stats": stats,
        "priors": priors,
        "computation": {
            "p_low": args.p_low,
            "p_high": args.p_high,
            "min_buffer": args.min_buffer,
            "max_buffer": args.max_buffer,
            "merge_gap_fixed": args.merge_gap_s,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote priors to {out_path}")
    print(f"  → pass to tune_postprocess_optuna_narrow.py via --priors {out_path}")


if __name__ == "__main__":
    main()
