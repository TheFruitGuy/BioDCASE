"""
run_loso_postprocess.py
=======================

Leave-one-site-out cross-validation driver for postprocess tuning. Runs
``tune_postprocess_optuna_narrow.py`` once per validation site,
holding that site out, and aggregates the held-out macro F1 across
folds.

Methodology (matches BPN paper Section II.C 3-fold CV):

    Fold 1: tune on {Kerguelen2014, Kerguelen2015} → eval on Casey2017
    Fold 2: tune on {Casey2017, Kerguelen2015}     → eval on Kerguelen2014
    Fold 3: tune on {Casey2017, Kerguelen2014}     → eval on Kerguelen2015

The reported headline number is the MEAN held-out macro F1 across the
three folds. This is the honest generalisation estimate — it measures
how well the postprocess config trained on two sites transfers to a
third site you didn't see during tuning. The gap between this number
and the all-sites-tuned-and-evaluated baseline quantifies overfitting
to the validation set.

Why subprocess: each fold runs in its own Python process so a crash
in one fold doesn't lose the others, and each fold can write its own
Optuna JournalFileStorage log for crash resilience and resume support.

Outputs
-------
- One ``runs/postprocess_loso_<site>.json`` per fold, written by the
  narrow tuner.
- One aggregated ``runs/postprocess_loso_summary.json`` with the
  per-fold results and the mean held-out macro F1.
- Console summary table.

Usage
-----
    python run_loso_postprocess.py \\
        --priors runs/duration_priors.json \\
        --checkpoints <10 ckpts> \\
        --weights-bmabz 1 1 1 1 1  0 0 0 0 0 \\
        --weights-d     0 0 0 0 0  1 1 1 1 1 \\
        --weights-bp    1 1 1 1 1  0 0 0 0 0 \\
        --segment-s 60 --use-fp16 \\
        --n-trials 100 --parallel-eval-workers 50
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--priors", type=str, required=True,
                   help="Path to JSON priors file from "
                        "analyze_call_durations.py.")
    p.add_argument("--checkpoints", nargs="+", required=True)

    # Ensemble combination.
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--d-from", type=int, default=None)
    p.add_argument("--weights-bmabz", nargs="+", type=float, default=None)
    p.add_argument("--weights-d", nargs="+", type=float, default=None)
    p.add_argument("--weights-bp", nargs="+", type=float, default=None)

    # Cache + inference.
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S)
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--cache-dir", type=str, default="runs/prob_cache")
    p.add_argument("--use-fp16", action="store_true")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)

    # Parallel eval (passed through to each fold).
    p.add_argument("--per-model-eval", action="store_true")
    p.add_argument("--parallel-eval-workers", type=int, default=1)

    # Optuna config (per-fold).
    p.add_argument("--n-trials", type=int, default=100,
                   help="Trials per fold. Default 100.")
    p.add_argument("--seed", type=int, default=cfg.SEED)

    # Output.
    p.add_argument("--output-dir", type=str, default="runs/loso",
                   help="Directory for per-fold JSONs and summary.")
    p.add_argument("--study-prefix", type=str,
                   default="loso",
                   help="Prefix for per-fold Optuna study names. "
                        "Each fold's study becomes "
                        "'<prefix>_<held_out_site>'.")
    p.add_argument("--sites", nargs="+", default=None,
                   help="Override the held-out site list. Default: "
                        "cfg.VAL_DATASETS (run one fold per val site).")
    p.add_argument("--resume", action="store_true",
                   help="If a fold's output JSON already exists, skip "
                        "that fold and re-use the existing result. "
                        "Useful for restarting after a partial run.")
    p.add_argument("--narrow-tuner", type=str,
                   default="tune_postprocess_optuna_narrow.py",
                   help="Path to the narrow tuner script.")
    return p.parse_args()


def build_fold_command(
    args: argparse.Namespace,
    held_out_site: str,
    output_path: Path,
    storage_path: Path,
    study_name: str,
) -> list[str]:
    """Construct the argv for one fold's invocation of the narrow tuner."""
    cmd = [
        sys.executable, args.narrow_tuner,
        "--priors", args.priors,
        "--checkpoints", *args.checkpoints,
        "--held-out-site", held_out_site,
        "--segment-s", str(args.segment_s),
        "--overlap-s", str(args.overlap_s),
        "--cache-dir", args.cache_dir,
        "--batch-size", str(args.batch_size),
        "--n-trials", str(args.n_trials),
        "--seed", str(args.seed),
        "--output", str(output_path),
        "--study-name", study_name,
        "--storage-path", str(storage_path),
        "--parallel-eval-workers", str(args.parallel_eval_workers),
    ]
    if args.use_fp16:
        cmd.append("--use-fp16")
    if args.per_model_eval:
        cmd.append("--per-model-eval")
    if args.weights is not None:
        cmd += ["--weights", *map(str, args.weights)]
    if args.d_from is not None:
        cmd += ["--d-from", str(args.d_from)]
    if args.weights_bmabz is not None:
        cmd += ["--weights-bmabz", *map(str, args.weights_bmabz)]
    if args.weights_d is not None:
        cmd += ["--weights-d", *map(str, args.weights_d)]
    if args.weights_bp is not None:
        cmd += ["--weights-bp", *map(str, args.weights_bp)]
    return cmd


def main() -> None:
    args = parse_args()
    sites = args.sites or list(cfg.VAL_DATASETS)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LOSO postprocess tuning across {len(sites)} folds:")
    for s in sites:
        print(f"  - hold out {s}; tune on {[x for x in sites if x != s]}")
    print(f"\nOutput dir: {out_dir}")
    print(f"Trials per fold: {args.n_trials}\n")

    # ------------------------------------------------------------------
    # Run each fold (subprocess, sequential — each one does its own
    # parallel per-model eval, so running folds in parallel would
    # oversubscribe).
    # ------------------------------------------------------------------
    per_fold_results: list[dict] = []
    overall_t0 = time.time()
    for i, held_out in enumerate(sites):
        fold_t0 = time.time()
        output_path = out_dir / f"postprocess_loso_{held_out}.json"
        storage_path = (
            out_dir / f"{args.study_prefix}_{held_out}.log"
        )
        study_name = f"{args.study_prefix}_{held_out}"

        print(f"\n{'#' * 64}")
        print(f"# Fold {i+1}/{len(sites)}: hold out {held_out}")
        print(f"#   output:  {output_path}")
        print(f"#   storage: {storage_path}")
        print(f"{'#' * 64}\n")

        if args.resume and output_path.exists():
            print(f"  --resume: {output_path} exists, skipping subprocess.")
        else:
            cmd = build_fold_command(
                args, held_out, output_path, storage_path, study_name,
            )
            print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}\n",
                  flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"\nFold {held_out} FAILED with exit code "
                      f"{result.returncode}. Continuing to remaining "
                      f"folds; re-run with --resume to fill in.")
                per_fold_results.append({
                    "held_out_site": held_out,
                    "status": "failed",
                    "exit_code": result.returncode,
                })
                continue

        if not output_path.exists():
            print(f"WARNING: fold {held_out} produced no output JSON. "
                  f"Skipping.")
            per_fold_results.append({
                "held_out_site": held_out,
                "status": "missing_output",
            })
            continue

        with open(output_path) as f:
            fold = json.load(f)
        fold["status"] = "ok"
        fold["wall_seconds"] = round(time.time() - fold_t0, 1)
        per_fold_results.append(fold)
        eval_macro = fold.get("macro_f1_held_out")
        tune_macro = fold.get("macro_f1_narrow_tune")
        cd_macro = fold.get("macro_f1_threshold_only_tune")
        print(f"\n  Fold {held_out} done in {fold['wall_seconds']:.0f}s:")
        print(f"    threshold-only (tune split): {cd_macro:.4f}")
        print(f"    narrow Optuna (tune split):  {tune_macro:.4f}")
        print(f"    HELD-OUT macro F1:           {eval_macro:.4f}")

    overall_wall = time.time() - overall_t0

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    ok_folds = [f for f in per_fold_results if f.get("status") == "ok"]
    held_out_macros = [
        f["macro_f1_held_out"] for f in ok_folds
        if f.get("macro_f1_held_out") is not None
    ]
    tune_macros = [
        f["macro_f1_narrow_tune"] for f in ok_folds
        if f.get("macro_f1_narrow_tune") is not None
    ]
    cd_macros = [
        f["macro_f1_threshold_only_tune"] for f in ok_folds
        if f.get("macro_f1_threshold_only_tune") is not None
    ]

    print("\n\n" + "=" * 64)
    print("LOSO SUMMARY")
    print("=" * 64)
    print(f"\n  {'fold':<24} {'cd-tune':>9} {'narrow-tune':>12} "
          f"{'HELD-OUT':>9} {'overfit Δ':>10}")
    for fold in per_fold_results:
        site = fold.get("held_out_site", "?")
        if fold.get("status") != "ok":
            print(f"  {site:<24} {'FAILED':>9}")
            continue
        cd = fold.get("macro_f1_threshold_only_tune", float("nan"))
        nt = fold.get("macro_f1_narrow_tune", float("nan"))
        ho = fold.get("macro_f1_held_out", float("nan"))
        delta = nt - ho if (
            nt is not None and ho is not None
        ) else float("nan")
        print(f"  {site:<24} {cd:>9.4f} {nt:>12.4f} {ho:>9.4f} "
              f"{delta:>+10.4f}")

    if held_out_macros:
        mean_held = float(np.mean(held_out_macros))
        std_held = float(np.std(held_out_macros))
        mean_tune = float(np.mean(tune_macros)) if tune_macros else None
        mean_cd = float(np.mean(cd_macros)) if cd_macros else None

        print(f"\n  Mean held-out macro F1:    {mean_held:.4f} "
              f"(± {std_held:.4f} over {len(held_out_macros)} folds)")
        if mean_tune is not None:
            print(f"  Mean narrow-tune macro F1: {mean_tune:.4f}")
            print(f"  Overfit gap (tune-held):   "
                  f"{mean_tune - mean_held:+.4f}")
        if mean_cd is not None:
            print(f"  Mean threshold-only macro: {mean_cd:.4f}")
            print(f"  Gain from narrow Optuna:   "
                  f"{(mean_tune - mean_cd):+.4f}" if mean_tune else "n/a")
    else:
        mean_held = mean_std = mean_tune = mean_cd = None
        print(f"\n  No successful folds — nothing to aggregate.")

    summary = {
        "sites": sites,
        "n_folds_ok": len(ok_folds),
        "n_folds_total": len(per_fold_results),
        "mean_macro_f1_held_out": mean_held,
        "std_macro_f1_held_out": std_held if held_out_macros else None,
        "mean_macro_f1_narrow_tune": mean_tune,
        "mean_macro_f1_threshold_only_tune": mean_cd,
        "overfit_gap_tune_minus_held": (
            (mean_tune - mean_held)
            if (mean_tune is not None and mean_held is not None)
            else None
        ),
        "per_fold": per_fold_results,
        "wall_seconds": round(overall_wall, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = out_dir / "postprocess_loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {summary_path}")
    print(f"Total wall time: {overall_wall:.0f}s")


if __name__ == "__main__":
    main()
