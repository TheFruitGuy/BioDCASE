"""
per_site_hybrid_predict.py
==========================

Per-site evaluation of the hybrid 4-model ensemble on the validation
partition. Same hybrid combination strategy as ``hybrid_ensemble_predict.py``
(BMABZ + BP weighted-averaged, D-class taken from a single nominated
model), but instead of reporting a single aggregated number it splits
the val GT and probabilities by ``cfg.VAL_DATASETS`` and reports each
site independently.

Two reports per site
--------------------
1. **Global thresholds, applied per site.**
   Thresholds are tuned once on the pooled val GT (this is what the
   system would actually use at test time — you can't tune on the held-
   out evaluation set). Per-site F1 with those thresholds tells you
   where the model is strong/weak across sites.

2. **Per-site tuned thresholds (oracle).**
   Thresholds are tuned independently on each site's GT. This is an
   *oracle* — it cannot be used as a real result because it leaks the
   per-site labels into hyperparameter selection. It only serves as an
   upper bound on what per-site threshold adaptation could buy.

The gap between (1) and (2) on each site quantifies how much per-site
threshold tuning would help if you had a way to estimate site-specific
thresholds without labels (e.g., from a small held-out portion of each
site).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=1 python per_site_hybrid_predict.py \\
        --checkpoints \\
            runs/whalevad_20260504_152450/best_model.pt \\
            runs/phase5_20260506_204358/best_model.pt \\
            runs/whalevad_20260507_191223/best_model.pt \\
            runs/phase5_20260507_211504/best_model.pt \\
        --weights 1 1 1 2 \\
        --d-from 3 \\
        --output-csv-dir val_per_site_predictions/

The ``--output-csv-dir`` writes one CSV per site (using global
thresholds — the realistic, test-time pipeline). Skip the flag to
report only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from dataset import build_dataloaders, load_annotations, get_file_manifest
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)
from hybrid_ensemble_predict import (
    build_model_for_ckpt,
    predict_probabilities,
    hybrid_combine,
    tune_thresholds_on_probs,
    evaluate_with_thresholds,
    save_predictions_csv,
)


# ----------------------------------------------------------------------
# Per-site filtering
# ----------------------------------------------------------------------

def filter_probs_by_site(
    all_probs: dict[tuple, np.ndarray],
    site: str,
) -> dict[tuple, np.ndarray]:
    """Keep only probability segments whose dataset == site."""
    return {k: v for k, v in all_probs.items() if k[0] == site}


def filter_events_by_site(
    events: list[Detection],
    site: str,
) -> list[Detection]:
    """Keep only events whose dataset == site."""
    return [e for e in events if e.dataset == site]


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------

def print_site_metrics(
    metrics: dict[str, dict[str, float]],
    thresholds: np.ndarray,
    site: str,
    title: str,
) -> None:
    """Print a per-site metric block in the same layout as the aggregated
    HYBRID RESULT block from hybrid_ensemble_predict.py."""
    print(f"\n  [{site}] {title}:")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():<6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5d} FP={m.get('fp', 0):5d} "
              f"FN={m.get('fn', 0):5d}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    overall = metrics.get("overall", {}).get("f1", 0.0)
    macro = float(np.mean([
        metrics.get(c, {}).get("f1", 0.0) for c in cfg.CALL_TYPES_3
    ]))
    print(f"    OVERALL F1 (micro) = {overall:.3f}   "
          f"MACRO F1 = {macro:.3f}")


def print_summary_table(
    per_site_results: dict[str, dict],
    aggregated: dict[str, dict[str, float]],
) -> None:
    """Compact end-of-run table comparing the two reporting modes."""
    print("\n" + "=" * 72)
    print("SUMMARY  (overall = micro F1 over all 3 classes for that subset)")
    print("=" * 72)
    print(f"  {'site':<18} {'thr-mode':<16}  "
          f"{'bmabz':>6} {'d':>6} {'bp':>6} {'micro':>6} {'macro':>6}")
    print(f"  {'-'*18} {'-'*16}  "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for site, blocks in per_site_results.items():
        for label, (metrics, _thr) in blocks.items():
            f_bm = metrics.get("bmabz", {}).get("f1", 0.0)
            f_d = metrics.get("d", {}).get("f1", 0.0)
            f_bp = metrics.get("bp", {}).get("f1", 0.0)
            f_micro = metrics.get("overall", {}).get("f1", 0.0)
            f_macro = float(np.mean([f_bm, f_d, f_bp]))
            print(f"  {site:<18} {label:<16}  "
                  f"{f_bm:>6.3f} {f_d:>6.3f} {f_bp:>6.3f} "
                  f"{f_micro:>6.3f} {f_macro:>6.3f}")
        print()

    # Aggregated (the original 0.516 line)
    f_bm = aggregated.get("bmabz", {}).get("f1", 0.0)
    f_d = aggregated.get("d", {}).get("f1", 0.0)
    f_bp = aggregated.get("bp", {}).get("f1", 0.0)
    f_micro = aggregated.get("overall", {}).get("f1", 0.0)
    f_macro = float(np.mean([f_bm, f_d, f_bp]))
    print(f"  {'ALL POOLED':<18} {'global thr':<16}  "
          f"{f_bm:>6.3f} {f_d:>6.3f} {f_bp:>6.3f} "
          f"{f_micro:>6.3f} {f_macro:>6.3f}   ← original 0.516 result")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-site evaluation of the hybrid 4-model ensemble."
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to .pt checkpoint files (typically 4).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for BMABZ+BP averaging. "
                        "Length must match --checkpoints. Default: equal.")
    p.add_argument("--d-from", type=int, default=None,
                   help="0-indexed position in --checkpoints of the model "
                        "to use for D-class predictions. Required for "
                        "ensemble runs (n_models > 1); auto-set to 0 for "
                        "single-model runs.")
    p.add_argument("--output-csv-dir", type=Path, default=None,
                   help="If given, write one predictions CSV per site "
                        "(using global thresholds) to this directory.")
    p.add_argument("--per-site-csv-with-oracle", action="store_true",
                   help="When writing CSVs, also write a second CSV per "
                        "site that uses the per-site (oracle) thresholds. "
                        "Useful for diagnostics. Default: off.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Skip wandb tracking (default: register as "
                        "final_eval report).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_models = len(args.checkpoints)
    if args.weights is None:
        weights = [1.0] * n_models
    else:
        assert len(args.weights) == n_models, \
            "len(weights) must equal len(checkpoints)"
        weights = args.weights

    # --d-from is mandatory for ensembles (you need to nominate which
    # model's D head to use), but redundant for single-model runs.
    if args.d_from is None:
        if n_models == 1:
            args.d_from = 0
        else:
            raise SystemExit(
                "--d-from is required when --checkpoints has more than one "
                "model. Pass the 0-indexed position of the model whose D "
                "head should be used.")
    elif not (0 <= args.d_from < n_models):
        raise SystemExit(
            f"--d-from={args.d_from} out of range for {n_models} checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Validation sites: {cfg.VAL_DATASETS}")

    # ------------------------------------------------------------------
    # Wandb setup. Registered as ``final_eval`` (not a phase number) —
    # this script reports per-site F1 on the chosen submission ensemble
    # rather than training a model. Each consumed checkpoint's path is
    # logged so the prof can trace which trained models fed the
    # ensemble; ``job_type="eval"`` distinguishes it visually from
    # training runs in the wandb UI.
    # ------------------------------------------------------------------
    run = None
    if not args.no_wandb:
        # Infer source phases for each consumed checkpoint, using the same
        # path-regex convention as train.py / train_bpn.py / train_phase_hnm.py.
        import re
        source_phases: list[str] = []
        for ckpt_path in args.checkpoints:
            src = str(ckpt_path)
            m = re.search(r"phase(\w+)_\d{8}_\d{6}", src)
            if m:
                source_phases.append(m.group(1))
            elif "whalevad" in src:
                source_phases.append("baseline")
            else:
                source_phases.append("unknown")
        unique_sources = sorted(set(source_phases))

        extra_tags = ["per_site_eval"]
        if n_models == 1:
            extra_tags.append("single_model")
        else:
            extra_tags.append("hybrid_ensemble")
            extra_tags.append(f"hybrid_{n_models}")
            extra_tags.append(f"d_from_{args.d_from}")

        for sp in unique_sources:
            extra_tags.append(f"source_{sp}")

        # Tag each consumed checkpoint by its short directory name so
        # the wandb table can filter "all runs that included this
        # specific trained model". For a single-model run this is the
        # main identity; for an ensemble it's how you'd find "all runs
        # where phase5_20260507_211504 was one of the members".
        model_short_names: list[str] = []
        for ckpt_path in args.checkpoints:
            short = Path(ckpt_path).parent.name
            model_short_names.append(short)
            extra_tags.append(f"model_{short}")

        run = wbu.init_phase(
            "final_eval",
            extra_tags=extra_tags,
            job_type="eval",
            config={
                "checkpoints":       [str(c) for c in args.checkpoints],
                "model_short_names": model_short_names,
                "weights":           weights,
                "n_models":          n_models,
                "is_single_model":   n_models == 1,
                "d_from":            args.d_from,
                "d_from_checkpoint": str(args.checkpoints[args.d_from]),
                "d_from_short":      model_short_names[args.d_from],
                "d_from_source":     source_phases[args.d_from],
                "source_phases":     source_phases,
                "unique_source_phases": unique_sources,
                "val_sites":         list(cfg.VAL_DATASETS),
                "output_csv_dir":    (str(args.output_csv_dir)
                                       if args.output_csv_dir else None),
                "per_site_csv_with_oracle": args.per_site_csv_with_oracle,
            },
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    _, _, val_loader = build_dataloaders()

    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    val_ann = load_annotations(cfg.VAL_DATASETS)
    gt_events: list[Detection] = []
    for _, row in val_ann.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"Validation: {len(gt_events)} ground-truth events "
          f"across {len(cfg.VAL_DATASETS)} sites")
    for site in cfg.VAL_DATASETS:
        n = sum(1 for e in gt_events if e.dataset == site)
        print(f"  {site:<18} : {n:6d} events")

    spec_extractor = SpectrogramExtractor().to(device)

    # ------------------------------------------------------------------
    # Inference for each checkpoint
    # ------------------------------------------------------------------
    all_prob_dicts: list[dict[tuple, np.ndarray]] = []

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{n_models}] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_type = build_model_for_ckpt(ckpt, device)
        is_d_source = (i == args.d_from)
        marker = "  ← D source" if is_d_source else ""
        print(f"  type: {model_type}{marker}")

        # Lazy-init feat_proj before load_state_dict (same as hybrid script)
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            _ = model(spec_extractor(dummy))

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False,
        )
        if unexpected:
            print(f"  WARNING: unexpected keys: {len(unexpected)}")
        if missing:
            non_bpn_missing = [k for k in missing if "bpn" not in k]
            if non_bpn_missing:
                print(f"  WARNING: missing non-BPN keys: {len(non_bpn_missing)}")

        probs = predict_probabilities(
            model, model_type, spec_extractor, val_loader, device,
        )
        probs = collapse_probs_to_3class(probs)
        print(f"  collected probabilities for {len(probs)} segments")
        all_prob_dicts.append(probs)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Hybrid combination (once, on pooled segments). For n_models == 1
    # the hybrid_combine call still works correctly — weighted avg of
    # one model is itself, and the D-class branch picks that same
    # model — but we print a different header so the report reads
    # sensibly.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    if n_models == 1:
        print("SINGLE-MODEL EVALUATION")
        print(f"  Model: {Path(args.checkpoints[0]).parent.name}")
    else:
        print(f"HYBRID COMBINATION")
        print(f"  BMABZ + BP : weighted avg (weights {weights})")
        print(f"  D          : from model #{args.d_from + 1} only "
              f"({Path(args.checkpoints[args.d_from]).parent.name})")
    print("=" * 72)
    hybrid_probs = hybrid_combine(
        all_prob_dicts, d_from_idx=args.d_from, weights=weights,
    )
    print(f"  combined probs for {len(hybrid_probs)} segments")

    # ------------------------------------------------------------------
    # (A) Global threshold tune on pooled val
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(A) GLOBAL THRESHOLDS  (tuned once on pooled val — realistic)")
    print("=" * 72)
    print("\nThreshold tuning on pooled hybrid probabilities:")
    global_thresholds = tune_thresholds_on_probs(hybrid_probs, gt_events)
    print(f"\nGlobal thresholds: bmabz={global_thresholds[0]:.2f} "
          f"d={global_thresholds[1]:.2f} bp={global_thresholds[2]:.2f}")

    # Aggregated (across all sites) — should match the 0.516 result
    aggregated_metrics = evaluate_with_thresholds(
        hybrid_probs, gt_events, global_thresholds,
    )

    # ------------------------------------------------------------------
    # Per-site evaluation: two reporting modes side-by-side
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(B) PER-SITE RESULTS")
    print("=" * 72)

    per_site_results: dict[str, dict] = {}
    for site in cfg.VAL_DATASETS:
        site_probs = filter_probs_by_site(hybrid_probs, site)
        site_gt = filter_events_by_site(gt_events, site)
        n_seg = len(site_probs)
        n_gt = len(site_gt)
        print(f"\n--- {site} ({n_seg} segments, {n_gt} GT events) ---")

        if n_seg == 0 or n_gt == 0:
            print(f"  SKIP: no probabilities or no GT for {site}")
            continue

        # Mode 1: apply global thresholds to this site
        m_global = evaluate_with_thresholds(
            site_probs, site_gt, global_thresholds,
        )
        print_site_metrics(m_global, global_thresholds, site,
                           "global thresholds applied")

        # Mode 2: tune thresholds on this site only (oracle)
        print(f"\n  [{site}] per-site threshold tune:")
        site_thresholds = tune_thresholds_on_probs(site_probs, site_gt)
        m_local = evaluate_with_thresholds(
            site_probs, site_gt, site_thresholds,
        )
        print_site_metrics(m_local, site_thresholds, site,
                           "per-site tuned (ORACLE)")

        per_site_results[site] = {
            "global thr": (m_global, global_thresholds),
            "site thr (oracle)": (m_local, site_thresholds),
        }

    # ------------------------------------------------------------------
    # Compact summary
    # ------------------------------------------------------------------
    print_summary_table(per_site_results, aggregated_metrics)

    # ------------------------------------------------------------------
    # CSV output (per site, using global thresholds = realistic pipeline)
    # ------------------------------------------------------------------
    if args.output_csv_dir is not None:
        args.output_csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting per-site predictions CSVs to "
              f"{args.output_csv_dir} ...")
        for site in cfg.VAL_DATASETS:
            site_probs = filter_probs_by_site(hybrid_probs, site)
            if not site_probs:
                continue

            # Realistic: global thresholds
            preds_global = postprocess_predictions(site_probs,
                                                   global_thresholds)
            out_path = args.output_csv_dir / f"{site}__global_thr.csv"
            save_predictions_csv(preds_global, out_path)

            # Optional: oracle thresholds
            if args.per_site_csv_with_oracle and site in per_site_results:
                site_thr = per_site_results[site]["site thr (oracle)"][1]
                preds_local = postprocess_predictions(site_probs, site_thr)
                out_path = args.output_csv_dir / f"{site}__oracle_thr.csv"
                save_predictions_csv(preds_local, out_path)

    print("\nDone.")

    # ------------------------------------------------------------------
    # Wandb finalize. Stamp per-site, per-class F1 on summary so the
    # prof can sort/filter/group runs by any individual cell of the
    # table without opening logs. Also captures the gap between
    # realistic (global) thresholds and oracle (per-site) thresholds —
    # the "how much would per-site threshold tuning buy us?" question.
    # ------------------------------------------------------------------
    if run is not None:
        summary: dict = {}

        # Aggregate (pooled) — the headline number for the submission.
        f_bm = aggregated_metrics.get("bmabz", {}).get("f1", 0.0)
        f_d  = aggregated_metrics.get("d",     {}).get("f1", 0.0)
        f_bp = aggregated_metrics.get("bp",    {}).get("f1", 0.0)
        agg_micro = aggregated_metrics.get("overall", {}).get("f1", 0.0)
        agg_macro = float(np.mean([f_bm, f_d, f_bp]))

        summary["pooled_f1_bmabz"]   = float(f_bm)
        summary["pooled_f1_d"]       = float(f_d)
        summary["pooled_f1_bp"]      = float(f_bp)
        summary["pooled_micro_f1"]   = float(agg_micro)
        summary["pooled_macro_f1"]   = float(agg_macro)
        summary["global_thresholds"] = list(map(float, global_thresholds))

        # Per-site fields. One pair (global / oracle) per (site, class)
        # plus per-site macro F1 under each threshold mode.
        for site, blocks in per_site_results.items():
            for label_short, (metrics, thr) in (
                ("global", blocks["global thr"]),
                ("oracle", blocks["site thr (oracle)"]),
            ):
                for cls in cfg.CALL_TYPES_3:
                    summary[f"{site}/f1_{cls}_{label_short}"] = float(
                        metrics.get(cls, {}).get("f1", 0.0))
                summary[f"{site}/micro_f1_{label_short}"] = float(
                    metrics.get("overall", {}).get("f1", 0.0))
                summary[f"{site}/macro_f1_{label_short}"] = float(
                    np.mean([metrics.get(c, {}).get("f1", 0.0)
                             for c in cfg.CALL_TYPES_3]))
                summary[f"{site}/thresholds_{label_short}"] = (
                    list(map(float, thr)))

            # Gap (oracle − global) macro F1: "how much per-site tuning
            # would buy us if we could pick site-specific thresholds
            # without leaking labels." Negative or near-zero means
            # global thresholds already do well on this site.
            macro_global = summary[f"{site}/macro_f1_global"]
            macro_oracle = summary[f"{site}/macro_f1_oracle"]
            summary[f"{site}/oracle_gap_macro"] = float(
                macro_oracle - macro_global)

        # Worst-site identifier (global thr) — concrete answer to "where
        # is the ensemble weakest on real-pipeline thresholds?"
        site_micro_global = {
            site: summary[f"{site}/micro_f1_global"]
            for site in per_site_results.keys()
        }
        if site_micro_global:
            worst_site = min(site_micro_global, key=site_micro_global.get)
            summary["worst_site_global"] = worst_site
            summary["worst_site_micro_f1"] = site_micro_global[worst_site]

        # Verdict — short, sortable, captures the headline numbers.
        per_site_micro = ", ".join(
            f"{s}={summary[f'{s}/micro_f1_global']:.3f}"
            for s in per_site_results.keys()
        )
        if n_models == 1:
            verdict = (
                f"Single model {model_short_names[0]}: pooled micro F1 "
                f"{agg_micro:.3f}, macro {agg_macro:.3f}. Per-site "
                f"micro F1 under global thresholds: {per_site_micro}."
            )
        else:
            verdict = (
                f"Hybrid {n_models}-model ensemble (weights {weights}, "
                f"D from #{args.d_from + 1}={model_short_names[args.d_from]}"
                f"): pooled micro F1 {agg_micro:.3f}, macro "
                f"{agg_macro:.3f}. Per-site micro F1 under global "
                f"thresholds: {per_site_micro}."
            )

        # Upload per-site CSVs as an artifact, if they were written.
        if args.output_csv_dir is not None and args.output_csv_dir.exists():
            csv_files = sorted(args.output_csv_dir.glob("*.csv"))
            if csv_files:
                import wandb
                csv_art = wandb.Artifact(
                    f"per-site-predictions-{run.name}",
                    type="predictions",
                    metadata={
                        "n_models":      n_models,
                        "d_from":        args.d_from,
                        "weights":       weights,
                        "global_thresholds": list(map(float, global_thresholds)),
                        "per_site_csv_with_oracle": args.per_site_csv_with_oracle,
                    },
                )
                for f in csv_files:
                    csv_art.add_file(str(f))
                run.log_artifact(csv_art,
                                 aliases=["final_eval", "per_site"])

        wbu.finalize_eval_phase(
            summary,
            verdict=verdict,
        )


if __name__ == "__main__":
    main()
