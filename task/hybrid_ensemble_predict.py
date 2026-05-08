"""
hybrid_ensemble_predict.py
==========================

Per-class hybrid ensemble for inference and submission.

Strategy
--------
The standard 4-model ensemble dilutes the BPN model's strong D-class
performance because three of four models have weak D. This script keeps
the ensemble for BMABZ and BP (where averaging genuinely helps) but
takes the D-class probabilities from a single nominated model — the
one with the strongest individual D-F1.

For your val results that means:
  - BMABZ : weighted average of all 4 models  (weights 1, 1, 1, 2)
  - D     : taken only from model index 3 (BPN R=8, D-F1=0.238)
  - BP    : weighted average of all 4 models  (weights 1, 1, 1, 2)

Two modes
---------
1. **Verification on val** (default): tunes thresholds on the hybrid
   probabilities, scores against val GT, prints per-class F1, and also
   writes a CSV. Use this to confirm the hybrid strategy beats the
   standard ensemble before applying to test.

2. **Test inference** (``--test-datasets X Y``): runs on the named test
   datasets, uses the frozen ``--thresholds`` you pass in, no scoring,
   writes the CSV.

Usage
-----
Verify on val (uses cfg.VAL_DATASETS, tunes thresholds, scores + CSV):

    python hybrid_ensemble_predict.py \\
        --checkpoints \\
            runs/whalevad_<ts>/best_model.pt \\
            runs/phase5_<ts>/best_model.pt \\
            runs/whalevad_<ts2>/best_model.pt \\
            runs/phase5_<ts2>/best_model.pt \\
        --weights 1 1 1 2 \\
        --d-from 3 \\
        --output-csv val_hybrid_predictions.csv

Apply to test (frozen thresholds from val, no scoring):

    python hybrid_ensemble_predict.py \\
        --checkpoints ... (same four) \\
        --weights 1 1 1 2 \\
        --d-from 3 \\
        --thresholds 0.25 0.35 0.30 \\
        --test-datasets kerguelen2020 ddu2021 \\
        --output-csv test_predictions.csv

Notes
-----
- ``--d-from`` is 0-indexed and refers to position in ``--checkpoints``.
- ``--thresholds`` order is BMABZ, D, BP (matches cfg.CALL_TYPES_3).
- In test mode, the script uses ``cfg.VAL_DATASETS`` infrastructure but
  monkey-patches it to point at the test dataset list. This avoids
  needing changes to ``dataset.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from dataset import build_dataloaders, load_annotations, get_file_manifest
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    collapse_probs_to_3class,
)
from model import WhaleVAD
from model_bpn import WhaleVADBPN, BPNConfig


# ----------------------------------------------------------------------
# Model construction (same logic as ensemble_predict.py)
# ----------------------------------------------------------------------

def detect_model_type(ckpt: dict[str, Any]) -> str:
    return "bpn" if "bpn_cfg" in ckpt else "baseline"


def build_model_for_ckpt(
    ckpt: dict[str, Any], device: torch.device,
) -> tuple[torch.nn.Module, str]:
    model_type = detect_model_type(ckpt)
    n_classes = cfg.n_classes()
    if model_type == "bpn":
        bpn_cfg = BPNConfig(**dict(ckpt["bpn_cfg"]))
        model = WhaleVADBPN(num_classes=n_classes, bpn_cfg=bpn_cfg).to(device)
    else:
        model = WhaleVAD(num_classes=n_classes).to(device)
    return model, model_type


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module, model_type: str,
    spec_extractor: SpectrogramExtractor,
    val_loader, device: torch.device,
) -> dict[tuple, np.ndarray]:
    model.eval()
    out_probs: dict[tuple, np.ndarray] = {}
    hop = spec_extractor.hop_length

    for audio, _, _, metas in tqdm(val_loader, desc="  inference", leave=False):
        audio = audio.to(device, non_blocking=True)
        spec = spec_extractor(audio)
        out = model(spec)

        if model_type == "bpn":
            probs = out["probs"]
        else:
            probs = torch.sigmoid(out)

        probs_np = probs.detach().float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs_np[j].shape[0])
            out_probs[key] = probs_np[j, :n_frames, :]
    return out_probs


# ----------------------------------------------------------------------
# Per-class hybrid combination (the key piece)
# ----------------------------------------------------------------------

def hybrid_combine(
    prob_dicts: list[dict[tuple, np.ndarray]],
    d_from_idx: int,
    weights: list[float],
) -> dict[tuple, np.ndarray]:
    """
    Combine per-segment probabilities with a per-class strategy:

      - BMABZ (col 0): weighted average across all models
      - D     (col 1): taken ONLY from prob_dicts[d_from_idx]
      - BP    (col 2): weighted average across all models

    Defensive about minor shape mismatches (truncates to shortest T and
    fewest classes across models).
    """
    if not prob_dicts:
        return {}
    assert 0 <= d_from_idx < len(prob_dicts), \
        f"--d-from index {d_from_idx} out of range for {len(prob_dicts)} models"
    assert len(weights) == len(prob_dicts), \
        "len(weights) must equal len(prob_dicts)"

    # Normalize weights for the BMABZ/BP averaging
    w_sum = sum(weights)
    norm_weights = [w / w_sum for w in weights]

    common_keys = set(prob_dicts[0].keys())
    for pd in prob_dicts[1:]:
        common_keys &= set(pd.keys())

    if len(common_keys) < len(prob_dicts[0]):
        n_missing = len(prob_dicts[0]) - len(common_keys)
        print(f"  WARNING: {n_missing} keys missing from at least one model; "
              f"dropping them.")

    out: dict[tuple, np.ndarray] = {}
    for key in common_keys:
        arrs = [pd[key] for pd in prob_dicts]
        min_T = min(a.shape[0] for a in arrs)
        min_C = min(a.shape[1] for a in arrs)

        # Step 1: weighted average of all classes (we'll overwrite D in step 2)
        combined = np.zeros((min_T, min_C), dtype=np.float32)
        for w, a in zip(norm_weights, arrs):
            combined += w * a[:min_T, :min_C].astype(np.float32)

        # Step 2: replace D-class column with the single nominated model
        if min_C >= 2:  # D is index 1
            combined[:, 1] = arrs[d_from_idx][:min_T, 1].astype(np.float32)

        out[key] = combined
    return out


# ----------------------------------------------------------------------
# Threshold tuning + scoring (only used when GT is available)
# ----------------------------------------------------------------------

def tune_thresholds_on_probs(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
) -> np.ndarray:
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    grids = [
        np.arange(0.20, 0.85, 0.05),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),
    ]
    for c, name in enumerate(cfg.CALL_TYPES_3):
        best_f1, best_t = -1.0, thresholds[c]
        for t in grids[c]:
            trial = thresholds.copy()
            trial[c] = float(t)
            preds = postprocess_predictions(all_probs, trial)
            metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
            f1 = metrics.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
        print(f"    {name:<6}  best t={best_t:.2f}  f1={best_f1:.3f}")
    return thresholds


def evaluate_with_thresholds(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
    thresholds: np.ndarray,
) -> dict[str, dict[str, float]]:
    pred_events = postprocess_predictions(all_probs, thresholds)
    return compute_metrics(pred_events, gt_events, iou_threshold=0.3)


def print_metrics(
    metrics: dict[str, dict[str, float]],
    thresholds: np.ndarray,
    title: str,
) -> None:
    print(f"\n  {title}:")
    for c, name in enumerate(cfg.CALL_TYPES_3):
        m = metrics.get(name, {})
        print(f"    {name.upper():<6} t={thresholds[c]:.2f}  "
              f"TP={m.get('tp', 0):5d} FP={m.get('fp', 0):5d} "
              f"FN={m.get('fn', 0):5d}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    print(f"    OVERALL F1={metrics.get('overall', {}).get('f1', 0):.3f}")


# ----------------------------------------------------------------------
# CSV output (DCASE event-level format)
# ----------------------------------------------------------------------

def save_predictions_csv(
    pred_events: list[Detection],
    output_path: Path,
) -> None:
    """
    Write predictions to CSV with columns:
        dataset, filename, onset, offset, event_label, confidence
    """
    pred_events = sorted(
        pred_events,
        key=lambda d: (d.dataset, d.filename, d.start_s, d.label),
    )
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "filename", "onset", "offset",
                         "event_label", "confidence"])
        for d in pred_events:
            writer.writerow([
                d.dataset, d.filename,
                f"{d.start_s:.3f}", f"{d.end_s:.3f}",
                d.label, f"{getattr(d, 'confidence', 1.0):.4f}",
            ])
    print(f"  wrote {len(pred_events)} predictions to {output_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to .pt checkpoint files.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for BMABZ+BP averaging. "
                        "Length must match --checkpoints. Default: equal.")
    p.add_argument("--d-from", type=int, required=True,
                   help="0-indexed position in --checkpoints of the model "
                        "to use for D-class predictions.")
    p.add_argument("--thresholds", nargs=3, type=float, default=None,
                   metavar=("BMABZ", "D", "BP"),
                   help="Frozen thresholds [BMABZ D BP]. If not provided, "
                        "thresholds are tuned on the (val) GT.")
    p.add_argument("--test-datasets", nargs="+", default=None,
                   help="Run inference on these dataset names instead of "
                        "cfg.VAL_DATASETS. In this mode no scoring is done; "
                        "you must supply --thresholds. Use this for the "
                        "actual test submission.")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Path to write predictions CSV. If omitted, no CSV "
                        "is written.")
    p.add_argument("--per-model-eval", action="store_true",
                   help="Also score each model individually for reference. "
                        "Ignored in test mode (no GT to score against).")
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

    # ------------------------------------------------------------------
    # Mode: test or val?
    # ------------------------------------------------------------------
    test_mode = args.test_datasets is not None
    if test_mode:
        if args.thresholds is None:
            raise SystemExit(
                "ERROR: --test-datasets requires --thresholds (no GT available "
                "to tune on in test mode). Pass thresholds tuned on val."
            )
        # Monkey-patch cfg.VAL_DATASETS so build_dataloaders/load_annotations
        # use the test datasets. This is a deliberate hack to avoid touching
        # dataset.py.
        print(f"TEST MODE: pointing val_datasets at {args.test_datasets}")
        cfg.VAL_DATASETS = list(args.test_datasets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data (val_loader builds whatever cfg.VAL_DATASETS now points to)
    # ------------------------------------------------------------------
    print("\nLoading data...")
    _, _, val_loader = build_dataloaders()

    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    # In test mode we skip GT loading entirely
    gt_events: list[Detection] = []
    if not test_mode:
        val_ann = load_annotations(cfg.VAL_DATASETS)
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
        print(f"Validation: {len(gt_events)} ground-truth events")

    spec_extractor = SpectrogramExtractor().to(device)

    # ------------------------------------------------------------------
    # Run inference for each checkpoint
    # ------------------------------------------------------------------
    all_prob_dicts: list[dict[tuple, np.ndarray]] = []
    individual_metrics: list[dict] = []

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{n_models}] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_type = build_model_for_ckpt(ckpt, device)
        is_d_source = (i == args.d_from)
        marker = "  ← D source" if is_d_source else ""
        print(f"  type: {model_type}{marker}")

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

        if args.per_model_eval and not test_mode:
            print(f"  individual threshold tune:")
            ind_thr = tune_thresholds_on_probs(probs, gt_events)
            ind_metrics = evaluate_with_thresholds(probs, gt_events, ind_thr)
            individual_metrics.append({
                "path": ckpt_path,
                "thresholds": ind_thr,
                "f1": ind_metrics.get("overall", {}).get("f1", 0.0),
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Hybrid combination
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"HYBRID COMBINATION")
    print(f"  BMABZ + BP : weighted avg (weights {weights})")
    print(f"  D          : from model #{args.d_from + 1} only "
          f"({Path(args.checkpoints[args.d_from]).parent.name})")
    print("=" * 64)
    hybrid_probs = hybrid_combine(
        all_prob_dicts, d_from_idx=args.d_from, weights=weights,
    )
    print(f"  combined probs for {len(hybrid_probs)} segments")

    # ------------------------------------------------------------------
    # Threshold tuning or use frozen
    # ------------------------------------------------------------------
    if args.thresholds is not None:
        thresholds = np.array(args.thresholds, dtype=np.float64)
        print(f"\nUsing frozen thresholds: {list(thresholds)}")
    else:
        print("\nThreshold tuning on hybrid probabilities:")
        thresholds = tune_thresholds_on_probs(hybrid_probs, gt_events)

    # ------------------------------------------------------------------
    # Score (if GT available) and print
    # ------------------------------------------------------------------
    if not test_mode:
        final_metrics = evaluate_with_thresholds(
            hybrid_probs, gt_events, thresholds,
        )
        print_metrics(final_metrics, thresholds, "HYBRID RESULT")

        if args.per_model_eval:
            print("\n" + "=" * 64)
            print("INDIVIDUAL MODEL F1 (FOR REFERENCE)")
            print("=" * 64)
            print(f"  {'#':<3}  {'F1':>6}  path")
            for i, im in enumerate(individual_metrics):
                short = Path(im["path"]).parent.name
                marker = " ← D source" if i == args.d_from else ""
                print(f"  {i+1:<3}  {im['f1']:>6.3f}  {short}{marker}")
            print(f"  {'HYB':<3}  "
                  f"{final_metrics.get('overall', {}).get('f1', 0):>6.3f}  "
                  f"hybrid")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    if args.output_csv is not None:
        print(f"\nWriting predictions CSV...")
        pred_events = postprocess_predictions(hybrid_probs, thresholds)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_predictions_csv(pred_events, args.output_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
