"""
ensemble_predict.py
===================

Load N trained checkpoints, run inference on the validation set, average
their per-frame probabilities, tune per-class thresholds on the averaged
predictions, and report event-level F1 (overall + per-class).

The script auto-detects each checkpoint's model type from its saved keys:
  - checkpoints saved by ``train.py`` use the ``WhaleVAD`` architecture
  - checkpoints saved by ``train_bpn.py`` carry a ``bpn_cfg`` dict and
    use the ``WhaleVADBPN`` architecture (BPN gate enabled or disabled
    according to that config)

Usage
-----
    python ensemble_predict.py \\
        --checkpoints runs/whalevad_<ts>/best_model.pt \\
                      runs/phase5_<ts>/best_model.pt \\
                      runs/whalevad_<ts2>/best_model.pt

Optional ``--weights`` lets you bias the average toward stronger models.
By default the script also reports each individual model's F1 for
reference, so you can see whether the ensemble actually helps.

Notes
-----
- All checkpoints must share the same class configuration (3-class or
  4-class). Mixing is not supported in this version.
- The script assumes single-GPU inference. For multi-GPU runs, pass
  ``CUDA_VISIBLE_DEVICES=0`` to pin to one device.
- The script consumes ~1.3 GB RAM per checkpoint (probabilities are kept
  in memory for averaging). For 3-4 checkpoints this is fine.
"""

from __future__ import annotations

import argparse
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
# Model construction
# ----------------------------------------------------------------------

def detect_model_type(ckpt: dict[str, Any]) -> str:
    """Return 'bpn' if checkpoint has a saved BPN config, else 'baseline'."""
    return "bpn" if "bpn_cfg" in ckpt else "baseline"


def build_model_for_ckpt(
    ckpt: dict[str, Any], device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """
    Construct the right model class for a checkpoint and prepare it for
    weight loading. Returns (model, model_type) where model_type is
    'bpn' or 'baseline'.

    The lazy projection layer is initialised by a dummy forward pass
    BEFORE load_state_dict so the saved feat_proj weights map cleanly.
    """
    model_type = detect_model_type(ckpt)
    n_classes = cfg.n_classes()

    if model_type == "bpn":
        bpn_cfg_dict = dict(ckpt["bpn_cfg"])
        # ``dilations`` round-trips as list via to_dict; BPNConfig accepts
        # both list and tuple, no conversion strictly needed.
        bpn_cfg = BPNConfig(**bpn_cfg_dict)
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
    """
    Run model on the full val loader. Returns a dict keyed by
    ``(dataset, filename, start_sample)`` with per-frame probability
    arrays of shape ``(T_frames, n_classes)``.
    """
    model.eval()
    out_probs: dict[tuple, np.ndarray] = {}
    hop = spec_extractor.hop_length

    for audio, _, _, metas in tqdm(val_loader, desc="  inference", leave=False):
        audio = audio.to(device, non_blocking=True)
        spec = spec_extractor(audio)
        out = model(spec)

        if model_type == "bpn":
            # WhaleVADBPN returns dict with already-gated probs in [0, 1]
            probs = out["probs"]
        else:
            # WhaleVAD returns raw logits — apply sigmoid here
            probs = torch.sigmoid(out)

        probs_np = probs.detach().float().cpu().numpy()

        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs_np[j].shape[0])
            out_probs[key] = probs_np[j, :n_frames, :]

    return out_probs


# ----------------------------------------------------------------------
# Probability averaging
# ----------------------------------------------------------------------

def average_prob_dicts(
    prob_dicts: list[dict[tuple, np.ndarray]],
    weights: list[float] | None = None,
) -> dict[tuple, np.ndarray]:
    """
    Weighted average of per-segment probability arrays across N models.

    Defensive about minor shape mismatches: if two models disagree on
    the number of frames for a segment by 1-2 frames (common when
    different model architectures produce slightly different time
    dimensions), the result is truncated to the shortest. Same for
    class counts, in case of mixed 3-class and collapsed 4-class outputs.
    """
    if not prob_dicts:
        return {}
    if weights is None:
        weights = [1.0 / len(prob_dicts)] * len(prob_dicts)
    assert len(weights) == len(prob_dicts), \
        "len(weights) must equal len(prob_dicts)"

    # Use the first dict's keys as the canonical set, but tolerate any
    # missing entries (extremely unusual; would indicate a dataloader bug).
    common_keys = set(prob_dicts[0].keys())
    for pd in prob_dicts[1:]:
        common_keys &= set(pd.keys())

    if len(common_keys) < len(prob_dicts[0]):
        n_missing = len(prob_dicts[0]) - len(common_keys)
        print(f"  WARNING: {n_missing} keys missing from at least one model; "
              f"dropping them from the ensemble.")

    out: dict[tuple, np.ndarray] = {}
    for key in common_keys:
        arrs = [pd[key] for pd in prob_dicts]
        min_T = min(a.shape[0] for a in arrs)
        min_C = min(a.shape[1] for a in arrs)
        # Allocate output array, accumulate weighted sum.
        averaged = np.zeros((min_T, min_C), dtype=np.float32)
        for w, a in zip(weights, arrs):
            averaged += w * a[:min_T, :min_C].astype(np.float32)
        out[key] = averaged
    return out


# ----------------------------------------------------------------------
# Threshold tuning + scoring (same coordinate-descent sweep as train.py)
# ----------------------------------------------------------------------

def tune_thresholds_on_probs(
    all_probs: dict[tuple, np.ndarray],
    gt_events: list[Detection],
) -> np.ndarray:
    """Per-class coordinate-descent threshold sweep (same grid as train.py)."""
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    grids = [
        np.arange(0.20, 0.85, 0.05),                                     # bmabz
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),                    # d
        np.concatenate([np.arange(0.05, 0.5, 0.05),
                        np.arange(0.5, 0.85, 0.10)]),                    # bp
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
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ensemble multiple checkpoints by averaging their "
                    "per-frame validation probabilities.",
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to .pt checkpoint files. The script auto-"
                        "detects each as either a baseline (WhaleVAD) "
                        "or a BPN run (WhaleVADBPN).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint weights for the average. "
                        "Length must match --checkpoints. Default: equal.")
    p.add_argument("--per-model-eval", action="store_true",
                   help="Also score each checkpoint individually for "
                        "reference. Costs ~10 seconds per model.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.weights is not None:
        assert len(args.weights) == len(args.checkpoints), \
            "len(weights) must equal len(checkpoints)"
        # Normalize to sum to 1
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
        print(f"Per-checkpoint weights (normalized): {weights}")
    else:
        weights = None
        print(f"Per-checkpoint weights: equal (1/{len(args.checkpoints)} each)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build val loader once — shared across all checkpoint inferences.
    print("\nLoading validation data...")
    _, _, val_loader = build_dataloaders()
    val_ann = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    spec_extractor = SpectrogramExtractor().to(device)

    # Build ground truth event list (always 3-class; same as train.py).
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
    print(f"Validation: {len(gt_events)} ground-truth events")

    # ------------------------------------------------------------------
    # Run inference for each checkpoint
    # ------------------------------------------------------------------
    all_prob_dicts: list[dict[tuple, np.ndarray]] = []
    individual_metrics: list[dict] = []

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_type = build_model_for_ckpt(ckpt, device)
        print(f"  type: {model_type}")
        if model_type == "bpn":
            bpn_meta = ckpt.get("bpn_cfg", {})
            print(f"  bpn config: enabled={bpn_meta.get('enabled')}, "
                  f"taps={bpn_meta.get('n_taps')}, rois={bpn_meta.get('n_rois')}")

        # Initialise lazy feat_proj BEFORE loading state_dict so its saved
        # weights have the correct in_features.
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            _ = model(spec_extractor(dummy))

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False,
        )
        if unexpected:
            print(f"  WARNING: unexpected state_dict keys: "
                  f"{len(unexpected)} (showing 3): {unexpected[:3]}")
        if missing:
            # A few "missing" keys are OK if the BPN module wasn't saved
            # (e.g. ckpt is from --no-bpn but we instantiated with bpn=None
            # — same result).
            non_bpn_missing = [k for k in missing if "bpn" not in k]
            if non_bpn_missing:
                print(f"  WARNING: missing non-BPN keys: "
                      f"{len(non_bpn_missing)} (showing 3): "
                      f"{non_bpn_missing[:3]}")

        probs = predict_probabilities(
            model, model_type, spec_extractor, val_loader, device,
        )
        # Collapse 4-class → 3-class if needed (no-op for 3-class checkpoints).
        probs = collapse_probs_to_3class(probs)
        print(f"  collected probabilities for {len(probs)} segments")
        all_prob_dicts.append(probs)

        # Optional per-model F1 for the reference table.
        if args.per_model_eval:
            print(f"  individual threshold tune:")
            ind_thr = tune_thresholds_on_probs(probs, gt_events)
            ind_metrics = evaluate_with_thresholds(probs, gt_events, ind_thr)
            individual_metrics.append({
                "path": ckpt_path,
                "thresholds": ind_thr,
                "metrics": ind_metrics,
                "f1": ind_metrics.get("overall", {}).get("f1", 0.0),
            })

        # Free GPU memory before next checkpoint loads.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("ENSEMBLE")
    print("=" * 64)
    avg_probs = average_prob_dicts(all_prob_dicts, weights=weights)
    print(f"Averaged probs across {len(all_prob_dicts)} models, "
          f"{len(avg_probs)} segments")

    print("\nThreshold tuning on ensemble probabilities:")
    thresholds = tune_thresholds_on_probs(avg_probs, gt_events)

    final_metrics = evaluate_with_thresholds(avg_probs, gt_events, thresholds)
    print_metrics(final_metrics, thresholds, "ENSEMBLE RESULT")
    print(f"    Tuned thresholds: {[f'{t:.2f}' for t in thresholds]}")

    # ------------------------------------------------------------------
    # Reference table
    # ------------------------------------------------------------------
    if args.per_model_eval:
        print("\n" + "=" * 64)
        print("INDIVIDUAL MODEL F1 (FOR REFERENCE)")
        print("=" * 64)
        print(f"  {'#':<3}  {'F1':>6}  {'thresholds':<22}  path")
        for i, im in enumerate(individual_metrics):
            t_str = " ".join(f"{t:.2f}" for t in im["thresholds"])
            short = Path(im["path"]).parent.name
            print(f"  {i+1:<3}  {im['f1']:>6.3f}  [{t_str}]  {short}")
        print(f"  {'ENS':<3}  {final_metrics.get('overall', {}).get('f1', 0):>6.3f}"
              f"  ensemble")

    print("\nDone.")


if __name__ == "__main__":
    main()
