"""
Two-Stage Inference: Stage-1 → Verifier → Event-Level F1
========================================================

Plugs the trained stage-1 (Whale-VAD CNN-BiLSTM) and stage-2 (verifier)
models together into the operational two-stage detector and reports
event-level F1 with 1D-IoU matching — the same metric the DCASE 2025
task 2 leaderboard uses, the same metric your stage-1 baseline of 0.469
is reported in.

Pipeline
--------
    1.  Stage 1 — already done by ``extract_candidates.py``. The
        candidates parquet has, per candidate event:
            (dataset, filename, start_s, end_s, predicted_class,
             stage1_score, GT-matching label, file path, file dur)
        i.e. detections post-merged at low threshold 0.3.

    2.  Stage 2 — for each candidate, this script runs the verifier on
        the centred audio crop and produces ``verifier_score`` ∈ [0, 1].

    3.  Combine — ``combined_score = stage1_score × verifier_score``.
        Multiplicative because both factors are post-sigmoid
        probabilities; the geometry-mean variant is equivalent up to
        monotone rescaling, which doesn't affect threshold sweeps.

    4.  Threshold tune — per class, sweep a confidence threshold on the
        detection list and pick the value that maximises event-level
        F1 against ground-truth annotations (1D-IoU ≥ 0.3 greedy
        matching, same as ``postprocess.compute_metrics``).

    5.  Report — side-by-side per-class numbers for stage-1-alone vs
        two-stage, plus macro and micro F1, plus the per-class best
        thresholds.

Honest evaluation
-----------------
By default this script restricts the evaluation set to the 20% of val
recordings the verifier did NOT see during training (re-derived from
the verifier checkpoint's saved seed and ``internal_val_frac``). Pass
``--full_val`` to score on all 587 val files, but be aware 80% of
those were in the verifier's training split.

A note on the stage-1 baseline number
-------------------------------------
The "stage-1" column reported below uses *event-level* threshold
tuning on the same threshold-0.3 candidate set — not the per-frame
threshold tuning that produced your published F1 = 0.469. The two are
not identical (the per-frame pipeline does smoothing → threshold →
merge differently). The "stage-1" number here is the internal point
of comparison: it shares the candidate set with the two-stage
pipeline, so the *delta* between them isolates the verifier's
contribution. Your 0.469 number stays the headline baseline.

Usage
-----
::

    python inference_two_stage.py \\
        --verifier_checkpoint runs_verifier/v3_val_sup_seed42/best.pt \\
        --candidates candidates_val.parquet \\
        --output_dir runs_verifier/v3_val_sup_seed42/eval_two_stage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import get_file_manifest, load_annotations
from dataset_verifier import (
    CandidateRecord, VerifierDataset, load_candidates, verifier_collate_fn,
)
from model_verifier import WhaleVerifier, count_parameters
from postprocess import Detection, compute_metrics
from spectrogram import SpectrogramExtractor
# Reuse the same recording-split function the trainer uses, so the
# held-out subset here is bit-identical to what the verifier never saw.
from train_verifier import split_by_recording


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier_checkpoint", type=str, required=True)
    p.add_argument("--candidates", type=str, required=True,
                   help="candidates_val.parquet (or whichever split you "
                        "want to evaluate on).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument(
        "--full_val", action="store_true",
        help="Evaluate on the full val candidates instead of just the "
             "20%% held out from verifier training. Use with caution: "
             "80%% of val recordings were the verifier's training set.")
    p.add_argument(
        "--iou_threshold", type=float, default=0.3,
        help="1D-IoU matching threshold. 0.3 is the DCASE default.")
    return p.parse_args()


# ======================================================================
# Verifier inference on candidates
# ======================================================================

@torch.no_grad()
def run_verifier(
    model: WhaleVerifier,
    spec_extractor: SpectrogramExtractor,
    records: list[CandidateRecord],
    device: torch.device,
    crop_s: float,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """
    Score every candidate. Returns ``verifier_score[i]`` aligned to
    ``records[i]``.
    """
    ds = VerifierDataset(records, crop_s=crop_s, train=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=verifier_collate_fn,
        pin_memory=True,
    )
    scores = []
    for audio, class_idx, _, metas in tqdm(loader, desc="Verifier"):
        audio = audio.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)
        aux = torch.tensor(
            [m["stage1_score"] for m in metas],
            device=device, dtype=torch.float32,
        ).unsqueeze(-1)
        spec = spec_extractor(audio)
        probs = torch.sigmoid(model(spec, class_idx, aux)).cpu().numpy()
        scores.append(probs)
    return np.concatenate(scores)


# ======================================================================
# Detection list builders
# ======================================================================

def records_to_detections(
    records: list[CandidateRecord], confidences: np.ndarray,
) -> list[Detection]:
    """
    Pair candidate records with a confidence score (typically stage1
    alone, or stage1 × verifier) and emit ``Detection`` objects ready
    for ``compute_metrics``.
    """
    out = []
    cls_names = cfg.CALL_TYPES_3
    for r, c in zip(records, confidences):
        out.append(Detection(
            dataset=r.dataset,
            filename=r.filename,
            label=cls_names[r.class_idx],
            start_s=r.start_s,
            end_s=r.end_s,
            confidence=float(c),
        ))
    return out


def build_gt_events(
    annotations: pd.DataFrame, manifest: pd.DataFrame,
    keep_files: set[tuple[str, str]] | None = None,
) -> list[Detection]:
    """
    Construct ground-truth ``Detection`` objects from val annotations.
    ``keep_files`` optionally restricts to a specific set.
    """
    file_starts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }
    out = []
    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        if keep_files is not None and key not in keep_files:
            continue
        fsd = file_starts.get(key)
        if fsd is None or pd.isna(fsd):
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        out.append(Detection(
            dataset=row["dataset"],
            filename=row["filename"],
            label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return out


# ======================================================================
# Per-class event-level threshold sweep
# ======================================================================

def sweep_per_class_thresholds(
    detections: list[Detection],
    gt_events: list[Detection],
    iou_threshold: float = 0.3,
    grid: np.ndarray | None = None,
) -> dict:
    """
    For each class, find the confidence threshold maximising event-level
    F1, holding the other classes' detections fixed (which is harmless
    since ``compute_metrics`` is per-class independent).

    Then report the joint metrics — each class filtered at its own best
    threshold — including the micro-averaged "overall" number.

    Returns
    -------
    dict
        ``{
          "<class>": {"best_threshold", "best_f1", "tp", "fp", "fn",
                      "precision", "recall"},
          ...
          "joint": {"per_class": {...}, "overall": {f1, precision, recall}},
        }``
    """
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)

    per_class: dict[str, dict] = {}
    best_thresholds: dict[str, float] = {}

    # Per-class sweep — independent because ``compute_metrics`` does not
    # mix classes when computing class F1.
    for cls in cfg.CALL_TYPES_3:
        cls_dets = [d for d in detections if d.label == cls]
        cls_gts = [g for g in gt_events if g.label == cls]
        if not cls_gts:
            per_class[cls] = {
                "best_threshold": 0.5, "best_f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0,
            }
            best_thresholds[cls] = 0.5
            continue

        best_f1 = -1.0
        best_thr = 0.5
        best_pack = None
        for thr in grid:
            kept = [d for d in cls_dets if d.confidence >= thr]
            m = compute_metrics(kept + cls_gts_dummy_other_classes(),
                                cls_gts, iou_threshold=iou_threshold)
            cm = m.get(cls, {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                             "tp": 0, "fp": 0, "fn": 0})
            if cm["f1"] > best_f1:
                best_f1 = cm["f1"]
                best_thr = float(thr)
                best_pack = cm
        per_class[cls] = {
            "best_threshold": best_thr,
            "best_f1": best_f1,
            "tp": best_pack["tp"], "fp": best_pack["fp"],
            "fn": best_pack["fn"],
            "precision": best_pack["precision"],
            "recall": best_pack["recall"],
        }
        best_thresholds[cls] = best_thr

    # Joint metric — apply each class's threshold and call
    # ``compute_metrics`` once on the merged list.
    merged = []
    for cls in cfg.CALL_TYPES_3:
        thr = best_thresholds[cls]
        merged.extend([d for d in detections
                       if d.label == cls and d.confidence >= thr])
    joint = compute_metrics(merged, gt_events, iou_threshold=iou_threshold)
    return {
        **per_class,
        "joint": joint,
    }


def cls_gts_dummy_other_classes() -> list[Detection]:
    """
    Trivial stand-in: ``compute_metrics`` skips classes it doesn't see,
    so we don't need anything here. Kept as a function for clarity at
    the call site.
    """
    return []


# ======================================================================
# Reporting
# ======================================================================

def fmt(x, digits=4):
    """Compact safe formatter."""
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "?"


def print_comparison(stage1_res: dict, combined_res: dict, header: str):
    """Side-by-side per-class table."""
    print(f"\n{header}")
    print(f"  {'class':<7} | "
          f"{'stage-1 F1':>11}  {'thr':>5}  {'P':>6}  {'R':>6}  | "
          f"{'two-stage F1':>13}  {'thr':>5}  {'P':>6}  {'R':>6}  | "
          f"{'Δ F1':>+8}")
    for cls in cfg.CALL_TYPES_3:
        s1 = stage1_res[cls]
        cb = combined_res[cls]
        delta = cb["best_f1"] - s1["best_f1"]
        print(f"  {cls:<7} | "
              f"{fmt(s1['best_f1']):>11}  "
              f"{fmt(s1['best_threshold'],3):>5}  "
              f"{fmt(s1['precision'],3):>6}  "
              f"{fmt(s1['recall'],3):>6}  | "
              f"{fmt(cb['best_f1']):>13}  "
              f"{fmt(cb['best_threshold'],3):>5}  "
              f"{fmt(cb['precision'],3):>6}  "
              f"{fmt(cb['recall'],3):>6}  | "
              f"{('+' if delta >= 0 else ''):>1}{fmt(delta):>7}")

    s1_micro = stage1_res["joint"]["overall"]
    cb_micro = combined_res["joint"]["overall"]
    print(f"  {'micro':<7} | "
          f"{fmt(s1_micro['f1']):>11}  "
          f"{'':>5}  "
          f"{fmt(s1_micro['precision'],3):>6}  "
          f"{fmt(s1_micro['recall'],3):>6}  | "
          f"{fmt(cb_micro['f1']):>13}  "
          f"{'':>5}  "
          f"{fmt(cb_micro['precision'],3):>6}  "
          f"{fmt(cb_micro['recall'],3):>6}  | "
          f"{('+' if cb_micro['f1'] - s1_micro['f1'] >= 0 else ''):>1}"
          f"{fmt(cb_micro['f1'] - s1_micro['f1']):>7}")

    # Macro = simple average of per-class F1
    macro_s1 = np.mean([stage1_res[c]["best_f1"] for c in cfg.CALL_TYPES_3])
    macro_cb = np.mean([combined_res[c]["best_f1"] for c in cfg.CALL_TYPES_3])
    print(f"  {'macro':<7} | "
          f"{fmt(macro_s1):>11}  "
          f"{'':>5}  {'':>6}  {'':>6}  | "
          f"{fmt(macro_cb):>13}  "
          f"{'':>5}  {'':>6}  {'':>6}  | "
          f"{('+' if macro_cb - macro_s1 >= 0 else ''):>1}"
          f"{fmt(macro_cb - macro_s1):>7}")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load verifier and recover its training config
    # ------------------------------------------------------------------
    print(f"\nLoading verifier: {args.verifier_checkpoint}")
    ckpt = torch.load(args.verifier_checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    crop_s = train_args.get("crop_s", 15.0)
    backbone_dropout = train_args.get("backbone_dropout", 0.5)
    head_dropout = train_args.get("head_dropout", 0.3)
    train_seed = train_args.get("seed", 42)
    train_val_frac = train_args.get("internal_val_frac", 0.2)
    print(f"  trained for {ckpt.get('epoch', '?')} epochs   "
          f"reported macro_F1 = {fmt(ckpt.get('macro_combined_f1'))}")
    print(f"  crop_s={crop_s}  seed={train_seed}  "
          f"internal_val_frac={train_val_frac}")

    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVerifier(
        n_classes=len(cfg.CALL_TYPES_3),
        backbone_dropout=backbone_dropout,
        head_dropout=head_dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total, _ = count_parameters(model)
    print(f"  loaded {total:,} params")

    # ------------------------------------------------------------------
    # Load candidates and (by default) restrict to held-out 20%
    # ------------------------------------------------------------------
    print(f"\nLoading candidates: {args.candidates}")
    all_records = load_candidates(args.candidates)
    print(f"  {len(all_records)} candidates total")

    if args.full_val:
        print("  --full_val: scoring on ALL candidates "
              "(80% were in the verifier's training set — caveat!)")
        records = all_records
        keep_files = None
    else:
        # Reproduce the verifier's 80/20 recording split exactly so we
        # only keep candidates from files the verifier never saw.
        train_recs, held_out_recs = split_by_recording(
            all_records, val_frac=train_val_frac, seed=train_seed,
        )
        held_out_files = {(r.dataset, r.filename) for r in held_out_recs}
        records = held_out_recs
        keep_files = held_out_files
        print(f"  --held-out only: kept {len(records)} candidates "
              f"from {len(held_out_files)} held-out files "
              f"({len(records) / max(len(all_records),1):.1%} of total)")

    # Per-class breakdown of the eval set, useful for sanity checks.
    for cn in cfg.CALL_TYPES_3:
        ci = cfg.CALL_TYPES_3.index(cn)
        sub = [r for r in records if r.class_idx == ci]
        n_tp = sum(1 for r in sub if r.label == 1)
        n_fp = sum(1 for r in sub if r.label == 0)
        print(f"    {cn}: candidates TP={n_tp}  FP={n_fp}")

    # ------------------------------------------------------------------
    # Verifier inference
    # ------------------------------------------------------------------
    print()
    verifier_scores = run_verifier(
        model, spec_extractor, records, device,
        crop_s=crop_s, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    stage1_scores = np.array([r.stage1_score for r in records],
                             dtype=np.float32)
    combined_scores = stage1_scores * verifier_scores

    # ------------------------------------------------------------------
    # Build detection lists and ground truth
    # ------------------------------------------------------------------
    stage1_dets = records_to_detections(records, stage1_scores)
    combined_dets = records_to_detections(records, combined_scores)

    print("\nLoading val annotations for ground truth…")
    manifest = get_file_manifest(cfg.VAL_DATASETS)
    annotations = load_annotations(cfg.VAL_DATASETS, manifest=manifest)
    gt_events = build_gt_events(annotations, manifest, keep_files=keep_files)
    print(f"  {len(gt_events)} ground-truth events in eval set")

    # ------------------------------------------------------------------
    # Threshold sweep — event-level F1 per class
    # ------------------------------------------------------------------
    print("\nSweeping per-class event-level thresholds (1D-IoU ≥ "
          f"{args.iou_threshold})…")
    stage1_res = sweep_per_class_thresholds(
        stage1_dets, gt_events, iou_threshold=args.iou_threshold,
    )
    combined_res = sweep_per_class_thresholds(
        combined_dets, gt_events, iou_threshold=args.iou_threshold,
    )

    eval_label = ("FULL VAL" if args.full_val
                  else "HELD-OUT 20% (leakage-free)")
    print_comparison(
        stage1_res, combined_res,
        header=(f"Event-level F1 — {eval_label}"
                f"  ({len(records)} candidates, "
                f"{len(gt_events)} GT events)"),
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    def _jsonable(o):
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    summary = {
        "verifier_checkpoint": args.verifier_checkpoint,
        "candidates": args.candidates,
        "eval_subset": "full_val" if args.full_val else "held_out_20pct",
        "n_candidates": len(records),
        "n_gt_events": len(gt_events),
        "iou_threshold": args.iou_threshold,
        "stage1": stage1_res,
        "combined": combined_res,
        "macro_F1_stage1": float(np.mean(
            [stage1_res[c]["best_f1"] for c in cfg.CALL_TYPES_3])),
        "macro_F1_combined": float(np.mean(
            [combined_res[c]["best_f1"] for c in cfg.CALL_TYPES_3])),
    }
    with open(out_dir / "two_stage_summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\nWrote {out_dir / 'two_stage_summary.json'}")


if __name__ == "__main__":
    main()
