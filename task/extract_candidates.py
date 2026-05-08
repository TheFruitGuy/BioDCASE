"""
Extract Stage-1 Candidate Events for Verifier Training
======================================================

Runs a trained Whale-VAD baseline checkpoint over a labelled split (train,
val, or both), gathers candidate event detections at a deliberately *low*
confidence threshold so that recall is high, and labels each candidate as
true-positive or false-positive by 1D-IoU matching against the ground-truth
annotations. The labelled candidate table is the training data for the
stage-2 verifier model.

Pipeline
--------
    1. Load checkpoint, init lazy projection, freeze model.
    2. Build the standard 30 s × 2 s overlap eval windows for the requested
       split via ``dataset.build_val_segments``.
    3. Run inference, accumulate per-window probabilities.
    4. Run ``postprocess_predictions`` with a low per-class threshold (0.3
       by default — same code path as evaluation, just looser).
    5. Match each candidate to the highest-IoU same-class ground-truth event
       on the same file using the GT-first greedy strategy of
       ``compute_metrics`` (so verifier labels match the eval semantics
       exactly).
    6. Emit a parquet file with one row per candidate.

Why low threshold
-----------------
At inference time the two-stage detector will use this same low threshold
to over-generate proposals; the verifier's job is to reject the FPs. So
the candidates produced here must be drawn from the same operating point
as inference. 0.3 is a reasonable starting point; tune later.

Output schema (parquet)
-----------------------
    cand_id           int64    unique row id
    dataset           string   site name
    filename          string   wav file
    path              string   absolute path to wav (for crop loading)
    file_dur_s        float64  full file duration in seconds
    start_s, end_s    float64  candidate event span (file-relative)
    predicted_class   string   "bmabz" / "d" / "bp"
    class_idx         int32    0 / 1 / 2
    stage1_score      float64  mean per-frame probability over the span
    label             int8     1 = TP (matched a GT), 0 = FP
    best_iou          float64  IoU with matched GT (0.0 for FPs)
    gt_start_s        float64  matched GT start, NaN for FPs
    gt_end_s          float64  matched GT end, NaN for FPs
    source_split      string   "train" / "val"

Usage
-----
::

    # Single-checkpoint mode (optimistic FPs on training split — fine for
    # an initial sanity check). For the final number, run K-fold via
    # train_kfold_stage1.py first and call this once per fold.
    python extract_candidates.py \\
        --checkpoint runs/whalevad_seed42/final_model.pt \\
        --split val \\
        --output candidates_val.parquet \\
        --low_threshold 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    WhaleDataset,
    build_val_segments,
    collate_fn,
    get_file_manifest,
    load_annotations,
)
from model import WhaleVAD
from postprocess import (
    Detection,
    collapse_probs_to_3class,
    compute_iou_1d,
    postprocess_predictions,
)
from spectrogram import SpectrogramExtractor


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a train.py checkpoint (must contain "
                        "'model_state_dict').")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "both"],
                   help="Which labelled split to extract candidates from. "
                        "'val' is the safe default for an initial sanity "
                        "check; 'train' produces optimistic FPs unless the "
                        "checkpoint was trained K-fold-style.")
    p.add_argument("--output", type=str, required=True,
                   help="Destination parquet path.")
    p.add_argument("--low_threshold", type=float, default=0.3,
                   help="Per-class probability threshold used to generate "
                        "candidates. Lower → more recall, more FPs for "
                        "verifier to reject. Applied to all classes.")
    p.add_argument("--iou_match", type=float, default=0.3,
                   help="IoU threshold used to label TPs. 0.3 matches the "
                        "DCASE evaluation metric.")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


# ======================================================================
# Inference
# ======================================================================

@torch.no_grad()
def run_inference(model, spec_extractor, loader, device):
    """
    Run stage-1 inference and return raw per-window probabilities.

    Mirrors ``tune_thresholds.collect_probs`` exactly so the candidates we
    extract here are bit-identical to what threshold tuning would see at
    the same operating point.

    Returns
    -------
    dict
        Maps ``(dataset, filename, start_sample)`` to ``(n_frames,
        n_classes)`` numpy arrays of post-sigmoid probabilities.
    """
    model.eval()
    all_probs: dict[tuple[str, str, int], np.ndarray] = {}
    hop = spec_extractor.hop_length

    for audio, _, _, metas in tqdm(loader, desc="Stage-1 inference"):
        audio = audio.to(device)
        logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Collapse 7→3 if needed; no-op when USE_3CLASS is True.
    return collapse_probs_to_3class(all_probs)


# ======================================================================
# Ground-truth event construction
# ======================================================================

def build_gt_events(annotations: pd.DataFrame, manifest: pd.DataFrame) -> list[Detection]:
    """
    Build a list of ground-truth ``Detection`` objects for IoU matching.

    Identical to ``tune_thresholds.build_gt_events`` but takes the manifest
    directly so we don't have to thread ``file_start_dts`` through the
    caller.

    Parameters
    ----------
    annotations : pd.DataFrame
        From ``load_annotations``.
    manifest : pd.DataFrame
        From ``get_file_manifest``.

    Returns
    -------
    list of Detection
    """
    file_starts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }

    gt_events: list[Detection] = []
    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_starts.get(key)
        if fsd is None or pd.isna(fsd):
            continue
        # Use coarse 3-class label for matching, since the candidate side
        # already lives in 3-class space (model output collapsed).
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"],
            filename=row["filename"],
            label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt_events


# ======================================================================
# Candidate ↔ ground-truth matching
# ======================================================================

def label_candidates(
    candidates: list[Detection],
    gt_events: list[Detection],
    iou_match: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Label each candidate as TP (1) or FP (0) by GT-first greedy matching.

    The matching policy mirrors ``postprocess.compute_metrics``: for each
    (file, class) bucket, sort GTs by start time; each GT picks the
    highest-IoU unmatched candidate; if best IoU ≥ ``iou_match`` the
    candidate is a TP, else the GT is unmatched (FN — irrelevant to the
    verifier, which only sees stage-1 outputs). All remaining candidates
    are FPs.

    This guarantees that the verifier's positive examples are exactly the
    detections that would count as TPs at evaluation time — no semantic
    drift between training labels and eval criterion.

    Parameters
    ----------
    candidates : list of Detection
        Stage-1 outputs after postprocessing.
    gt_events : list of Detection
        Ground-truth annotations.
    iou_match : float
        Minimum IoU to accept a TP match.

    Returns
    -------
    labels : np.ndarray, int8, shape (n_candidates,)
        1 for TP, 0 for FP.
    ious : np.ndarray, float64, shape (n_candidates,)
        Best matched IoU (0.0 for FPs).
    matched_gt_idx : np.ndarray, int32, shape (n_candidates,)
        Index into ``gt_events`` of the matched GT (-1 for FPs).
    """
    n = len(candidates)
    labels = np.zeros(n, dtype=np.int8)
    ious = np.zeros(n, dtype=np.float64)
    matched = np.full(n, -1, dtype=np.int32)

    # Bucket candidates by (file, class) with their global indices preserved.
    cand_buckets: dict[tuple[str, str, str], list[tuple[int, Detection]]] = {}
    for i, c in enumerate(candidates):
        cand_buckets.setdefault(
            (c.dataset, c.filename, c.label), []
        ).append((i, c))

    # Same buckets for GT, plus global indices for diagnostic output.
    gt_buckets: dict[tuple[str, str, str], list[tuple[int, Detection]]] = {}
    for j, g in enumerate(gt_events):
        gt_buckets.setdefault(
            (g.dataset, g.filename, g.label), []
        ).append((j, g))

    for key, cand_list in cand_buckets.items():
        gt_list = gt_buckets.get(key, [])
        if not gt_list:
            continue  # all FPs — labels already 0

        # Sort GTs by start time so the iteration order matches compute_metrics.
        gt_list_sorted = sorted(gt_list, key=lambda x: x[1].start_s)
        used_global: set[int] = set()

        for j, gt in gt_list_sorted:
            best_iou = 0.0
            best_global_idx = -1
            for global_idx, c in cand_list:
                if global_idx in used_global:
                    continue
                iou = compute_iou_1d(c.start_s, c.end_s, gt.start_s, gt.end_s)
                if iou > best_iou:
                    best_iou = iou
                    best_global_idx = global_idx
            if best_iou >= iou_match and best_global_idx >= 0:
                labels[best_global_idx] = 1
                ious[best_global_idx] = best_iou
                matched[best_global_idx] = j
                used_global.add(best_global_idx)

    return labels, ious, matched


# ======================================================================
# Per-split processing
# ======================================================================

def process_split(
    datasets: list[str],
    split_name: str,
    model: torch.nn.Module,
    spec_extractor: torch.nn.Module,
    device: torch.device,
    args,
) -> pd.DataFrame:
    """
    Run stage-1 + matching on one split. Returns a DataFrame ready to
    concat with other splits before parquet write.
    """
    print(f"\n=== Split: {split_name} ({len(datasets)} sites) ===")

    manifest = get_file_manifest(datasets)
    annotations = load_annotations(datasets, manifest=manifest)
    print(f"  {len(manifest)} files, {len(annotations)} annotations")

    segments = build_val_segments(manifest, annotations)
    print(f"  {len(segments)} eval-style 30s segments")

    loader = DataLoader(
        WhaleDataset(segments),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    all_probs = run_inference(model, spec_extractor, loader, device)

    # Postprocess at the LOW threshold to over-generate proposals. Same
    # code path as evaluation: stitch → 500 ms median → threshold → merge
    # → duration filter. This ensures candidates are drawn from the same
    # distribution that inference will face.
    n_classes = 3 if cfg.USE_3CLASS or any(
        p.shape[1] == 3 for p in all_probs.values()
    ) else cfg.n_classes()
    thresholds = np.full(n_classes, args.low_threshold)
    candidates = postprocess_predictions(all_probs, thresholds)
    print(f"  {len(candidates)} candidate events at threshold "
          f"{args.low_threshold}")

    # Build GT, match, label.
    gt_events = build_gt_events(annotations, manifest)
    print(f"  {len(gt_events)} ground-truth events")

    labels, ious, matched = label_candidates(candidates, gt_events,
                                             args.iou_match)
    n_tp = int(labels.sum())
    n_fp = len(labels) - n_tp
    print(f"  TP={n_tp}  FP={n_fp}  TP/FP ratio={n_tp / max(n_fp, 1):.3f}")

    # Per-class TP/FP breakdown — useful for sanity-checking that D-class
    # really does have an FP-dominated distribution.
    for cls_name in cfg.CALL_TYPES_3:
        cls_mask = np.array([c.label == cls_name for c in candidates])
        if not cls_mask.any():
            continue
        cls_labels = labels[cls_mask]
        n_cls_tp = int(cls_labels.sum())
        n_cls_fp = int(len(cls_labels) - n_cls_tp)
        print(f"    {cls_name}: TP={n_cls_tp}  FP={n_cls_fp}  "
              f"precision={n_cls_tp / max(n_cls_tp + n_cls_fp, 1):.3f}")

    # Per-file path lookup for the crop loader downstream.
    path_lookup = {
        (r["dataset"], r["filename"]): (r["path"], r["duration_s"])
        for _, r in manifest.iterrows()
    }

    # Build the output rows. Use NaN for un-matched GT fields so the
    # parquet stays float (no mixed types). class_idx is the index of the
    # predicted class in cfg.CALL_TYPES_3 (or CALL_TYPES_7 in 7-class
    # mode).
    cls_to_idx = {n: i for i, n in enumerate(cfg.CALL_TYPES_3)}

    rows = []
    for i, c in enumerate(candidates):
        path, dur = path_lookup.get((c.dataset, c.filename), (None, np.nan))
        if path is None:
            # Defensive: should never happen because candidates come from
            # the same manifest. Skip with a warning.
            print(f"  WARN: no path for ({c.dataset}, {c.filename}) — skip")
            continue

        if labels[i] == 1 and matched[i] >= 0:
            gt = gt_events[matched[i]]
            gt_start = gt.start_s
            gt_end = gt.end_s
        else:
            gt_start = float("nan")
            gt_end = float("nan")

        rows.append({
            "cand_id": i,                          # local to this split
            "dataset": c.dataset,
            "filename": c.filename,
            "path": path,
            "file_dur_s": float(dur),
            "start_s": float(c.start_s),
            "end_s": float(c.end_s),
            "predicted_class": c.label,
            "class_idx": int(cls_to_idx.get(c.label, -1)),
            "stage1_score": float(c.confidence),
            "label": int(labels[i]),
            "best_iou": float(ious[i]),
            "gt_start_s": gt_start,
            "gt_end_s": gt_end,
            "source_split": split_name,
        })

    return pd.DataFrame(rows)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Initialise the lazy projection layer before loading weights.
    dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
    model(spec_extractor(dummy))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------
    # Per-split processing
    # ------------------------------------------------------------------
    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", cfg.TRAIN_DATASETS))
    if args.split in ("val", "both"):
        splits.append(("val", cfg.VAL_DATASETS))

    dfs = []
    cand_id_offset = 0
    for split_name, datasets in splits:
        df = process_split(datasets, split_name, model, spec_extractor,
                           device, args)
        # Make cand_id globally unique across the merged parquet.
        df["cand_id"] = df["cand_id"] + cand_id_offset
        cand_id_offset += len(df)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"\nWrote {len(out)} candidates → {out_path}")
    print(f"  Total TPs: {(out['label'] == 1).sum()}")
    print(f"  Total FPs: {(out['label'] == 0).sum()}")
    print("\nPer-class summary across all splits:")
    for cls_name in cfg.CALL_TYPES_3:
        sub = out[out["predicted_class"] == cls_name]
        n_tp = int((sub["label"] == 1).sum())
        n_fp = int((sub["label"] == 0).sum())
        prec = n_tp / max(n_tp + n_fp, 1)
        print(f"  {cls_name}: TP={n_tp}  FP={n_fp}  precision={prec:.3f}")


if __name__ == "__main__":
    main()
