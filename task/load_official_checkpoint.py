"""
Load and Evaluate the Official WhaleVAD Checkpoint
==================================================

Loads the trained checkpoint distributed by Geldenhuys et al.
(``WhaleVAD_ATBFL_3P-c6f6a07a.pt``) into our ``WhaleVAD`` architecture
and evaluates it on the BioDCASE 2025 development set, using our own
post-processing and metric pipeline.

The point of this script is to **decouple** the two halves of any
reproduction gap:

    1. If this script reproduces F1≈0.44 (the value reported in Table 2
       of the DCASE 2025 tech report), then our evaluation pipeline,
       data loading, post-processing, and metrics are all correct — and
       any gap between our trained model and the paper's number is
       therefore due to **training**, not evaluation.

    2. If this script reports a substantially different F1, the gap is
       in our evaluation/post-processing/metrics — and that is much
       cheaper to fix than retraining.

Either outcome is informative. The script does NOT modify the checkpoint
or write any new model files; it only prints metrics.

Checkpoint key remapping
------------------------
The official checkpoint uses different module names than our ``model.py``::

    fbank.*                     →  filterbank.*
    cnn_blocks.0.*              →  feat_extractor.*
    cnn_blocks.1.blocks.0.*     →  residual_stack.blocks.0.*  (bottleneck)
    cnn_blocks.1.blocks.1.*     →  residual_stack.blocks.1.*  (depthwise)
    feat_proj.*                 →  feat_proj.*               (unchanged)
    bb_proj.*                   →  SKIPPED (bounding-box head, unused)
    lstm.*                      →  lstm.*                     (unchanged)
    classifier.*                →  classifier.*               (unchanged; 7-way)

Note that the official model has 7 output classes, so we build with
``num_classes=7`` regardless of ``cfg.USE_3CLASS`` and collapse to
3 classes at evaluation time via ``COLLAPSE_MAP``.

Usage
-----
::

    python load_official_checkpoint.py \
        --checkpoint /path/to/WhaleVAD_ATBFL_3P-c6f6a07a.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD
from dataset import (
    WhaleDataset, build_val_segments, get_file_manifest, load_annotations,
    collate_fn,
)
from postprocess import (
    Detection, stitch_segments, smooth_probabilities, merge_and_filter,
    compute_metrics,
)


# Mapping from coarse 3-class label name to the indices of the
# corresponding fine-grained 7-class outputs. Order of the 7 classes in
# the official checkpoint matches CALL_TYPES_7 from config.py.
SEVEN_TO_THREE = {
    "bmabz": [
        cfg.CALL_TYPES_7.index("bma"),
        cfg.CALL_TYPES_7.index("bmb"),
        cfg.CALL_TYPES_7.index("bmz"),
    ],
    "d": [
        cfg.CALL_TYPES_7.index("bmd"),
        cfg.CALL_TYPES_7.index("bpd"),
    ],
    "bp": [
        cfg.CALL_TYPES_7.index("bp20"),
        cfg.CALL_TYPES_7.index("bp20plus"),
    ],
}


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the official WhaleVAD checkpoint "
                        "(e.g. WhaleVAD_ATBFL_3P-c6f6a07a.pt).")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE,
                   help="Inference batch size. Adjust if you OOM.")
    p.add_argument("--iou", type=float, default=0.3,
                   help="IoU threshold for event-level matching "
                        "(default 0.3, matching DCASE evaluation).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Single classification threshold for ALL classes. "
                        "If omitted, sweeps a small grid per class to "
                        "find each class's best F1 (matching the paper's "
                        "post-hoc threshold-on-PR-curve protocol).")
    return p.parse_args()


# ======================================================================
# Checkpoint loading
# ======================================================================

def remap_checkpoint_keys(state_dict: dict) -> dict:
    """
    Translate official-checkpoint key names into our model's key names.

    Drops any ``bb_proj.*`` keys (the bounding-box regressor head, which
    we do not include in our model since the multi-objective ablation in
    DCASE Table 2 showed it hurt F1 by 22.1% and was excluded from the
    final 0.440 model).

    Parameters
    ----------
    state_dict : dict
        Raw state_dict from ``torch.load`` of the official checkpoint.

    Returns
    -------
    dict
        State_dict with our model's key naming and no ``bb_proj.*`` keys.
    """
    def rename(k: str) -> str:
        if k.startswith("fbank."):
            return k.replace("fbank.", "filterbank.")
        if k.startswith("cnn_blocks.0."):
            return k.replace("cnn_blocks.0.", "feat_extractor.")
        if k.startswith("cnn_blocks.1.blocks.0."):
            return k.replace("cnn_blocks.1.blocks.0.", "residual_stack.blocks.0.")
        if k.startswith("cnn_blocks.1.blocks.1."):
            return k.replace("cnn_blocks.1.blocks.1.", "residual_stack.blocks.1.")
        return k

    out = {}
    for k, v in state_dict.items():
        if k.startswith("bb_proj"):
            # Bounding-box regressor head — explicitly not part of our model.
            continue
        out[rename(k)] = v
    return out


def load_official_model(ckpt_path: str, device: torch.device) -> WhaleVAD:
    """
    Build a 7-class WhaleVAD and load the official checkpoint into it.

    Performs a dummy forward pass before ``load_state_dict`` so that the
    lazy projection layer in our model is materialized and its weight
    tensor is allocated; otherwise ``load_state_dict`` would complain
    about a missing ``feat_proj.weight`` key.

    Parameters
    ----------
    ckpt_path : str
        Filesystem path to the official ``.pt`` file.
    device : torch.device

    Returns
    -------
    WhaleVAD
        Model with weights loaded, set to eval mode, on the given device.
    """
    spec_extractor = SpectrogramExtractor().to(device)
    # 7 outputs to match the checkpoint's classifier head.
    model = WhaleVAD(num_classes=7).to(device)

    # Trigger lazy projection materialisation. The shape of the projection
    # input depends on the spectrogram dimensions, so we do it via an
    # actual forward pass rather than guessing.
    dummy_audio = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
    with torch.no_grad():
        model(spec_extractor(dummy_audio))

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    remapped = remap_checkpoint_keys(raw)

    # strict=True: any key mismatch indicates a bug in the remap above
    # and we want to know about it loudly.
    model.load_state_dict(remapped, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded official checkpoint: {ckpt_path}")
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Output classes: 7 (collapsed to 3 at evaluation time)")
    return model, spec_extractor


# ======================================================================
# 7→3 class collapsing
# ======================================================================

def collapse_7_to_3(probs_7: np.ndarray) -> np.ndarray:
    """
    Collapse a (T, 7) probability array into (T, 3) by max-pooling within
    each coarse-class group.

    Per the DCASE tech report Section 2.9, the 7-class output is
    "collapsed into the 3-class variant" for evaluation. Max-pooling
    inside each group is the most defensible reduction: a frame is
    counted as "bmabz" if any of its constituent fine-grained outputs
    fired strongly, and similarly for d and bp.

    Parameters
    ----------
    probs_7 : np.ndarray
        Array of shape ``(T, 7)`` with per-frame fine-grained probabilities.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, 3)`` with per-frame coarse-class probabilities,
        with column order ``[bmabz, d, bp]`` matching ``CALL_TYPES_3``.
    """
    out = np.zeros((probs_7.shape[0], 3), dtype=probs_7.dtype)
    for i, name in enumerate(cfg.CALL_TYPES_3):
        idxs = SEVEN_TO_THREE[name]
        out[:, i] = probs_7[:, idxs].max(axis=1)
    return out


# ======================================================================
# Inference + metrics
# ======================================================================

@torch.no_grad()
def run_inference(model, spec_extractor, loader, device):
    """
    Run the model over the loader and return per-window 3-class probability
    arrays, keyed by ``(dataset, filename, start_sample)``.

    The 7-class outputs are collapsed to 3-class via ``collapse_7_to_3``
    before being returned, so downstream code can treat the result as if
    it came from a native 3-class model.
    """
    all_probs = {}
    hop = spec_extractor.hop_length
    for audio, _, _, metas in tqdm(loader, desc="Inference"):
        audio = audio.to(device)
        logits = model(spec_extractor(audio))            # (B, T, 7)
        probs_7 = torch.sigmoid(logits).cpu().numpy()    # (B, T, 7)
        for j, meta in enumerate(metas):
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs_7[j].shape[0])
            collapsed = collapse_7_to_3(probs_7[j, :n_frames, :])
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            all_probs[key] = collapsed
    return all_probs


def probs_to_detections(file_probs: dict, thresholds: np.ndarray) -> list[Detection]:
    """
    Apply thresholds, smoothing, and merge-and-filter to produce final
    event-level detections.

    This mirrors the post-processing logic used by the regular inference
    paths so the official checkpoint is evaluated under exactly the same
    conditions our trained models would be.
    """
    all_dets = []
    class_names = cfg.CALL_TYPES_3
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        for c, name in enumerate(class_names):
            active = probs[:, c] > thresholds[c]
            diffs = np.diff(active.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                all_dets.append(Detection(
                    dataset=ds, filename=fn, label=name,
                    start_s=s * cfg.FRAME_STRIDE_S,
                    end_s=e * cfg.FRAME_STRIDE_S,
                    confidence=float(probs[s:e, c].mean()),
                ))
    return merge_and_filter(all_dets)


def build_gt_events(annotations, file_start_dts):
    """Construct ground-truth Detection objects from the validation CSV."""
    gt = []
    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        # Always use the 3-class label here regardless of cfg.USE_3CLASS,
        # because the entire purpose of this script is 3-class evaluation
        # of a 7-class model.
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt


def find_best_thresholds(file_probs: dict, gt_events: list, iou: float):
    """
    Per-class threshold sweep over a coarse grid, choosing each class's
    threshold to maximise its own F1.

    The DCASE tech report Section 2.9 selects thresholds from the
    precision-recall curve at the point of best F1 on a held-out set.
    Here we approximate that by sweeping a fixed grid; this is the same
    logic that ``tune_thresholds.py`` uses for our own trained models.

    Parameters
    ----------
    file_probs : dict
    gt_events : list of Detection
    iou : float

    Returns
    -------
    np.ndarray of shape (3,)
        Per-class thresholds in CALL_TYPES_3 order.
    """
    grid = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.95, 0.1)])
    best = np.array([0.5, 0.5, 0.5])
    for c, name in enumerate(cfg.CALL_TYPES_3):
        best_f1 = -1.0
        for t in grid:
            trial = best.copy()
            trial[c] = t
            preds = probs_to_detections(file_probs, trial)
            m = compute_metrics(preds, gt_events, iou_threshold=iou)
            f1 = m.get(name, {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best[c] = t
        print(f"  {name:6} best threshold = {best[c]:.3f}  → F1 = {best_f1:.3f}")
    return best


# ======================================================================
# Main
# ======================================================================

def main():
    """Entry point: load checkpoint, run inference, print metrics."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, spec_extractor = load_official_model(args.checkpoint, device)

    # ------------------------------------------------------------------
    # Validation data
    # ------------------------------------------------------------------
    print(f"\nLoading validation sets: {cfg.VAL_DATASETS}")
    manifest = get_file_manifest(cfg.VAL_DATASETS)
    annotations = load_annotations(cfg.VAL_DATASETS, manifest=manifest)
    print(f"  {len(manifest)} files, {len(annotations)} annotations")

    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }
    segments = build_val_segments(manifest, annotations)
    print(f"  {len(segments)} validation segments (30s × 2s overlap)")

    loader = DataLoader(
        WhaleDataset(segments), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Inference (7-class → collapsed to 3-class probabilities)
    # ------------------------------------------------------------------
    print("\nRunning inference and stitching overlapping windows...")
    raw_probs = run_inference(model, spec_extractor, loader, device)
    file_probs = stitch_segments(raw_probs)
    print(f"  {len(file_probs)} files of stitched probabilities")

    gt_events = build_gt_events(annotations, file_start_dts)
    print(f"  {len(gt_events)} ground-truth events")

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    if args.threshold is not None:
        thresholds = np.array([args.threshold] * 3)
        print(f"\nUsing single threshold {args.threshold} for all classes")
    else:
        print("\nSweeping per-class thresholds for best F1...")
        thresholds = find_best_thresholds(file_probs, gt_events, args.iou)

    print(f"\nFinal thresholds: bmabz={thresholds[0]:.3f}, "
          f"d={thresholds[1]:.3f}, bp={thresholds[2]:.3f}")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    preds = probs_to_detections(file_probs, thresholds)
    print(f"\nTotal predicted events: {len(preds)}")

    print(f"\n=== OVERALL ===  (IoU threshold {args.iou})")
    metrics = compute_metrics(preds, gt_events, iou_threshold=args.iou)
    for name in cfg.CALL_TYPES_3:
        if name in metrics:
            m = metrics[name]
            print(f"  {name:6} TP={m['tp']:5} FP={m['fp']:6} FN={m['fn']:6}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"  OVERALL F1 = {metrics['overall']['f1']:.3f}")

    # ------------------------------------------------------------------
    # Per-dataset breakdown — this is what to compare against Table 3
    # ------------------------------------------------------------------
    print(f"\n=== PER-DATASET (matches DCASE tech report Table 3 layout) ===")
    print("\n  Tech report numbers for reference:")
    print("    casey2017      bmabz F1=0.624   d F1=0.054   bp F1=0.025")
    print("    kerguelen2014  bmabz F1=0.672   d F1=0.141   bp F1=0.480")
    print("    kerguelen2015  bmabz F1=0.565   d F1=0.165   bp F1=0.581")
    print("    overall (avg across sites and classes): F1=0.440\n")

    for ds_name in cfg.VAL_DATASETS:
        ds_preds = [d for d in preds if d.dataset == ds_name]
        ds_gts = [d for d in gt_events if d.dataset == ds_name]
        m = compute_metrics(ds_preds, ds_gts, iou_threshold=args.iou)
        print(f"  {ds_name}:")
        for name in cfg.CALL_TYPES_3:
            if name in m:
                r = m[name]
                print(f"    {name:6} TP={r['tp']:5} FP={r['fp']:6} FN={r['fn']:6}  "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
        print(f"    overall F1 = {m['overall']['f1']:.3f}\n")


if __name__ == "__main__":
    main()
