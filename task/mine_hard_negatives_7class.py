"""
Hard-Negative Mining for 7-Class Models
========================================

Same purpose as ``mine_hard_negatives.py``: run one or more trained
checkpoints on the labeled training data, identify event-level false
positives of a target class, persist their locations for HNM fine-tuning.

What changes for 7-class
------------------------

The 3-class version of this script has a silent bug for 7-class
checkpoints. After collapsing 7-channel probabilities to 3 channels,
it calls ``threshold_to_detections``, which reads ``cfg.class_names()``
to label the resulting events. In 7-class mode that returns
``["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]`` — so the
THREE collapsed channels get labelled bma/bmb/bmz, and merge_and_filter
then maps all three back to ``bmabz``. The script produces zero D-FPs
and zero BP-FPs, period. Run it once on a 7-class checkpoint and the
output JSON looks plausible but only contains "bmabz" events.

This version does mining entirely in 7-class space. The target is a
single CALL_TYPES_7 name (bma, bmb, bmz, bmd, bpd, bp20, bp20plus).
GT for matching comes from the raw fine-grained ``annotation`` field
of the training set. The output JSON records ``target_class`` as the
7-class name, so the consuming trainer (``train_phase_hnm_7class.py``)
can do subclass-level PGI directly. For group-level PGI on the
3-class space, mine one JSON per subclass and pass them all to the
trainer with ``--isolate_classes`` — the trainer expands group-target
JSONs and single-channel JSONs uniformly.

Config handling
---------------

The script auto-overrides ``cfg.USE_3CLASS = False`` at the start of
``main()`` if it was True. The override is in-process only — config.py
on disk is never modified and concurrent 3-class jobs in OTHER
processes are unaffected. You do not need to edit config.py to switch
between 3-class and 7-class mining runs.

Usage
-----
::

    # Mine BmD false positives (subclass of the D group):
    python mine_hard_negatives_7class.py \\
        --checkpoints runs/phase0r_20260520_xxxxxx/best_model.pt \\
        --target bmd --threshold 0.2 --top_k 1500 \\
        --output runs/hardnegs_7c/bmd_<src>.json

    # Mine BpD false positives:
    python mine_hard_negatives_7class.py \\
        --checkpoints runs/phase0r_20260520_xxxxxx/best_model.pt \\
        --target bpd --threshold 0.2 --top_k 1500 \\
        --output runs/hardnegs_7c/bpd_<src>.json

    # Full ensemble (consensus FP signal), all 7 subclasses:
    for t in bma bmb bmz bmd bpd bp20 bp20plus; do
      python mine_hard_negatives_7class.py \\
        --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \\
        --target $t --threshold 0.2 --top_k 1500 \\
        --output runs/hardnegs_7c/${t}_ensemble.json
    done
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import (
    Detection, compute_iou_1d, smooth_probabilities,
    stitch_segments, threshold_to_detections,
)
from spectrogram import SpectrogramExtractor

import wandb_utils as wbu

# Reuse the ensemble's model/inference plumbing — the file already
# handles baseline vs BPN auto-detection and BPN's dict return value.
from ensemble_predict import (
    average_prob_dicts, build_model_for_ckpt, predict_probabilities,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint paths. Multiple → "
                        "averaged probabilities (ensemble mining).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint weights (must match "
                        "len(checkpoints)). Default: equal.")
    p.add_argument("--target", type=str, default="bmd",
                   choices=cfg.CALL_TYPES_7,
                   help="Subclass to mine FPs for. The 7-class group "
                        "structure under cfg.COLLAPSE_MAP is: "
                        "bmabz={bma,bmb,bmz}, d={bmd,bpd}, "
                        "bp={bp20,bp20plus}.")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Frame-level threshold for proposal generation. "
                        "Lower than tuned operating point on purpose.")
    p.add_argument("--max_iou_for_fp", type=float, default=0.1)
    p.add_argument("--top_k", type=int, default=1500)
    p.add_argument("--datasets", type=str, nargs="+", default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for reproducible mining. Affects any "
                        "stochastic ops in the inference pipeline. "
                        "Use the same seed across mining runs you want "
                        "to compare directly.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Skip wandb tracking for this mining run "
                        "(default: track as phase 6 mining job).")
    return p.parse_args()


def build_target_class_thresholds(target_idx: int, threshold: float) -> np.ndarray:
    """Threshold array for the 7-class output: `threshold` for target, 0.999 otherwise."""
    thr = np.full(len(cfg.CALL_TYPES_7), 0.999, dtype=np.float64)
    thr[target_idx] = threshold
    return thr


def build_gt_events_for_class(annotations, file_start_dts, target_class: str):
    """Build GT events whose fine-grained annotation matches target_class.

    Reads ``row["annotation"]`` directly (not label_3class), since the
    target is a 7-class name.
    """
    gt = []
    for _, row in annotations.iterrows():
        if row["annotation"] != target_class:
            continue
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=target_class,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt


def predictions_to_fps(predictions, gt, target_class: str, max_iou_for_fp: float):
    gt_by_file = {}
    for g in gt:
        gt_by_file.setdefault((g.dataset, g.filename), []).append(g)
    fps = []
    for p in predictions:
        if p.label != target_class:
            continue
        candidates = gt_by_file.get((p.dataset, p.filename), [])
        max_iou = max(
            (compute_iou_1d(p.start_s, p.end_s, g.start_s, g.end_s)
             for g in candidates),
            default=0.0,
        )
        if max_iou < max_iou_for_fp:
            fps.append(p)
    return fps


def merge_and_filter_7class(detections):
    """Skip the COLLAPSE_MAP step of postprocess.merge_and_filter.

    Otherwise identical: per (file, class) sort + greedy merge of events
    within MERGE_GAP_S, then duration filter against POST_MIN_DUR_S /
    POST_MAX_DUR_S. We need this because the standard merge_and_filter
    unconditionally rewrites every 7-class label to its 3-class group,
    which here would erase the subclass distinction we are mining for.
    """
    groups: dict[tuple, list[Detection]] = {}
    for d in detections:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    final = []
    for _, events in groups.items():
        events.sort(key=lambda x: x.start_s)
        merged: list[Detection] = []
        for e in events:
            if not merged:
                merged.append(e)
            else:
                last = merged[-1]
                if e.start_s - last.end_s <= cfg.MERGE_GAP_S:
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(e)
        for m in merged:
            dur = m.end_s - m.start_s
            if cfg.POST_MIN_DUR_S <= dur <= cfg.POST_MAX_DUR_S:
                final.append(m)
    return final


def main():
    args = parse_args()
    # Auto-override the class-axis flag instead of requiring the user
    # to edit config.py. cfg attributes are read dynamically by
    # downstream callers (cfg.n_classes, cfg.class_names, dataset/loss
    # paths), so a single in-process flip is enough — config.py on
    # disk is never modified, and concurrent 3-class jobs in OTHER
    # processes are unaffected. Restore-on-exit is unnecessary
    # because the script then terminates.
    if getattr(cfg, "USE_4CLASS_D_SPLIT", False):
        raise RuntimeError(
            "USE_4CLASS_D_SPLIT must be False for 7-class mining; "
            "got True. Check config.py.")
    if cfg.USE_3CLASS:
        print(f"[mine 7c] cfg.USE_3CLASS True -> False (in-process "
              f"only; config.py unchanged). Required because the "
              f"source checkpoint has 7 output channels.")
        cfg.USE_3CLASS = False
    assert cfg.n_classes() == 7, (
        f"expected n_classes() == 7 after override, got {cfg.n_classes()}")
    wbu.seed_everything(args.seed, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_idx = cfg.CALL_TYPES_7.index(args.target)
    print(f"Target subclass: '{args.target}' (7-class idx {target_idx})")
    print(f"  rolls up to 3-class group '{cfg.COLLAPSE_MAP[args.target]}'")
    print(f"Proposal threshold: {args.threshold}")
    print(f"Mining from {len(args.checkpoints)} checkpoint(s)")
    print(f"Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Wandb setup. Mining runs are tagged with phase 6 (the HNM phase
    # they feed into) but with ``job_type="mining"`` to distinguish
    # them from the actual fine-tuning runs. Each source checkpoint
    # gets its phase inferred from the path so ``source_phase`` tags
    # like ``source_5`` / ``source_baseline`` are queryable.
    # ------------------------------------------------------------------
    run = None
    if not args.no_wandb:
        import re
        # Infer source phase from the FIRST checkpoint path. For ensemble
        # mining (multiple checkpoints), we additionally tag with the
        # number of source models so ensemble vs single-model mining
        # runs are distinguishable in the runs table.
        first_src = str(args.checkpoints[0])
        m = re.search(r"phase(\w+)_\d{8}_\d{6}", first_src)
        if m:
            source_phase = m.group(1)
        elif "whalevad" in first_src:
            source_phase = "baseline"
        else:
            source_phase = "unknown"

        ensemble_size = len(args.checkpoints)
        extra_tags = [
            f"target_{args.target}",
            f"target_group_{cfg.COLLAPSE_MAP[args.target]}",
            f"source_{source_phase}",
            "mining",
            "7class",
        ]
        if ensemble_size > 1:
            extra_tags.append(f"ensemble_{ensemble_size}")
        else:
            extra_tags.append("single_model")

        run = wbu.init_phase(
            "6-7c",
            extra_tags=extra_tags,
            job_type="mining",
            config={
                "target":           args.target,
                "target_idx":       target_idx,
                "target_group_3c":  cfg.COLLAPSE_MAP[args.target],
                "threshold":        args.threshold,
                "max_iou_for_fp":   args.max_iou_for_fp,
                "top_k":            args.top_k,
                "n_checkpoints":    ensemble_size,
                "source_phase":     source_phase,
                "checkpoints":      [str(c) for c in args.checkpoints],
                "weights":          args.weights,
                "datasets":         (args.datasets or list(cfg.TRAIN_DATASETS)),
                "batch_size":       args.batch_size,
                "seed":             args.seed,
                "output_path":      str(args.output),
            },
        )

    # ------------------------------------------------------------------
    # Weights handling — same convention as ensemble_predict.py
    # ------------------------------------------------------------------
    if args.weights is not None:
        assert len(args.weights) == len(args.checkpoints)
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
        print(f"Per-checkpoint weights (normalized): {weights}")
    else:
        weights = None

    # ------------------------------------------------------------------
    # Build inference loader — same fixed 30s tiles as validation
    # ------------------------------------------------------------------
    datasets = args.datasets or cfg.TRAIN_DATASETS
    print(f"Mining on: {datasets}")

    manifest = get_file_manifest(datasets)
    annotations = load_annotations(datasets, manifest=manifest)
    print(f"  {len(manifest)} files, {len(annotations)} annotations")

    segments = build_val_segments(manifest, annotations)
    print(f"  {len(segments)} 30s tiles to score")

    loader = DataLoader(
        WhaleDataset(segments),
        batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    spec_extractor = SpectrogramExtractor().to(device)

    # ------------------------------------------------------------------
    # Run inference with each checkpoint, collecting prob dicts
    # ------------------------------------------------------------------
    all_prob_dicts = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_type = build_model_for_ckpt(ckpt, device)
        print(f"  type: {model_type}")

        # Materialize lazy projection layer before load_state_dict.
        with torch.no_grad():
            dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
            spec = spec_extractor(dummy)
            _ = model(spec)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        t0 = time.time()
        probs = predict_probabilities(
            model, model_type, spec_extractor, loader, device)
        # NB: do NOT collapse to 3-class here — we are mining in 7-class
        # space so threshold_to_detections can label each subclass channel
        # correctly via cfg.class_names() (= CALL_TYPES_7).
        all_prob_dicts.append(probs)
        print(f"  inference {time.time()-t0:.0f}s, {len(probs)} prob arrays")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Average probabilities (single checkpoint → identity)
    # ------------------------------------------------------------------
    if len(all_prob_dicts) == 1:
        all_probs = all_prob_dicts[0]
        print(f"\nSingle-checkpoint mode: skipping averaging.")
    else:
        all_probs = average_prob_dicts(all_prob_dicts, weights=weights)
        print(f"\nAveraged probs across {len(all_prob_dicts)} models, "
              f"{len(all_probs)} segments")

    # ------------------------------------------------------------------
    # Stitch + smooth + threshold + merge
    # ------------------------------------------------------------------
    file_probs = stitch_segments(all_probs)
    thr = build_target_class_thresholds(target_idx, args.threshold)
    raw_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        raw_dets.extend(threshold_to_detections(probs, thr, ds, fn))
    pred_events = merge_and_filter_7class(raw_dets)
    print(f"  {len(pred_events)} '{args.target}' proposals "
          f"at threshold {args.threshold}")

    # ------------------------------------------------------------------
    # Match to GT, identify FPs, top-k by confidence
    # ------------------------------------------------------------------
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }
    gt = build_gt_events_for_class(annotations, file_start_dts, args.target)
    print(f"  {len(gt)} GT '{args.target}' events on training sites")

    fps = predictions_to_fps(pred_events, gt, args.target, args.max_iou_for_fp)
    print(f"  {len(fps)} hard FPs (max IoU < {args.max_iou_for_fp})")

    fps.sort(key=lambda d: d.confidence, reverse=True)
    if args.top_k > 0 and len(fps) > args.top_k:
        fps = fps[:args.top_k]
        print(f"  truncated to top {args.top_k} by confidence")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoints": [str(c) for c in args.checkpoints],
        "weights": weights,
        "target_class": args.target,
        "threshold": args.threshold,
        "max_iou_for_fp": args.max_iou_for_fp,
        "datasets": list(datasets),
        "n_fps": len(fps),
        "fps": [
            {"dataset": d.dataset, "filename": d.filename,
             "start_s": float(d.start_s), "end_s": float(d.end_s),
             "confidence": float(d.confidence),
             "target_class": args.target}
            for d in fps
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {len(fps)} hard negatives → {out_path}")
    if fps:
        confs = np.array([d.confidence for d in fps])
        print(f"  confidence range: {confs.min():.3f}–{confs.max():.3f} "
              f"(median {np.median(confs):.3f})")

    # ------------------------------------------------------------------
    # Wandb finalize: stamp summary metrics + verdict + upload the JSON
    # as a typed artifact so phase 6 fine-tuning runs can pick it up
    # via ``run.use_artifact`` for true lineage. The artifact is
    # aliased with target class + source phase so the consumer can
    # request ``hardnegs:target_d_source_5:latest`` etc.
    # ------------------------------------------------------------------
    if run is not None:
        n_total_proposals = len(pred_events)
        n_gt = len(gt)
        n_fps = len(fps)
        confs = np.array([d.confidence for d in fps]) if fps else np.array([])

        summary = {
            "n_proposals":       n_total_proposals,
            "n_gt_target_class": n_gt,
            "n_fps_raw":         "computed_in_loop",  # placeholder
            "n_fps_kept":        n_fps,
            "fp_rate":           n_total_proposals and (n_fps / n_total_proposals),
        }
        if confs.size > 0:
            summary.update({
                "fp_confidence_min":    float(confs.min()),
                "fp_confidence_max":    float(confs.max()),
                "fp_confidence_median": float(np.median(confs)),
                "fp_confidence_mean":   float(confs.mean()),
            })

        # Verdict — short, comparable across mining runs at a glance.
        verdict = (
            f"Mined {n_fps} {args.target}-class FPs (top_k={args.top_k}) "
            f"from {len(args.checkpoints)} checkpoint(s) at threshold "
            f"{args.threshold}, max_iou_for_fp={args.max_iou_for_fp}. "
            f"GT events of this class on the mining datasets: {n_gt}."
        )

        wbu.finalize_eval_phase(
            summary,
            verdict=verdict,
            artifact_path=out_path,
            artifact_type="hardnegs",
            artifact_metadata={
                "target":          args.target,
                "threshold":       args.threshold,
                "max_iou_for_fp":  args.max_iou_for_fp,
                "n_fps":           n_fps,
                "n_checkpoints":   len(args.checkpoints),
                "source_phase":    run.config.get("source_phase"),
                "checkpoints":     [str(c) for c in args.checkpoints],
            },
        )


if __name__ == "__main__":
    main()