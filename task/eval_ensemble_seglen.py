"""
Ensemble eval at multiple tile lengths (30s / 60s) — _final-native
==================================================================

Scores an ENSEMBLE of _final checkpoints (averaged per-segment 3-class
posteriors) at one or more eval-tile lengths, through the exact
rescore_base_epochs paper-F1 path that aggregate_numbers uses.

For each --seconds value:
  build_val_segments at that segment_s  ->  run each checkpoint
  ->  average the per-segment 3-class posteriors across members
  ->  tune thresholds + evaluate + macro paper-F1 + per-class P/R/F1/TP/FP/FN.

Averaging per-segment then scoring is linear-equivalent to stitch-then-average,
so the 30s number here should MATCH aggregate_numbers' FM-stack 30s row
(a built-in cross-check). 60s re-tiles the val files at 60s windows — which is
fresh inference the saved best_posteriors.npz (30s) can't give you.

Each checkpoint's model_state_dict is a plain WhaleVAD (for MT / stack runs that
is the EMA teacher; EMATeacher.state_dict() returns the inner model), so members
all load the same way as a base model. Models are built once and reused across
seg-lengths; the GT is seg-length-independent and built once.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 python eval_ensemble_seglen.py \\
        --checkpoints runs/fmfb_3class_s*_ens_fixlabel_mtstack/best_model.pt \\
        --seconds 30 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config_final as cfg
from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from rescore_base_epochs import (
    build_model, build_gt_events, collapse_to_3class,
    tune_thresholds, evaluate, paper_f1, per_class_block,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Ensemble members (averaged). Shell-glob OK, e.g. "
                        "runs/fmfb_3class_s*_ens_fixlabel_mtstack/best_model.pt")
    p.add_argument("--seconds", nargs="+", type=float, default=[30.0, 60.0],
                   help="Eval-tile lengths to score (default: 30 60).")
    p.add_argument("--overlap", type=float, default=cfg.EVAL_OVERLAP_S,
                   help=f"Tile overlap in seconds (default cfg.EVAL_OVERLAP_S="
                        f"{cfg.EVAL_OVERLAP_S}, matching the trainer's saved 30s "
                        f"posteriors so the 30s row cross-checks aggregate_numbers).")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                   help="Lower this for 60s tiles if you OOM (2x the activations of 30s).")
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--val-workers", type=int, default=13,
                   help="Threshold-sweep workers (CPU).")
    return p.parse_args()


@torch.no_grad()
def predict_probs3(model, spec, loader, device, n_classes, is_3class):
    """validate()'s inference path WITHOUT the per-model threshold sweep:
    per-segment sigmoid posteriors keyed by (dataset, filename, start_sample),
    then collapse to 3-class (leaves cfg.USE_3CLASS=True)."""
    cfg.USE_3CLASS = is_3class
    model.eval()
    hop = spec.hop_length
    probs = {}
    for audio, _targets, _mask, metas in loader:
        audio = audio.to(device, non_blocking=True)
        p = torch.sigmoid(model(spec(audio))).cpu().numpy()
        for j, m in enumerate(metas):
            nf = min((m["end_sample"] - m["start_sample"]) // hop, p[j].shape[0])
            probs[(m["dataset"], m["filename"], m["start_sample"])] = p[j, :nf, :]
    return collapse_to_3class(probs, n_classes)


def average_probs3(dicts):
    """Mean over members at matching per-segment keys (min-T aligned). All
    members run the SAME val tiling, so the key sets are identical."""
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    out = {}
    for k in keys:
        T = min(d[k].shape[0] for d in dicts)
        out[k] = np.mean(np.stack([d[k][:T] for d in dicts]), axis=0)
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [torch.load(c, map_location=device, weights_only=False)
             for c in args.checkpoints]
    n_classes = int(ckpts[0]["model_state_dict"]["classifier.weight"].shape[0])
    for c, ck in zip(args.checkpoints, ckpts):
        nc = int(ck["model_state_dict"]["classifier.weight"].shape[0])
        if nc != n_classes:
            raise SystemExit(f"{c} is {nc}-class but the first is {n_classes}-class.")
    is_3class = (n_classes == 3)
    print(f"Device: {device} | ensemble of {len(args.checkpoints)} | {n_classes}-class "
          f"| seconds={[f'{s:g}' for s in args.seconds]} | overlap={args.overlap}")
    for c in args.checkpoints:
        print(f"  - {c}")

    # build models once; reuse across all seg-lengths
    models, spec = [], None
    for ck in ckpts:
        m, sp = build_model(n_classes, device)
        if spec is None:
            spec = sp
        m.load_state_dict(ck["model_state_dict"], strict=False)
        m.eval()
        models.append(m)

    # val manifest/annotations + GT (seg-length independent) built once
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)

    results = {}
    for sec in args.seconds:
        val_segs = build_val_segments(val_manifest, val_anns,
                                      segment_s=sec, overlap_s=args.overlap)
        loader = DataLoader(WhaleDataset(val_segs), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True)
        t0 = time.time()
        per_model = [predict_probs3(m, spec, loader, device, n_classes, is_3class)
                     for m in models]
        avg = average_probs3(per_model)
        thr = tune_thresholds(avg, gt_events, args.val_workers)
        metrics = evaluate(avg, gt_events, thr)
        mf = paper_f1(metrics)
        pcb = per_class_block(metrics)
        results[sec] = (mf, thr, pcb, len(val_segs), time.time() - t0)

        print(f"\n{'='*66}\n{sec:g}s tiles | {len(val_segs)} segments | "
              f"ensemble macro paper-F1 = {mf:.4f}  ({results[sec][4]:.0f}s)\n{'='*66}")
        print("  thresholds: " + ", ".join(
            f"{c}={thr[i]:.3f}" for i, c in enumerate(cfg.CALL_TYPES_3)))
        for c in cfg.CALL_TYPES_3:
            b = pcb[c]
            print(f"  {c.upper():6} P={b['precision']:.3f} R={b['recall']:.3f} "
                  f"F1={b['f1']:.3f}  TP={b['tp']:4d} FP={b['fp']:4d} FN={b['fn']:4d}")

    if len(args.seconds) > 1:
        print(f"\n{'='*66}\nSEG-LENGTH COMPARISON (ensemble macro paper-F1)\n{'='*66}")
        for sec in args.seconds:
            mf, _, pcb, _, _ = results[sec]
            ds = " ".join(f"{c}={pcb[c]['f1']:.3f}" for c in cfg.CALL_TYPES_3)
            print(f"  {sec:g}s : {mf:.4f}   ({ds})")
        best = max(args.seconds, key=lambda s: results[s][0])
        print(f"  -> best: {best:g}s at {results[best][0]:.4f}")


if __name__ == "__main__":
    main()
