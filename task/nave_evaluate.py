"""
NAVE single-checkpoint evaluation.
==================================
Loads a NAVE checkpoint, runs one inference pass over the validation sites,
tunes per-class thresholds and reports the macro F1 (the official challenge
metric, the mean of the three per-class F1s).

    CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/nave_s42_*/nave_best.pt --workers 13

Self-contained
--------------
Depends only on the other NAVE files (``nave_config`` / ``nave_features`` /
``nave_model`` / ``nave_train``). The val-set loader, post-processing, metrics
and per-class threshold tuner are imported from ``nave_train`` (where they are
inlined). Anything not in those four files is third-party (``torch``, ``numpy``).
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_model import NAVE
from nave_features import NAVEFeatureExtractor
from nave_train import (
    CLASS_NAMES,
    Detection,
    WhaleDataset,
    build_val_segments,
    collapse_probs_to_3class,
    collate_fn,
    compute_metrics,
    get_file_manifest,
    load_annotations,
    postprocess_predictions,
    tune_thresholds_per_class,
)


# ----------------------------------------------------------------------
# Checkpoint loader
# ----------------------------------------------------------------------

def build_nave_from_ckpt(path, device):
    """Load a NAVE checkpoint, restoring its ablation config FIRST so the
    rebuilt architecture matches (strict load). Old checkpoints without a
    stored config default to the full NAVE recipe. Returns (model, stored_thr)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    conf = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    cfg.apply_ablation(
        use_pcen=conf.get("use_pcen", True),
        use_fdy=conf.get("use_fdy", True),
        conv_kernel=conf.get("conv_kernel", cfg.CONV_KERNEL),
    )
    model = NAVE()
    sd = (ckpt["model_state_dict"]
          if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.load_state_dict(sd, strict=True)
    stored_thr = ckpt.get("thresholds") if isinstance(ckpt, dict) else None
    return model.to(device).eval(), stored_thr


# ----------------------------------------------------------------------
# Validation-set inference (one forward pass over the val sites)
# ----------------------------------------------------------------------

@torch.no_grad()
def collect_val_probs(model, spec_extractor, device, use_fp16: bool = False,
                      segment_s: float = cfg.EVAL_SEGMENT_S):
    """Reproduce the val-set probability collection from training. Returns
    ``(probs3, gt_events)`` where ``probs3`` is a dict keyed by
    ``(dataset, filename, start_sample)`` with ``(n_frames, 3)`` float arrays
    and ``gt_events`` is a list of :class:`Detection` ground truths."""
    val_sites = list(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations, segment_s=segment_s)
    loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}

    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_fp16 and device.type == "cuda") else nullcontext())
    hop = spec_extractor.hop_length
    raw: dict = {}
    for audio, _t, _m, metas in loader:
        audio = audio.to(device)
        with autocast:
            logits = model(spec_extractor(audio))
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            raw[key] = probs[j, :n_frames, :]

    probs3 = collapse_probs_to_3class(raw)             # no-op for the 3-class NAVE head

    gt_events: list[Detection] = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return probs3, gt_events


# ----------------------------------------------------------------------
# Threshold application + macro F1 reductions
# ----------------------------------------------------------------------

def evaluate_with_thresholds(probs, gt_events, thresholds):
    """Apply per-class thresholds and compute the event-level metrics dict."""
    preds = postprocess_predictions(probs, np.asarray(thresholds, dtype=np.float64))
    return compute_metrics(preds, gt_events, iou_threshold=cfg.IOU_THRESHOLD)


def f1(metrics) -> float:
    """Macro F1 (the official challenge metric): mean of the three per-class F1s."""
    return float(np.mean([metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES]))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", type=Path, help="NAVE checkpoint.")
    p.add_argument("--workers", type=int, default=8, help="Threshold-tuner workers.")
    p.add_argument("--fp16", action="store_true", help="fp16 inference.")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Eval tile length in seconds (use 60 to match the paper).")
    p.add_argument("--out", type=Path, default=None,
                   help="Write tuned thresholds JSON here.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _stored = build_nave_from_ckpt(args.ckpt, device)
    spec = NAVEFeatureExtractor().to(device)
    print(f"[NAVE eval] {args.ckpt}  "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  ablation: {cfg.ablation_tag()}  PCEN={cfg.USE_PCEN}  FDY={cfg.USE_FDY}"
          f"  | segment={args.segment_s:.0f}s")

    probs, gt = collect_val_probs(model, spec, device, args.fp16,
                                  segment_s=args.segment_s)
    thr = tune_thresholds_per_class(probs, gt, workers=args.workers)
    metrics = evaluate_with_thresholds(probs, gt, thr)

    names = CLASS_NAMES
    macro_a = float(np.mean([metrics.get(c, {}).get("f1", 0.0) for c in names]))
    p_bar = float(np.mean([metrics.get(c, {}).get("precision", 0.0) for c in names]))
    r_bar = float(np.mean([metrics.get(c, {}).get("recall", 0.0) for c in names]))
    macro_b = 2 * p_bar * r_bar / (p_bar + r_bar + 1e-8)

    print(f"\n  thresholds: {[round(float(t), 3) for t in thr]}")
    print(f"  {'class':6} {'P':>7} {'R':>7} {'F1':>7}")
    for c in names:
        m = metrics.get(c, {})
        print(f"  {c.upper():6} {m.get('precision', 0.0):7.3f} "
              f"{m.get('recall', 0.0):7.3f} {m.get('f1', 0.0):7.3f}")
    print(f"  macro F1 (mean per-class, A) = {macro_a:.4f}")
    print(f"  official B (F1 of mean P, R) = {macro_b:.4f}")

    if args.out:
        args.out.write_text(json.dumps(
            {n: float(t) for n, t in zip(CLASS_NAMES, thr)}, indent=2))
        print(f"  thresholds -> {args.out}")


if __name__ == "__main__":
    main()
