"""
NAVE single-checkpoint evaluation.
==================================
Loads a NAVE checkpoint (native, or legacy ``train_phase13r``), runs one
inference pass over the validation sites, tunes per-class thresholds and reports
per-class P/R/F1 + macro F1 (the official metric) and macro_paper.

    CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/nave_s42_*/nave_best.pt --workers 13
    CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/phase13r_3c_s42_*/phase13r_best.pt

Reuses the verified probability collector + tuner from ``eval_conformer`` (it
takes any (model, spec) pair, so NAVE plugs straight in).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import nave_config as cfg
from nave_model import NAVE
from nave_features import NAVEFeatureExtractor
from eval_conformer import (
    collect_val_probs, tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)


def build_nave_from_ckpt(path, device):
    """Auto-detect legacy vs native NAVE checkpoint and return (model, stored_thr)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_legacy = any(k.startswith("_inner.") for k in sd)
    if is_legacy:
        model, meta = NAVE.from_legacy_checkpoint(path, map_location=device)
        stored_thr = meta.get("thresholds")
    else:
        model = NAVE()
        model.load_checkpoint(path, map_location=device)
        stored_thr = ckpt.get("thresholds") if isinstance(ckpt, dict) else None
    return model.to(device).eval(), stored_thr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", type=Path, help="NAVE checkpoint (native or legacy).")
    p.add_argument("--workers", type=int, default=8, help="Threshold-tuner workers.")
    p.add_argument("--fp16", action="store_true", help="fp16 inference.")
    p.add_argument("--out", type=Path, default=None, help="Write tuned thresholds JSON here.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _stored = build_nave_from_ckpt(args.ckpt, device)
    spec = NAVEFeatureExtractor().to(device)
    print(f"[NAVE eval] {args.ckpt}  ({sum(p.numel() for p in model.parameters()):,} params)")

    probs, gt = collect_val_probs(model, spec, device, args.fp16)
    thr = tune_thresholds_per_class(probs, gt, workers=args.workers)
    metrics = evaluate_with_thresholds(probs, gt, thr)

    print(f"\n  thresholds: {[round(float(t), 3) for t in thr]}")
    for name in CLASS_NAMES:
        m = metrics.get(name, {})
        print(f"    {name.upper():6} P={m.get('precision',0):.3f} "
              f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")
    print(f"\n  MACRO F1 (official) = {macro_f1(metrics):.4f}")
    print(f"  macro_paper         = {macro_f1_paper(metrics):.4f}")

    if args.out:
        args.out.write_text(json.dumps(
            {n: float(t) for n, t in zip(CLASS_NAMES, thr)}, indent=2))
        print(f"  thresholds -> {args.out}")


if __name__ == "__main__":
    main()
