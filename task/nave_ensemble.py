"""
NAVE multi-seed ensemble (probability averaging).
=================================================
Averages per-frame probabilities across NAVE checkpoints (native and/or legacy
``train_phase13r``), tunes per-class thresholds on the averaged probabilities,
and reports the ensemble macro F1 plus each member's tuned macro and the lift
over the best member. Gains are diversity-driven and mostly land on D
(consensus suppresses uncorrelated per-seed false positives).

    CUDA_VISIBLE_DEVICES=0 python nave_ensemble.py \
        runs/nave_s42_*/nave_best.pt runs/nave_s2024_*/nave_best.pt \
        runs/nave_s7777_*/nave_best.pt --workers 13

Mixing legacy and native checkpoints is fine; the architecture is identical, so
the segment keys and frame counts line up exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from eval_conformer import (
    collect_val_probs, tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)
from ensemble_conformer import average_probs            # verified, unit-tested
from nave_evaluate import build_nave_from_ckpt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", type=Path, nargs="+", help="Two or more NAVE checkpoints.")
    p.add_argument("--workers", type=int, default=8, help="Threshold-tuner workers.")
    p.add_argument("--fp16", action="store_true", help="fp16 inference.")
    p.add_argument("--out", type=Path, default=None, help="Write ensemble thresholds JSON.")
    return p.parse_args()


def _tuned_macro(probs, gt, workers):
    thr = tune_thresholds_per_class(probs, gt, workers=workers)
    return macro_f1(evaluate_with_thresholds(probs, gt, thr)), thr


def main():
    args = parse_args()
    if len(args.ckpts) < 2:
        print("[NAVE ensemble] need >= 2 checkpoints; use nave_evaluate.py for one.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = NAVEFeatureExtractor().to(device)

    member_probs, gt = [], None
    member_macros = []
    for ckpt in args.ckpts:
        model, _ = build_nave_from_ckpt(ckpt, device)
        probs, gt_i = collect_val_probs(model, spec, device, args.fp16)
        gt = gt_i if gt is None else gt
        member_probs.append(probs)
        m, _ = _tuned_macro(probs, gt_i, args.workers)
        member_macros.append(m)
        print(f"  member {ckpt.parent.name:>28}  tuned macro {m:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg = average_probs(member_probs)
    ens_thr = tune_thresholds_per_class(avg, gt, workers=args.workers)
    metrics = evaluate_with_thresholds(avg, gt, ens_thr)
    ens_macro = macro_f1(metrics)

    print(f"\n  ensemble ({len(args.ckpts)} members)  thresholds {[round(float(t),3) for t in ens_thr]}")
    for name in CLASS_NAMES:
        m = metrics.get(name, {})
        print(f"    {name.upper():6} P={m.get('precision',0):.3f} "
              f"R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f}")
    best_member = max(member_macros)
    print(f"\n  ENSEMBLE MACRO F1 = {ens_macro:.4f}   "
          f"(best member {best_member:.4f}, lift {ens_macro-best_member:+.4f})")
    print(f"  macro_paper       = {macro_f1_paper(metrics):.4f}")

    if args.out:
        import json
        args.out.write_text(json.dumps(
            {n: float(t) for n, t in zip(CLASS_NAMES, ens_thr)}, indent=2))
        print(f"  thresholds -> {args.out}")


if __name__ == "__main__":
    main()
