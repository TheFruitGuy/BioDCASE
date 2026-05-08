"""
Eval-only helper for the stage-2 verifier.
==========================================

Loads a verifier checkpoint and runs the same per-class binary F1
evaluation that ``train_verifier.py`` does at the end of training, but
on whatever candidates parquet you point it at. Use it to:

  - Compare different checkpoints on the same eval set
    (e.g. v2 trained on val-80%, v3 trained on train, both scored on
    the full val parquet)
  - Sanity-check generalization to a held-out parquet without re-training
  - Run inference for downstream end-to-end event-level evaluation
    (the printed scores are also saved as a pickle for later)

Usage
-----
::

    python eval_verifier.py \\
        --checkpoint runs_verifier/v2_seed42/best.pt \\
        --candidates candidates_val.parquet \\
        --output_dir runs_verifier/eval/v2_on_full_val
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset_verifier import (
    VerifierDataset, load_candidates, verifier_collate_fn,
)
from model_verifier import WhaleVerifier, count_parameters
from spectrogram import SpectrogramExtractor
# Reuse the trainer's metric helpers so eval and training agree exactly.
from train_verifier import (
    collect_val_scores, per_class_summary, print_summary,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a verifier best.pt file.")
    p.add_argument("--candidates", type=str, required=True,
                   help="Candidates parquet to evaluate on.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--crop_s", type=float, default=None,
                   help="Override crop length. Defaults to whatever was "
                        "stored in the checkpoint args.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load checkpoint and recover the model config.
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    crop_s = args.crop_s or train_args.get("crop_s", 15.0)
    backbone_dropout = train_args.get("backbone_dropout", 0.5)
    head_dropout = train_args.get("head_dropout", 0.3)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  trained for {ckpt.get('epoch', '?')} epochs")
    print(f"  crop_s={crop_s}  "
          f"backbone_dropout={backbone_dropout}  "
          f"head_dropout={head_dropout}")
    if "macro_combined_f1" in ckpt:
        print(f"  reported best macro F1 = {ckpt['macro_combined_f1']:.4f}")

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
    # Load candidates and run forward pass.
    # ------------------------------------------------------------------
    print(f"\nCandidates: {args.candidates}")
    records = load_candidates(args.candidates)
    print(f"  {len(records)} candidates total")

    # Per-class TP/FP printout for the eval set.
    for cls_name in cfg.CALL_TYPES_3:
        cls_idx = cfg.CALL_TYPES_3.index(cls_name)
        sub = [r for r in records if r.class_idx == cls_idx]
        n_tp = sum(1 for r in sub if r.label == 1)
        n_fp = sum(1 for r in sub if r.label == 0)
        print(f"  {cls_name}: TP={n_tp:5d}  FP={n_fp:5d}")

    eval_ds = VerifierDataset(records, crop_s=crop_s, train=False)
    loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=verifier_collate_fn,
        pin_memory=True,
    )

    arrs = collect_val_scores(model, spec_extractor, loader, device)
    summary = per_class_summary(arrs)
    print_summary(summary, "\nEvaluation on the supplied candidates:")

    # ------------------------------------------------------------------
    # Persist results: JSON for the headline numbers, pickle of raw
    # scores for downstream end-to-end event-level evaluation.
    # ------------------------------------------------------------------
    def _jsonable(obj):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_dir / "summary.json", "w") as f:
        json.dump(_jsonable({
            "checkpoint": args.checkpoint,
            "candidates": args.candidates,
            "summary": summary,
        }), f, indent=2)

    # cand_id-aligned arrays so a downstream rescorer can join them with
    # the candidates parquet.
    cand_ids = np.array([m["cand_id"] for m in
                         [{"cand_id": r.cand_id} for r in records]],
                        dtype=np.int64)
    # Note: collect_val_scores returns arrays in dataloader iteration
    # order, which (with shuffle=False) matches dataset order — i.e.,
    # the order of `records`. We zip them to cand_ids accordingly.
    with open(out_dir / "scores.pkl", "wb") as f:
        pickle.dump({
            "cand_id":  cand_ids,
            "verifier": arrs["verifier"],
            "stage1":   arrs["stage1"],
            "label":    arrs["label"],
            "class":    arrs["class"],
        }, f)

    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'scores.pkl'}")


if __name__ == "__main__":
    main()
