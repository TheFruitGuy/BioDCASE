"""
check_d_pr.py — is D recall-limited or precision-limited?  (Block E pre-flight)
==============================================================================

Loads one or more ``paper_best.pt`` checkpoints, runs the EXACT same validation
path the trainers use (``train_hnm_final.validate`` -> rescore's tuned paper-F1),
and prints the per-class precision / recall / F1 with **D** called out, plus a
verdict on whether D is recall- or precision-limited.

Why this matters for Block E
----------------------------
SAT lowers D's consistency gate on the premise that D is RECALL-limited — it
misses real calls and a lower gate lets the teacher's weak-but-real D positives
through. If the base D is instead PRECISION-limited (firing on noise), lowering
the gate makes it worse, and you'd lean on the DA cap + verifier-gate or skip
SAT for D. This is the one number to check before committing the E1 matrix.

The metric is identical to A/B/C selection (tuned per-class thresholds, paper
convention), so the numbers are directly comparable to the run tables. ``use_focal``
only affects the printed loss, not P/R, so it is off by default (matches the
base checkpoints).

Usage
-----
::

    # one checkpoint
    python check_d_pr.py --checkpoint runs/final_3c_s42_20260527_200054/paper_best.pt

    # all five 3-class seeds at once (shell expands the glob) -> per-seed rows + mean
    python check_d_pr.py --checkpoint runs/3class_seed*/paper_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config_final as cfg
from dataset_final import (
    WhaleDataset, collate_fn, get_file_manifest, load_annotations,
    build_val_segments,
)
# Pull the proven primitives straight from the trainer so the metric is byte
# -identical to checkpoint selection. build_model / build_gt_events are
# re-exported by train_hnm_final from rescore_base_epochs.
from train_hnm_final import validate, build_model, build_gt_events


def d_verdict(p: float, r: float) -> str:
    """One-line read on D's failure mode (the SAT premise check)."""
    if r < 0.05 and p < 0.05:
        return ("DEGENERATE (P and R ~0) — D is collapsed; fix base D first, "
                "no consistency lever recovers it")
    if r < 0.8 * p:
        return ("RECALL-limited (R << P) -> SAT is the right lever: lower D's "
                "gate to admit the teacher's missed D positives")
    if p < 0.8 * r:
        return ("PRECISION-limited (P << R) -> SAT would likely HURT D; rely on "
                "the DA cap + verifier-gate, or drop D from --gate/SAT")
    return ("BALANCED (P ~ R) -> modest SAT upside; keep DA on and watch D "
            "precision as the gate drops")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="One or more paper_best.pt files (globs ok via the shell).")
    p.add_argument("--val-workers", type=int, default=13,
                   help="Fork workers for the per-class threshold sweep.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--focal", action="store_true",
                   help="Use focal loss for the printed val loss only (does NOT "
                        "affect P/R). Off matches the base checkpoints.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | {len(args.checkpoint)} checkpoint(s)\n")

    # ---- val data built ONCE (flag-independent); GT is always 3-class -----
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_segments = build_val_segments(val_manifest, val_anns)
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)   # always 3-class GT (flag-independent)

    rows = []   # (name, head, paper_f1, thresholds, per_class)
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
        is_3class = (n_classes == 3)
        head = "3class" if is_3class else "7class"

        # WhaleDataset bakes target WIDTH (cfg.n_classes()) and the class map at
        # CONSTRUCTION time, so it must be (re)built AFTER the flag is set for
        # this checkpoint's head — otherwise a 3-class model gets 7-wide targets.
        cfg.USE_3CLASS = is_3class
        val_ds = WhaleDataset(val_segments)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn,
                                pin_memory=True)

        model, spec = build_model(n_classes, device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        pos_weight = (torch.tensor(ckpt["pos_weight"], dtype=torch.float32, device=device)
                      if ckpt.get("pos_weight") is not None else None)

        v = validate(model, spec, val_loader, device, gt_events, pos_weight,
                     n_classes, is_3class, args.val_workers, args.focal)

        name = Path(ckpt_path).parent.name
        rows.append((name, head, v["paper_f1"], v["thresholds"], v["per_class"]))

        print(f"{'='*64}")
        print(f"{name}  [{head}]   paper-F1 = {v['paper_f1']:.4f}")
        print(f"  tuned thresholds {['%.2f' % t for t in v['thresholds']]}  "
              f"({'/'.join(cfg.CALL_TYPES_3)})")
        print(f"  {'class':6} {'P':>7} {'R':>7} {'F1':>7}")
        for c in cfg.CALL_TYPES_3:
            m = v["per_class"].get(c, {})
            flag = "  <- D" if c == "d" else ""
            print(f"  {c.upper():6} {m.get('precision',0):7.3f} "
                  f"{m.get('recall',0):7.3f} {m.get('f1',0):7.3f}{flag}")
        dm = v["per_class"].get("d", {})
        print(f"  D verdict: {d_verdict(dm.get('precision',0), dm.get('recall',0))}")

    # ---- aggregate across checkpoints (the decision is on the mean) -------
    if len(rows) > 1:
        print(f"\n{'='*64}\nAGGREGATE over {len(rows)} checkpoints (mean ± std)")
        for c in cfg.CALL_TYPES_3:
            P = np.array([r[4].get(c, {}).get("precision", 0.0) for r in rows])
            R = np.array([r[4].get(c, {}).get("recall", 0.0) for r in rows])
            F = np.array([r[4].get(c, {}).get("f1", 0.0) for r in rows])
            flag = "  <- D" if c == "d" else ""
            print(f"  {c.upper():6} P={P.mean():.3f}±{P.std():.3f}  "
                  f"R={R.mean():.3f}±{R.std():.3f}  F1={F.mean():.3f}±{F.std():.3f}{flag}")
        Pd = np.array([r[4].get('d', {}).get('precision', 0.0) for r in rows]).mean()
        Rd = np.array([r[4].get('d', {}).get('recall', 0.0) for r in rows]).mean()
        pf = np.array([r[2] for r in rows])
        print(f"  paper-F1 {pf.mean():.4f} ± {pf.std():.4f}")
        print(f"  D verdict (mean): {d_verdict(Pd, Rd)}")


if __name__ == "__main__":
    main()
