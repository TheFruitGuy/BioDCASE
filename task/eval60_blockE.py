"""
eval60_blockE.py — re-run Block E inference at 60 s eval tiles
==============================================================

The cached Block E posteriors were produced at EVAL_SEGMENT_S = 30 s. This
script re-tiles the val set at ``--segment-s`` (default 60) and re-runs each
model's forward pass, then scores single-model and ensemble paper-F1 the same
way as the 30 s analysis so you can compare 30 s vs 60 s directly.

Why this needs a forward pass (unlike analyze/ensemble_blockE): tile length
changes the temporal context the CNN→BiLSTM sees, so the per-frame posteriors
differ — the 30 s ``best_posteriors.npz`` can't be reused. The model is
fully-convolutional + BiLSTM, so longer tiles are fine.

Per-model 60 s posteriors are cached to ``--cache-dir`` (keyed by run + tile
length), so the ensemble combos reuse each model's inference instead of
recomputing it. Metric is paper-F1 (F1 of mean-P/mean-R), same as the rest.

GPU note: run on the Ampere/Ada cards. The rk10 RTX PRO 6000 Blackwell (sm_120,
GPUs 2/7) crash the bio cu118 build at the LSTM — pin a good GPU with
CUDA_VISIBLE_DEVICES.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 python eval60_blockE.py                       # 60 s, default combos
    CUDA_VISIBLE_DEVICES=0 python eval60_blockE.py --segment-s 60 --combos A0 A2 A3 A0+A3
    CUDA_VISIBLE_DEVICES=0 python eval60_blockE.py --a0-pattern "whatever_s{seed}_..."
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from dataset_final import (
    WhaleDataset, collate_fn, get_file_manifest, load_annotations,
    build_val_segments,
)
from train_hnm_final import validate, build_model, build_gt_events
from rescore_base_epochs import evaluate, paper_f1, per_class_block, tune_thresholds

SEEDS = [42, 2024, 7777, 9999, 1337]
ARM_PATTERN = {
    "A0": "mtfb_3class_s{seed}_ens_asymmetric_mse",
    "A1": "mtE_A1_3class_s{seed}_ens",
    "A2": "mtE_A2_3class_s{seed}_ens",
    "A3": "mtE_A3_3class_s{seed}_ens",
    "A4": "mtE_A4_3class_s{seed}_ens",
}
ARM_DESC = {"A0": "baseline asym", "A1": "+DA", "A2": "SAT", "A3": "SAT+DA",
            "A4": "SAT+DA+gate"}


def infer_probs3(run_dir, val_segs, gt_events, device, val_workers,
                 segment_s, cache_dir, batch_size, num_workers):
    """Return segment-keyed 3-class probs for one run at the given tile length.
    Cached per (run, segment_s). Forward pass via the proven `validate` (also
    handles the USE_3CLASS fork protocol); we keep only its probs3."""
    cp = cache_dir / f"{run_dir.name}_probs_{int(segment_s)}s.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    ckpt_p = run_dir / "best_model.pt"
    if not ckpt_p.exists():
        return None
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
    is_3class = (n_classes == 3)

    cfg.USE_3CLASS = is_3class                      # workers fork with right flag
    loader = DataLoader(WhaleDataset(val_segs), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    model, spec = build_model(n_classes, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    pos_weight = (torch.tensor(ckpt["pos_weight"], dtype=torch.float32, device=device)
                  if ckpt.get("pos_weight") is not None else None)

    v = validate(model, spec, loader, device, gt_events, pos_weight,
                 n_classes, is_3class, val_workers, False)
    probs3 = v["probs3"]                            # {(ds,fn,start): (T,3)}, USE_3CLASS=True

    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump({k: vv.astype(np.float16) for k, vv in probs3.items()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: vv.astype(np.float16) for k, vv in probs3.items()}


def average_probs3(dicts: list[dict]) -> dict:
    """Per-segment mean across members (intersect keys, align to min T)."""
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    out = {}
    for k in keys:
        arrs = [d[k] for d in dicts]
        L = min(a.shape[0] for a in arrs)
        out[k] = np.mean([a[:L].astype(np.float32) for a in arrs], axis=0)
    return out


def score(probs3, gt_events, val_workers):
    cfg.USE_3CLASS = True
    thr = tune_thresholds(probs3, gt_events, val_workers)
    metrics = evaluate(probs3, gt_events, thr)
    return paper_f1(metrics), per_class_block(metrics), np.asarray(thr, dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--segment-s", type=float, default=60.0,
                   help="Eval tile length in seconds (default 60).")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--combos", nargs="+",
                   default=["A0", "A2", "A3", "A0+A3", "A0+A1+A2+A3"],
                   help="'+'-joined arm combos for the ensemble table.")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--a0-pattern", default=None, help="Override A0 dir pattern ({seed}).")
    p.add_argument("--cache-dir", default="runs/prob_cache_60s")
    p.add_argument("--val-workers", type=int, default=13)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.a0_pattern:
        ARM_PATTERN["A0"] = args.a0_pattern
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_root, cache_dir = Path(args.runs_root), Path(args.cache_dir)
    print(f"Device: {device} | eval tiles: {args.segment_s:.0f}s "
          f"(overlap {args.overlap_s:.0f}s) | cache: {cache_dir}")

    # ---- val set tiled at the requested length ----------------------
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_segs = build_val_segments(val_manifest, val_anns,
                                  segment_s=args.segment_s, overlap_s=args.overlap_s)
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)
    print(f"  {len(val_segs)} tiles @ {args.segment_s:.0f}s, {len(gt_events)} GT events")

    # ---- unique runs across all combos -> infer/caches probs3 -------
    arms_needed = sorted({a for combo in args.combos for a in combo.split("+")})
    probs_by_run: dict[tuple, dict] = {}        # (arm, seed) -> probs3
    for arm in arms_needed:
        for seed in args.seeds:
            d = runs_root / ARM_PATTERN[arm].format(seed=seed)
            pr = infer_probs3(d, val_segs, gt_events, device, args.val_workers,
                              args.segment_s, cache_dir, args.batch_size, args.num_workers)
            if pr is None:
                print(f"  [missing] {arm} s{seed}: {d}")
                continue
            probs_by_run[(arm, seed)] = pr
            print(f"  ready {arm} s{seed}  ({len(pr)} segment arrays)")

    # ---- single-model table (60 s) ----------------------------------
    print(f"\n{'='*78}\nPER-ARM @ {args.segment_s:.0f}s (single-model, mean ± std)\n{'='*78}")
    print(f"  {'arm':22} {'paperF1':>12} {'D P':>9} {'D R':>9} {'D F1':>9} "
          f"{'BMABZ':>8} {'BP':>8}")
    for arm in arms_needed:
        rows = [(s, probs_by_run[(arm, s)]) for s in args.seeds if (arm, s) in probs_by_run]
        if not rows:
            continue
        mps, dP, dR, dF, bF, pF = [], [], [], [], [], []
        for _, pr in rows:
            mp, pc, _ = score(pr, gt_events, args.val_workers)
            mps.append(mp); dP.append(pc["d"]["precision"]); dR.append(pc["d"]["recall"])
            dF.append(pc["d"]["f1"]); bF.append(pc["bmabz"]["f1"]); pF.append(pc["bp"]["f1"])
        a = lambda x: (np.mean(x), np.std(x))
        m = a(mps); p = a(dP); r = a(dR); f = a(dF); b = a(bF); q = a(pF)
        print(f"  {arm+' '+ARM_DESC[arm]:22} {m[0]:.4f}±{m[1]:.3f} {p[0]:.3f}±{p[1]:.2f} "
              f"{r[0]:.3f}±{r[1]:.2f} {f[0]:.3f}±{f[1]:.2f} {b[0]:.3f}±{b[1]:.2f} "
              f"{q[0]:.3f}±{q[1]:.2f}  (n={len(rows)})")

    # ---- ensemble combos (60 s) -------------------------------------
    print(f"\n{'='*78}\nENSEMBLE @ {args.segment_s:.0f}s (posterior-average)\n{'='*78}")
    print(f"  {'combo':16} {'mem':>4} {'paperF1':>9} {'D P':>7} {'D R':>7} "
          f"{'D F1':>7} {'BMABZ':>7} {'BP':>7} {'ΔpaperF1 vs A0':>15}")
    base_mp = None
    for combo in args.combos:
        members = [probs_by_run[(a, s)] for a in combo.split("+")
                   for s in args.seeds if (a, s) in probs_by_run]
        if not members:
            print(f"  {combo:16} (no members)")
            continue
        ens = average_probs3(members)
        mp, pc, thr = score(ens, gt_events, args.val_workers)
        if combo == "A0":
            base_mp = mp
        dM = f"{mp - base_mp:+.4f}" if base_mp is not None else "   —"
        print(f"  {combo:16} {len(members):>4} {mp:>9.4f} {pc['d']['precision']:>7.3f} "
              f"{pc['d']['recall']:>7.3f} {pc['d']['f1']:>7.3f} "
              f"{pc['bmabz']['f1']:>7.3f} {pc['bp']['f1']:>7.3f} {dM:>15}")

    print(f"\nCompare these to your 30 s numbers. If 60 s lifts the A0 ensemble "
          "paper-F1, it's a tiling win independent of Block E (apply it to the "
          "baseline too). Metric = paper-F1 (F1 of mean-P/mean-R).")


if __name__ == "__main__":
    main()
