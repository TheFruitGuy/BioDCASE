"""
aggregate_mt.py — the from-base MT+HNM runs as a paper-ready comparison
=======================================================================

Scores the 10 from-base MT runs against the 5 base models and the 20 Block-B
HNM runs, all from their saved val posteriors (`best_posteriors.npz` /
`paper_best_posteriors.npz`), and prints the base → +HNM → +MT+HNM ladder with
per-class F1, per-model mean ± std, the uniform-ensemble of each stage, and the
+/- deltas. This is the "for the paper" overview, not the ensemble selector.

Methodology is identical to `ensemble_blockE.py`: load the cached file-level
3-class posteriors, re-tune per-class thresholds with the same `_final` sweep,
score paper-macro F1 + per-class P/R/F1. No GPU, no inference, no stack
mismatch — every number comes from the exact posteriors each run produced, so
base/HNM/MT are directly comparable.

Per-model deltas:
  Δbase = MT(seed)        − base(seed)
  ΔHNM  = MT(seed,source) − best HNM(seed,source)   (matched seed + mining source)

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task
    python aggregate_mt.py
    python aggregate_mt.py --csv runs/mt_summary.csv --val-workers 13
"""

from __future__ import annotations

import argparse
import csv as _csv
import re
import statistics
from pathlib import Path

import numpy as np

import config_final as cfg
cfg.USE_3CLASS = True

from dataset_final import load_annotations, get_file_manifest
from rescore_base_epochs import (
    build_gt_events, evaluate, paper_f1, per_class_block, tune_thresholds,
)

C3 = cfg.CALL_TYPES_3                       # ["bmabz", "d", "bp"]
SEEDS = [42, 2024, 7777, 9999, 1337]


# ---------------------------------------------------------------------
# posteriors + scoring (same path as ensemble_blockE.py)
# ---------------------------------------------------------------------

def load_member(run_dir: Path):
    """{'ds|fn': (T,3) float32} from a run's saved posteriors, or None."""
    for name in ("best_posteriors.npz", "paper_best_posteriors.npz"):
        p = run_dir / name
        if p.exists():
            npz = np.load(p)
            return {k: npz[k].astype(np.float32) for k in npz.files}
    return None


def rekey(members: list[dict]) -> dict:
    """Mean posteriors across members, re-keyed {(ds,fn,0): (T,3)} for the metric path."""
    keys = set(members[0])
    for m in members[1:]:
        keys &= set(m)
    out = {}
    for k in keys:
        arrs = [m[k] for m in members]
        L = min(a.shape[0] for a in arrs)
        ds, fn = k.split("|", 1)
        out[(ds, fn, 0)] = np.mean([a[:L] for a in arrs], axis=0).astype(np.float32)
    return out


def score(members: list[dict], gt_events, workers):
    """-> (paper_f1, per_class_dict, thresholds)."""
    probs = rekey(members)
    thr = tune_thresholds(probs, gt_events, workers)
    metrics = evaluate(probs, gt_events, thr)
    return paper_f1(metrics), per_class_block(metrics), np.asarray(thr, dtype=np.float64)


def seed_of(name: str):
    m = re.search(r"s(\d+)", name)
    return int(m.group(1)) if m else None


def src_of(name: str) -> str:
    return "ens" if "_ens" in name else ("solo" if "_solo" in name else "-")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def pm(vals) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "   --"
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mu:.4f} ± {sd:.4f}"


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def f1s(scored, key):
    """mean per-class F1 across a list of scored rows."""
    return {c: mean_or_none([r["pc"][c]["f1"] for r in scored]) for c in C3}


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs", type=Path)
    p.add_argument("--mt-glob", default="mtfb_3class_s*")
    p.add_argument("--base-glob", default="final_3c_s*")
    p.add_argument("--hnm-glob", default="hnm_3class_s*")
    p.add_argument("--val-workers", type=int, default=13)
    p.add_argument("--csv", default=None, help="Write the full per-model table here.")
    return p.parse_args()


def main():
    args = parse_args()
    rr = args.runs_root

    print("Building val ground-truth events ...")
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt = build_gt_events(val_anns, val_fsd)

    def discover(glob_pat):
        out = []
        for d in sorted(rr.glob(glob_pat)):
            if not d.is_dir():
                continue
            m = load_member(d)
            if m is None:
                print(f"  [skip] {d.name}: no posteriors")
                continue
            out.append({"name": d.name, "seed": seed_of(d.name),
                        "src": src_of(d.name), "members": m})
        return out

    print("Discovering + scoring runs (base / HNM / MT) ...")
    base = discover(args.base_glob)
    hnm = discover(args.hnm_glob)
    mt = discover(args.mt_glob)
    print(f"  base={len(base)}  hnm={len(hnm)}  mt={len(mt)}")
    if not mt:
        raise SystemExit("No MT runs found — check --mt-glob / --runs-root.")

    for r in base + hnm + mt:
        mp, pc, thr = score([r["members"]], gt, args.val_workers)
        r["f1"], r["pc"], r["thr"] = mp, pc, thr

    base_by_seed = {r["seed"]: r for r in base}

    def best_hnm(seed, src):
        cand = [r for r in hnm if r["seed"] == seed and r["src"] == src]
        if not cand:
            cand = [r for r in hnm if r["seed"] == seed]
        return max(cand, key=lambda r: r["f1"]) if cand else None

    # ---- per-MT-run table ------------------------------------------
    print("\n" + "=" * 92)
    print("PER-MT-RUN  (tuned paper-F1 + per-class F1; Δ vs same-seed base / matched HNM)")
    print("=" * 92)
    print(f"{'run':40} {'paperF1':>8} {'bmabz':>7} {'d':>7} {'bp':>7} {'Δbase':>8} {'ΔHNM':>8}")
    for r in sorted(mt, key=lambda x: x["f1"], reverse=True):
        b = base_by_seed.get(r["seed"])
        h = best_hnm(r["seed"], r["src"])
        db = f"{r['f1']-b['f1']:+.4f}" if b else "   --"
        dh = f"{r['f1']-h['f1']:+.4f}" if h else "   --"
        pc = r["pc"]
        print(f"{r['name']:40} {r['f1']:8.4f} {pc['bmabz']['f1']:7.3f} "
              f"{pc['d']['f1']:7.3f} {pc['bp']['f1']:7.3f} {db:>8} {dh:>8}")

    solo = [r["f1"] for r in mt if r["src"] == "solo"]
    ens = [r["f1"] for r in mt if r["src"] == "ens"]
    print(f"\n  MT mean paper-F1:  solo={mean_or_none(solo) or float('nan'):.4f}  "
          f"ens={mean_or_none(ens) or float('nan'):.4f}")

    # ---- stage ladder ----------------------------------------------
    stages = [("base", base), ("HNM", hnm), ("MT", mt)]
    ens_score = {}
    for tag, rows in stages:
        if rows:
            ens_score[tag] = score([r["members"] for r in rows], gt, args.val_workers)

    print("\n" + "=" * 92)
    print("STAGE LADDER  (per-class columns = uniform-ENSEMBLE per-class F1)")
    print("=" * 92)
    print(f"{'stage':6} {'n':>3} {'single mean±std':>18} {'best':>7} "
          f"{'ensemble':>9} {'bmabz':>7} {'d':>7} {'bp':>7}")
    for tag, rows in stages:
        if not rows:
            continue
        f1v = [r["f1"] for r in rows]
        emp, epc, _ = ens_score[tag]
        print(f"{tag:6} {len(rows):>3} {pm(f1v):>18} {max(f1v):7.4f} "
              f"{emp:9.4f} {epc['bmabz']['f1']:7.3f} {epc['d']['f1']:7.3f} "
              f"{epc['bp']['f1']:7.3f}")

    # ---- deltas ----------------------------------------------------
    def ens_f1(tag):
        return ens_score[tag][0] if tag in ens_score else None

    def single_mean(rows):
        return mean_or_none([r["f1"] for r in rows])

    bE, hE, mE = ens_f1("base"), ens_f1("HNM"), ens_f1("MT")
    bS, hS, mS = single_mean(base), single_mean(hnm), single_mean(mt)

    def d(a, b):
        return f"{a-b:+.4f}" if (a is not None and b is not None) else "  --"

    print("\n" + "-" * 92)
    print("DELTAS  (for the paper)")
    print("-" * 92)
    print(f"  ensemble paper-F1:    HNM−base {d(hE,bE)}   MT−base {d(mE,bE)}   MT−HNM {d(mE,hE)}")
    print(f"  single-model mean:    HNM−base {d(hS,bS)}   MT−base {d(mS,bS)}   MT−HNM {d(mS,hS)}")
    if all(t in ens_score for t in ("base", "HNM", "MT")):
        bpc, hpc, mpc = (ens_score[t][1] for t in ("base", "HNM", "MT"))
        print("  per-class ensemble F1  (base → HNM → MT, and MT−HNM):")
        for c in C3:
            print(f"    {c:6} {bpc[c]['f1']:.3f} → {hpc[c]['f1']:.3f} → {mpc[c]['f1']:.3f}"
                  f"   (HNM−base {hpc[c]['f1']-bpc[c]['f1']:+.3f}, "
                  f"MT−HNM {mpc[c]['f1']-hpc[c]['f1']:+.3f})")
        print("  MT ensemble per-class P/R:")
        for c in C3:
            print(f"    {c:6} P={mpc[c]['precision']:.3f}  R={mpc[c]['recall']:.3f}")

    # ---- CSV -------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["stage", "run", "seed", "src", "paper_f1",
                        "bmabz_f1", "d_f1", "bp_f1",
                        "bmabz_p", "bmabz_r", "d_p", "d_r", "bp_p", "bp_r",
                        "thr_bmabz", "thr_d", "thr_bp"])
            for tag, rows in stages:
                for r in sorted(rows, key=lambda x: x["f1"], reverse=True):
                    pc = r["pc"]
                    w.writerow([tag, r["name"], r["seed"], r["src"], f"{r['f1']:.4f}"]
                               + [f"{pc[c]['f1']:.4f}" for c in C3]
                               + [f"{pc[c][k]:.4f}" for c in C3 for k in ("precision", "recall")]
                               + [f"{t:.3f}" for t in r["thr"]])
            for tag in ("base", "HNM", "MT"):
                if tag in ens_score:
                    mp, pc, thr = ens_score[tag]
                    w.writerow([tag, f"{tag}_ENSEMBLE", "", "uniform", f"{mp:.4f}"]
                               + [f"{pc[c]['f1']:.4f}" for c in C3]
                               + [f"{pc[c][k]:.4f}" for c in C3 for k in ("precision", "recall")]
                               + [f"{t:.3f}" for t in thr])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
