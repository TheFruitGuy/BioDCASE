"""
aggregate_numbers.py — base / HNM / MT / FlatMatch as one paper-ready ladder
============================================================================

Adapted from aggregate_mt.py. Scores every *finished* run from its saved val
posteriors (`best_posteriors.npz` / `paper_best_posteriors.npz`) and prints the
base -> +HNM -> +MT -> +FlatMatch ladder, with FlatMatch split into its three
variants:

  FM-vanilla    pure cross-sharpness            (fmfb_* ... _e / _full, no fixlabel)
  FM-fixlabel   Fix-label cross-sharpness       (fmfb_* ... _fixlabel)
  FM-stack      MT + Fix-label cross-sharpness  (fmfb_* ... _fixlabel_mtstack)

Runs that have not finished yet (no posteriors on disk) are reported as MISSING
and left out of every number, so this is safe to run in the middle of a sweep —
re-run it as seeds land and the ladder fills in.

Methodology is identical to aggregate_mt.py / ensemble_blockE.py: load the
cached file-level 3-class posteriors, re-tune per-class thresholds with the same
`_final` sweep, score paper-macro F1 + per-class P/R/F1. No GPU, no inference —
every number comes from the exact posteriors each run produced, so all stages
are directly comparable.

NOTE on the leak: globs only match direct children of --runs-root, so anything
you moved into runs/_contaminated_aadc_leak_*/ is correctly ignored. For the
FM-vs-MT deltas to be meaningful, the MT runs in runs/ must be the CLEAN redo —
if they're still the leaked ones, archive them and MT shows up as MISSING until
you rerun it clean.

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task
    python aggregate_numbers.py
    python aggregate_numbers.py --csv runs/all_summary.csv --seeds 42 2024 7777 9999 1337
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
DEFAULT_SEEDS = [42, 2024, 7777, 9999, 1337]

# Ladder order. base/HNM/MT use a fixed glob; the three FM variants are split
# out of a single fmfb_* glob by name (see fm_variant).
FM_VARIANTS = ["FM-vanilla", "FM-fixlabel", "FM-stack"]


# ---------------------------------------------------------------------
# posteriors + scoring (same path as aggregate_mt.py / ensemble_blockE.py)
# ---------------------------------------------------------------------

def load_member(run_dir: Path):
    """{'ds|fn': (T,3) float32} from a run's saved posteriors, or None if unfinished."""
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


def fm_variant(name: str) -> str:
    """Classify a fmfb_* run by name."""
    if "mtstack" in name:
        return "FM-stack"
    if "fixlabel" in name:
        return "FM-fixlabel"
    return "FM-vanilla"


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


def fin(rows):
    """Finished (scorable) rows of a stage."""
    return [r for r in rows if r.get("finished")]


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs", type=Path)
    p.add_argument("--base-glob", default="final_3c_s*")
    p.add_argument("--hnm-glob", default="hnm_3class_s*")
    p.add_argument("--mt-glob", default="mtfb_3class_s*")
    p.add_argument("--fm-glob", default="fmfb_*")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seeds expected per stage; absent ones are reported MISSING.")
    p.add_argument("--val-workers", type=int, default=13)
    p.add_argument("--csv", default=None, help="Write the full per-model table here.")
    return p.parse_args()


def main():
    args = parse_args()
    rr = args.runs_root
    seeds = args.seeds

    print("Building val ground-truth events ...")
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt = build_gt_events(val_anns, val_fsd)

    def discover(glob_pat, stage_tag=None):
        out = []
        for d in sorted(rr.glob(glob_pat)):
            if not d.is_dir():
                continue
            m = load_member(d)
            out.append({"name": d.name, "seed": seed_of(d.name), "src": src_of(d.name),
                        "members": m, "finished": m is not None, "stage": stage_tag})
        return out

    print("Discovering runs (base / HNM / MT / FlatMatch) ...")
    base = discover(args.base_glob, "base")
    hnm = discover(args.hnm_glob, "HNM")
    mt = discover(args.mt_glob, "MT")
    fm_all = discover(args.fm_glob)
    fm = {v: [] for v in FM_VARIANTS}
    for r in fm_all:
        r["stage"] = fm_variant(r["name"])
        fm[r["stage"]].append(r)

    stages = [("base", base), ("HNM", hnm), ("MT", mt)] + \
             [(v, fm[v]) for v in FM_VARIANTS]

    # ---- score finished runs only ----------------------------------
    for _tag, rows in stages:
        for r in fin(rows):
            mp, pc, thr = score([r["members"]], gt, args.val_workers)
            r["f1"], r["pc"], r["thr"] = mp, pc, thr

    # ---- run status (finished / in-progress / MISSING) -------------
    print("\n" + "=" * 96)
    print("RUN STATUS  (MISSING = no posteriors on disk yet; left out of all numbers)")
    print("=" * 96)
    for tag, rows in stages:
        done = fin(rows)
        seeds_present = {r["seed"] for r in rows}            # dir exists (any state)
        seeds_done = sorted({r["seed"] for r in done})
        in_prog = [r["name"] for r in rows if not r["finished"]]
        missing = [s for s in seeds if s not in seeds_present]
        line = f"  {tag:12} {len(done):>2} done"
        if seeds_done:
            line += f"  seeds={seeds_done}"
        if in_prog:
            line += f"  | in-progress: {', '.join(in_prog)}"
        if missing:
            line += f"  | MISSING: {', '.join('s'+str(s) for s in missing)}"
        print(line)

    base_by_seed = {r["seed"]: r for r in fin(base)}
    mt_by = {(r["seed"], r["src"]): r for r in fin(mt)}

    def matched_mt(r):
        return (mt_by.get((r["seed"], r["src"])) or mt_by.get((r["seed"], "ens"))
                or mt_by.get((r["seed"], "solo")))

    # ---- per-run table: the from-base SSL family (MT + FlatMatch) --
    ssl_runs = fin(mt) + [r for v in FM_VARIANTS for r in fin(fm[v])]
    if ssl_runs:
        print("\n" + "=" * 96)
        print("PER-RUN  (from-base SSL: MT + FlatMatch; Δ vs same-seed base / same-seed MT)")
        print("=" * 96)
        print(f"{'run':42} {'variant':12} {'paperF1':>8} {'bmabz':>6} {'d':>6} "
              f"{'bp':>6} {'Δbase':>8} {'ΔMT':>8}")
        for r in sorted(ssl_runs, key=lambda x: x["f1"], reverse=True):
            b = base_by_seed.get(r["seed"])
            db = f"{r['f1']-b['f1']:+.4f}" if b else "   --"
            if r["stage"] == "MT":
                dm = "   --"                       # MT is the reference
            else:
                mm = matched_mt(r)
                dm = f"{r['f1']-mm['f1']:+.4f}" if mm else "   --"
            pc = r["pc"]
            print(f"{r['name']:42} {r['stage']:12} {r['f1']:8.4f} "
                  f"{pc['bmabz']['f1']:6.3f} {pc['d']['f1']:6.3f} {pc['bp']['f1']:6.3f} "
                  f"{db:>8} {dm:>8}")

    # ---- ensembles per stage (finished runs only) ------------------
    ens_score = {}
    for tag, rows in stages:
        rfin = fin(rows)
        if rfin:
            ens_score[tag] = score([r["members"] for r in rfin], gt, args.val_workers)

    print("\n" + "=" * 96)
    print("STAGE LADDER  (per-class columns = uniform-ENSEMBLE per-class F1, finished runs)")
    print("=" * 96)
    print(f"{'stage':12} {'n':>3} {'single mean±std':>18} {'best':>7} "
          f"{'ensemble':>9} {'bmabz':>7} {'d':>7} {'bp':>7}")
    for tag, rows in stages:
        rfin = fin(rows)
        if not rfin:
            print(f"{tag:12} {'0':>3}  (none finished)")
            continue
        f1v = [r["f1"] for r in rfin]
        emp, epc, _ = ens_score[tag]
        print(f"{tag:12} {len(rfin):>3} {pm(f1v):>18} {max(f1v):7.4f} "
              f"{emp:9.4f} {epc['bmabz']['f1']:7.3f} {epc['d']['f1']:7.3f} "
              f"{epc['bp']['f1']:7.3f}")

    # ---- deltas ----------------------------------------------------
    def edelta(a, b):
        ea, eb = ens_score.get(a), ens_score.get(b)
        return f"{ea[0]-eb[0]:+.4f}" if (ea and eb) else "  --"

    print("\n" + "-" * 96)
    print("DELTAS  (ensemble paper-F1; only pairs where both stages have finished runs)")
    print("-" * 96)
    print(f"  HNM − base          {edelta('HNM','base')}")
    print(f"  MT  − base          {edelta('MT','base')}")
    print(f"  MT  − HNM           {edelta('MT','HNM')}")
    print(f"  FM-vanilla  − MT    {edelta('FM-vanilla','MT')}")
    print(f"  FM-fixlabel − MT    {edelta('FM-fixlabel','MT')}   <- does Fix-label beat MT?")
    print(f"  FM-stack    − MT    {edelta('FM-stack','MT')}   <- does the stack beat MT?")
    print(f"  FM-fixlabel − FM-vanilla   {edelta('FM-fixlabel','FM-vanilla')}")
    print(f"  FM-stack    − FM-fixlabel  {edelta('FM-stack','FM-fixlabel')}")

    # per-class ensemble progression for the key stages that exist
    prog = [t for t in ("base", "HNM", "MT", "FM-fixlabel", "FM-stack") if t in ens_score]
    if len(prog) >= 2:
        print("\n  per-class ensemble F1 across stages:")
        header = "    " + "class ".ljust(8) + "  ".join(f"{t:>12}" for t in prog)
        print(header)
        for c in C3:
            cells = "  ".join(f"{ens_score[t][1][c]['f1']:>12.3f}" for t in prog)
            print(f"    {c:6}   {cells}")
        # P/R for the best available FM stage
        for t in ("FM-stack", "FM-fixlabel"):
            if t in ens_score:
                print(f"\n  {t} ensemble per-class P/R:")
                for c in C3:
                    pc = ens_score[t][1][c]
                    print(f"    {c:6} P={pc['precision']:.3f}  R={pc['recall']:.3f}")
                break

    # ---- CSV -------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["stage", "run", "seed", "src", "paper_f1",
                        "bmabz_f1", "d_f1", "bp_f1",
                        "bmabz_p", "bmabz_r", "d_p", "d_r", "bp_p", "bp_r",
                        "thr_bmabz", "thr_d", "thr_bp"])
            for tag, rows in stages:
                for r in sorted(fin(rows), key=lambda x: x["f1"], reverse=True):
                    pc = r["pc"]
                    w.writerow([tag, r["name"], r["seed"], r["src"], f"{r['f1']:.4f}"]
                               + [f"{pc[c]['f1']:.4f}" for c in C3]
                               + [f"{pc[c][k]:.4f}" for c in C3 for k in ("precision", "recall")]
                               + [f"{t:.3f}" for t in r["thr"]])
            for tag, _rows in stages:
                if tag in ens_score:
                    mp, pc, thr = ens_score[tag]
                    w.writerow([tag, f"{tag}_ENSEMBLE", "", "uniform", f"{mp:.4f}"]
                               + [f"{pc[c]['f1']:.4f}" for c in C3]
                               + [f"{pc[c][k]:.4f}" for c in C3 for k in ("precision", "recall")]
                               + [f"{t:.3f}" for t in thr])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
