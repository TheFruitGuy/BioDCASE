"""
Ensemble evaluator (post-hoc, no retraining) — 3-class paper-F1
===============================================================

Every run (base, Block-B HNM, from-base MT+HNM) cached its val posteriors as
``*_posteriors.npz`` (file-level, stitched, 3-class). This loads them, scores
each model on its own, then tries different ensemble groupings — uniform
averages of groups, top-K by individual F1, best-per-seed, and per-class
routing (the §8.2 lever that gave 0.490/0.504). Pure post-processing on cached
probabilities: a full sweep runs in minutes.

Scoring is byte-exact to the trainers. The saved arrays are already the
post-``stitch_segments`` per-file streams, so we run only the post-stitch tail
(smooth -> threshold_to_detections -> merge_and_filter), then compute_metrics +
paper_f1 — identical to what each trainer logged. So a 1-member "ensemble"
reproduces that run's aggregate-table number exactly, and groups are directly
comparable.

Reads (per run dir under --runs-root):
  base  final_3c_s*/paper_best_posteriors.npz
  hnm   hnm_3class_*/best_posteriors.npz
  mtfb  mtfb_3class_*/best_posteriors.npz

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task
    python ensemble_eval.py                 # per-model table + all default groups
    python ensemble_eval.py --topk 8 --route-topm 3 --csv ensemble_results.csv

Caveat: thresholds are tuned on val and groups are picked by val F1, so the
best group's number is optimistic (selection on a 3-site val set). Use it to
choose a direction; confirm the chosen ensemble on the held-out test set.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

import config_final as cfg
cfg.USE_3CLASS = True                       # the cached posteriors are 3-class

import postprocess_final as pp
from rescore_base_epochs import (
    THRESHOLD_GRIDS, IOU_THRESHOLD, paper_f1, build_gt_events,
)
from dataset_final import load_annotations, get_file_manifest

C3 = cfg.CALL_TYPES_3                        # ["bmabz", "d", "bp"]


# ---------------------------------------------------------------------
# posteriors I/O + combination
# ---------------------------------------------------------------------

def load_posteriors(npz_path: Path) -> dict:
    """{(ds, fn): (T, 3) float32} from a saved '<ds>|<fn>' -> array npz."""
    d = np.load(npz_path)
    out = {}
    for k in d.files:
        ds, fn = k.split("|", 1)
        out[(ds, fn)] = np.asarray(d[k], dtype=np.float32)
    return out


def common_keys(dicts: list[dict]) -> set:
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    return keys


def average(dicts: list[dict], weights=None) -> dict:
    """Uniform (or weighted) mean of per-file probs over a member list."""
    w = [1.0] * len(dicts) if weights is None else list(weights)
    sw = float(sum(w))
    out = {}
    for k in common_keys(dicts):
        T = min(d[k].shape[0] for d in dicts)
        acc = np.zeros((T, 3), dtype=np.float64)
        for wi, d in zip(w, dicts):
            acc += wi * d[k][:T]
        out[k] = (acc / sw).astype(np.float32)
    return out


def route_per_class(members_by_class: dict) -> dict:
    """Per-class column fusion: column c = mean of that class's selected members."""
    pools = [m for ms in members_by_class.values() for m in ms]
    keys = common_keys(pools)
    out = {}
    for k in keys:
        T = min(m[k].shape[0] for m in pools)
        col = np.zeros((T, 3), dtype=np.float64)
        for ci, ms in members_by_class.items():
            col[:, ci] = np.mean([m[k][:T, ci] for m in ms], axis=0)
        out[k] = col.astype(np.float32)
    return out


# ---------------------------------------------------------------------
# byte-exact post-stitch scoring (saved arrays are already stitched)
# ---------------------------------------------------------------------

def postprocess_file_level(file_probs: dict, thresholds: np.ndarray):
    """smooth -> threshold -> merge, exactly as postprocess_predictions does
    AFTER stitch_segments (which the cached arrays already went through)."""
    thr = np.asarray(thresholds, dtype=np.float64)
    dets = []
    for (ds, fn), probs in file_probs.items():
        probs = pp.smooth_probabilities(probs)
        dets.extend(pp.threshold_to_detections(probs, thr, ds, fn))
    return pp.merge_and_filter(dets)


def tune_thresholds(file_probs: dict, gt_events) -> np.ndarray:
    """Per-class coordinate descent on the same grids as rescore_base_epochs."""
    thr = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    for c in range(3):
        best_f1, best_t = -1.0, thr[c]
        for t in THRESHOLD_GRIDS[c]:
            trial = thr.copy(); trial[c] = float(t)
            m = pp.compute_metrics(postprocess_file_level(file_probs, trial),
                                   gt_events, iou_threshold=IOU_THRESHOLD)
            f1 = m.get(C3[c], {}).get("f1", 0.0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[c] = best_t
    return thr


def score(file_probs: dict, gt_events) -> dict:
    thr = tune_thresholds(file_probs, gt_events)
    m = pp.compute_metrics(postprocess_file_level(file_probs, thr),
                           gt_events, iou_threshold=IOU_THRESHOLD)
    return {"paper_f1": paper_f1(m), "thr": thr, "per_class": m}


# ---------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------

def _find_npz(run_dir: Path):
    for name in ("best_posteriors.npz", "paper_best_posteriors.npz"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _seed_of(name: str):
    for tok in name.split("_"):
        if tok.startswith("s") and tok[1:].isdigit():
            return int(tok[1:])
    return None


def discover(runs_root: Path):
    """-> list of {group, run, seed, src, posteriors}."""
    specs = [("base", "final_3c_s*"), ("hnm", "hnm_3class_*"), ("mtfb", "mtfb_3class_*")]
    runs = []
    for group, pat in specs:
        for d in sorted(glob.glob(str(runs_root / pat))):
            d = Path(d)
            if not d.is_dir():
                continue
            npz = _find_npz(d)
            if npz is None:
                print(f"  [skip] {d.name}: no posteriors npz")
                continue
            src = "ens" if "_ens" in d.name else ("solo" if "_solo" in d.name else "-")
            runs.append({"group": group, "run": d.name, "seed": _seed_of(d.name),
                         "src": src, "posteriors": load_posteriors(npz)})
    return runs


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs", type=Path)
    p.add_argument("--topk", type=int, default=8, help="top-K-by-F1 ensemble size(s).")
    p.add_argument("--route-topm", type=int, default=3,
                   help="members per class for the per-class routed ensemble.")
    p.add_argument("--baseline", type=float, default=0.504,
                   help="reference ensemble F1 to beat.")
    p.add_argument("--csv", default=None, help="write the group table to this CSV.")
    p.add_argument("--no-per-model", action="store_true",
                   help="skip the individual-model table (still needed for routing/top-K).")
    return p.parse_args()


def main():
    args = parse_args()

    print("Building val ground-truth events ...")
    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)

    print(f"Discovering runs under {args.runs_root}/ ...")
    runs = discover(args.runs_root)
    by_group = {}
    for r in runs:
        by_group.setdefault(r["group"], []).append(r)
    print(f"  base={len(by_group.get('base',[]))} "
          f"hnm={len(by_group.get('hnm',[]))} "
          f"mtfb={len(by_group.get('mtfb',[]))}  ({len(runs)} total)\n")
    if not runs:
        raise SystemExit("No runs with cached posteriors found.")

    # --- per-model scoring (also drives top-K / per-seed / routing) ---
    print("Scoring each model individually ...")
    for r in runs:
        s = score(r["posteriors"], gt_events)
        r["f1"] = s["paper_f1"]
        r["pc"] = {c: s["per_class"].get(c, {}) for c in C3}
    runs_sorted = sorted(runs, key=lambda x: x["f1"], reverse=True)

    if not args.no_per_model:
        print("\n" + "=" * 78 + "\nPER-MODEL  (paper-F1, per-class F1)\n" + "=" * 78)
        print(f"{'run':40} {'grp':5} {'F1':>7} {'bmabz':>7} {'d':>7} {'bp':>7}")
        for r in runs_sorted:
            pc = r["pc"]
            print(f"{r['run']:40} {r['group']:5} {r['f1']:7.4f} "
                  f"{pc['bmabz'].get('f1',0):7.3f} {pc['d'].get('f1',0):7.3f} "
                  f"{pc['bp'].get('f1',0):7.3f}")

    # --- ensemble groups -------------------------------------------------
    def members(group):
        return [r["posteriors"] for r in by_group.get(group, [])]

    sets = {}
    if by_group.get("base"):  sets["base_all"] = members("base")
    if by_group.get("hnm"):   sets["hnm_all"] = members("hnm")
    if by_group.get("mtfb"):
        sets["mtfb_all"] = members("mtfb")
        sets["mtfb_solo"] = [r["posteriors"] for r in by_group["mtfb"] if r["src"] == "solo"]
        sets["mtfb_ens"] = [r["posteriors"] for r in by_group["mtfb"] if r["src"] == "ens"]
    if by_group.get("hnm") and by_group.get("mtfb"):
        sets["hnm+mtfb"] = members("hnm") + members("mtfb")
    sets["all"] = [r["posteriors"] for r in runs]

    # best individual run per seed (across hnm+mtfb, the improved models)
    improved = [r for r in runs if r["group"] in ("hnm", "mtfb")]
    best_by_seed, seen = [], {}
    for r in sorted(improved, key=lambda x: x["f1"], reverse=True):
        if r["seed"] not in seen:
            seen[r["seed"]] = r; best_by_seed.append(r)
    if best_by_seed:
        sets["best_per_seed"] = [r["posteriors"] for r in best_by_seed]

    # top-K by individual F1
    k = max(1, args.topk)
    sets[f"top{k}"] = [r["posteriors"] for r in runs_sorted[:k]]

    # per-class routed: top-M members for each class by that class's F1
    m = max(1, args.route_topm)
    members_by_class = {}
    for ci, c in enumerate(C3):
        ranked = sorted(runs, key=lambda x: x["pc"][c].get("f1", 0.0), reverse=True)[:m]
        members_by_class[ci] = [r["posteriors"] for r in ranked]
    routed = route_per_class(members_by_class)

    print("\nScoring ensemble groups ...")
    rows = []
    for name, dicts in sets.items():
        if not dicts:
            continue
        s = score(average(dicts), gt_events)
        rows.append((name, len(dicts), s))
    s_routed = score(routed, gt_events)
    rows.append((f"routed_top{m}_perclass", "pc", s_routed))

    rows.sort(key=lambda x: x[2]["paper_f1"], reverse=True)
    print("\n" + "=" * 86 + f"\nENSEMBLE GROUPS  (vs baseline {args.baseline:.3f})\n" + "=" * 86)
    print(f"{'group':24} {'n':>4} {'F1':>7} {'Δbase':>7}  {'bmabz P/R':>13} "
          f"{'d P/R':>13} {'bp P/R':>13}  thr(b,d,p)")
    for name, n, s in rows:
        pc, thr = s["per_class"], s["thr"]
        def pr(c):
            mm = pc.get(c, {})
            return f"{mm.get('precision',0):.2f}/{mm.get('recall',0):.2f}"
        mark = " *" if s["paper_f1"] >= args.baseline else ""
        print(f"{name:24} {str(n):>4} {s['paper_f1']:7.4f} "
              f"{s['paper_f1']-args.baseline:+7.4f}  {pr('bmabz'):>13} {pr('d'):>13} "
              f"{pr('bp'):>13}  {thr[0]:.2f},{thr[1]:.2f},{thr[2]:.2f}{mark}")

    best = rows[0]
    print(f"\nBest group: {best[0]}  paper-F1 {best[2]['paper_f1']:.4f} "
          f"({best[2]['paper_f1']-args.baseline:+.4f} vs {args.baseline:.3f})")
    print("(val-tuned + val-selected -> optimistic; confirm the winner on the test set.)")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "n_members", "paper_f1",
                        "bmabz_p", "bmabz_r", "d_p", "d_r", "bp_p", "bp_r",
                        "thr_bmabz", "thr_d", "thr_bp"])
            for name, n, s in rows:
                pc, thr = s["per_class"], s["thr"]
                w.writerow([name, n, f"{s['paper_f1']:.4f}"] +
                           [f"{pc.get(c,{}).get(k2,0):.4f}" for c in C3 for k2 in ("precision", "recall")] +
                           [f"{t:.3f}" for t in thr])
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
