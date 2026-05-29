"""
Block B aggregator — the 40 HNM runs as one ablation table
===========================================================

Reads each Block B run from disk and prints a per-run table + the aggregate
breakdowns that answer the Block B questions: does PGI help, does ensemble-
mining beat solo, 3-class vs 7-class, and which individual runs are the best
ensemble candidates. Works on partial results — runs still going show as
``running``, unstarted ones as ``pending``.

Sources (no wandb needed)
-------------------------
  * ``<runs>/<run-name>/best_model.pt`` -> authoritative best ``paper_f1`` +
    meta (epoch, n_classes, isolate_classes, thresholds, n hard-negs).
  * ``<logdir>/<run-name>.log`` (from launch_blockB.py) -> the epoch-0 start
    F1 and the per-class ``start -> best (delta)`` lines the trainer prints.
If torch is unavailable the checkpoint read is skipped and best-F1 falls back
to the log's verdict line.

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task
    python aggregate_blockB.py
    python aggregate_blockB.py --csv runs/blockB_summary.csv
"""

from __future__ import annotations

import argparse
import csv as _csv
import re
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

SEEDS = [42, 2024, 7777, 9999, 1337]
CALL_TYPES_3 = ["bmabz", "d", "bp"]

# Block A baseline: per-seed tuned best-epoch paper-F1 from the rescore, for a
# no-HNM reference. (3c mean 0.4004, 7c mean 0.3677.)
BASELINE_A = {
    ("3class", 42): 0.4076, ("3class", 2024): 0.3802, ("3class", 7777): 0.4133,
    ("3class", 9999): 0.4049, ("3class", 1337): 0.3961,
    ("7class", 42): 0.3604, ("7class", 7777): 0.3833, ("7class", 1337): 0.3678,
    ("7class", 2024): 0.3566, ("7class", 9999): 0.3705,
}

_RE_START = re.compile(r"start paper-F1\s*=\s*([0-9.]+)")
_RE_VERDICT = re.compile(r"paper-F1\s+([0-9.]+)\s*->\s*([0-9.]+)")
_RE_PERCLASS = re.compile(
    r"^\s*(BMABZ|D|BP)\s+F1\s+([0-9.]+)\s*->\s*([0-9.]+)\s*\(([+-][0-9.]+)\)"
    r"(\s*\[target\])?", re.MULTILINE)


def enumerate_names(heads, seeds, sources, pgis):
    names = []
    for head in heads:
        for seed in seeds:
            for source in sources:
                for pgi in pgis:
                    names.append((head, seed, source, pgi,
                                  f"hnm_{head}_s{seed}_{source}_{'pgi' if pgi else 'nopgi'}"))
    return names


def read_checkpoint(path: Path):
    if torch is None or not path.exists():
        return None
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
        return {"paper_f1": float(c.get("paper_f1", float("nan"))),
                "epoch": c.get("epoch"),
                "n_classes": c.get("n_classes"),
                "isolate_classes": c.get("isolate_classes"),
                "n_hard_negs": sum(m.get("n_fps", 0) for m in c.get("hnm_meta", []))}
    except Exception:
        return None


def parse_log(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(errors="replace")
    out = {"start_f1": None, "best_f1": None, "per_class": {}, "done": False}
    ms = _RE_START.search(txt)
    if ms:
        out["start_f1"] = float(ms.group(1))
    verdicts = _RE_VERDICT.findall(txt)
    if verdicts:                              # last verdict-style line = final
        out["start_f1"] = out["start_f1"] or float(verdicts[-1][0])
        out["best_f1"] = float(verdicts[-1][1])
    for c, s, b, d, tgt in _RE_PERCLASS.findall(txt):
        out["per_class"][c.lower()] = {"start": float(s), "best": float(b),
                                       "delta": float(d), "is_target": bool(tgt)}
    out["done"] = bool(out["per_class"]) or "succeeded" in txt or bool(verdicts)
    return out


def gather(runs_root, logdir, names):
    rows = []
    for head, seed, source, pgi, name in names:
        ck = read_checkpoint(Path(runs_root) / name / "best_model.pt")
        lg = parse_log(Path(logdir) / f"{name}.log")
        best = (ck and ck["paper_f1"]) or (lg and lg["best_f1"])
        start = (lg and lg["start_f1"]) or BASELINE_A.get((head, seed))
        if ck:
            status = "done"
        elif lg and not lg["done"]:
            status = "running"
        elif lg and lg["done"]:
            status = "done?"          # log finished but no checkpoint
        else:
            status = "pending"
        rows.append({"head": head, "seed": seed, "source": source, "pgi": pgi,
                     "name": name, "start": start, "best": best,
                     "delta": (best - start) if (best and start) else None,
                     "per_class": (lg or {}).get("per_class", {}),
                     "epoch": ck["epoch"] if ck else None, "status": status})
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, w=6, p=4):
    return (f"{x:.{p}f}".rjust(w)) if isinstance(x, float) else "  -- ".rjust(w)


def print_table(rows):
    for head in ["3class", "7class"]:
        hr = [r for r in rows if r["head"] == head]
        if not hr:
            continue
        hr.sort(key=lambda r: (r["best"] is None, -(r["best"] or 0)))
        print(f"\n{'='*94}\n{head.upper()}\n{'='*94}")
        print(f"{'run':38} {'src':4} {'pgi':4} {'start':>6} {'best':>6} "
              f"{'Δ':>7}  {'bmabz':>6} {'d':>6} {'bp':>6}  {'ep':>3} status")
        for r in hr:
            pc = r["per_class"]
            def cd(c):
                return f"{pc[c]['delta']:+.3f}" if c in pc else "   -- "
            print(f"{r['name']:38} {r['source']:4} {'on ' if r['pgi'] else 'off':4} "
                  f"{_fmt(r['start'])} {_fmt(r['best'])} {_fmt(r['delta'],7,4):>7}  "
                  f"{cd('bmabz'):>6} {cd('d'):>6} {cd('bp'):>6}  "
                  f"{str(r['epoch'] or '-'):>3} {r['status']}")


def print_aggregates(rows):
    done = [r for r in rows if r["best"] is not None]
    n_done = len(done)
    n_run = sum(1 for r in rows if r["status"] == "running")
    print(f"\n{'='*94}\nAGGREGATES  ({n_done}/{len(rows)} with results"
          f"{f', {n_run} running' if n_run else ''})\n{'='*94}")
    if not done:
        print("  (no finished runs yet — re-run me as they complete)")
        return

    def grp(key, val):
        return _mean([r["best"] for r in done if r[key] == val])

    pgi_on, pgi_off = grp("pgi", True), grp("pgi", False)
    ens, solo = grp("source", "ens"), grp("source", "solo")
    c3, c7 = grp("head", "3class"), grp("head", "7class")

    def line(label, a, an, b, bn):
        d = (a - b) if (a is not None and b is not None) else None
        ds = f"  (Δ {d:+.4f})" if d is not None else ""
        print(f"  {label:22} {an}={_fmt(a)}   {bn}={_fmt(b)}{ds}")

    print("  mean best paper-F1:")
    line("PGI", pgi_on, "on", pgi_off, "off")
    line("mining source", ens, "ens ", solo, "solo")
    line("head", c3, "3c ", c7, "7c ")

    print("\n  best individual runs (ensemble candidates):")
    for r in sorted(done, key=lambda r: -(r["best"] or 0))[:6]:
        print(f"    {_fmt(r['best'])}  {r['name']}  (Δ {_fmt(r['delta'],7,4).strip()} "
              f"vs its start)")

    print("\n  vs Block A baseline (no HNM):")
    for head, base in [("3class", 0.4004), ("7class", 0.3677)]:
        hb = _mean([r["best"] for r in done if r["head"] == head])
        if hb is not None:
            print(f"    {head}: baseline mean {base:.4f}  ->  HNM best mean "
                  f"{hb:.4f}  ({hb-base:+.4f})")

    print("\n  per-class: mean Δ on TARGETED groups (did HNM fix the FPs it mined?):")
    for c in CALL_TYPES_3:
        deltas = [r["per_class"][c]["delta"] for r in done
                  if c in r["per_class"] and r["per_class"][c]["is_target"]]
        if deltas:
            print(f"    {c:6} mean Δ {_mean(deltas):+.4f}  (n={len(deltas)} targeted runs)")


def write_csv(rows, path):
    cols = ["name", "head", "seed", "source", "pgi", "start", "best", "delta",
            "epoch", "status"]
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols + [f"delta_{c}" for c in CALL_TYPES_3])
        for r in rows:
            w.writerow([r[k] for k in cols] +
                       [r["per_class"].get(c, {}).get("delta") for c in CALL_TYPES_3])
    print(f"\nWrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--logdir", default="runs/blockB_logs")
    p.add_argument("--heads", nargs="+", choices=["3class", "7class"],
                   default=["3class", "7class"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--sources", nargs="+", choices=["solo", "ens"],
                   default=["solo", "ens"])
    p.add_argument("--pgi", nargs="+", choices=["on", "off"], default=["on", "off"])
    p.add_argument("--csv", default=None, help="Also write a CSV here.")
    return p.parse_args()


def main():
    args = parse_args()
    pgis = [g == "on" for g in args.pgi]
    names = enumerate_names(args.heads, args.seeds, args.sources, pgis)
    if torch is None:
        print("[note] torch not importable — reading logs only "
              "(best-F1 from the verdict line, no checkpoint).")
    rows = gather(args.runs_root, args.logdir, names)
    print_table(rows)
    print_aggregates(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
