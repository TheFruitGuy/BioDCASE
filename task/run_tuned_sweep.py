"""
Tuned-threshold sweep over selected epochs of 13n(smooth=0.01) / 13o / 13p / 13q.
================================================================================

Runs each checkpoint sequentially through eval_conformer's own pipeline
(get_probs -> per-class coordinate-descent threshold tuning -> per-class F1),
then prints ONE aligned table at the end and writes it to disk. Same eval config
as the 0.025 reference (cfg.EVAL_SEGMENT_S=30, in-sample thresholds), so the
numbers are directly comparable to 13n smooth=0.025 ep31 (D 0.336 / MACRO 0.463).

    CUDA_VISIBLE_DEVICES=6 \
        /home/matthias-nagl/.conda/envs/bio/bin/python run_tuned_sweep.py

The macro_paper column is shown but NOT used for ranking (it is the gameable
metric); rows are grouped by run and sorted by D F1, and the bar to beat is the
REF row at the top. A trailing star marks rows that clear the reference on MACRO.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import eval_conformer as ec
from postprocess_final import Detection

# ---- what to evaluate: label -> (run dir, checkpoint prefix, [epochs]) --------
SWEEP = [
    ("13o learnable-PCEN", "runs/phase13o_3c_s42_20260605_134311", "phase13o",
     [10, 12, 14, 15, 17, 24]),
    ("13q wide-kernel",    "runs/phase13q_3c_s42_20260605_134607", "phase13q",
     [9, 12, 13, 14]),
    ("13n smooth=0.01",    "runs/phase13n_3c_s42_20260605_075957", "phase13n",
     [24, 29, 30, 31, 34]),
    ("13p dual-res-5ch",   "runs/phase13p_3c_s42_20260605_134445", "phase13p",
     [5, 8, 10]),
]

# ---- the bar to beat: 13n smooth=0.025 ep31 (same eval config) ----------------
REF = dict(label="REF 13n smooth0.025", ep=31, bmabz=0.540, d=0.336, d_p=0.392,
           d_r=0.293, bp=0.514, macro=0.463, micro=0.504, mpaper=0.480,
           thr=(0.45, 0.60, 0.20))

OUT_TXT = Path("runs/tuned_sweep_results.txt")
OUT_JSON = Path("runs/tuned_sweep_results.json")


def _row_from_metrics(label, ep, metrics, thr):
    g = lambda n, k: float(metrics.get(n, {}).get(k, 0.0))
    bm, d, bp = ec.CLASS_NAMES                      # ('bmabz','d','bp')
    return dict(
        label=label, ep=ep,
        bmabz=g(bm, "f1"),
        d=g(d, "f1"), d_p=g(d, "precision"), d_r=g(d, "recall"),
        bp=g(bp, "f1"),
        macro=ec.macro_f1(metrics),
        micro=metrics.get("overall", {}).get("f1", 0.0),
        mpaper=ec.macro_f1_paper(metrics),
        thr=(float(thr[0]), float(thr[1]), float(thr[2])),
    )


def _fmt_row(r, beats):
    star = " *" if beats else ""
    return (f" {r['label']:<20} {r['ep']:>2} | "
            f"{r['bmabz']:.3f}  {r['d']:.3f} ({r['d_p']:.2f}/{r['d_r']:.2f})  "
            f"{r['bp']:.3f} | {r['macro']:.3f}  {r['micro']:.3f} | "
            f"{r['mpaper']:.3f} | {r['thr'][0]:.2f}/{r['thr'][1]:.2f}/{r['thr'][2]:.2f}{star}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = min((os.cpu_count() or 1), 13)
    args = SimpleNamespace(cache_dir="runs/conformer_eval_cache",
                           no_cache=False, use_fp16=False, workers=workers)
    print(f"device={device}  workers={workers}  "
          f"EVAL_SEGMENT_S={getattr(ec.cfg, 'EVAL_SEGMENT_S', '?')}")

    results = {}          # label -> list of row dicts
    t_start = time.time()
    for label, rundir, prefix, epochs in SWEEP:
        results[label] = []
        for ep in epochs:
            ckpt = Path(rundir) / f"{prefix}_epoch_{ep:02d}.pt"
            if not ckpt.exists():
                print(f"  [skip] missing {ckpt}")
                continue
            print(f"\n>>> {label}  epoch {ep}  ({ckpt.name})")
            t0 = time.time()
            probs3, gt_tuples, ncl = ec.get_probs(ckpt, args, device)
            gt = [Detection(dataset=d, filename=f, label=lab, start_s=s, end_s=e)
                  for (d, f, lab, s, e) in gt_tuples]
            thr = ec.tune_thresholds_per_class(probs3, gt, start=[0.5, 0.5, 0.5],
                                               workers=workers)
            tuned = ec.evaluate_with_thresholds(probs3, gt, thr)
            row = _row_from_metrics(label, ep, tuned, thr)
            results[label].append(row)
            print(f"    D F1={row['d']:.3f}  MACRO={row['macro']:.3f}  "
                  f"micro={row['micro']:.3f}  ({time.time() - t0:.0f}s)")

    # ----------------------------- final table --------------------------------
    W = 96
    lines = []
    lines.append("=" * W)
    lines.append(" TUNED PER-CLASS F1  (val: casey2017/kerguelen2014/kerguelen2015 @ "
                 f"{getattr(ec.cfg, 'EVAL_SEGMENT_S', '?')}s, in-sample thresholds)")
    lines.append("=" * W)
    lines.append(" run                  ep | BMABZ    D  (P / R)       BP   | "
                 "MACRO  micro | mpap. | thr b/d/bp")
    lines.append(" " + "-" * (W - 1))
    lines.append(_fmt_row(REF, beats=False) + "   <- bar to beat")
    lines.append(" " + "-" * (W - 1))

    flat = []
    for label, _rd, _pf, _eps in SWEEP:
        rows = sorted(results.get(label, []), key=lambda r: -r["d"])  # best D first
        for i, r in enumerate(rows):
            beats = r["macro"] >= REF["macro"]
            lines.append(_fmt_row(r, beats))
            flat.append(r)
        if rows:
            best_d = max(rows, key=lambda r: r["d"])
            best_m = max(rows, key=lambda r: r["macro"])
            lines.append(f"   -> best D F1 ep{best_d['ep']}={best_d['d']:.3f} | "
                         f"best MACRO ep{best_m['ep']}={best_m['macro']:.3f}")
        lines.append(" " + "-" * (W - 1))

    if flat:
        gd = max(flat, key=lambda r: r["d"])
        gm = max(flat, key=lambda r: r["macro"])
        lines.append(f" WINNER by D F1 : {gd['label']} ep{gd['ep']}  "
                     f"D={gd['d']:.3f} (ref 0.336)  MACRO={gd['macro']:.3f}")
        lines.append(f" WINNER by MACRO: {gm['label']} ep{gm['ep']}  "
                     f"MACRO={gm['macro']:.3f} (ref 0.463)  D={gm['d']:.3f}")
        cleared = [r for r in flat if r["macro"] >= REF["macro"]]
        lines.append(f" cleared MACRO 0.463: "
                     + (", ".join(f"{r['label']} ep{r['ep']} ({r['macro']:.3f})"
                                  for r in sorted(cleared, key=lambda r: -r['macro']))
                        if cleared else "none"))
    lines.append("=" * W)
    lines.append(f" total sweep time {time.time() - t_start:.0f}s")

    table = "\n".join(lines)
    print("\n" + table)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(table + "\n")
    OUT_JSON.write_text(json.dumps({"ref": REF, "results": results}, indent=2,
                                   default=float))
    print(f"\nwrote {OUT_TXT}  and  {OUT_JSON}")


if __name__ == "__main__":
    main()
