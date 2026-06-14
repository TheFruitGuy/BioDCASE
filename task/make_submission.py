"""
make_submission.py — build the BioDCASE 2026 Task 2 submission CSV
==================================================================

Runs the routed ensemble's *post-processing* on the evaluation set and writes the
official submission file. It consumes the per-model evaluation posteriors (the
same ``{"ds|fn": (T,3)}`` npz your pipeline already saves for val as
``best_posteriors.npz``), combines them with the per-class routing weights,
applies your deployed post-processing (``postprocess_final.postprocess_predictions``:
stitch -> 500 ms median -> per-class threshold -> 0.5 s merge -> [0.5,30] s
duration), converts each event to an absolute UTC timestamp, and emits:

    dataset,filename,annotation,start_datetime,end_datetime

verified against the official scorer (``evaluation_submissions.py``):
  * labels are submitted as your 3 classes **bmabz / d / bp** — the scorer's
    ``joining_dict`` maps ``d -> fm`` (and keeps bmabz, bp), matching its
    ``labels_ground_truth = ['bmabz','fm','bp']``. No fine 7-class labels needed.
  * ``start_datetime`` / ``end_datetime`` are ISO8601 UTC (``...+00:00``); the
    scorer parses them with ``format='ISO8601'`` and localises to UTC.

Thresholds are **frozen** — passed in, never tuned here. The evaluation set has
no labels, so the operating point must come from your val tuning (the all-val /
R09 thresholds that ``loso_eval.py`` prints on its "all-val" line). Tuning on the
eval set would be leakage.

Prerequisite — per-model eval posteriors
----------------------------------------
This script does NOT run the audio->posteriors forward pass (that's your
validated ``ensemble_predict`` pipeline, which owns the spectrogram frontend,
segment tiling and stitching). Produce the eval posteriors the same way you made
``best_posteriors.npz``, but on the evaluation audio and **at 60 s tiles** (to
match R09/R10), one npz per run dir, keyed ``"<dataset>|<filename.wav>"`` -> (T,3)
float. Point ``--posterior-name`` at them.

Usage
-----
::

    python make_submission.py \
        --checkpoints \
            runs/hnm_3class_s{42,2024,7777,9999,1337}_ens_nopgi/best_model.pt \
            runs/mtfb_3class_s{42,2024,7777,9999,1337}_solo_asymmetric_mse/best_model.pt \
        --weights-bmabz 1 1 1 1 1 0 0 0 0 0 \
        --weights-bp    1 1 1 1 1 0 0 0 0 0 \
        --weights-d     0 0 0 0 0 1 1 1 1 1 \
        --posterior-name eval_posteriors.npz \
        --thr-bmabz 0.30 --thr-d 0.40 --thr-bp 0.30 \
        --out Nagl_JKU_task2_1.output.csv

    python make_submission.py --selftest      # no data needed
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import config_final as cfg
from postprocess_final import postprocess_predictions

CLASSES = ["bmabz", "d", "bp"]


# --- filename -> file start datetime (mirrors dataset_final._parse_file_start_dt) ---
def parse_file_start_dt(filename: str) -> datetime:
    """ATBFL convention: '2014-06-29T23-00-00_000.wav' -> 2014-06-29 23:00:00 UTC."""
    stem = Path(filename).stem.split("_")[0]
    return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)


# --- per-class routed combine (same as loso_eval; keyed (ds,fn,0) for stitch) ---
def _find_posteriors(run_dir: Path, name):
    names = [name] if name else ["eval_posteriors.npz", "best_posteriors.npz",
                                  "paper_best_posteriors.npz"]
    for n in names:
        if n and (run_dir / n).exists():
            return run_dir / n
    return None


def load_combined(checkpoints, weights, posterior_name):
    """Per-class weighted average of members' eval posteriors.
    Returns {(ds, fn, 0): (T,3) float32} (the (ds,fn,start) key stitch expects)."""
    members = []
    for ck in checkpoints:
        rd = Path(ck).parent
        p = _find_posteriors(rd, posterior_name)
        if p is None:
            raise FileNotFoundError(
                f"no eval posteriors npz in {rd} (looked for "
                f"{posterior_name or 'eval_posteriors.npz / best_posteriors.npz'}). "
                "Generate per-model eval posteriors with your forward pipeline first.")
        npz = np.load(p)
        members.append({k: npz[k].astype(np.float32) for k in npz.files})
        print(f"  [load] {rd.name:46s} {len(npz.files):4d} files  <- {p.name}")

    keys = set(members[0])
    for m in members[1:]:
        keys &= set(m)
    if not keys:
        raise ValueError("members share no common file keys — check the posterior set.")
    w = {c: np.asarray(weights[c], float) for c in CLASSES}
    for c in CLASSES:
        if w[c].shape[0] != len(members):
            raise ValueError(f"--weights-{c} has {w[c].shape[0]} entries, expected {len(members)}")
        if w[c].sum() <= 0:
            raise ValueError(f"--weights-{c} sums to 0")

    combined = {}
    for k in keys:
        arrs = [m[k] for m in members]
        L = min(a.shape[0] for a in arrs)
        out = np.zeros((L, 3), dtype=np.float32)
        for ci in range(3):
            wc = w[CLASSES[ci]]
            num = np.zeros(L, dtype=np.float64)
            for i, a in enumerate(arrs):
                if wc[i] != 0:
                    num += wc[i] * a[:L, ci]
            out[:, ci] = (num / wc.sum()).astype(np.float32)
        ds, fn = k.split("|", 1)
        combined[(ds, fn, 0)] = out
    print(f"  [combine] {len(combined)} files across {len(members)} members")
    return combined


# --- detections -> submission rows (relative seconds -> absolute UTC) ---
def detections_to_rows(dets):
    rows, start_cache = [], {}
    for d in dets:
        if d.filename not in start_cache:
            start_cache[d.filename] = parse_file_start_dt(d.filename)
        t0 = start_cache[d.filename]
        rows.append((
            d.dataset, d.filename, d.label,
            (t0 + timedelta(seconds=float(d.start_s))).isoformat(),
            (t0 + timedelta(seconds=float(d.end_s))).isoformat(),
        ))
    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    return rows


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "filename", "annotation", "start_datetime", "end_datetime"])
        wr.writerows(rows)


def build_submission(combined, thresholds, out_path):
    cfg.USE_3CLASS = True                       # so class_names() == [bmabz, d, bp]
    dets = postprocess_predictions(combined, thresholds)
    rows = detections_to_rows(dets)
    write_csv(rows, out_path)
    return rows


# --- self-test (full pipeline on synthetic posteriors; torch-free) ---
def selftest():
    print("[selftest] datetime parse + postprocess + csv ...")
    assert parse_file_start_dt("2014-06-29T23-00-00_000.wav") == \
        datetime(2014, 6, 29, 23, 0, 0, tzinfo=timezone.utc)

    cfg.USE_3CLASS = True
    stride = float(cfg.FRAME_STRIDE_S)
    fn = "2014-06-29T23-00-00_000.wav"
    T = 1500
    arr = np.zeros((T, 3), dtype=np.float32)
    # bmabz event ~5 s (frames 100..350), d event ~1 s (frames 800..850)
    arr[100:350, 0] = 0.8
    arr[800:850, 1] = 0.9
    combined = {("kerguelen2019", fn, 0): arr}
    thr = np.array([0.3, 0.4, 0.3])
    rows = build_submission(combined, thr, "/tmp/_sub_selftest.csv")
    labels = {r[2] for r in rows}
    assert "bmabz" in labels and "d" in labels, f"missing events: {labels}"
    # check the bmabz row's absolute start ~ file_start + 100*0.02 = +2.0 s
    b = next(r for r in rows if r[2] == "bmabz")
    exp = (datetime(2014, 6, 29, 23, 0, 0, tzinfo=timezone.utc)
           + timedelta(seconds=100 * stride)).isoformat()
    assert b[3] == exp, f"{b[3]} != {exp}"
    assert b[4].endswith("+00:00") and "T" in b[3]            # ISO8601 UTC
    # re-read CSV: 5 columns, right header, labels in {bmabz,d,bp}
    import csv as _csv
    with open("/tmp/_sub_selftest.csv") as f:
        r = list(_csv.reader(f))
    assert r[0] == ["dataset", "filename", "annotation", "start_datetime", "end_datetime"]
    assert all(row[2] in CLASSES for row in r[1:]) and len(r[0]) == 5
    print(f"  produced {len(rows)} rows; sample: {rows[0]}")
    print("[selftest] OK")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+")
    p.add_argument("--weights-bmabz", nargs="+", type=float)
    p.add_argument("--weights-d", nargs="+", type=float)
    p.add_argument("--weights-bp", nargs="+", type=float)
    p.add_argument("--posterior-name", default=None,
                   help="per-model eval posteriors npz filename (60 s tiles, "
                        "{'ds|fn':(T,3)}). Default tries eval_/best_posteriors.npz.")
    p.add_argument("--thr-bmabz", type=float, help="frozen val-tuned threshold (required for a real run)")
    p.add_argument("--thr-d", type=float)
    p.add_argument("--thr-bp", type=float)
    p.add_argument("--out", default="submission.output.csv")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.checkpoints:
        raise SystemExit("provide --checkpoints (or --selftest)")
    if None in (args.thr_bmabz, args.thr_d, args.thr_bp):
        raise SystemExit("--thr-bmabz/--thr-d/--thr-bp are required (frozen val/R09 thresholds)")

    n = len(args.checkpoints)
    weights = {
        "bmabz": args.weights_bmabz if args.weights_bmabz else [1.0] * n,
        "d":     args.weights_d if args.weights_d else [1.0] * n,
        "bp":    args.weights_bp if args.weights_bp else [1.0] * n,
    }
    thresholds = np.array([args.thr_bmabz, args.thr_d, args.thr_bp], dtype=float)

    print(f"{'='*78}\nBUILD SUBMISSION  (frozen thresholds bmabz/d/bp = "
          f"{args.thr_bmabz}/{args.thr_d}/{args.thr_bp})\n{'='*78}")
    print("Loading + combining per-model eval posteriors ...")
    combined = load_combined(args.checkpoints, weights, args.posterior_name)

    print("Post-processing (your deployed pipeline) + writing CSV ...")
    rows = build_submission(combined, thresholds, args.out)

    by_ds, by_cls = {}, {c: 0 for c in CLASSES}
    for r in rows:
        by_ds[r[0]] = by_ds.get(r[0], 0) + 1
        by_cls[r[2]] += 1
    print(f"\nwrote {args.out}  —  {len(rows)} events")
    print("  per dataset:", ", ".join(f"{k}={v}" for k, v in sorted(by_ds.items())))
    print("  per class:  ", ", ".join(f"{k}={by_cls[k]}" for k in CLASSES))
    print("\nReminders: labels are bmabz/d/bp (scorer maps d->fm); thresholds are the")
    print("frozen val/R09 operating point (not tuned on eval); pair this CSV with your")
    print("Nagl_JKU_task2_N.meta.yaml + technical report for the official submission.")


if __name__ == "__main__":
    main()
