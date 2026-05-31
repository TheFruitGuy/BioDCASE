"""
analyze_blockE.py — score the Block E ablation and answer "run A4?"
====================================================================

Fast, GPU-free analysis. Each MT run saved ``best_posteriors.npz`` (the stitched
3-class val posteriors at the selected epoch) and the tuned ``thresholds`` in
``best_model.pt``. We re-feed each per-file stream as a single segment at
start_sample=0 (so ``stitch_segments`` is identity) and re-score with the exact
``evaluate`` the trainer selected on — no forward pass. Each run's recomputed
macro paper-F1 is checked against the value stored in ``best_model.pt``; a
mismatch beyond ``--tol`` is flagged (float16 rounding should keep it < 0.005).

What it reports
---------------
* Per-arm absolute table: macro paper-F1 + per-class P/R/F1 (D called out), mean±std.
* Paired vs A0 (same seeds): Δ for D-F1 / D-recall / D-precision / macro, with
  how many seeds moved up. Headline is D's P/R, not macro (the threshold sweep
  already exploits the threshold axis — only a PR-curve shift counts).
* A data-driven A4 (verifier-gate) recommendation from the A3-vs-A0 D pattern.

Arm -> run-dir naming (override A0 if your baseline runs are named differently)
-------------------------------------------------------------------------------
  A0  mtfb_3class_s{seed}_ens_asymmetric_mse   baseline (your finished asym MT+HNM)
  A1  mtE_A1_3class_s{seed}_ens                +DA
  A2  mtE_A2_3class_s{seed}_ens                SAT
  A3  mtE_A3_3class_s{seed}_ens                SAT+DA
  A4  mtE_A4_3class_s{seed}_ens                SAT+DA+gate (if/when run)

Usage
-----
::

    python analyze_blockE.py                      # A0..A3, default seeds
    python analyze_blockE.py --arms A0 A1 A2 A3 A4
    python analyze_blockE.py --a0-pattern "mt_3class_s{seed}_chained_asym"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config_final as cfg
from dataset_final import load_annotations, get_file_manifest
from rescore_base_epochs import build_gt_events, evaluate, paper_f1, per_class_block

SEEDS = [42, 2024, 7777, 9999, 1337]
ARM_PATTERN = {
    "A0": "mtfb_3class_s{seed}_ens_asymmetric_mse",
    "A1": "mtE_A1_3class_s{seed}_ens",
    "A2": "mtE_A2_3class_s{seed}_ens",
    "A3": "mtE_A3_3class_s{seed}_ens",
    "A4": "mtE_A4_3class_s{seed}_ens",
}
ARM_DESC = {"A0": "baseline asym", "A1": "+DA", "A2": "SAT",
            "A3": "SAT+DA", "A4": "SAT+DA+gate"}


def score_run(run_dir: Path, gt_events, tol: float):
    """Score one run from its saved posteriors + thresholds. Returns dict or None."""
    ckpt_p, npz_p = run_dir / "best_model.pt", run_dir / "best_posteriors.npz"
    if not ckpt_p.exists() or not npz_p.exists():
        return None
    import torch
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    thr = np.asarray(ckpt["thresholds"], dtype=np.float64)
    saved_mp = float(ckpt.get("paper_f1", float("nan")))

    npz = np.load(npz_p)
    probs = {}
    for k in npz.files:
        ds, fn = k.split("|", 1)
        probs[(ds, fn, 0)] = npz[k].astype(np.float32)   # one segment/file -> stitch=identity

    cfg.USE_3CLASS = True
    metrics = evaluate(probs, gt_events, thr)
    mp = paper_f1(metrics)
    pc = per_class_block(metrics)
    match = (not np.isnan(saved_mp)) and abs(mp - saved_mp) <= tol
    return {"macro": mp, "saved_macro": saved_mp, "match": match, "pc": pc,
            "thr": np.asarray(thr, dtype=np.float64)}


def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"],
                   choices=list(ARM_PATTERN))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--a0-pattern", default=None,
                   help="Override the A0 run-dir pattern (use {seed}).")
    p.add_argument("--tol", type=float, default=0.01,
                   help="Max |recomputed - saved| macro-F1 before flagging a run.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.a0_pattern:
        ARM_PATTERN["A0"] = args.a0_pattern
    runs_root = Path(args.runs_root)

    val_anns = load_annotations(list(cfg.VAL_DATASETS))
    val_manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    val_fsd = {(r["dataset"], r["filename"]): r["start_dt"]
               for _, r in val_manifest.iterrows()}
    gt_events = build_gt_events(val_anns, val_fsd)

    # arm -> seed -> result
    res: dict[str, dict[int, dict]] = {a: {} for a in args.arms}
    flagged = []
    for arm in args.arms:
        for seed in args.seeds:
            d = runs_root / ARM_PATTERN[arm].format(seed=seed)
            r = score_run(d, gt_events, args.tol)
            if r is None:
                print(f"  [missing] {arm} s{seed}: {d}")
                continue
            res[arm][seed] = r
            if not r["match"]:
                flagged.append(f"{arm} s{seed}: recomputed {r['macro']:.4f} vs "
                               f"saved {r['saved_macro']:.4f}")

    if flagged:
        print("\n!! macro-F1 self-check mismatches (float16 drift or naming/format):")
        for m in flagged:
            print(f"   {m}")

    # ---- per-arm absolute table -------------------------------------
    print(f"\n{'='*72}\nPER-ARM (mean ± std over seeds present)\n{'='*72}")
    print(f"  {'arm':22} {'macroF1':>14} {'D P':>11} {'D R':>11} {'D F1':>11} "
          f"{'BMABZ F1':>10} {'BP F1':>9}")
    for arm in args.arms:
        seeds_here = sorted(res[arm])
        if not seeds_here:
            print(f"  {arm+' '+ARM_DESC[arm]:22} (no runs found)")
            continue
        macro = agg([res[arm][s]["macro"] for s in seeds_here])
        dP = agg([res[arm][s]["pc"]["d"]["precision"] for s in seeds_here])
        dR = agg([res[arm][s]["pc"]["d"]["recall"] for s in seeds_here])
        dF = agg([res[arm][s]["pc"]["d"]["f1"] for s in seeds_here])
        bF = agg([res[arm][s]["pc"]["bmabz"]["f1"] for s in seeds_here])
        pF = agg([res[arm][s]["pc"]["bp"]["f1"] for s in seeds_here])
        print(f"  {arm+' '+ARM_DESC[arm]:22} {macro[0]:.4f}±{macro[1]:.3f} "
              f"{dP[0]:.3f}±{dP[1]:.2f} {dR[0]:.3f}±{dR[1]:.2f} {dF[0]:.3f}±{dF[1]:.2f} "
              f"{bF[0]:.3f}±{bF[1]:.2f} {pF[0]:.3f}±{pF[1]:.2f}  (n={len(seeds_here)})")

    # ---- tuned thresholds per arm (diagnostic: did posteriors shift?) ----
    print(f"\n{'='*72}\nTUNED THRESHOLDS (mean ± std, bmabz / d / bp)\n{'='*72}")
    for arm in args.arms:
        sh = sorted(res[arm])
        if not sh:
            continue
        thrs = np.array([res[arm][s]["thr"] for s in sh])      # (n, 3)
        m, sd = thrs.mean(0), thrs.std(0)
        print(f"  {arm+' '+ARM_DESC[arm]:22} bmabz {m[0]:.2f}±{sd[0]:.2f}   "
              f"d {m[1]:.2f}±{sd[1]:.2f}   bp {m[2]:.2f}±{sd[2]:.2f}")
    print("  (if SAT's tuned D threshold ≈ A0's, the consistency change barely "
          "moved the D posteriors; if it shifts but D-F1 doesn't, the sweep is "
          "absorbing the change — same frontier either way.)")

    # ---- paired vs A0 -----------------------------------------------
    deltas = {}   # arm -> dict of mean deltas
    if "A0" in res and res["A0"]:
        print(f"\n{'='*72}\nPAIRED vs A0 (same seeds): mean Δ [seeds up / n]\n{'='*72}")
        print(f"  {'arm':22} {'ΔD-F1':>16} {'ΔD-recall':>16} {'ΔD-prec':>16} {'Δmacro':>16}")
        for arm in args.arms:
            if arm == "A0":
                continue
            common = sorted(set(res[arm]) & set(res["A0"]))
            if not common:
                continue

            def pair(field, cls=None):
                xs = []
                for s in common:
                    a = (res[arm][s]["pc"][cls][field] if cls
                         else res[arm][s]["macro"])
                    b = (res["A0"][s]["pc"][cls][field] if cls
                         else res["A0"][s]["macro"])
                    xs.append(a - b)
                xs = np.asarray(xs)
                return float(xs.mean()), int((xs > 0).sum()), len(xs)

            dF = pair("f1", "d"); dR = pair("recall", "d")
            dP = pair("precision", "d"); dM = pair(None)
            deltas[arm] = {"dF": dF[0], "dR": dR[0], "dP": dP[0], "dM": dM[0]}

            def cell(t):
                return f"{t[0]:+.4f} [{t[1]}/{t[2]}]"
            print(f"  {arm+' '+ARM_DESC[arm]:22} {cell(dF):>16} {cell(dR):>16} "
                  f"{cell(dP):>16} {cell(dM):>16}")
    else:
        print("\n(no A0 runs found — pass --a0-pattern to enable the paired "
              "comparison and the A4 recommendation; absolute D P/R is above)")

    # ---- A4 recommendation ------------------------------------------
    print(f"\n{'='*72}\nA4 (verifier-gate) recommendation\n{'='*72}")
    ref = "A3" if "A3" in deltas else ("A2" if "A2" in deltas else None)
    if ref is None:
        print("  Need A0 + (A2 or A3) to advise. From the absolute D P/R above: "
              "run A4 only if D recall is decent but precision is the limiter.")
    else:
        d = deltas[ref]
        print(f"  Using {ref} ({ARM_DESC[ref]}) vs A0:  "
              f"ΔD-recall={d['dR']:+.4f}  ΔD-prec={d['dP']:+.4f}  ΔD-F1={d['dF']:+.4f}")
        if d["dR"] >= 0.02 and d["dP"] <= -0.01:
            print("  -> RUN A4. SAT lifts D recall but pays in precision — the "
                  "verifier-gate is built to recover exactly that leak. (Needs a "
                  "trained verifier; gate filters SAT's D positives.)")
        elif d["dR"] >= 0.02 and d["dP"] > -0.01:
            print("  -> A4 OPTIONAL. D recall rose with no precision penalty; the "
                  "gate has little to fix. Run only to complete the ablation.")
        elif d["dF"] < 0.01 and d["dR"] < 0.02:
            print("  -> SKIP A4 (likely). SAT didn't move D recall, so the lever "
                  "isn't firing — the gate can only filter SAT's D positives, it "
                  "can't create recall. Spend effort elsewhere.")
        else:
            print("  -> MIXED. Recall/precision moved ambiguously; decide by whether "
                  "the recall gain is real and precision-limited.")
    print("\nNote: scored from saved posteriors at the selected (best-macro) epoch; "
          "D metrics are at that operating point.")


if __name__ == "__main__":
    main()
