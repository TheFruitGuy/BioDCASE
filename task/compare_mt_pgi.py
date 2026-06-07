"""
compare_mt_pgi.py — does per-class gradient isolation (PGI) help the MT D-specialist?
====================================================================================

Compares the MT+HNM-from-base runs with PGI on vs off, for both hard-negative
sources (solo, ens), per seed and as the 5-seed ensemble. Each variant is scored
*standalone* on the val set with your own pipeline — ``tune_thresholds`` +
``evaluate`` + ``per_class_block`` from ``rescore_base_epochs`` (median 500 ms ->
per-class threshold -> merge 0.5 s -> dur [0.5,30] s, greedy 1-D IoU @ 0.3), exactly
as ``ensemble_blockE`` does — so the numbers line up with everything else.

D is broken out because that is the only class the MT models route in the hybrid;
the per-class gradient-isolation question is really "does PGI change MT's D".

Run directories (auto-discovered under ``--runs-root``):
    PGI on  : runs/mtfb_3class_s{seed}_{source}_asymmetric_mse
    PGI off : runs/mtfb_3class_s{seed}_{source}_asymmetric_mse_nopgi
Missing variants are reported, not fatal — so this works before the ens-nopgi
arm is trained and fills in once it is.

For each (source) it prints:
  * a per-seed table: standalone D-F1 and macro-F1, PGI on vs off, with Δ(off-on);
  * the 5-seed-ensemble row (members averaged), same columns;
  * the ensemble per-class P/R/F1 (on -> off), so you see where any change lives.
Δ is signed off - on: **positive means turning PGI off helped** (PGI was hurting).

Usage
-----
::

    python compare_mt_pgi.py                              # solo + ens, 5 seeds
    python compare_mt_pgi.py --sources solo               # just solo
    python compare_mt_pgi.py --posterior-name best_posteriors_60s.npz   # 60 s
    python compare_mt_pgi.py --selftest                   # no data/server needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config_final as cfg

CLASSES = ["bmabz", "d", "bp"]
SEEDS = [42, 2024, 7777, 9999, 1337]
SOURCES = ["solo", "ens"]


def f1_of(p, r):
    return 2 * p * r / (p + r + 1e-8)


def run_dir_name(seed, source, pgi):
    return f"mtfb_3class_s{seed}_{source}_asymmetric_mse" + ("" if pgi else "_nopgi")


# =============================================================================
# Loading / discovery
# =============================================================================
def find_posteriors(run_dir: Path, override):
    names = [override] if override else ["best_posteriors.npz", "paper_best_posteriors.npz"]
    for n in names:
        if n and (run_dir / n).exists():
            return run_dir / n
    return None


def load_posteriors(path):
    """Saved npz {"ds|fn": (T,3)} -> {(ds, fn, 0): (T,3) float32} (evaluate's key)."""
    npz = np.load(path)
    out = {}
    for k in npz.files:
        ds, fn = k.split("|", 1)
        out[(ds, fn, 0)] = npz[k].astype(np.float32)
    return out


def ensemble_avg(prob_dicts):
    """Uniform mean over members on the shared file keys -> {(ds,fn,0): (T,3)}."""
    keys = set(prob_dicts[0])
    for d in prob_dicts[1:]:
        keys &= set(d)
    out = {}
    for k in keys:
        arrs = [d[k] for d in prob_dicts]
        L = min(a.shape[0] for a in arrs)
        out[k] = np.mean([a[:L] for a in arrs], axis=0).astype(np.float32)
    return out


def discover(runs_root, sources, seeds, posterior_name):
    """{(source, pgi_bool): {seed: npz_path}} for runs that exist on disk."""
    reg = {}
    for source in sources:
        for pgi in (True, False):
            found = {}
            for seed in seeds:
                p = find_posteriors(Path(runs_root) / run_dir_name(seed, source, pgi),
                                    posterior_name)
                if p is not None:
                    found[seed] = p
            reg[(source, pgi)] = found
    return reg


# =============================================================================
# Scoring (scoring fns injected so the orchestration is testable)
# =============================================================================
def score(probs, gt, tune_fn, eval_fn, perclass_fn, paperf1_fn, workers):
    cfg.USE_3CLASS = True
    thr = tune_fn(probs, gt, workers)
    m = eval_fn(probs, gt, thr)
    return dict(macro=float(paperf1_fn(m)), pc=perclass_fn(m),
                thr=[float(t) for t in np.asarray(thr)])


def run_compare(reg, gt, tune_fn, eval_fn, perclass_fn, paperf1_fn, workers):
    """For each source: score every available (pgi, seed) standalone + the
    members-averaged ensemble per pgi. Returns a nested results dict."""
    out = {}
    sources = sorted({s for s, _ in reg})
    for source in sources:
        on, off = reg[(source, True)], reg[(source, False)]
        per_seed = {}
        for seed in sorted(set(on) | set(off)):
            row = {}
            if seed in on:
                row["on"] = score(load_posteriors(on[seed]), gt, tune_fn, eval_fn,
                                  perclass_fn, paperf1_fn, workers)
            if seed in off:
                row["off"] = score(load_posteriors(off[seed]), gt, tune_fn, eval_fn,
                                   perclass_fn, paperf1_fn, workers)
            per_seed[seed] = row
        ens = {}
        for tag, reg_d in (("on", on), ("off", off)):
            if reg_d:
                avg = ensemble_avg([load_posteriors(p) for p in reg_d.values()])
                ens[tag] = score(avg, gt, tune_fn, eval_fn, perclass_fn, paperf1_fn, workers)
                ens[tag]["n"] = len(reg_d)
        out[source] = dict(per_seed=per_seed, ens=ens,
                           n_on=len(on), n_off=len(off))
    return out


# =============================================================================
# Reporting
# =============================================================================
def _d_macro(s):
    return (s["pc"]["d"]["f1"], s["macro"]) if s else (None, None)


def _cell(x):
    return f"{x:6.3f}" if x is not None else "   -- "


def report(results, posterior_name):
    bar = "=" * 80
    print(f"\n{bar}\nMT PGI vs NO-PGI  —  does per-class gradient isolation help the D-specialist?\n{bar}")
    print(f"  sites: {', '.join(cfg.VAL_DATASETS)}  |  standalone, val-tuned (all-val)  |  IoU 0.3  |  {posterior_name}")
    print("  Δ is off - on:  + means PGI OFF is better (PGI was hurting);  D is the routed class.")

    for source in sorted(results):
        r = results[source]
        print(f"\n{'-'*80}\nSOURCE: {source}    (PGI-on seeds: {r['n_on']}/5, PGI-off seeds: {r['n_off']}/5)")

        if not r["per_seed"]:
            print("  no runs found.")
            continue

        print(f"\n  per-seed standalone        {'D-F1':^22}     {'macro-F1':^22}")
        print(f"    {'seed':8} {'on':>7} {'off':>7} {'Δ':>7}     {'on':>7} {'off':>7} {'Δ':>7}")
        for seed in sorted(r["per_seed"], key=lambda s: (SEEDS.index(s) if s in SEEDS else 999, s)):
            don, mon = _d_macro(r["per_seed"][seed].get("on"))
            dof, mof = _d_macro(r["per_seed"][seed].get("off"))
            dd = f"{dof-don:+6.3f}" if (don is not None and dof is not None) else "   -- "
            dm = f"{mof-mon:+6.3f}" if (mon is not None and mof is not None) else "   -- "
            print(f"    {seed:<8} {_cell(don)} {_cell(dof)} {dd}     {_cell(mon)} {_cell(mof)} {dm}")

        eon, eof = r["ens"].get("on"), r["ens"].get("off")
        don, mon = _d_macro(eon)
        dof, mof = _d_macro(eof)
        dd = f"{dof-don:+6.3f}" if (don is not None and dof is not None) else "   -- "
        dm = f"{mof-mon:+6.3f}" if (mon is not None and mof is not None) else "   -- "
        non = eon["n"] if eon else 0
        noff = eof["n"] if eof else 0
        print(f"    {'ENS':<8} {_cell(don)} {_cell(dof)} {dd}     {_cell(mon)} {_cell(mof)} {dm}"
              f"   <- {non}-seed / {noff}-seed avg")

        if eon and eof:
            print(f"\n  ensemble per-class (PGI on -> off)")
            print(f"    {'class':6} {'P_on':>6} {'R_on':>6} {'F1_on':>6}   {'P_off':>6} {'R_off':>6} {'F1_off':>6}   {'ΔF1':>7}")
            for lab in CLASSES:
                a, b = eon["pc"][lab], eof["pc"][lab]
                mark = "  <- routed" if lab == "d" else ""
                print(f"    {lab:6} {a['precision']:6.3f} {a['recall']:6.3f} {a['f1']:6.3f}   "
                      f"{b['precision']:6.3f} {b['recall']:6.3f} {b['f1']:6.3f}   {b['f1']-a['f1']:+7.3f}{mark}")
        elif eon or eof:
            have = "PGI-on" if eon else "PGI-off"
            need = "off" if eon else "on"
            print(f"\n  only {have} present — run the other arm to complete the comparison:")
            print(f"    python launch_mt_3c_frombase.py --pgi {need} --sources {source} ...")

    print(f"\n{bar}\nVERDICT (ensemble D-F1, the routed quantity)\n{bar}")
    for source in sorted(results):
        eon, eof = results[source]["ens"].get("on"), results[source]["ens"].get("off")
        if eon and eof:
            don, dof = eon["pc"]["d"]["f1"], eof["pc"]["d"]["f1"]
            d = dof - don
            tag = ("PGI off is better — drop isolation for this D route" if d > 0.005 else
                   "PGI on is better — keep isolation" if d < -0.005 else
                   "tie — PGI makes no real difference here")
            print(f"  {source:4}: D-F1 on {don:.3f} -> off {dof:.3f}  (Δ {d:+.3f})  →  {tag}")
        else:
            print(f"  {source:4}: incomplete (need both PGI on and off)")
    print("  Deployment check: point the D-routed MT ckpts in loso_eval.py at the winning")
    print("  variant and rerun — standalone D-F1 predicts, the routed LOSO confirms.")


# =============================================================================
# Self-test (filesystem discovery + orchestration; scoring fns mocked)
# =============================================================================
def selftest():
    import tempfile
    print("[selftest] naming / discover / ensemble_avg / compare ...")
    assert run_dir_name(42, "solo", True) == "mtfb_3class_s42_solo_asymmetric_mse"
    assert run_dir_name(42, "solo", False) == "mtfb_3class_s42_solo_asymmetric_mse_nopgi"

    # ensemble_avg: mean over members on shared keys
    a = {("S", "f", 0): np.array([[0.2, 0.4, 0.6]], np.float32)}
    b = {("S", "f", 0): np.array([[0.4, 0.8, 0.2]], np.float32), ("S", "g", 0): np.zeros((1, 3), np.float32)}
    m = ensemble_avg([a, b])
    assert set(m) == {("S", "f", 0)} and np.allclose(m[("S", "f", 0)], [[0.3, 0.6, 0.4]])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # solo: 5 on + 5 off ; ens: 5 on, 0 off  (mirrors current state)
        for seed in SEEDS:
            for source, pgi in [("solo", True), ("solo", False), ("ens", True)]:
                d = root / run_dir_name(seed, source, pgi)
                d.mkdir(parents=True)
                arr = np.random.default_rng(seed).uniform(0, 1, size=(5, 3)).astype(np.float32)
                np.savez(d / "best_posteriors.npz", **{f"casey2017|f{seed}": arr})
        reg = discover(root, SOURCES, SEEDS, None)
        assert len(reg[("solo", True)]) == 5 and len(reg[("solo", False)]) == 5
        assert len(reg[("ens", True)]) == 5 and len(reg[("ens", False)]) == 0

        # mock scoring: D-F1 = 0.3 + 0.02*(off) so 'off' looks slightly better; macro fixed
        def tune_fn(p, g, w):
            return np.array([0.3, 0.4, 0.3])

        def eval_fn(p, g, thr):  # carries an 'is_off' marker injected via a sentinel key count
            return {"_probs": p}

        def perclass_fn(m):
            return {lab: dict(precision=0.5, recall=0.5, f1=0.5) for lab in CLASSES}

        def paperf1_fn(m):
            return 0.45

        res = run_compare(reg, [], tune_fn, eval_fn, perclass_fn, paperf1_fn, 1)
        assert set(res) == {"solo", "ens"}
        assert res["solo"]["ens"]["on"]["n"] == 5 and res["solo"]["ens"]["off"]["n"] == 5
        assert "on" in res["ens"]["ens"] and "off" not in res["ens"]["ens"]
        assert len(res["solo"]["per_seed"]) == 5 and "on" in res["solo"]["per_seed"][42]
        report(res, "best_posteriors.npz")   # exercise rendering (incl. missing-arm branch)
    print("[selftest] OK")


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--sources", nargs="+", choices=["solo", "ens"], default=SOURCES)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--posterior-name", default=None,
                   help="npz filename (default auto best_/paper_best_posteriors.npz)")
    p.add_argument("--workers", type=int, default=13, help="passed to tune_thresholds")
    p.add_argument("--out", default="compare_mt_pgi_result.json")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return

    reg = discover(args.runs_root, args.sources, args.seeds, args.posterior_name)
    total = sum(len(v) for v in reg.values())
    if total == 0:
        raise SystemExit(f"no mtfb posteriors found under {args.runs_root} "
                         f"for sources={args.sources}, seeds={args.seeds}")
    print(f"{'='*80}\nDiscovered runs (with {args.posterior_name or 'best_/paper_best_posteriors.npz'}):")
    for (source, pgi), d in sorted(reg.items()):
        print(f"  {source:4} PGI {'on ' if pgi else 'off'}: {sorted(d)}")

    print("\nBuilding val ground truth ...")
    cfg.USE_3CLASS = True
    from dataset_final import load_annotations, get_file_manifest
    from rescore_base_epochs import (
        build_gt_events, evaluate, paper_f1, per_class_block, tune_thresholds,
    )
    anns = load_annotations(list(cfg.VAL_DATASETS))
    manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    fsd = {(r["dataset"], r["filename"]): r["start_dt"] for _, r in manifest.iterrows()}
    gt = build_gt_events(anns, fsd)

    print("Scoring each variant (standalone + ensemble) ...")
    results = run_compare(reg, gt, tune_thresholds, evaluate, per_class_block, paper_f1, args.workers)
    pname = args.posterior_name or "best_posteriors.npz"
    report(results, pname)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
