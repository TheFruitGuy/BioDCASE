"""
loso_eval.py — honest leave-one-site-out evaluation of the ensemble
===================================================================

Your reported ~0.504 is computed by tuning the per-class thresholds on all three
val sites and scoring those same three sites — i.e. it is selection-*optimistic*.
This script reports the selection-unbiased number: tune the thresholds on two of
the three val sites, score the held-out third, rotate over all three turns, and
recompute the paper macro-F1 from precision/recall averaged across classes and
folds (the WhaleVAD-BPN / BioDCASE protocol).

It uses *your own* post-processing and tuner — ``tune_thresholds`` and
``evaluate`` from ``rescore_base_epochs`` and ``build_gt_events`` /
``load_annotations`` / ``get_file_manifest`` from your stack, called exactly as
``ensemble_blockE`` calls them — so every number is byte-identical to your
pipeline (median 500 ms -> per-class threshold -> merge 0.5 s -> dur [0.5,30] s,
greedy 1-D IoU @ 0.3). Only the *split* changes.

It prints three things:
  * per-fold detail (mirrors paper Table V): the tuned thresholds, the dev-fold
    paper-F1 (optimistic fit) and the held-out paper-F1 (generalisation);
  * per-class precision/recall/F1 on the held-out folds (P/R averaged over folds);
  * the headline: LOSO paper-macro (honest) vs all-val paper-macro (optimistic,
    = your reported number) and the selection-bias gap.

It combines the members exactly like your hybrid (per-class routing weights), so
pass the same ``--checkpoints`` + ``--weights-*`` you use elsewhere. It reads
whatever posteriors you point it at, so the *same* script does 30 s now and 60 s
once you save 60 s-tiled posteriors (``--posterior-name``).

Usage
-----
::

    # your 10-model routed ensemble (5 HNM -> bmabz/bp, 5 MT -> d), 30 s posteriors
    python loso_eval.py \
        --checkpoints <5 HNM best_model.pt> <5 MT best_model.pt> \
        --weights-bmabz 1 1 1 1 1 0 0 0 0 0 \
        --weights-bp    1 1 1 1 1 0 0 0 0 0 \
        --weights-d     0 0 0 0 0 1 1 1 1 1

    # later, on 60 s posteriors saved under a different name
    python loso_eval.py --checkpoints ... --weights-... --posterior-name best_posteriors_60s.npz

    python loso_eval.py --selftest          # orchestration check, no data/server needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config_final as cfg

CLASSES = ["bmabz", "d", "bp"]


def f1_of(p, r):
    return 2 * p * r / (p + r + 1e-8)


# =============================================================================
# Member loading + per-class routed combine  (keyed (ds, fn, 0) for evaluate)
# =============================================================================
def _find_posteriors(run_dir: Path, override):
    names = [override] if override else ["best_posteriors.npz", "paper_best_posteriors.npz"]
    for n in names:
        if n and (run_dir / n).exists():
            return run_dir / n
    return None


def load_combined(checkpoints, weights, posterior_name):
    """Per-class weighted average of members' saved posteriors.

    weights: dict class -> (n_members,). Returns {(ds, fn, 0): (T, 3) float32}."""
    members = []
    for ck in checkpoints:
        rd = Path(ck).parent
        npz_p = _find_posteriors(rd, posterior_name)
        if npz_p is None:
            raise FileNotFoundError(f"no posteriors npz in {rd} "
                                    "(best_posteriors.npz / paper_best_posteriors.npz)")
        npz = np.load(npz_p)
        members.append({k: npz[k].astype(np.float32) for k in npz.files})
        print(f"  [load] {rd.name:42s} {len(npz.files):4d} files  <- {npz_p.name}")

    keys = set(members[0])
    for m in members[1:]:
        keys &= set(m)
    w = {c: np.asarray(weights[c], dtype=np.float64) for c in CLASSES}
    for c in CLASSES:
        if w[c].shape[0] != len(members):
            raise ValueError(f"--weights-{c} has {w[c].shape[0]} entries, expected {len(members)}")
        if w[c].sum() <= 0:
            raise ValueError(f"--weights-{c} sums to 0")

    ens = {}
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
        ens[(ds, fn, 0)] = out
    print(f"  [combine] {len(ens)} files across {len(members)} members")
    return ens


def subset_sites(ens, gt, sites):
    """Restrict the (ds,fn,0)-keyed probs and the gt Detection list to `sites`."""
    sset = set(sites)
    e = {k: v for k, v in ens.items() if k[0] in sset}
    g = [d for d in gt if d.dataset in sset]
    return e, g


# =============================================================================
# Aggregation -> paper macro-F1 (P/R averaged across classes AND test folds)
# =============================================================================
def aggregate(fold_pcs):
    """fold_pcs: list of per_class_block dicts (one per held-out fold).
    Returns (paper_macro_f1, {class: (P_bar, R_bar, F1)})."""
    per_class = {}
    Ps, Rs = [], []
    for lab in CLASSES:
        p = float(np.mean([fc[lab]["precision"] for fc in fold_pcs]))
        r = float(np.mean([fc[lab]["recall"] for fc in fold_pcs]))
        per_class[lab] = (p, r, f1_of(p, r))
        Ps.append(p)
        Rs.append(r)
    return f1_of(float(np.mean(Ps)), float(np.mean(Rs))), per_class


# =============================================================================
# The LOSO + all-val computations (scoring functions injected for testability)
# =============================================================================
def run_loso(ens, gt, sites, tune_fn, eval_fn, perclass_fn, paperf1_fn, workers):
    cfg.USE_3CLASS = True
    folds = []
    for held in sites:
        dev = [s for s in sites if s != held]
        dev_probs, dev_gt = subset_sites(ens, gt, dev)
        test_probs, test_gt = subset_sites(ens, gt, [held])

        thr = tune_fn(dev_probs, dev_gt, workers)              # tune on the 2 dev sites
        dev_f1 = paperf1_fn(eval_fn(dev_probs, dev_gt, thr))   # optimistic dev fit
        test_metrics = eval_fn(test_probs, test_gt, thr)       # score held-out site
        folds.append(dict(held=held,
                          thr=[float(t) for t in np.asarray(thr)],
                          dev_f1=float(dev_f1),
                          test_f1=float(paperf1_fn(test_metrics)),
                          pc=perclass_fn(test_metrics)))
    return folds


def run_allval(ens, gt, sites, tune_fn, eval_fn, perclass_fn, paperf1_fn, workers):
    """Tune + score on all sites — the selection-optimistic number (your ~0.504).
    Also scores each site separately at those same (deployed) thresholds."""
    cfg.USE_3CLASS = True
    probs, g = subset_sites(ens, gt, sites)
    thr = tune_fn(probs, g, workers)
    metrics = eval_fn(probs, g, thr)
    per_site = {}
    for s in sites:
        sp, sg = subset_sites(ens, gt, [s])
        sm = eval_fn(sp, sg, thr)
        per_site[s] = dict(macro=float(paperf1_fn(sm)), pc=perclass_fn(sm))
    return (float(paperf1_fn(metrics)), perclass_fn(metrics),
            [float(t) for t in np.asarray(thr)], per_site)


# =============================================================================
# Reporting
# =============================================================================
def report(loso_folds, allval, posterior_name, n_members):
    loso_macro, loso_pc = aggregate([f["pc"] for f in loso_folds])
    av_macro, av_pc, av_thr, per_site = allval
    bar = "=" * 80

    print(f"\n{bar}\nENSEMBLE LOSO EVALUATION  (your deployed threshold-only pipeline)\n{bar}")
    print(f"  sites: {', '.join(cfg.VAL_DATASETS)}   |   {n_members} members   |   {posterior_name}")
    print(f"  pipeline: median {cfg.SMOOTH_KERNEL_MS} ms -> per-class threshold (tuned) "
          f"-> merge {cfg.MERGE_GAP_S} s -> dur [{cfg.POST_MIN_DUR_S},{cfg.POST_MAX_DUR_S}] s   |   IoU 0.3")

    print(f"\n  PER-FOLD (tune on the 2 dev sites, score the held-out — cf. paper Table V)")
    print(f"    {'held-out':16} {'thr bmabz/d/bp':>20}   {'dev paperF1':>12} {'test paperF1':>13}")
    for f in loso_folds:
        t = f["thr"]
        print(f"    {f['held']:16} {t[0]:5.2f} /{t[1]:5.2f} /{t[2]:5.2f}    "
              f"{f['dev_f1']:>12.3f} {f['test_f1']:>13.3f}")

    print(f"\n  PER-CLASS on held-out folds (P/R averaged across folds)")
    print(f"    {'class':6} {'P':>7} {'R':>7} {'F1':>7}")
    for lab in CLASSES:
        p, r, fct = loso_pc[lab]
        print(f"    {lab:6} {p:7.3f} {r:7.3f} {fct:7.3f}")

    # ---- per-site, the honest view: each site as the held-out fold -----------
    fold_by_site = {f["held"]: f for f in loso_folds}
    print(f"\n  PER-SITE — held-out (LOSO: thresholds tuned on the other 2 sites) [honest]")
    print(f"    {'site':16} {'bmabz':>7} {'d':>7} {'bp':>7} {'macro':>8}")
    for s in cfg.VAL_DATASETS:
        f = fold_by_site.get(s)
        if not f:
            continue
        pc = f["pc"]
        print(f"    {s:16} {pc['bmabz']['f1']:7.3f} {pc['d']['f1']:7.3f} {pc['bp']['f1']:7.3f}"
              f" {f['test_f1']:8.3f}")

    # ---- per-site at the deployed thresholds (tuned once on all 3) -----------
    print(f"\n  PER-SITE — deployed thresholds (tuned once on all 3 val sites)")
    for s in cfg.VAL_DATASETS:
        ps = per_site.get(s)
        if not ps:
            continue
        print(f"    {s}  (macro {ps['macro']:.3f})")
        for lab in CLASSES:
            v = ps["pc"][lab]
            print(f"      {lab:6} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}")

    print(f"\n  {'-'*76}")
    print(f"  PAPER-MACRO   LOSO (honest)         : {loso_macro:.4f}")
    print(f"                all-val (optimistic)  : {av_macro:.4f}   "
          f"<- the selection-optimistic number (thr {'/'.join(f'{t:.2f}' for t in av_thr)})")
    print(f"                selection-bias gap    : {loso_macro - av_macro:+.4f}")
    print(bar)
    if av_macro - loso_macro > 0.02:
        print("  The all-val figure overstates generalisation by more than a noise margin;")
        print("  report the LOSO number (or the challenge test set), not the val-tuned one.")


# =============================================================================
# Self-test (orchestration only; the scoring fns are your tested code)
# =============================================================================
def selftest():
    print("[selftest] combine / subset / aggregate / loso-loop ...")
    # combine: 2 members, per-class routing -> exact weighted average
    m0 = {"A|f": np.array([[0.2, 0.8, 0.4]], dtype=np.float32)}
    m1 = {"A|f": np.array([[0.6, 0.4, 0.2]], dtype=np.float32)}
    import numpy as _np
    members = [m0, m1]
    # emulate load_combined's math directly
    w = {"bmabz": _np.array([1.0, 0.0]), "d": _np.array([0.0, 1.0]), "bp": _np.array([0.5, 0.5])}
    L = 1
    out = _np.zeros((L, 3))
    for ci, c in enumerate(CLASSES):
        wc = w[c]
        num = sum(wc[i] * members[i]["A|f"][:L, ci] for i in range(2))
        out[:, ci] = num / wc.sum()
    assert abs(out[0, 0] - 0.2) < 1e-6 and abs(out[0, 1] - 0.4) < 1e-6 and abs(out[0, 2] - 0.3) < 1e-6, out

    # subset by site
    ens = {("A", "f", 0): 1, ("B", "g", 0): 2, ("C", "h", 0): 3}
    gt = [type("D", (), {"dataset": s})() for s in ("A", "B", "C")]
    e, g = subset_sites(ens, gt, ["A", "C"])
    assert set(e) == {("A", "f", 0), ("C", "h", 0)} and {d.dataset for d in g} == {"A", "C"}

    # aggregate: paper-macro = F1(mean P, mean R) over classes, with per-class P/R
    # averaged over folds first.
    fold_pcs = [
        {"bmabz": dict(precision=0.6, recall=0.6), "d": dict(precision=0.3, recall=0.2),
         "bp": dict(precision=0.6, recall=0.4)},
        {"bmabz": dict(precision=0.6, recall=0.6), "d": dict(precision=0.3, recall=0.2),
         "bp": dict(precision=0.6, recall=0.4)},
    ]
    macro, pc = aggregate(fold_pcs)
    mp = (0.6 + 0.3 + 0.6) / 3
    mr = (0.6 + 0.2 + 0.4) / 3
    assert abs(macro - f1_of(mp, mr)) < 1e-9, (macro, f1_of(mp, mr))
    assert abs(pc["d"][2] - f1_of(0.3, 0.2)) < 1e-9

    # run_loso smoke test with deterministic mock scoring fns
    sites = list(cfg.VAL_DATASETS)
    ens2 = {(s, f"f{i}", 0): np.zeros((10, 3), np.float32) for s in sites for i in range(2)}
    gt2 = [type("D", (), {"dataset": s})() for s in sites for _ in range(3)]
    PR = {"casey2017": (0.5, 0.5), "kerguelen2014": (0.4, 0.3), "kerguelen2015": (0.6, 0.4)}

    def tune_fn(p, g, w):
        return np.array([0.3, 0.4, 0.3])

    def eval_fn(p, g, thr):
        held = sorted({k[0] for k in p})           # site(s) present
        # mean P/R over the present sites, same for all classes (mock)
        mp_ = np.mean([PR[s][0] for s in held])
        mr_ = np.mean([PR[s][1] for s in held])
        return {lab: dict(precision=mp_, recall=mr_) for lab in CLASSES}

    def perclass_fn(m):
        return {lab: dict(precision=m[lab]["precision"], recall=m[lab]["recall"],
                          f1=f1_of(m[lab]["precision"], m[lab]["recall"])) for lab in CLASSES}

    def paperf1_fn(m):
        mp_ = np.mean([m[lab]["precision"] for lab in CLASSES])
        mr_ = np.mean([m[lab]["recall"] for lab in CLASSES])
        return f1_of(mp_, mr_)

    folds = run_loso(ens2, gt2, sites, tune_fn, eval_fn, perclass_fn, paperf1_fn, 1)
    assert len(folds) == 3
    # held-out casey2017 -> P/R should be casey's (0.5,0.5) since test subset = casey only
    cf = next(f for f in folds if f["held"] == "casey2017")
    assert abs(cf["pc"]["d"]["precision"] - 0.5) < 1e-9
    macro, _ = aggregate([f["pc"] for f in folds])
    # honest LOSO macro: mean P over folds = mean(0.5,0.4,0.6)=0.5, same R=0.4 -> F1(0.5,0.4)
    assert abs(macro - f1_of(0.5, 0.4)) < 1e-9, macro
    print(f"  loso smoke macro={macro:.3f}  (expected {f1_of(0.5,0.4):.3f})")

    # run_allval returns per-site at the deployed thresholds; each site's mock P/R
    av = run_allval(ens2, gt2, sites, tune_fn, eval_fn, perclass_fn, paperf1_fn, 1)
    assert len(av) == 4 and set(av[3]) == set(sites)
    assert abs(av[3]["kerguelen2014"]["pc"]["d"]["precision"] - 0.4) < 1e-9   # site-only eval
    report(folds, av, "best_posteriors.npz", 6)   # exercises both per-site print paths
    print("[selftest] OK")


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+")
    p.add_argument("--weights-bmabz", nargs="+", type=float)
    p.add_argument("--weights-d", nargs="+", type=float)
    p.add_argument("--weights-bp", nargs="+", type=float)
    p.add_argument("--posterior-name", default=None,
                   help="npz filename (default auto best_/paper_best_posteriors.npz); "
                        "point at your 60 s posteriors here")
    p.add_argument("--workers", type=int, default=13, help="passed to tune_thresholds")
    p.add_argument("--out", default="loso_eval_result.json")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.checkpoints:
        raise SystemExit("provide --checkpoints (or --selftest)")

    n = len(args.checkpoints)
    weights = {
        "bmabz": args.weights_bmabz if args.weights_bmabz else [1.0] * n,
        "d":     args.weights_d if args.weights_d else [1.0] * n,
        "bp":    args.weights_bp if args.weights_bp else [1.0] * n,
    }

    print(f"{'='*80}\nLOSO evaluation over {len(cfg.VAL_DATASETS)} val sites\n{'='*80}")
    print("Loading members + combining posteriors ...")
    ens = load_combined(args.checkpoints, weights, args.posterior_name)

    print("Building val ground truth ...")
    cfg.USE_3CLASS = True
    from dataset_final import load_annotations, get_file_manifest
    from rescore_base_epochs import (
        build_gt_events, evaluate, paper_f1, per_class_block, tune_thresholds,
    )
    anns = load_annotations(list(cfg.VAL_DATASETS))
    manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    fsd = {(r["dataset"], r["filename"]): r["start_dt"] for _, r in manifest.iterrows()}
    gt = build_gt_events(anns, fsd)

    sites = list(cfg.VAL_DATASETS)
    print("\nAll-val (optimistic) ...")
    allval = run_allval(ens, gt, sites, tune_thresholds, evaluate, per_class_block, paper_f1, args.workers)
    print("LOSO (honest), 3 turns ...")
    loso_folds = run_loso(ens, gt, sites, tune_thresholds, evaluate, per_class_block, paper_f1, args.workers)

    pname = args.posterior_name or "best_posteriors.npz"
    report(loso_folds, allval, pname, n)

    loso_macro, loso_pc = aggregate([f["pc"] for f in loso_folds])
    Path(args.out).write_text(json.dumps(dict(
        loso_paper_f1=loso_macro, allval_paper_f1=allval[0],
        selection_bias_gap=loso_macro - allval[0],
        loso_per_class={k: dict(precision=v[0], recall=v[1], f1=v[2]) for k, v in loso_pc.items()},
        per_site_deployed=allval[3],
        loso_folds=loso_folds, posterior_name=pname), indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
