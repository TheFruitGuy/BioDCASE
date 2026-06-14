"""
make_val_holdout.py — stratified in-distribution validation holdout
===================================================================

Carve a small *file-level* validation set out of all 11 labelled sites so that
every site still appears in training (generalisation + kerguelen-domain
coverage) while a SPREAD of held-out files gives a stable checkpoint-selection
/ collapse-monitoring signal for the all-11 retrain.

READ THIS BEFORE TRUSTING ANY NUMBER IT PRODUCES
------------------------------------------------
This split is *in-distribution*: every held-out file comes from a site that is
also in training. It is a TRAINING MONITOR — use it to pick the epoch and to
catch a run that has collapsed. It is NOT a generalisation estimate: the
challenge eval set (kerguelen2019/2020, ddu2019/2021) is *unseen sites*. Keep
the LOSO site-held-out number as the reported estimate. Do not tune thresholds
on this set.

Selection model (rewritten — was an event-target grab, which pathologically
pulled the 1-2 densest chorus files and gutted training)
---------------------------------------------------------------------------
The monitor is built by *sampling representative files*, not by hitting big
event totals:

1. **Dense-chorus files stay in training.** Any file holding more than
   EXCLUDE_DENSE_FRAC of a class's total events is removed from candidacy.
   These are your most valuable training files (and unrepresentative as a
   monitor); they are never held out.
2. **One representative file from EVERY site** (base pass), so the monitor is
   spread across all 11 soundscapes and not two choruses. Eval-domain
   (Kerguelen) sites are capped tighter to protect scarce D.
3. **Per-class floors, not targets.** After the base pass, under-floor classes
   are topped up scarcest-first — floors are *minimums* for a stable F1, not
   amounts to maximise.
4. **Training-retention cap.** No more than RETAIN_FRAC of any class's total
   events may leave training. This is the hard guard that the old version
   lacked — it is why a 200-event Kerguelen D file can no longer be held out.
5. **Eval-domain D floor.** A small floor of Kerguelen D events in the monitor
   so the D signal is eval-relevant; if it can't be met within the caps, the
   report says so and you lean on LOSO.
6. **Bout spacing.** Multiple files from one site stay >= MIN_FILE_GAP apart in
   time order.

Holding out whole ~1 h files is the leakage mitigation; residual within-site
optimism is expected (it's a monitor, not a metric).

Outputs
-------
* ``val_holdout.json`` : {"holdout": {dataset: [filename, ...]}, "meta": {...}}
* ``val_holdout.txt``  : one "dataset/filename" per line

Wiring (train_final.py --holdout val_holdout.json does this for you)
--------------------------------------------------------------------
    import make_val_holdout as mvh
    holdout = mvh.load_holdout("val_holdout.json")
    tr_manifest, tr_anns, va_manifest, va_anns = mvh.split_by_holdout(
        manifest, anns, holdout)

Usage
-----
    python make_val_holdout.py
    python make_val_holdout.py --files-per-site 1 --max-per-site 2 \
        --exclude-dense-frac 0.02 --retain-frac 0.08
    python make_val_holdout.py --selftest
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config_final as cfg

CLASSES = ["bmabz", "d", "bp"]

# ----------------------------------------------------------------------
# Defaults (overridable on the command line)
# ----------------------------------------------------------------------
ALL_SITES = list(cfg.TRAIN_DATASETS) + list(cfg.VAL_DATASETS)

#: Per-class minimum held-out events (FLOOR, not a target). Just enough for a
#: non-noisy per-class F1; the per-site base pass usually clears these anyway.
FLOOR_PER_CLASS = {"bmabz": 50, "d": 30, "bp": 50}

#: Fill the scarcest class first when topping up to floors.
PREFER_ORDER = ["d", "bp", "bmabz"]

#: Base pass: representative files held out per (non-eval) site.
FILES_PER_SITE = 1
#: Hard cap per (non-eval) site (floor pass may add up to this).
MAX_FILES_PER_SITE = 2
#: Min separation (time-ordered index) between two held-out files in one site.
MIN_FILE_GAP_WITHIN_SITE = 2

#: A file holding more than this fraction of ANY class's total events is a
#: dense chorus — kept in training, never a monitor candidate.
EXCLUDE_DENSE_FRAC = 0.02
#: Never hold more than this fraction of any class's total events out of
#: training. The hard training-retention guard.
RETAIN_FRAC = 0.08

#: Eval-domain (Kerguelen) sites: scarce + the eval domain.
EVAL_DOMAIN_SITES = ["kerguelen2005", "kerguelen2014", "kerguelen2015"]
#: Tighter file cap for eval-domain sites (protect Kerguelen training D).
EVAL_DOMAIN_MAX_PER_SITE = 1
#: Floor on held-out D events drawn from eval-domain sites (0 disables).
MIN_EVAL_DOMAIN_D = 12

SEED = cfg.SEED
OUT_JSON = "val_holdout.json"
OUT_TXT = "val_holdout.txt"


# ======================================================================
# Counting + ordering
# ======================================================================
def per_file_counts(anns) -> dict[tuple[str, str], dict[str, int]]:
    """{(dataset, filename): {class: n_events}} over the 3 coarse classes."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    grp = (anns.groupby(["dataset", "filename", "label_3class"])
                .size().reset_index(name="n"))
    for _, r in grp.iterrows():
        cls = r["label_3class"]
        if cls not in CLASSES:
            continue
        key = (r["dataset"], r["filename"])
        counts.setdefault(key, {c: 0 for c in CLASSES})[cls] += int(r["n"])
    return counts


def ordered_files_per_site(manifest) -> dict[str, list[str]]:
    """{dataset: [filename, ...]} sorted by file start time (NaT/unparsed last,
    ordered by filename as a stable fallback)."""
    out: dict[str, list[str]] = {}
    for ds, sub in manifest.groupby("dataset"):
        sub = sub.copy()
        sub["_nat"] = sub["start_dt"].isna()
        sub = sub.sort_values(["_nat", "start_dt", "filename"])
        out[ds] = sub["filename"].tolist()
    return out


# ======================================================================
# Selection: representative per-site sampling + floors + guards
# ======================================================================
def select_holdout(counts, ordered, floors, files_per_site, max_per_site,
                   min_gap, prefer_order, eval_domain_sites=frozenset(),
                   eval_domain_max_per_site=1, min_eval_domain_d=0,
                   exclude_dense_frac=0.02, retain_frac=0.08, seed=0):
    """Sample a spread of representative files. Returns
    (selected, achieved, exhausted, eval_info)."""
    rng = random.Random(seed)
    eval_domain_sites = set(eval_domain_sites)
    pos = {(ds, fn): i for ds, fns in ordered.items() for i, fn in enumerate(fns)}

    totals = {c: 0 for c in CLASSES}
    for v in counts.values():
        for c in CLASSES:
            totals[c] += v.get(c, 0)

    def has_events(key):
        return sum(counts.get(key, {}).get(c, 0) for c in CLASSES) > 0

    def is_outlier(key):
        c = counts.get(key, {})
        return any(totals[cl] > 0 and c.get(cl, 0) > exclude_dense_frac * totals[cl]
                   for cl in CLASSES)

    candidate = {k for k in counts if has_events(k) and not is_outlier(k)}
    n_outliers = sum(1 for k in counts if has_events(k) and is_outlier(k))

    selected = {ds: [] for ds in ordered}
    sel_idx = {ds: [] for ds in ordered}
    chosen: set = set()
    achieved = {c: 0 for c in CLASSES}
    eval_domain_d = 0

    def site_cap(ds):
        return (eval_domain_max_per_site if ds in eval_domain_sites
                else max_per_site)

    def spaced(ds, fn):
        i = pos[(ds, fn)]
        return all(abs(i - j) >= min_gap for j in sel_idx[ds])

    def retain_ok(key):
        c = counts[key]
        return all(totals[cl] == 0 or
                   achieved[cl] + c.get(cl, 0) <= retain_frac * totals[cl] + 1e-9
                   for cl in CLASSES)

    def eligible(key, limit):
        ds, fn = key
        return (key in candidate and key not in chosen
                and len(selected[ds]) < limit
                and spaced(ds, fn) and retain_ok(key))

    def take(key):
        nonlocal eval_domain_d
        ds, fn = key
        chosen.add(key)
        selected[ds].append(fn)
        sel_idx[ds].append(pos[key])
        for c in CLASSES:
            achieved[c] += counts[key].get(c, 0)
        if ds in eval_domain_sites:
            eval_domain_d += counts[key].get("d", 0)

    # --- Base pass: one representative file from EVERY site -------------
    for ds in sorted(ordered):
        limit = (eval_domain_max_per_site if ds in eval_domain_sites
                 else files_per_site)
        while len(selected[ds]) < limit:
            cands = [(ds, fn) for fn in ordered[ds]
                     if eligible((ds, fn), limit)]
            if not cands:
                break
            # representative = carries the scarce class, then most total
            best = max(cands, key=lambda k: (counts[k].get("d", 0),
                                             sum(counts[k].values()),
                                             rng.random()))
            take(best)

    # --- Eval-domain D floor (Kerguelen D in the monitor) --------------
    eval_domain_short = False
    if eval_domain_sites and min_eval_domain_d > 0:
        while eval_domain_d < min_eval_domain_d:
            cands = [k for k in candidate
                     if k[0] in eval_domain_sites and k not in chosen
                     and counts[k].get("d", 0) > 0
                     and eligible(k, eval_domain_max_per_site)]
            if not cands:
                eval_domain_short = True
                break
            take(max(cands, key=lambda k: (counts[k]["d"], rng.random())))

    # --- Floor pass: top up under-floor classes, scarcest first --------
    exhausted: set = set()
    while True:
        under = [c for c in CLASSES
                 if achieved[c] < floors.get(c, 0) and c not in exhausted]
        if not under:
            break
        target_c = min(under, key=lambda c: (prefer_order.index(c)
                                             if c in prefer_order else 99))
        cands = [k for k in candidate
                 if k not in chosen and counts[k].get(target_c, 0) > 0
                 and eligible(k, site_cap(k[0]))]
        if not cands:
            exhausted.add(target_c)  # floor unmeetable (retain cap or no files)
            continue
        take(max(cands, key=lambda k: (counts[k][target_c], rng.random())))

    selected = {ds: fns for ds, fns in selected.items() if fns}
    eval_info = {
        "eval_domain_d": eval_domain_d,
        "min_eval_domain_d": min_eval_domain_d,
        "eval_domain_short": eval_domain_short,
        "eval_domain_sites": sorted(eval_domain_sites),
        "totals": totals,
        "held_frac": {c: (achieved[c] / totals[c] if totals[c] else 0.0)
                      for c in CLASSES},
        "n_candidates": len(candidate),
        "n_outliers_excluded": n_outliers,
        "retain_frac": retain_frac,
        "exclude_dense_frac": exclude_dense_frac,
    }
    return selected, achieved, exhausted, eval_info


# ======================================================================
# Reporting + IO
# ======================================================================
def report(selected, achieved, exhausted, eval_info, floors, counts,
           manifest, eval_domain_sites):
    bar = "=" * 78
    eval_domain_sites = set(eval_domain_sites)
    dur = {(r["dataset"], r["filename"]): r["duration_s"]
           for _, r in manifest.iterrows()}
    n_files = sum(len(v) for v in selected.values())
    tot_h = sum(dur.get((ds, fn), 0.0)
                for ds, fns in selected.items() for fn in fns) / 3600.0

    print(f"\n{bar}\nSTRATIFIED VAL HOLDOUT  (in-distribution monitor — NOT a "
          f"generalisation metric)\n{bar}")
    print(f"  files: {n_files}   across {len(selected)} site(s)   "
          f"~{tot_h:.1f} h held out")
    print(f"  candidates after dense-file exclusion: {eval_info['n_candidates']} "
          f"({eval_info['n_outliers_excluded']} dense files kept in training, "
          f">{eval_info['exclude_dense_frac']:.0%} of a class)")

    print(f"\n  per-class held-out events (floor -> held, % of class total):")
    for c in CLASSES:
        frac = eval_info["held_frac"][c]
        flag = ""
        if c in exhausted:
            near = frac >= eval_info["retain_frac"] - 1e-6
            flag = ("  <-- floor not met: retain cap bound" if near
                    else "  <-- floor not met: no eligible files")
        print(f"    {c:6} floor {floors.get(c, 0):4d} -> held {achieved[c]:5d}  "
              f"({frac:5.1%} of {eval_info['totals'][c]:,}){flag}")
    print(f"  (retain cap = {eval_info['retain_frac']:.0%} of each class stays "
          f"available for training)")

    floor = eval_info["min_eval_domain_d"]
    if floor > 0:
        got = eval_info["eval_domain_d"]
        if eval_info["eval_domain_short"]:
            print(f"\n  eval-domain D floor: {got}/{floor}  <-- UNDER within "
                  f"the eval-domain cap. Raise --eval-domain-max-per-site "
                  f"(costs training Kerguelen D) or lower --min-eval-domain-d "
                  f"and lean on LOSO for the domain number.")
        else:
            print(f"\n  eval-domain D floor: {got}/{floor}  (from "
                  f"{', '.join(eval_info['eval_domain_sites'])})")

    print(f"\n  per-site selection (time-ordered):")
    for ds in sorted(selected):
        tag = "  [eval-domain]" if ds in eval_domain_sites else ""
        print(f"    {ds}{tag}")
        for fn in selected[ds]:
            c = counts.get((ds, fn), {})
            print(f"        {fn:40s}  "
                  f"bmabz={c.get('bmabz', 0):3d} d={c.get('d', 0):3d} "
                  f"bp={c.get('bp', 0):3d}")
    print(bar)


def write_outputs(selected, achieved, exhausted, eval_info, floors, params,
                  out_json, out_txt):
    meta = dict(floors=floors, achieved=achieved,
                exhausted=sorted(exhausted),
                held_frac=eval_info["held_frac"],
                class_totals=eval_info["totals"],
                n_candidates=eval_info["n_candidates"],
                n_outliers_excluded=eval_info["n_outliers_excluded"],
                eval_domain_d=eval_info["eval_domain_d"],
                eval_domain_short=eval_info["eval_domain_short"],
                params=params,
                note="in-distribution monitor; not a generalisation estimate")
    Path(out_json).write_text(json.dumps(
        dict(holdout=selected, meta=meta), indent=2))
    lines = [f"{ds}/{fn}" for ds in sorted(selected) for fn in selected[ds]]
    Path(out_txt).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_json}  and  {out_txt}  ({len(lines)} files)")


def load_holdout(path="val_holdout.json") -> dict[str, list[str]]:
    """Load the {dataset: [filename, ...]} holdout map written by this script."""
    return json.loads(Path(path).read_text())["holdout"]


def split_by_holdout(manifest, anns, holdout):
    """Split manifest + annotations into (train, val) by the holdout file set.
    Returns (train_manifest, train_anns, val_manifest, val_anns)."""
    hold = {(ds, fn) for ds, fns in holdout.items() for fn in fns}
    m_is_val = manifest.apply(
        lambda r: (r["dataset"], r["filename"]) in hold, axis=1)
    a_is_val = anns.apply(
        lambda r: (r["dataset"], r["filename"]) in hold, axis=1)
    return (manifest[~m_is_val].reset_index(drop=True),
            anns[~a_is_val].reset_index(drop=True),
            manifest[m_is_val].reset_index(drop=True),
            anns[a_is_val].reset_index(drop=True))


# ======================================================================
# Self-test (synthetic; no disk / loaders)
# ======================================================================
def selftest():
    print("[selftest] selection logic ...")
    sites = [f"site{s}" for s in range(4)]
    ordered = {s: [f"f{i}" for i in range(25)] for s in sites}
    counts = {}
    for s in sites:
        for i in range(25):
            counts[(s, f"f{i}")] = {"bmabz": 4, "d": 2, "bp": 4}
    # one dense chorus file -> must be excluded and kept in training
    counts[("site0", "f12")] = {"bmabz": 40, "d": 40, "bp": 40}
    floors = {"bmabz": 10, "d": 10, "bp": 10}

    sel, ach, exh, info = select_holdout(
        counts, ordered, floors, files_per_site=1, max_per_site=2, min_gap=2,
        prefer_order=["d", "bp", "bmabz"], exclude_dense_frac=0.02,
        retain_frac=0.08, seed=0)

    # dense chorus never held out
    assert ("site0", "f12") not in {(ds, fn) for ds, fns in sel.items()
                                    for fn in fns}, "dense file was held out!"
    # spread: every site represented
    assert set(sel) == set(sites), ("missing sites", sel.keys())
    # floors met
    assert all(ach[c] >= floors[c] for c in CLASSES), (ach, floors)
    # training-retention cap respected
    assert all(info["held_frac"][c] <= info["retain_frac"] + 1e-9
               for c in CLASSES), info["held_frac"]
    # spacing
    pos = {(ds, fn): i for ds, fns in ordered.items() for i, fn in enumerate(fns)}
    for ds, fns in sel.items():
        idx = sorted(pos[(ds, fn)] for fn in fns)
        assert all(b - a >= 2 for a, b in zip(idx, idx[1:])), (ds, idx)
    # determinism
    sel2, *_ = select_holdout(counts, ordered, floors, 1, 2, 2,
                              ["d", "bp", "bmabz"], exclude_dense_frac=0.02,
                              retain_frac=0.08, seed=0)
    assert sel == sel2, "non-deterministic for fixed seed"

    # eval-domain D floor: meetable
    _, _, _, info_e = select_holdout(
        counts, ordered, floors, 1, 2, 2, ["d", "bp", "bmabz"],
        eval_domain_sites={"site0"}, eval_domain_max_per_site=1,
        min_eval_domain_d=2, exclude_dense_frac=0.02, retain_frac=0.08, seed=0)
    assert info_e["eval_domain_d"] >= 2 and not info_e["eval_domain_short"], info_e

    # eval-domain D floor: unmeetable within cap -> flagged, not looped
    sel_s, _, _, info_s = select_holdout(
        counts, ordered, floors, 1, 2, 2, ["d", "bp", "bmabz"],
        eval_domain_sites={"site0"}, eval_domain_max_per_site=1,
        min_eval_domain_d=999, exclude_dense_frac=0.02, retain_frac=0.08, seed=0)
    assert info_s["eval_domain_short"], info_s
    assert len(sel_s.get("site0", [])) <= 1, sel_s

    # retention cap binds when a floor exceeds the cap
    _, ach_r, exh_r, info_r = select_holdout(
        counts, ordered, {"bmabz": 0, "d": 9999, "bp": 0}, 1, 5, 2,
        ["d", "bp", "bmabz"], exclude_dense_frac=0.02, retain_frac=0.08, seed=0)
    assert "d" in exh_r and info_r["held_frac"]["d"] <= 0.08 + 1e-9, info_r

    # split_by_holdout filters both frames
    import pandas as pd
    manifest = pd.DataFrame([{"dataset": "site0", "filename": f"f{i}",
                              "start_dt": pd.Timestamp("2014-01-01", tz="UTC"),
                              "duration_s": 3600.0} for i in range(25)])
    annsdf = pd.DataFrame([{"dataset": "site0", "filename": f"f{i % 25}",
                            "label_3class": "d"} for i in range(40)])
    trm, tra, vam, vaa = split_by_holdout(manifest, annsdf, {"site0": ["f1"]})
    assert set(vam["filename"]) == {"f1"} and "f1" not in set(trm["filename"])
    assert set(vaa["filename"]) == {"f1"} and "f1" not in set(tra["filename"])

    print(f"  base: {sum(len(v) for v in sel.values())} files across "
          f"{len(sel)} sites; held_frac={ {k: round(v,3) for k,v in info['held_frac'].items()} }")
    print(f"  dense files excluded: {info['n_outliers_excluded']}; "
          f"eval floor meetable={not info_e['eval_domain_short']}; "
          f"retain-cap binds on forced floor={'d' in exh_r}")
    print("[selftest] OK")


# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bmabz", type=int, default=FLOOR_PER_CLASS["bmabz"],
                   help="Min held-out BMABZ events (floor).")
    p.add_argument("--d", type=int, default=FLOOR_PER_CLASS["d"],
                   help="Min held-out D events (floor).")
    p.add_argument("--bp", type=int, default=FLOOR_PER_CLASS["bp"],
                   help="Min held-out BP events (floor).")
    p.add_argument("--files-per-site", type=int, default=FILES_PER_SITE,
                   help=f"Representative files per site in the base pass "
                        f"(default {FILES_PER_SITE}).")
    p.add_argument("--max-per-site", type=int, default=MAX_FILES_PER_SITE,
                   help=f"Hard cap of held-out files per site (default "
                        f"{MAX_FILES_PER_SITE}).")
    p.add_argument("--min-gap", type=int, default=MIN_FILE_GAP_WITHIN_SITE)
    p.add_argument("--exclude-dense-frac", type=float, default=EXCLUDE_DENSE_FRAC,
                   help="Files holding more than this fraction of a class's "
                        f"total events are kept in training (default "
                        f"{EXCLUDE_DENSE_FRAC}).")
    p.add_argument("--retain-frac", type=float, default=RETAIN_FRAC,
                   help="Max fraction of any class's events that may leave "
                        f"training (default {RETAIN_FRAC}).")
    p.add_argument("--eval-domain-sites", nargs="+", default=EVAL_DOMAIN_SITES,
                   metavar="SITE")
    p.add_argument("--eval-domain-max-per-site", type=int,
                   default=EVAL_DOMAIN_MAX_PER_SITE)
    p.add_argument("--min-eval-domain-d", type=int, default=MIN_EVAL_DOMAIN_D)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-txt", default=OUT_TXT)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return

    unknown = [s for s in args.eval_domain_sites if s not in ALL_SITES]
    if unknown:
        raise SystemExit(f"--eval-domain-sites contains unknown site(s) "
                         f"{unknown}; valid: {ALL_SITES}")

    floors = {"bmabz": args.bmabz, "d": args.d, "bp": args.bp}
    print(f"Loading manifest + annotations for {len(ALL_SITES)} sites ...")
    from dataset_final import get_file_manifest, load_annotations
    manifest = get_file_manifest(ALL_SITES)
    anns = load_annotations(ALL_SITES, manifest=manifest)

    counts = per_file_counts(anns)
    ordered = ordered_files_per_site(manifest)

    selected, achieved, exhausted, eval_info = select_holdout(
        counts, ordered, floors,
        files_per_site=args.files_per_site, max_per_site=args.max_per_site,
        min_gap=args.min_gap, prefer_order=PREFER_ORDER,
        eval_domain_sites=args.eval_domain_sites,
        eval_domain_max_per_site=args.eval_domain_max_per_site,
        min_eval_domain_d=args.min_eval_domain_d,
        exclude_dense_frac=args.exclude_dense_frac,
        retain_frac=args.retain_frac, seed=args.seed)

    report(selected, achieved, exhausted, eval_info, floors, counts,
           manifest, args.eval_domain_sites)

    params = dict(files_per_site=args.files_per_site,
                  max_per_site=args.max_per_site, min_gap=args.min_gap,
                  exclude_dense_frac=args.exclude_dense_frac,
                  retain_frac=args.retain_frac, seed=args.seed,
                  eval_domain_sites=args.eval_domain_sites,
                  eval_domain_max_per_site=args.eval_domain_max_per_site,
                  min_eval_domain_d=args.min_eval_domain_d)
    write_outputs(selected, achieved, exhausted, eval_info, floors, params,
                  args.out_json, args.out_txt)


if __name__ == "__main__":
    main()
