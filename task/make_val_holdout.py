"""
make_val_holdout.py — stratified in-distribution validation holdout
===================================================================

Carve a small *file-level* validation set out of all 11 labelled sites so that
every site still appears in training (generalisation + kerguelen-domain
coverage) while a handful of held-out files give a checkpoint-selection /
collapse-monitoring signal for the all-11 retrain.

READ THIS BEFORE TRUSTING ANY NUMBER IT PRODUCES
------------------------------------------------
This split is *in-distribution*: every held-out file comes from a site that is
also in training. It is a TRAINING MONITOR — use it to pick the epoch and to
catch a run that has collapsed. It is NOT a generalisation estimate: the
challenge eval set (kerguelen2019/2020, ddu2019/2021) is *unseen sites*, and a
within-site split systematically overstates eval macro-F1 (worst for D, a
soundscape-sensitivity problem). Keep the LOSO site-held-out number as the
reported estimate. Do not tune thresholds on this set — use the LOSO-honest
thresholds.

What the selector guarantees
----------------------------
1. Per-class call-count targets, scarcest class filled FIRST, so D is not
   starved (the usual failure mode of a "few random files" holdout).
2. Spread across sites (cap per site) rather than concentrated in one.
3. Bout spacing: selected files within a site are kept >= MIN_FILE_GAP apart in
   time order, so two held-out files are not two halves of the same chorus.
4. Eval-domain handling (NEW). The challenge eval set is entirely Kerguelen +
   DDU, and the only labelled Kerguelen data are kerguelen2005/2014/2015 (there
   is no labelled DDU at all). Kerguelen D is therefore both scarce AND the
   eval domain, so it is wanted in two competing places — training and the
   monitor. The split resolves the tension by:
     * capping eval-domain sites at EVAL_DOMAIN_MAX_PER_SITE files (default 1),
       so the bulk of the scarce Kerguelen D stays in TRAINING;
     * enforcing a small floor of MIN_EVAL_DOMAIN_D held-out D events drawn
       from eval-domain sites, so the D monitor still carries Kerguelen signal
       rather than being filled entirely from the D-rich elephantisland sites
       (which together hold ~90% of all D but are the wrong domain).
   If the floor cannot be met within the cap, the report warns and you should
   lean on the LOSO number rather than spend more scarce Kerguelen D on the
   monitor.

It does NOT remove a held-out file's training-side temporal neighbours (that
would defeat "all sites in training"); holding out the whole ~1 h file is the
leakage mitigation, and residual within-site optimism is expected.

Outputs
-------
* ``val_holdout.json`` : {"holdout": {dataset: [filename, ...]}, "meta": {...}}
* ``val_holdout.txt``  : one "dataset/filename" per line

Wiring into a training run (see ``split_by_holdout`` at the bottom)
-------------------------------------------------------------------
    import make_val_holdout as mvh
    holdout = mvh.load_holdout("val_holdout.json")
    tr_manifest, tr_anns, va_manifest, va_anns = mvh.split_by_holdout(
        manifest, anns, holdout)
    # feed tr_* to build_positive/negative_segments, va_* to build_val_segments
(train_final.py --holdout val_holdout.json does this for you.)

Usage
-----
    python make_val_holdout.py                 # use the defaults below
    python make_val_holdout.py --d 40 --bmabz 60 --bp 60 --max-per-site 3
    python make_val_holdout.py --min-eval-domain-d 12 \
        --eval-domain-sites kerguelen2005 kerguelen2014 kerguelen2015 \
        --eval-domain-max-per-site 1
    python make_val_holdout.py --min-eval-domain-d 0   # disable the D floor
    python make_val_holdout.py --selftest      # synthetic-data logic check
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
#: All labelled sites = the 8 train + 3 (formerly) val sites.
ALL_SITES = list(cfg.TRAIN_DATASETS) + list(cfg.VAL_DATASETS)

#: Target number of held-out call EVENTS per coarse class. D is low because it
#: is scarce; raise it only if the report says it could be met.
TARGET_PER_CLASS = {"bmabz": 60, "d": 40, "bp": 60}

#: Fill the scarcest class first so D drives selection.
PREFER_ORDER = ["d", "bp", "bmabz"]

#: Max held-out files per site (spread, don't concentrate).
MAX_FILES_PER_SITE = 3

#: Min separation (in time-ordered file index) between two held-out files in
#: the same site. 2 => never two consecutive files => different bouts.
MIN_FILE_GAP_WITHIN_SITE = 2

#: Sites that proxy the (unseen) eval domain. The eval set is Kerguelen + DDU;
#: these are the labelled Kerguelen sites. Add casey* if you also want to
#: protect the East-Antarctic (DDU-proxy) sites from being held out.
EVAL_DOMAIN_SITES = ["kerguelen2005", "kerguelen2014", "kerguelen2015"]

#: Tighter file cap for eval-domain sites: keep the scarce Kerguelen D in
#: training, allow at most this many files into the monitor.
EVAL_DOMAIN_MAX_PER_SITE = 1

#: Floor on held-out D events that must come from eval-domain sites, so the D
#: monitor carries Kerguelen signal. Set 0 to disable.
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
# Greedy stratified selection
# ======================================================================
def select_holdout(counts, ordered, targets, max_per_site, min_gap,
                   prefer_order, eval_domain_sites=frozenset(),
                   eval_domain_max_per_site=1, min_eval_domain_d=0, seed=0):
    """Greedily pick files to hit per-class targets, scarcest class first,
    respecting per-site caps and within-site spacing, with an optional
    eval-domain D floor selected first.

    Returns (selected: {ds: [fn]}, achieved: {cls: int}, exhausted: set[str],
             eval_info: dict).
    """
    rng = random.Random(seed)
    eval_domain_sites = set(eval_domain_sites)
    pos = {(ds, fn): i for ds, fns in ordered.items() for i, fn in enumerate(fns)}
    selected: dict[str, list[str]] = {ds: [] for ds in ordered}
    sel_idx: dict[str, list[int]] = {ds: [] for ds in ordered}
    chosen: set[tuple[str, str]] = set()
    achieved = {c: 0 for c in CLASSES}
    exhausted: set[str] = set()
    eval_domain_d = 0

    def site_cap(ds):
        return (eval_domain_max_per_site if ds in eval_domain_sites
                else max_per_site)

    def eligible(ds, fn):
        if len(selected[ds]) >= site_cap(ds):
            return False
        i = pos[(ds, fn)]
        return all(abs(i - j) >= min_gap for j in sel_idx[ds])

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

    # --- Phase 1: eval-domain D floor (keep Kerguelen D in the monitor) ---
    eval_domain_short = False
    if eval_domain_sites and min_eval_domain_d > 0:
        while eval_domain_d < min_eval_domain_d:
            cands = [k for k in counts
                     if k not in chosen and k[0] in eval_domain_sites
                     and counts[k].get("d", 0) > 0 and eligible(*k)]
            if not cands:
                eval_domain_short = True
                break
            best = max(cands, key=lambda k: (counts[k]["d"], rng.random()))
            take(best)

    # --- Phase 2: general stratified fill (scarcest class first) ---
    def useful(key):
        c = counts.get(key, {})
        return sum(min(max(targets[cl] - achieved[cl], 0), c.get(cl, 0))
                   for cl in CLASSES)

    while True:
        under = [c for c in CLASSES
                 if achieved[c] < targets.get(c, 0) and c not in exhausted]
        if not under:
            break
        target_c = min(under, key=lambda c: (prefer_order.index(c)
                                             if c in prefer_order else 99))
        cands = [k for k in counts
                 if k not in chosen and counts[k].get(target_c, 0) > 0
                 and eligible(*k)]
        if not cands:
            exhausted.add(target_c)
            continue
        best = max(cands, key=lambda k: (counts[k][target_c], useful(k),
                                         rng.random()))
        take(best)

    selected = {ds: fns for ds, fns in selected.items() if fns}
    eval_info = {"eval_domain_d": eval_domain_d,
                 "min_eval_domain_d": min_eval_domain_d,
                 "eval_domain_short": eval_domain_short,
                 "eval_domain_sites": sorted(eval_domain_sites)}
    return selected, achieved, exhausted, eval_info


# ======================================================================
# Reporting + IO
# ======================================================================
def report(selected, achieved, exhausted, eval_info, targets, counts,
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

    print(f"\n  per-class held-out call events (target -> achieved):")
    for c in CLASSES:
        flag = "  <-- UNDER (could not be met under caps)" if c in exhausted else ""
        print(f"    {c:6} {targets.get(c, 0):4d} -> {achieved[c]:4d}{flag}")

    # eval-domain D floor status
    floor = eval_info["min_eval_domain_d"]
    if floor > 0:
        got = eval_info["eval_domain_d"]
        if eval_info["eval_domain_short"]:
            print(f"\n  eval-domain D floor: {got}/{floor}  <-- UNDER. Could "
                  f"not reach it within --eval-domain-max-per-site. Either "
                  f"raise that cap (costs training Kerguelen D) or lower "
                  f"--min-eval-domain-d and lean on LOSO for the domain number.")
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
    if exhausted:
        print(f"\n  NOTE: {sorted(exhausted)} below target. Raise "
              f"--max-per-site or lower that target; do not lower --min-gap "
              f"below 2 (that reintroduces bout leakage).")
    print(bar)


def write_outputs(selected, achieved, eval_info, targets, eval_cfg,
                  out_json, out_txt):
    meta = dict(targets=targets, achieved=achieved,
                max_files_per_site=eval_cfg["max_per_site"],
                min_file_gap=eval_cfg["min_gap"], seed=eval_cfg["seed"],
                eval_domain_sites=eval_cfg["eval_domain_sites"],
                eval_domain_max_per_site=eval_cfg["eval_domain_max_per_site"],
                min_eval_domain_d=eval_info["min_eval_domain_d"],
                eval_domain_d=eval_info["eval_domain_d"],
                eval_domain_short=eval_info["eval_domain_short"],
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
    """Split a manifest + annotations DataFrame into (train, val) by the holdout
    file set. Feed the train_* pair to the positive/negative segment builders
    and the val_* pair to build_val_segments.

    Returns (train_manifest, train_anns, val_manifest, val_anns).
    """
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
# Self-test (synthetic data; no disk / loaders needed)
# ======================================================================
def selftest():
    print("[selftest] selection logic ...")
    ordered = {f"site{s}": [f"f{i}" for i in range(8)] for s in range(3)}
    counts = {}
    for s in range(3):
        for i in range(8):
            counts[(f"site{s}", f"f{i}")] = {
                "bmabz": 5 if i % 2 == 0 else 1,
                "d": 4 if i in (1, 2, 6) else 0,
                "bp": 3,
            }
    targets = {"bmabz": 20, "d": 9, "bp": 15}
    pos = {(ds, fn): i for ds, fns in ordered.items() for i, fn in enumerate(fns)}

    # --- base case (no eval-domain): scarce-first, caps, spacing, determinism
    sel, ach, exh, _ = select_holdout(counts, ordered, targets,
                                      max_per_site=3, min_gap=2,
                                      prefer_order=["d", "bp", "bmabz"], seed=0)
    assert ach["d"] >= targets["d"], (ach, "D target not met")
    assert all(len(v) <= 3 for v in sel.values()), sel
    for ds, fns in sel.items():
        idx = sorted(pos[(ds, fn)] for fn in fns)
        assert all(b - a >= 2 for a, b in zip(idx, idx[1:])), (ds, idx)
    sel2, *_ = select_holdout(counts, ordered, targets, 3, 2,
                              ["d", "bp", "bmabz"], seed=0)
    assert sel == sel2, "non-deterministic for fixed seed"

    # --- eval-domain D floor: meetable within cap=1 across two eval sites
    ecounts = dict(counts)
    ecounts[("site0", "f0")] = {"bmabz": 5, "d": 5, "bp": 3}
    ecounts[("site1", "f0")] = {"bmabz": 5, "d": 5, "bp": 3}
    eval_sites = {"site0", "site1"}
    sel_e, ach_e, exh_e, info_e = select_holdout(
        ecounts, ordered, targets, max_per_site=3, min_gap=2,
        prefer_order=["d", "bp", "bmabz"], eval_domain_sites=eval_sites,
        eval_domain_max_per_site=1, min_eval_domain_d=8, seed=0)
    assert info_e["eval_domain_d"] >= 8, info_e
    assert not info_e["eval_domain_short"], info_e
    assert all(len(sel_e.get(s, [])) <= 1 for s in eval_sites), sel_e

    # --- eval-domain floor unmeetable under cap: flagged, not looped
    sel_s, ach_s, exh_s, info_s = select_holdout(
        ecounts, ordered, targets, max_per_site=3, min_gap=2,
        prefer_order=["d", "bp", "bmabz"], eval_domain_sites=eval_sites,
        eval_domain_max_per_site=1, min_eval_domain_d=99, seed=0)
    assert info_s["eval_domain_short"], info_s
    assert all(len(sel_s.get(s, [])) <= 1 for s in eval_sites), sel_s

    # --- split_by_holdout filters both frames
    import pandas as pd
    manifest = pd.DataFrame([{"dataset": "site0", "filename": f"f{i}",
                              "start_dt": pd.Timestamp("2014-01-01", tz="UTC"),
                              "duration_s": 3600.0} for i in range(8)])
    annsdf = pd.DataFrame([{"dataset": "site0", "filename": f"f{i % 8}",
                            "label_3class": "d"} for i in range(20)])
    trm, tra, vam, vaa = split_by_holdout(manifest, annsdf, {"site0": ["f1"]})
    assert set(vam["filename"]) == {"f1"} and "f1" not in set(trm["filename"])
    assert set(vaa["filename"]) == {"f1"} and "f1" not in set(tra["filename"])

    print(f"  base achieved={ach}  exhausted={sorted(exh)}")
    print(f"  eval-domain floor met: {info_e['eval_domain_d']} D "
          f"(short={info_e['eval_domain_short']}); "
          f"unmeetable case short={info_s['eval_domain_short']}")
    print("[selftest] OK")


# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bmabz", type=int, default=TARGET_PER_CLASS["bmabz"])
    p.add_argument("--d", type=int, default=TARGET_PER_CLASS["d"])
    p.add_argument("--bp", type=int, default=TARGET_PER_CLASS["bp"])
    p.add_argument("--max-per-site", type=int, default=MAX_FILES_PER_SITE)
    p.add_argument("--min-gap", type=int, default=MIN_FILE_GAP_WITHIN_SITE)
    p.add_argument("--eval-domain-sites", nargs="+", default=EVAL_DOMAIN_SITES,
                   metavar="SITE",
                   help="Labelled sites that proxy the (unseen) eval domain. "
                        f"Default: {EVAL_DOMAIN_SITES}.")
    p.add_argument("--eval-domain-max-per-site", type=int,
                   default=EVAL_DOMAIN_MAX_PER_SITE,
                   help="Max held-out files per eval-domain site (keeps scarce "
                        f"Kerguelen D in training). Default {EVAL_DOMAIN_MAX_PER_SITE}.")
    p.add_argument("--min-eval-domain-d", type=int, default=MIN_EVAL_DOMAIN_D,
                   help="Floor on held-out D events from eval-domain sites, so "
                        "the D monitor carries Kerguelen signal. 0 disables. "
                        f"Default {MIN_EVAL_DOMAIN_D}.")
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

    # Typo guard: every named eval-domain site must be a real labelled site.
    unknown = [s for s in args.eval_domain_sites if s not in ALL_SITES]
    if unknown:
        raise SystemExit(f"--eval-domain-sites contains unknown site(s) "
                         f"{unknown}; valid sites: {ALL_SITES}")

    targets = {"bmabz": args.bmabz, "d": args.d, "bp": args.bp}
    print(f"Loading manifest + annotations for {len(ALL_SITES)} sites ...")
    from dataset_final import get_file_manifest, load_annotations
    manifest = get_file_manifest(ALL_SITES)
    anns = load_annotations(ALL_SITES, manifest=manifest)

    counts = per_file_counts(anns)
    ordered = ordered_files_per_site(manifest)

    selected, achieved, exhausted, eval_info = select_holdout(
        counts, ordered, targets,
        max_per_site=args.max_per_site, min_gap=args.min_gap,
        prefer_order=PREFER_ORDER,
        eval_domain_sites=args.eval_domain_sites,
        eval_domain_max_per_site=args.eval_domain_max_per_site,
        min_eval_domain_d=args.min_eval_domain_d, seed=args.seed)

    report(selected, achieved, exhausted, eval_info, targets, counts,
           manifest, args.eval_domain_sites)

    eval_cfg = dict(max_per_site=args.max_per_site, min_gap=args.min_gap,
                    seed=args.seed, eval_domain_sites=args.eval_domain_sites,
                    eval_domain_max_per_site=args.eval_domain_max_per_site)
    write_outputs(selected, achieved, eval_info, targets, eval_cfg,
                  args.out_json, args.out_txt)


if __name__ == "__main__":
    main()
