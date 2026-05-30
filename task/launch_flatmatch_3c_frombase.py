"""
FlatMatch+HNM from-base orchestrator — five 3class base models
==============================================================

The FlatMatch sibling of ``launch_mt_3c_frombase.py``. Warm-starts a pure
cross-sharpness (FlatMatch, Huang et al. NeurIPS 2023) fine-tune from each of
the five 3class *base* models, with BOTH the seed-specific (solo) and the
ensemble hard negatives, HNM+PGI on. Everything except the unlabeled-side
objective is identical to the MT-from-base run — same base checkpoints, same
hard negatives, same AADC sites, same 20-epoch budget, same paper-F1
selection — so the per-seed deltas are PAIRED and directly comparable to the
MT numbers (FlatMatch's cross-sharpness vs MT's consistency, all else fixed).

Pure FlatMatch by default: lambda_xs=1.0, lambda_mt=0 (no MT term), mode 'e'
(the paper's efficient EMA-gradient variant), rho=0.1. To run the optional
"does FlatMatch stack on MT" A/B later, add ``--extra --lambda-mt 1.0``.

The matrix (10 runs, default)
-----------------------------
3class x {seed 42, 2024, 7777, 9999, 1337} x {solo, ens}, mode=e.
  solo -> runs/hardnegs_final/3class/seed{S}/*.json
  ens  -> runs/hardnegs_final/3class/ensemble/*.json
Warm start per seed -> runs/<base-3c-dir>/paper_best.pt.

GPUs
----
The GPU pool defaults to whatever ``CUDA_VISIBLE_DEVICES`` is set to when you
launch, so you stay flexible::

    CUDA_VISIBLE_DEVICES=3       python launch_flatmatch_3c_frombase.py ...   # 1 GPU
    CUDA_VISIBLE_DEVICES=3,4,8,9 python launch_flatmatch_3c_frombase.py ...   # 4 GPUs

Falls back to 3,4,8,9 if the env var is unset. ``--gpus`` overrides either.
Each run is pinned to one GPU via a per-subprocess CUDA_VISIBLE_DEVICES
override (physical ids — the launcher itself never touches CUDA).

AADC data
---------
train_flatmatch_final.py needs the unlabeled AADC audio: pass --aadc-root +
--aadc-sites. Use the SAME sites you gave the MT run so the comparison is
clean. --dry-run works without them.

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task

    python launch_flatmatch_3c_frombase.py --dry-run                  # inspect, no AADC

    # PILOT one cell first — watch epochs 1-2 for displacement (Phase-8 discipline):
    CUDA_VISIBLE_DEVICES=3 python launch_flatmatch_3c_frombase.py \
        --seeds 42 --sources ens \
        --aadc-root /home/matthias-nagl/BioDCASE/task/data_pretrain/audio \
        --aadc-sites Casey2018 DDU2018 DDU2019 Kerguelen2018 Kerguelen2019

    # full 10 across four GPUs, once the pilot looks sane:
    CUDA_VISIBLE_DEVICES=3,4,8,9 python launch_flatmatch_3c_frombase.py \
        --aadc-root /home/matthias-nagl/BioDCASE/task/data_pretrain/audio \
        --aadc-sites Casey2018 DDU2018 DDU2019 Kerguelen2018 Kerguelen2019 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

SEEDS = [42, 2024, 7777, 9999, 1337]
SOURCES = ["solo", "ens"]
MODES = ["e"]                              # FlatMatch eps* source (paper efficient variant)
TRAINER = "train_flatmatch_final.py"

# Base 3class run dirs by seed (same mapping as the MT launcher / miner).
CKPT_DIR_3C = {
    42:   "final_3c_s42_20260527_200054",
    2024: "final_3c_s2024_20260527_200309",
    7777: "final_3c_s7777_20260527_200401",
    9999: "final_3c_s9999_20260527_200852",
    1337: "final_3c_s1337_20260528_202023",
}

# rk10-specific: indices of RTX PRO 6000 Blackwell cards (sm_120) that crash the
# bio cu118 build. Warned about if present in the pool. Harmless on other boxes.
BLACKWELL_GPUS = {2, 7}

# Default GPU pool: honour a CUDA_VISIBLE_DEVICES prepended at launch so the
# user stays flexible; fall back to rk10's free L40S (3,4) + A6000 (8,9).
DEFAULT_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES") or "3,4,8,9"


def enumerate_jobs(runs_root, hn_root, ckpt_name, seeds, sources, modes, pgi):
    head = "3class"
    jobs = []
    for seed in seeds:
        base = Path(runs_root) / CKPT_DIR_3C[seed] / ckpt_name
        for source in sources:
            sub = f"seed{seed}" if source == "solo" else "ensemble"
            hn_glob = str(Path(hn_root) / head / sub / "*.json")
            hn_files = sorted(glob.glob(hn_glob))
            for mode in modes:
                name = f"fmfb_3class_s{seed}_{source}_{mode}"
                jobs.append({
                    "seed": seed, "source": source, "mode": mode, "pgi": pgi,
                    "ckpt": base, "hn_glob": hn_glob, "hn_files": hn_files,
                    "name": name,
                })
    return jobs


def build_cmd(job, val_workers, aadc_root, aadc_sites, out_dir, rho, lambda_xs, extra):
    cmd = [sys.executable, TRAINER,
           "--checkpoint", str(job["ckpt"]),
           "--hard-negatives", *job["hn_files"],
           "--seed", str(job["seed"]),
           "--run-name", job["name"],
           "--out-dir", out_dir,
           "--flatmatch-mode", job["mode"],
           "--rho", str(rho),
           "--lambda-xs", str(lambda_xs),
           "--aadc-root", str(aadc_root),
           "--aadc-sites", *aadc_sites,
           "--val-workers", str(val_workers)]
    if job["pgi"]:
        cmd.append("--isolate-classes")
    return cmd + extra


def tail_verdict(log_path: Path) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception:
        return ""
    for ln in reversed(lines[-60:]):
        if "paper-F1" in ln and "->" in ln:
            return ln.strip()
    return ""


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--hn-root", default="runs/hardnegs_final")
    p.add_argument("--ckpt-name", default="paper_best.pt",
                   help="Base-model checkpoint filename to warm-start from.")
    p.add_argument("--out-dir", default="runs",
                   help="Where the FlatMatch trainer writes run dirs.")
    p.add_argument("--gpus", default=DEFAULT_GPUS,
                   help="Comma-separated GPU ids. Defaults to CUDA_VISIBLE_DEVICES "
                        "if set at launch, else 3,4,8,9.")
    p.add_argument("--val-workers", type=int, default=0,
                   help="Threshold-sweep workers per run (0 = auto: cores//n_gpus, "
                        "capped at 13).")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--sources", nargs="+", choices=["solo", "ens"], default=SOURCES)
    p.add_argument("--modes", nargs="+", choices=["e", "full"], default=MODES,
                   help="FlatMatch eps* source(s). Default 'e' (paper efficient variant).")
    p.add_argument("--rho", type=float, default=0.1,
                   help="Perturbation magnitude (paper default 0.1). Forwarded to every run.")
    p.add_argument("--lambda-xs", type=float, default=1.0,
                   help="Cross-sharpness weight. Forwarded to every run.")
    p.add_argument("--pgi", choices=["on", "off"], default="on",
                   help="Per-class gradient isolation (HNM+PGI). Default on.")
    p.add_argument("--aadc-root", default=None,
                   help="Root dir of AADC unlabeled audio (forwarded to every run).")
    p.add_argument("--aadc-sites", nargs="+", default=None,
                   help="AADC site subfolders (forwarded to every run). Match the MT run.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose <out-dir>/<run-name>/best_model.pt exists.")
    p.add_argument("--logdir", default="runs/fmfb_logs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is passed through to the trainer "
                        "(e.g. --extra --lambda-mt 1.0 to stack MT alongside FlatMatch).")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    pgi = args.pgi == "on"
    n_cpu = os.cpu_count() or 8
    val_workers = args.val_workers or min(13, max(2, n_cpu // max(1, len(gpus))))
    omp = max(1, n_cpu // max(1, len(gpus)))

    bw = sorted(set(gpus) & BLACKWELL_GPUS)
    if bw:
        print(f"!! WARNING: GPU(s) {bw} look like RTX PRO 6000 Blackwell (sm_120) on "
              f"rk10 — the bio cu118 build crashes there at the LSTM. Drop them from "
              f"the pool unless you're on a different box.\n")

    jobs = enumerate_jobs(args.runs_root, args.hn_root, args.ckpt_name,
                          args.seeds, args.sources, args.modes, pgi)

    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs
                if not (Path(args.out_dir) / j["name"] / "best_model.pt").exists()]
        print(f"--skip-existing: {before - len(jobs)} already done, {len(jobs)} to run")

    # ---- plan + preflight -------------------------------------------
    print(f"GPUs: {gpus} | val-workers/run: {val_workers} | OMP threads/run: {omp} "
          f"| {n_cpu} cores")
    print(f"FlatMatch from-base 3class | sources={args.sources} | modes={args.modes} "
          f"| rho={args.rho} lambda_xs={args.lambda_xs} | PGI={'on' if pgi else 'off'}")
    if any(tok == "--lambda-mt" for tok in args.extra):
        print("  (note: --lambda-mt passed via --extra → stacking MT alongside FlatMatch)")
    print(f"{len(jobs)} run(s):")
    problems = []
    for i, j in enumerate(jobs):
        tag = f"{len(j['hn_files'])} json" if j["hn_files"] else "NO JSON"
        print(f"  [{i:2d}] {j['name']:42} <- {j['ckpt'].parent.name}/{args.ckpt_name}"
              f"  ({tag})")
        if not j["ckpt"].exists():
            problems.append(f"missing base checkpoint: {j['ckpt']}")
        if not j["hn_files"]:
            problems.append(f"no hard-neg JSONs: {j['hn_glob']} (run the miner first?)")

    aadc_ok = bool(args.aadc_root) and bool(args.aadc_sites)
    if not args.dry_run and not aadc_ok:
        problems.append("FlatMatch runs need --aadc-root and --aadc-sites "
                        "(omit only with --dry-run).")
    if args.aadc_root and not Path(args.aadc_root).exists():
        problems.append(f"AADC root does not exist: {args.aadc_root}")

    if problems:
        print("\nPREFLIGHT PROBLEMS:")
        for m in sorted(set(problems)):
            print(f"  - {m}")
        if not args.dry_run:
            raise SystemExit("Fix the above (or use --dry-run to inspect). Aborting.")
    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return
    if not jobs:
        print("Nothing to run.")
        return

    # ---- run: one worker thread per GPU, shared job queue -----------
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    results = []
    lock = threading.Lock()
    t_start = time.time()

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            cmd = build_cmd(j, val_workers, args.aadc_root, args.aadc_sites,
                            args.out_dir, args.rho, args.lambda_xs, args.extra)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OMP_NUM_THREADS"] = str(omp)
            env["MKL_NUM_THREADS"] = str(omp)
            log = logdir / f"{j['name']}.log"
            with lock:
                print(f"[GPU {gpu}] START {j['name']}  (log: {log})")
            t0 = time.time()
            with open(log, "w") as f:
                rc = subprocess.run(cmd, env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            dt = time.time() - t0
            verdict = tail_verdict(log)
            with lock:
                results.append((j["name"], gpu, rc, dt))
                flag = "OK " if rc == 0 else f"FAIL(rc={rc})"
                print(f"[GPU {gpu}] {flag} {j['name']}  {dt/60:.1f}min  {verdict}")
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ---- summary ----------------------------------------------------
    ok = sum(1 for _, _, rc, _ in results if rc == 0)
    print(f"\n{'='*66}\nFlatMatch+HNM from-base done in {(time.time()-t_start)/60:.1f} min — "
          f"{ok}/{len(results)} succeeded\n{'='*66}")
    for name, gpu, rc, dt in sorted(results):
        if rc != 0:
            print(f"  FAILED  {name}  (GPU {gpu}, rc={rc}) — see {logdir/(name+'.log')}")
    print(f"Logs: {logdir}/   |   checkpoints: {args.out_dir}/<run-name>/best_model.pt")


if __name__ == "__main__":
    main()
