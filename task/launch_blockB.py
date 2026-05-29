"""
Block B launcher — 40 HNM runs sharded across GPUs (sxrk09: 4x A40)
====================================================================

Enumerates the Block B matrix and runs it through ``train_hnm_final.py``,
one job per free GPU at a time. A40 is Ampere (sm_86) like the A6000s, so the
``bio`` cu118 PyTorch runs on all four — no Blackwell to avoid here.

The matrix (40 runs)
--------------------
{3class, 7class} x 5 seeds x {solo, ens} x {PGI on, off}.

Per (head, seed): the SAME base ``paper_best.pt``, four runs varying only the
hard-neg source and PGI:
  * solo -> runs/hardnegs_final/<head>/seed{S}/*.json   (3 files 3c, 7 files 7c)
  * ens  -> runs/hardnegs_final/<head>/ensemble/*.json
PGI on adds ``--isolate-classes``. Everything else is the trainer's locked
default (lr 1e-5, 15 epochs, oversample 5, macro-paper selection).

Scheduling
----------
A queue of jobs + one worker thread per GPU. Each worker pops the next job,
runs it with ``CUDA_VISIBLE_DEVICES`` pinned to its GPU, and pops again when
done — so a GPU that finishes early picks up the next job instead of idling.
Each run's stdout goes to its own log (4 concurrent runs would interleave
otherwise). ``OMP/MKL_NUM_THREADS`` is capped per process so the concurrent
torch processes don't oversubscribe the CPU.

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task

    # see the full plan + GPU assignment, run nothing
    python launch_blockB.py --dry-run

    # pilot ONE cell through the launcher first (3c, seed 42, ensemble, PGI on)
    python launch_blockB.py --heads 3class --seeds 42 --sources ens --pgi on

    # the full 40, across all four A40s
    python launch_blockB.py

    # resume: skip runs that already have a best_model.pt
    python launch_blockB.py --skip-existing
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

# Base run dirs keyed by seed (same mapping as the miner; symlink-independent).
CKPT_DIR = {
    "3class": {
        42:   "final_3c_s42_20260527_200054",
        2024: "final_3c_s2024_20260527_200309",
        7777: "final_3c_s7777_20260527_200401",
        9999: "final_3c_s9999_20260527_200852",
        1337: "final_3c_s1337_20260528_202023",
    },
    "7class": {
        42:   "final_7c_s42_20260528_204414",
        7777: "final_20260526_074308",
        9999: "final_20260526_074410",
        2024: "final_20260527_060332",
        1337: "final_20260527_141710",
    },
}

TRAINER = "train_hnm_final.py"


def enumerate_jobs(runs_root, hn_root, ckpt_name, heads, seeds, sources, pgis):
    jobs = []
    for head in heads:
        for seed in seeds:
            ckpt = Path(runs_root) / CKPT_DIR[head][seed] / ckpt_name
            for source in sources:
                sub = f"seed{seed}" if source == "solo" else "ensemble"
                hn_glob = str(Path(hn_root) / head / sub / "*.json")
                hn_files = sorted(glob.glob(hn_glob))
                for pgi in pgis:
                    name = f"hnm_{head}_s{seed}_{source}_{'pgi' if pgi else 'nopgi'}"
                    jobs.append({
                        "head": head, "seed": seed, "source": source, "pgi": pgi,
                        "ckpt": ckpt, "hn_glob": hn_glob, "hn_files": hn_files,
                        "name": name,
                    })
    return jobs


def build_cmd(job, val_workers, extra):
    cmd = [sys.executable, TRAINER,
           "--checkpoint", str(job["ckpt"]),
           "--hard-negatives", *job["hn_files"],
           "--seed", str(job["seed"]),
           "--run-name", job["name"],
           "--val-workers", str(val_workers)]
    if job["pgi"]:
        cmd.append("--isolate-classes")
    return cmd + extra


def tail_verdict(log_path: Path) -> str:
    """Pull the trainer's final 'paper-F1 X -> Y (+Z)' verdict line from a log."""
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
    p.add_argument("--ckpt-name", default="paper_best.pt")
    p.add_argument("--out-dir", default="runs",
                   help="Where the trainer writes run dirs (for --skip-existing).")
    p.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU ids.")
    p.add_argument("--val-workers", type=int, default=0,
                   help="Threshold-sweep workers per run (0 = auto: cores//n_gpus, "
                        "capped at 13 = grid points/class).")
    p.add_argument("--heads", nargs="+", choices=["3class", "7class"],
                   default=["3class", "7class"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--sources", nargs="+", choices=["solo", "ens"],
                   default=["solo", "ens"])
    p.add_argument("--pgi", nargs="+", choices=["on", "off"], default=["on", "off"])
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose <out-dir>/<run-name>/best_model.pt exists.")
    p.add_argument("--logdir", default="runs/blockB_logs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is passed through to the trainer.")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    pgis = [g == "on" for g in args.pgi]
    n_cpu = os.cpu_count() or 8
    val_workers = args.val_workers or min(13, max(2, n_cpu // max(1, len(gpus))))
    omp = max(1, n_cpu // max(1, len(gpus)))

    jobs = enumerate_jobs(args.runs_root, args.hn_root, args.ckpt_name,
                          args.heads, args.seeds, args.sources, pgis)

    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs
                if not (Path(args.out_dir) / j["name"] / "best_model.pt").exists()]
        print(f"--skip-existing: {before - len(jobs)} already done, {len(jobs)} to run")

    # ---- plan + preflight -------------------------------------------
    print(f"GPUs: {gpus} | val-workers/run: {val_workers} | OMP threads/run: {omp} "
          f"| {n_cpu} cores")
    print(f"{len(jobs)} run(s):")
    problems = []
    for i, j in enumerate(jobs):
        tag = f"{len(j['hn_files'])} json" if j["hn_files"] else "NO JSON"
        print(f"  [{i:2d}] {j['name']:34} pgi={'on ' if j['pgi'] else 'off'} "
              f"<- {j['hn_glob']}  ({tag})")
        if not j["ckpt"].exists():
            problems.append(f"missing checkpoint: {j['ckpt']}")
        if not j["hn_files"]:
            problems.append(f"no hard-neg JSONs: {j['hn_glob']} "
                            f"(run the miner first?)")

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
            cmd = build_cmd(j, val_workers, args.extra)
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
    print(f"\n{'='*66}\nBlock B done in {(time.time()-t_start)/60:.1f} min — "
          f"{ok}/{len(results)} succeeded\n{'='*66}")
    for name, gpu, rc, dt in sorted(results):
        if rc != 0:
            print(f"  FAILED  {name}  (GPU {gpu}, rc={rc}) — see {logdir/(name+'.log')}")
    print(f"Logs: {logdir}/   |   checkpoints: {args.out_dir}/<run-name>/best_model.pt")


if __name__ == "__main__":
    main()
