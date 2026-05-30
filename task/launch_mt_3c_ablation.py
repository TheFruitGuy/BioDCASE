"""
Block E ablation orchestrator — SAT / DA / verifier-gate on the MT+HNM baseline
===============================================================================

Runs the Block E *ablation ladder* (see whale_sed_experiment_plan.md §11) on the
five 3class base models, warm-started from base exactly like the from-base
diversity arm. The baseline is `asymmetric_mse` (highest d-class ensemble lift);
each arm toggles ONE thing on the same harness so any delta is attributable.

The ladder (arm -> consistency / lambda_da / verifier-gate)
----------------------------------------------------------
  A0  asymmetric_mse / 0.0 / off   baseline (usually REUSED from Block C; not in
                                   the default arms — pass --arms A0 to re-run it)
  A1  asymmetric_mse / 0.5 / off   + DA on the baseline           (isolates DA)
  A2  sat            / 0.0 / off   SAT vs fixed up-weighting       (isolates SAT)
  A3  sat            / 0.5 / off   SAT + DA               (SAT's intended pairing)
  A4  sat            / 0.5 / on    SAT + DA + verifier-gate    (D-precision rescue)

Default matrix = {A1,A2,A3,A4} x {seed 42,2024,7777,9999,1337} x {ens} = 20 runs.
Cheap-first per §11.5: smoke A3 one seed, then A3-vs-A4 one seed, then the full
ladder on the winner. Examples below.

GPUs — driven by CUDA_VISIBLE_DEVICES
-------------------------------------
The GPU pool is whatever you put in front:

    CUDA_VISIBLE_DEVICES=3,4,8 python launch_mt_3c_ablation.py ...

Those physical ids are parsed and one worker thread is pinned per id (each child
gets CUDA_VISIBLE_DEVICES=<that id>). `--gpus 3,4,8` overrides the env; if neither
is set it falls back to rk10's free set with a note. GPU 2/7 on rk10 are RTX PRO
6000 Blackwell (sm_120) that crash the bio cu118 build at the LSTM — a warning
fires if they're in the pool.

A4 needs the verifier
---------------------
Any arm with the gate on (A4) requires --verifier-checkpoint <verifier best.pt>;
preflight refuses to launch a gate arm without it. Pick the best verifier by
eval_verifier macro-F1 on the val candidates.

AADC data
---------
Forward --aadc-root + --aadc-sites to every run (SAFE sites only: Casey2018
Scott2019 Prydz2013 — the trainer hard-fails on the quarantined Kerguelen2020 /
DDU2021). --dry-run works without them.

Usage
-----
::

    conda activate bio
    cd /home/matthias-nagl/BioDCASE/task

    # inspect, no AADC, no GPUs needed
    CUDA_VISIBLE_DEVICES=3,4,8 python launch_mt_3c_ablation.py --dry-run

    # smoke A3 on one seed
    CUDA_VISIBLE_DEVICES=3 python launch_mt_3c_ablation.py \
        --arms A3 --seeds 42 --aadc-root /data/aadc --aadc-sites Casey2018 Scott2019 Prydz2013

    # A3-vs-A4 head-to-head on one seed (A4 needs the verifier)
    CUDA_VISIBLE_DEVICES=3,4 python launch_mt_3c_ablation.py \
        --arms A3 A4 --seeds 42 \
        --verifier-checkpoint runs_verifier/<best>/best.pt \
        --aadc-root /data/aadc --aadc-sites Casey2018 Scott2019 Prydz2013

    # full ladder on the winner across 5 seeds (drop A4 if the gate didn't help)
    CUDA_VISIBLE_DEVICES=3,4,8,9 python launch_mt_3c_ablation.py \
        --arms A1 A2 A3 A4 \
        --verifier-checkpoint runs_verifier/<best>/best.pt \
        --aadc-root /data/aadc --aadc-sites Casey2018 Scott2019 Prydz2013
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
SOURCES = ["ens"]                 # ablation holds the mining source fixed (ens)
TRAINER = "train_mt_final.py"
DEFAULT_GPUS = "3,4,8,9"          # rk10 free L40S+A6000, used only if nothing set

# arm -> (consistency_type, lambda_da, verifier_gate)
ARMS = {
    "A0": ("asymmetric_mse", 0.0, False),
    "A1": ("asymmetric_mse", 0.5, False),
    "A2": ("sat",            0.0, False),
    "A3": ("sat",            0.5, False),
    "A4": ("sat",            0.5, True),
}
DEFAULT_ARMS = ["A1", "A2", "A3", "A4"]     # A0 reused from Block C

# Base 3class run dirs by seed (same mapping as the Block B / from-base launchers).
CKPT_DIR_3C = {
    42:   "final_3c_s42_20260527_200054",
    2024: "final_3c_s2024_20260527_200309",
    7777: "final_3c_s7777_20260527_200401",
    9999: "final_3c_s9999_20260527_200852",
    1337: "final_3c_s1337_20260528_202023",
}

BLACKWELL_GPUS = {2, 7}           # sm_120 cards that crash the bio cu118 build


def resolve_gpus(args):
    """GPU pool: --gpus overrides; else CUDA_VISIBLE_DEVICES (what you put in
    front); else the rk10 fallback. Returns (list[int], source_label)."""
    if args.gpus:
        raw, src = args.gpus, "--gpus"
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raw, src = os.environ["CUDA_VISIBLE_DEVICES"], "CUDA_VISIBLE_DEVICES"
    else:
        raw, src = DEFAULT_GPUS, "fallback default"
    gpus = [int(g) for g in raw.split(",") if g.strip() != ""]
    return gpus, src


def enumerate_jobs(runs_root, hn_root, ckpt_name, seeds, sources, arms, pgi):
    head = "3class"
    jobs = []
    for seed in seeds:
        base = Path(runs_root) / CKPT_DIR_3C[seed] / ckpt_name
        for source in sources:
            sub = f"seed{seed}" if source == "solo" else "ensemble"
            hn_glob = str(Path(hn_root) / head / sub / "*.json")
            hn_files = sorted(glob.glob(hn_glob))
            for arm in arms:
                cons, lam, gate = ARMS[arm]
                jobs.append({
                    "seed": seed, "source": source, "arm": arm,
                    "cons": cons, "lambda_da": lam, "gate": gate, "pgi": pgi,
                    "ckpt": base, "hn_glob": hn_glob, "hn_files": hn_files,
                    "name": f"mtE_{arm}_3class_s{seed}_{source}",
                })
    return jobs


def build_cmd(job, val_workers, aadc_root, aadc_sites, out_dir,
              verifier_ckpt, gate_classes, extra):
    cmd = [sys.executable, TRAINER,
           "--checkpoint", str(job["ckpt"]),
           "--hard-negatives", *job["hn_files"],
           "--seed", str(job["seed"]),
           "--run-name", job["name"],
           "--out-dir", out_dir,
           "--consistency-type", job["cons"],
           "--lambda-da", str(job["lambda_da"]),
           "--aadc-root", str(aadc_root),
           "--aadc-sites", *aadc_sites,
           "--val-workers", str(val_workers)]
    if job["gate"]:
        cmd += ["--verifier-gate",
                "--verifier-checkpoint", str(verifier_ckpt),
                "--gate-classes", *gate_classes]
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
                   help="Where the MT trainer writes run dirs.")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated physical GPU ids. OVERRIDES "
                        "CUDA_VISIBLE_DEVICES; if unset, the env var is used.")
    p.add_argument("--val-workers", type=int, default=0,
                   help="Threshold-sweep workers per run (0 = auto: cores//n_gpus, "
                        "capped at 13).")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--sources", nargs="+", choices=["solo", "ens"], default=SOURCES,
                   help="Hard-neg source. Ablation default holds it at ens.")
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=DEFAULT_ARMS,
                   help="Ablation arms to run (default A1 A2 A3 A4; A0 is reused).")
    p.add_argument("--pgi", choices=["on", "off"], default="on",
                   help="Per-class gradient isolation (HNM+PGI). Default on.")
    p.add_argument("--verifier-checkpoint", default=None,
                   help="Verifier best.pt — REQUIRED if any gate arm (A4) is run.")
    p.add_argument("--gate-classes", nargs="+", default=["d"],
                   help="Classes the verifier gates in A4 (default: d).")
    p.add_argument("--aadc-root", default=None,
                   help="Root dir of AADC unlabeled audio (forwarded to every run).")
    p.add_argument("--aadc-sites", nargs="+", default=None,
                   help="SAFE AADC sites only (forwarded). Casey2018 Scott2019 Prydz2013.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose <out-dir>/<run-name>/best_model.pt exists.")
    p.add_argument("--logdir", default="runs/mtE_logs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is passed through to the trainer.")
    return p.parse_args()


def main():
    args = parse_args()
    gpus, gpu_src = resolve_gpus(args)
    pgi = args.pgi == "on"
    n_cpu = os.cpu_count() or 8
    val_workers = args.val_workers or min(13, max(2, n_cpu // max(1, len(gpus))))
    omp = max(1, n_cpu // max(1, len(gpus)))

    if not gpus:
        raise SystemExit("No GPUs resolved. Set CUDA_VISIBLE_DEVICES or pass --gpus.")
    bw = sorted(set(gpus) & BLACKWELL_GPUS)
    if bw:
        print(f"!! WARNING: GPU(s) {bw} look like RTX PRO 6000 Blackwell (sm_120) on "
              f"rk10 — the bio cu118 build crashes there at the LSTM. Drop them "
              f"unless you're on a different box.\n")

    jobs = enumerate_jobs(args.runs_root, args.hn_root, args.ckpt_name,
                          args.seeds, args.sources, args.arms, pgi)

    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs
                if not (Path(args.out_dir) / j["name"] / "best_model.pt").exists()]
        print(f"--skip-existing: {before - len(jobs)} already done, {len(jobs)} to run")

    # ---- plan + preflight -------------------------------------------
    print(f"GPUs: {gpus} (from {gpu_src}) | val-workers/run: {val_workers} | "
          f"OMP threads/run: {omp} | {n_cpu} cores")
    print(f"Block E ablation 3class | arms={args.arms} | sources={args.sources} | "
          f"PGI={'on' if pgi else 'off'}")
    if "A0" not in args.arms:
        print("  (A0 baseline not in arms — pairing against your existing Block C "
              "asymmetric_mse runs; add --arms A0 to re-run it identically)")
    print(f"{len(jobs)} run(s):")
    problems = []
    for i, j in enumerate(jobs):
        tag = f"{len(j['hn_files'])} json" if j["hn_files"] else "NO JSON"
        gate = " +gate" if j["gate"] else ""
        print(f"  [{i:2d}] {j['name']:34} <- {j['ckpt'].parent.name}/{args.ckpt_name}"
              f"  [{j['cons']}, da={j['lambda_da']}{gate}]  ({tag})")
        if not j["ckpt"].exists():
            problems.append(f"missing base checkpoint: {j['ckpt']}")
        if not j["hn_files"]:
            problems.append(f"no hard-neg JSONs: {j['hn_glob']} (run the miner first?)")

    if any(j["gate"] for j in jobs):
        if not args.verifier_checkpoint:
            problems.append("gate arm (A4) selected but --verifier-checkpoint is unset.")
        elif not Path(args.verifier_checkpoint).exists():
            problems.append(f"verifier checkpoint not found: {args.verifier_checkpoint}")

    aadc_ok = bool(args.aadc_root) and bool(args.aadc_sites)
    if not args.dry_run and not aadc_ok:
        problems.append("MT runs need --aadc-root and --aadc-sites "
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
                            args.out_dir, args.verifier_checkpoint,
                            args.gate_classes, args.extra)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)   # pin this child to one physical id
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
    print(f"\n{'='*66}\nBlock E ablation done in {(time.time()-t_start)/60:.1f} min — "
          f"{ok}/{len(results)} succeeded\n{'='*66}")
    for name, gpu, rc, dt in sorted(results):
        if rc != 0:
            print(f"  FAILED  {name}  (GPU {gpu}, rc={rc}) — see {logdir/(name+'.log')}")
    print(f"Logs: {logdir}/   |   checkpoints: {args.out_dir}/<run-name>/best_model.pt")
    print("Next: per-arm D P/R vs A0 (paired). Headline = PR-curve shift, not D F1.")


if __name__ == "__main__":
    main()
