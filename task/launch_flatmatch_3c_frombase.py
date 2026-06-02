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

Small GPUs (~11 GB)
-------------------
A run uses ~15 GB at bs-32, so on an 11 GB card pass ``--grad-accum-steps 2``
(micro-batch 16) or ``--grad-accum-steps 4`` (micro-batch 8). The effective
batch stays 32 and the two-pass accumulation is mathematically identical to
the bs-32 run, so results remain directly comparable to the MT runs — you
just free the 48 GB cards. Forwarded to every run.

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

    # full 10 on a pool of ~11 GB cards (effective batch still 32):
    CUDA_VISIBLE_DEVICES=10,11,12,13 python launch_flatmatch_3c_frombase.py \
        --grad-accum-steps 2 \
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


def enumerate_jobs(runs_root, hn_root, ckpt_name, seeds, sources, modes, pgi,
                   fixed_labels_pattern, lambda_mt):
    head = "3class"
    use_fix = fixed_labels_pattern is not None
    use_stack = use_fix and lambda_mt > 0.0
    jobs = []
    for seed in seeds:
        base = Path(runs_root) / CKPT_DIR_3C[seed] / ckpt_name
        fixed = fixed_labels_pattern.format(seed=seed) if use_fix else None
        for source in sources:
            sub = f"seed{seed}" if source == "solo" else "ensemble"
            hn_glob = str(Path(hn_root) / head / sub / "*.json")
            hn_files = sorted(glob.glob(hn_glob))
            for mode in modes:
                if use_stack:
                    name = f"fmfb_3class_s{seed}_{source}_fixlabel_mtstack"
                elif use_fix:
                    name = f"fmfb_3class_s{seed}_{source}_fixlabel"
                else:
                    name = f"fmfb_3class_s{seed}_{source}_{mode}"
                jobs.append({
                    "seed": seed, "source": source, "mode": mode, "pgi": pgi,
                    "ckpt": base, "hn_glob": hn_glob, "hn_files": hn_files,
                    "fixed_labels": fixed, "name": name,
                })
    return jobs


def build_cmd(job, val_workers, aadc_root, aadc_sites, out_dir, rho, lambda_xs,
              grad_accum_steps, lambda_mt, consistency_type, conf_threshold,
              pos_weight, neg_weight, extra):
    cmd = [sys.executable, TRAINER,
           "--checkpoint", str(job["ckpt"]),
           "--hard-negatives", *job["hn_files"],
           "--seed", str(job["seed"]),
           "--run-name", job["name"],
           "--out-dir", out_dir,
           "--flatmatch-mode", job["mode"],
           "--rho", str(rho),
           "--lambda-xs", str(lambda_xs),
           "--grad-accum-steps", str(grad_accum_steps),
           "--aadc-root", str(aadc_root),
           "--aadc-sites", *aadc_sites,
           "--val-workers", str(val_workers)]
    if job["pgi"]:
        cmd.append("--isolate-classes")
    if job.get("fixed_labels"):
        cmd += ["--fixed-labels", job["fixed_labels"]]
    if lambda_mt > 0.0:                       # the MT + Fix-label stack
        cmd += ["--lambda-mt", str(lambda_mt),
                "--consistency-type", consistency_type,
                "--conf-threshold", str(conf_threshold),
                "--pos-weight", str(pos_weight),
                "--neg-weight", str(neg_weight)]
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
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Two-pass gradient accumulation forwarded to every run. "
                        "K=1 = bs-32 (needs a ~16 GB+ card); K=2 (micro 16) or "
                        "K=4 (micro 8) fits an ~11 GB card with the SAME effective "
                        "batch 32, so results stay identical/comparable to MT.")
    p.add_argument("--pgi", choices=["on", "off"], default="on",
                   help="Per-class gradient isolation (HNM+PGI). Default on.")
    p.add_argument("--aadc-root", default=None,
                   help="Root dir of AADC unlabeled audio (forwarded to every run).")
    p.add_argument("--aadc-sites", nargs="+", default=None,
                   help="AADC site subfolders (forwarded to every run). Match the MT run.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose <out-dir>/<run-name>/best_model.pt exists.")
    p.add_argument("--fixed-labels-pattern", default=None,
                   help="Per-seed fixed-label npz with {seed} (FlatMatch Fix-label), "
                        "e.g. runs/fixed_labels/fixed_3c_s{seed}.npz. Run "
                        "select_fixed_labels.py for each seed first. Omit = vanilla.")
    p.add_argument("--lambda-mt", type=float, default=0.0,
                   help="MT-consistency weight = the stack. 0 = pure cross-sharpness. "
                        "Set to your MT lambda_max (1.0) to layer MT onto Fix-label.")
    p.add_argument("--consistency-type", default="mse",
                   choices=["mse", "confident", "asymmetric_mse"],
                   help="MT consistency form for the stack (match your MT runs).")
    p.add_argument("--conf-threshold", type=float, default=0.7)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--neg-weight", type=float, default=1.0)
    p.add_argument("--logdir", default="runs/fmfb_logs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Anything after --extra is forwarded verbatim to every run.")
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
                          args.seeds, args.sources, args.modes, pgi,
                          args.fixed_labels_pattern, args.lambda_mt)

    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs
                if not (Path(args.out_dir) / j["name"] / "best_model.pt").exists()]
        print(f"--skip-existing: {before - len(jobs)} already done, {len(jobs)} to run")

    # ---- plan + preflight -------------------------------------------
    if args.lambda_mt > 0.0 and args.fixed_labels_pattern:
        variant = f"STACK (MT lambda_mt={args.lambda_mt} {args.consistency_type} + Fix-label cross-sharpness)"
    elif args.fixed_labels_pattern:
        variant = "Fix-label FlatMatch (pure cross-sharpness)"
    elif args.lambda_mt > 0.0:
        variant = f"MT-stacked vanilla (lambda_mt={args.lambda_mt}, no fixed labels)"
    else:
        variant = "vanilla FlatMatch (pure cross-sharpness)"
    print(f"GPUs: {gpus} | val-workers/run: {val_workers} | OMP threads/run: {omp} "
          f"| {n_cpu} cores")
    print(f"FlatMatch from-base 3class | {variant} | sources={args.sources} "
          f"| modes={args.modes} | rho={args.rho} lambda_xs={args.lambda_xs} "
          f"| PGI={'on' if pgi else 'off'}")
    print(f"{len(jobs)} run(s):")
    problems = []
    for i, j in enumerate(jobs):
        tag = f"{len(j['hn_files'])} json" if j["hn_files"] else "NO JSON"
        fx = ""
        if j.get("fixed_labels"):
            ok = Path(j["fixed_labels"]).exists()
            fx = f"  fix={'ok' if ok else 'MISSING'}"
            if not ok:
                problems.append(f"missing fixed-label npz: {j['fixed_labels']} "
                                f"(run select_fixed_labels.py --checkpoint <s{j['seed']} base> "
                                f"--out {j['fixed_labels']})")
        print(f"  [{i:2d}] {j['name']:38} <- {j['ckpt'].parent.name}/{args.ckpt_name}"
              f"  ({tag}){fx}")
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
                            args.out_dir, args.rho, args.lambda_xs,
                            args.grad_accum_steps, args.lambda_mt,
                            args.consistency_type, args.conf_threshold,
                            args.pos_weight, args.neg_weight, args.extra)
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
