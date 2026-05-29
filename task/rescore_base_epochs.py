"""
Re-score base-model epochs by tuned paper-convention macro-F1
=============================================================

The base models (``train_final.py``) were selected/saved under the WRONG
metric (micro F1, old pre-fix runs) and validated at a FIXED 0.3 threshold.
This script walks every per-epoch checkpoint, runs val inference, tunes
per-class thresholds (coordinate descent, exactly the grids/order from
``validation_core``), and computes the paper-convention macro-F1
(F1 of mean-P and mean-R — the Geldenhuys et al. 0.440 metric) at the
TUNED operating point. It then picks the best epoch per run.

Why this and not ``validation_core``
------------------------------------
``validation_core`` imports ``config`` / ``postprocess`` (the non-final
modules). This script lives in the final pipeline and imports the
``*_final`` modules. To avoid two different ``config`` objects disagreeing
on ``USE_3CLASS`` (which silently breaks the 7->3 collapse + labelling),
the coordinate-descent tuner is inlined here against ``postprocess_final``.
Numbers stay identical to ``validation_core`` (same grids, same IoU=0.3,
same BMABZ->D->BP order).

What it does per run
--------------------
1. Determine head width (3 or 7) from the run dir name (``final_3c_*`` /
   ``final_7c_*``).
2. Score every ``final_epoch_NN.pt``: infer -> collapse to 3-class -> tune
   per-class thresholds -> event-level metrics -> paper-F1 (tuned).
3. Select the best epoch by tuned paper-F1.
4. Side effects (all on by default):
   - copy the winning epoch to ``paper_best.pt`` in the run dir
   - write ``rescore_summary.json`` (every epoch's metrics + the winner)
   - cache the winning epoch's stitched 3-class val posteriors to
     ``paper_best_posteriors.npz`` (float16) for the Block-D window sweep

Multi-GPU
---------
Inference is GPU work; threshold tuning is CPU work (the bigger cost).
Shard runs across GPUs with one process each. Example for 3 GPUs::

    CUDA_VISIBLE_DEVICES=0 python rescore_base_epochs.py --shard 0/3 --tune-workers 8 &
    CUDA_VISIBLE_DEVICES=1 python rescore_base_epochs.py --shard 1/3 --tune-workers 8 &
    CUDA_VISIBLE_DEVICES=2 python rescore_base_epochs.py --shard 2/3 --tune-workers 8 &
    wait
    python rescore_base_epochs.py --summarize     # aggregate all per-run JSONs

Keep ``(num GPU processes) x (tune-workers) <~ CPU core count`` to avoid
oversubscribing the tuner. If you hit a fork-after-CUDA issue, run with
``--tune-workers 1`` (sequential tuning, slower but no forking).

Run this from your task dir (where ``config_final.py`` etc. live).
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config_final as cfg
from dataset_final import (
    build_val_segments, get_file_manifest, load_annotations,
)
from model_final import WhaleVAD
from spectrogram_final import SpectrogramExtractor
import postprocess_final as pp
from postprocess_final import Detection, collapse_probs_to_3class


# ======================================================================
# Inlined coordinate-descent threshold tuner
# (copied verbatim from validation_core: same grids, IoU, BMABZ->D->BP order)
# ======================================================================

BMABZ_GRID = np.arange(0.20, 0.85, 0.05)
D_GRID = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.85, 0.10)])
BP_GRID = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.85, 0.10)])
THRESHOLD_GRIDS = [BMABZ_GRID, D_GRID, BP_GRID]
IOU_THRESHOLD = 0.3

# Worker-side fork-COW globals (set in parent before the pool is created).
_W_PROBS: dict | None = None
_W_GT: list | None = None


def _grid_task(payload):
    """Worker: evaluate one (class_idx, threshold, trial_vec); return class F1."""
    c, t, trial = payload
    preds = pp.postprocess_predictions(_W_PROBS, trial)
    m = pp.compute_metrics(preds, _W_GT, iou_threshold=IOU_THRESHOLD)
    return c, float(t), float(m.get(cfg.CALL_TYPES_3[c], {}).get("f1", 0.0))


def tune_thresholds(probs3, gt_events, workers: int) -> np.ndarray:
    """Per-class coordinate descent. Assumes cfg.USE_3CLASS is True already."""
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    if workers <= 1:
        for c in range(3):
            best_f1, best_t = -1.0, thresholds[c]
            for t in THRESHOLD_GRIDS[c]:
                trial = thresholds.copy()
                trial[c] = float(t)
                preds = pp.postprocess_predictions(probs3, trial)
                m = pp.compute_metrics(preds, gt_events, iou_threshold=IOU_THRESHOLD)
                f1 = m.get(cfg.CALL_TYPES_3[c], {}).get("f1", 0.0)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            thresholds[c] = best_t
        return thresholds

    global _W_PROBS, _W_GT
    ctx = mp.get_context("fork")
    _W_PROBS = probs3
    _W_GT = list(gt_events)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for c in range(3):
                tasks = []
                for t in THRESHOLD_GRIDS[c]:
                    trial = thresholds.copy()
                    trial[c] = float(t)
                    tasks.append((c, float(t), trial))
                best_f1, best_t = -1.0, thresholds[c]
                for fut in as_completed([pool.submit(_grid_task, tk) for tk in tasks]):
                    _, t_val, f1 = fut.result()
                    if f1 > best_f1:
                        best_f1, best_t = f1, t_val
                thresholds[c] = best_t
    finally:
        _W_PROBS = None
        _W_GT = None
    return thresholds


def evaluate(probs3, gt_events, thresholds) -> dict:
    """Postprocess at thresholds + event-level metrics. cfg.USE_3CLASS True."""
    preds = pp.postprocess_predictions(probs3, np.asarray(thresholds, dtype=np.float64))
    return pp.compute_metrics(preds, gt_events, iou_threshold=IOU_THRESHOLD)


def paper_f1(metrics: dict) -> float:
    """F1 of mean-P and mean-R across the 3 classes (paper convention)."""
    P = [metrics.get(c, {}).get("precision", 0.0) for c in cfg.CALL_TYPES_3]
    R = [metrics.get(c, {}).get("recall", 0.0) for c in cfg.CALL_TYPES_3]
    pb, rb = float(np.mean(P)), float(np.mean(R))
    return 2.0 * pb * rb / (pb + rb + 1e-8)


def per_class_block(metrics: dict) -> dict:
    return {
        name: {
            "precision": metrics.get(name, {}).get("precision", 0.0),
            "recall": metrics.get(name, {}).get("recall", 0.0),
            "f1": metrics.get(name, {}).get("f1", 0.0),
            "tp": int(metrics.get(name, {}).get("tp", 0)),
            "fp": int(metrics.get(name, {}).get("fp", 0)),
            "fn": int(metrics.get(name, {}).get("fn", 0)),
        }
        for name in cfg.CALL_TYPES_3
    }


# ======================================================================
# In-memory validation set (preloaded once, reused across runs/epochs)
# ======================================================================

class InMemVal(Dataset):
    """Holds all val audio in RAM so 500 inference passes don't re-read disk."""

    def __init__(self, segments):
        self.items = []
        for seg in tqdm(segments, desc="Preloading val audio", leave=False):
            audio, sr = sf.read(
                seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32",
            )
            assert sr == cfg.SAMPLE_RATE, f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}"
            self.items.append((
                torch.from_numpy(audio),
                {"dataset": seg.dataset, "filename": seg.filename,
                 "start_sample": seg.start_sample, "end_sample": seg.end_sample},
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_infer(batch):
    audios, metas = zip(*batch)
    max_samp = max(a.size(0) for a in audios)
    out = torch.zeros(len(audios), max_samp)
    for i, a in enumerate(audios):
        out[i, : a.size(0)] = a
    return out, list(metas)


@torch.no_grad()
def infer(model, spec_extractor, loader, device) -> dict:
    """Return {(dataset, filename, start_sample): (n_frames, n_classes)} probs."""
    model.eval()
    hop = spec_extractor.hop_length
    probs = {}
    for audio, metas in loader:
        audio = audio.to(device, non_blocking=True)
        logits = model(spec_extractor(audio))
        p = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            n_samp = meta["end_sample"] - meta["start_sample"]
            nf = min(n_samp // hop, p[j].shape[0])
            probs[(meta["dataset"], meta["filename"], meta["start_sample"])] = p[j, :nf, :]
    return probs


# ======================================================================
# GT events (USE_3CLASS-independent: always label_3class)
# ======================================================================

def build_gt_events(val_annotations, file_start_dts) -> list:
    gt = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt


# ======================================================================
# Model construction (per run)
# ======================================================================

def build_model(num_classes: int, device):
    cfg.USE_3CLASS = (num_classes == 3)
    model = WhaleVAD(num_classes=num_classes).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():  # materialise the lazy projection layer
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec(dummy))
    return model, spec


def collapse_to_3class(probs_raw: dict, num_classes: int) -> dict:
    """Collapse to 3-class and leave cfg.USE_3CLASS=True for downstream eval."""
    if num_classes == 7:
        cfg.USE_3CLASS = False           # so collapse actually pools 7->3
        probs3 = collapse_probs_to_3class(probs_raw)
        cfg.USE_3CLASS = True            # so postprocess labels in 3-class space
    else:
        cfg.USE_3CLASS = True
        probs3 = probs_raw
    return probs3


# ======================================================================
# Per-run processing
# ======================================================================

# Your 9 base runs: 4x 3-class (properly tagged, classifier -> 3 outputs,
# ~1,032,131 params) and 5x 7-class (old-style names with no class token,
# classifier -> 7 outputs, ~1,033,159 params). Used as the default set when
# neither --only nor --runs-glob is given. Edit here if your dirs differ.
DEFAULT_RUNS = [
    # --- 3-class (5 seeds; classifier -> 3 outputs, ~1,032,131 params) ---
    "runs/final_3c_s42_20260527_200054",     # seed 42
    "runs/final_3c_s2024_20260527_200309",   # seed 2024
    "runs/final_3c_s7777_20260527_200401",   # seed 7777
    "runs/final_3c_s9999_20260527_200852",   # seed 9999
    "runs/final_3c_s1337_20260528_202023",   # seed 1337
    # --- 7-class (5; classifier -> 7 outputs, ~1,033,159 params) ---
    "runs/final_7c_s42_20260528_204414",     # seed 42 (new-style name)
    "runs/final_20260526_074308",            # seed 7777 (recovered) -- old-style
    "runs/final_20260527_141710",            # seed ? -- old-style
    "runs/final_20260527_060332",            # seed ? -- old-style
    "runs/final_20260526_074410",            # seed ? -- old-style
]


def find_any_checkpoint(run_dir: Path):
    """A representative checkpoint: the first per-epoch file, else final_best.pt."""
    eps = discover_epoch_files(run_dir)
    if eps:
        return eps[0][1]
    best = run_dir / "final_best.pt"
    return best if best.exists() else None


def detect_num_classes(run_dir: Path) -> int | None:
    """
    Read the training class count straight from a checkpoint (name-independent).
    `classifier.weight` has shape (num_classes, 2*LSTM_HIDDEN) and `pos_weight`
    has length num_classes. The config log line ("7-class training...") prints
    even for 3-class runs, so it is NOT used.
    """
    ckpt = find_any_checkpoint(run_dir)
    if ckpt is None:
        return None
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    w = ck.get("model_state_dict", {}).get("classifier.weight")
    if w is not None:
        return int(w.shape[0])
    pw = ck.get("pos_weight")
    if pw is not None:
        return int(len(pw))
    if "_7c_" in run_dir.name:
        return 7
    if "_3c_" in run_dir.name:
        return 3
    return None


def discover_epoch_files(run_dir: Path):
    files = []
    for p in run_dir.glob("final_epoch_*.pt"):
        m = re.match(r"final_epoch_(\d+)\.pt$", p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return files


def old_micro_best_epoch(run_dir: Path, device) -> int | None:
    best = run_dir / "final_best.pt"
    if not best.exists():
        return None
    try:
        ck = torch.load(best, map_location="cpu", weights_only=False)
        return int(ck.get("epoch")) if ck.get("epoch") is not None else None
    except Exception:
        return None


def process_run(run_dir: Path, loader, gt_events, args, device):
    name = run_dir.name

    num_classes = detect_num_classes(run_dir)
    if num_classes is None:
        print(f"  !! no usable checkpoint in {name}, skipping")
        return None

    # Build the candidate (epoch, path) list. Prefer per-epoch checkpoints
    # (full sweep). Fall back to final_best.pt (single-checkpoint rescore) for
    # older runs that did not save every epoch.
    epoch_files = discover_epoch_files(run_dir)
    single_mode = False
    if epoch_files:
        if isinstance(args.candidates, int):
            ranked = rank_epochs_at_fixed_threshold(run_dir, epoch_files)
            keep = set(ep for ep, _ in ranked[: args.candidates])
            epoch_files = [(ep, p) for ep, p in epoch_files if ep in keep]
            print(f"  [{num_classes}c] candidates: top-{args.candidates} by "
                  f"0.3 paper-F1 -> {sorted(keep)}")
        else:
            print(f"  [{num_classes}c] candidates: ALL {len(epoch_files)} "
                  f"epochs (full sweep, option B)")
    else:
        best_pt = run_dir / "final_best.pt"
        if not best_pt.exists():
            print(f"  !! no final_epoch_*.pt and no final_best.pt in {name}, "
                  f"skipping")
            return None
        ep0 = old_micro_best_epoch(run_dir, device)
        epoch_files = [(ep0 if ep0 is not None else -1, best_pt)]
        single_mode = True
        print(f"  [{num_classes}c] no per-epoch checkpoints; single-checkpoint "
              f"rescore of final_best.pt (epoch {ep0}) -- can re-tune its "
              f"thresholds but cannot pick a better epoch.")

    model, spec = build_model(num_classes, device)

    records = []
    best = {"paper_f1": -1.0}
    best_probs3 = None
    best_path = None

    for ep, path in tqdm(epoch_files, desc=f"  {name} [{num_classes}c]", leave=False):
        ck = torch.load(path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ck["model_state_dict"], strict=True)
        except RuntimeError as e:
            print(f"    epoch {ep}: state_dict load failed ({e}); skipping")
            continue

        probs_raw = infer(model, spec, loader, device)
        probs3 = collapse_to_3class(probs_raw, num_classes)  # USE_3CLASS now True
        thr = tune_thresholds(probs3, gt_events, args.tune_workers)
        metrics = evaluate(probs3, gt_events, thr)

        rec = {
            "epoch": ep,
            "paper_f1": paper_f1(metrics),
            "micro_f1": metrics.get("overall", {}).get("f1", 0.0),
            "thresholds": [float(t) for t in thr],
            "per_class": per_class_block(metrics),
        }
        records.append(rec)
        if rec["paper_f1"] > best["paper_f1"]:
            best = rec
            best_probs3 = probs3  # keep for posterior cache
            best_path = path

    if not records:
        return None

    # ----- print per-run table -----
    old_ep = old_micro_best_epoch(run_dir, device)
    print(f"\n  === {name}  ({num_classes}-class) ===")
    print(f"  {'ep':>3}  {'paperF1':>8} {'micro':>7}  "
          f"{'bmabz':>6} {'d':>6} {'bp':>6}   thresholds")
    for r in sorted(records, key=lambda x: x["epoch"]):
        pc = r["per_class"]
        star = "  <-- best" if r["epoch"] == best["epoch"] else ""
        old = "  (old micro-best)" if r["epoch"] == old_ep else ""
        print(f"  {r['epoch']:>3}  {r['paper_f1']:>8.4f} {r['micro_f1']:>7.3f}  "
              f"{pc['bmabz']['f1']:>6.3f} {pc['d']['f1']:>6.3f} {pc['bp']['f1']:>6.3f}   "
              f"{['%.2f' % t for t in r['thresholds']]}{star}{old}")
    print(f"  selected epoch {best['epoch']}: tuned paper-F1 "
          f"{best['paper_f1']:.4f} (old micro-best was epoch {old_ep})")

    # ----- side effects -----
    if not args.no_copy_best and best_path is not None:
        dst = run_dir / "paper_best.pt"
        if best_path.resolve() != dst.resolve():
            shutil.copy2(best_path, dst)
        print(f"  copied {best_path.name} -> paper_best.pt")

    if not args.no_json:
        summary = {
            "run_dir": str(run_dir),
            "num_classes": num_classes,
            "single_checkpoint": single_mode,
            "tune_workers": args.tune_workers,
            "candidates": ("all" if args.candidates == "all" else args.candidates),
            "n_epochs_scored": len(records),
            "selected": best,
            "old_micro_best_epoch": old_ep,
            "epochs": sorted(records, key=lambda x: x["epoch"]),
        }
        with open(run_dir / "rescore_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote rescore_summary.json")

    if not args.no_cache_posteriors and best_probs3 is not None:
        # Re-infer the winner if it wasn't the last epoch scored (best_probs3
        # is correct as long as we kept it; it is). Stitch to per-file and save.
        cfg.USE_3CLASS = True
        stitched = pp.stitch_segments(best_probs3)
        npz = {f"{ds}|{fn}": arr.astype(np.float16) for (ds, fn), arr in stitched.items()}
        np.savez_compressed(run_dir / "paper_best_posteriors.npz", **npz)
        print(f"  cached paper_best_posteriors.npz ({len(npz)} files)")

    return {"run_dir": str(run_dir), "num_classes": num_classes,
            "best_epoch": best["epoch"], "paper_f1": best["paper_f1"],
            "micro_f1": best["micro_f1"]}


def rank_epochs_at_fixed_threshold(run_dir: Path, epoch_files):
    """Free 0.3 paper-F1 ranking from the stored history (option C candidates)."""
    # Any late checkpoint's history holds all epochs' per-class P/R at 0.3.
    last_path = epoch_files[-1][1]
    ck = torch.load(last_path, map_location="cpu", weights_only=False)
    hist = ck.get("history") or []
    ranked = []
    for h in hist:
        pc = h.get("per_class", {})
        P = [pc.get(c, {}).get("precision", 0.0) for c in cfg.CALL_TYPES_3]
        R = [pc.get(c, {}).get("recall", 0.0) for c in cfg.CALL_TYPES_3]
        pb, rb = float(np.mean(P)), float(np.mean(R))
        f1 = 2.0 * pb * rb / (pb + rb + 1e-8)
        ranked.append((int(h["epoch"]), f1))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ======================================================================
# Discovery + sharding + summary
# ======================================================================

def discover_runs(runs_glob, only):
    if only:
        dirs = [Path(d) for d in only]
    elif runs_glob:
        dirs = [Path(d) for d in sorted(glob.glob(runs_glob)) if Path(d).is_dir()]
    else:
        dirs = [Path(d) for d in DEFAULT_RUNS]

    def has_ckpt(d: Path) -> bool:
        return d.is_dir() and (
            bool(list(d.glob("final_epoch_*.pt"))) or (d / "final_best.pt").exists()
        )

    kept = [d for d in dirs if has_ckpt(d)]
    missing = [d for d in dirs if d not in kept]
    if missing:
        print("  (skipping -- no checkpoint found / dir missing):")
        for m in missing:
            print(f"    - {m}")
    return sorted(kept, key=lambda d: d.name)


def apply_shard(dirs, shard):
    if not shard:
        return dirs
    i, n = (int(x) for x in shard.split("/"))
    return dirs[i::n]


def summarize(runs_glob):
    if runs_glob:
        jsons = sorted(glob.glob(runs_glob.rstrip("/") + "/rescore_summary.json"))
    else:
        jsons = [str(Path(d) / "rescore_summary.json") for d in DEFAULT_RUNS
                 if (Path(d) / "rescore_summary.json").exists()]
    by_class = {3: [], 7: []}
    print(f"\n{'=' * 70}\nRESCORE SUMMARY ({len(jsons)} runs)\n{'=' * 70}")
    print(f"{'run':45} {'cls':>3} {'best_ep':>7} {'paperF1':>8} {'micro':>7}")
    for jp in jsons:
        with open(jp) as f:
            s = json.load(f)
        nc = s["num_classes"]
        b = s["selected"]
        by_class[nc].append(b["paper_f1"])
        print(f"{Path(s['run_dir']).name:45} {nc:>3} {b['epoch']:>7} "
              f"{b['paper_f1']:>8.4f} {b['micro_f1']:>7.3f}")
    print(f"{'-' * 70}")
    for nc in (3, 7):
        v = by_class[nc]
        if v:
            print(f"  {nc}-class: mean paper-F1 = {np.mean(v):.4f} "
                  f"± {np.std(v):.4f}  (n={len(v)})")
    allv = by_class[3] + by_class[7]
    if allv:
        print(f"  overall: mean paper-F1 = {np.mean(allv):.4f} "
              f"± {np.std(allv):.4f}  (n={len(allv)})")


# ======================================================================
# Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-glob", default=None,
                   help="Glob for run dirs. If omitted (and --only not given), "
                        "the built-in DEFAULT_RUNS list of your 9 base runs is "
                        "used. Example: --runs-glob 'runs/final_*'.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Explicit run dirs to process (overrides --runs-glob).")
    p.add_argument("--shard", default=None,
                   help="Process a slice of discovered runs, e.g. '0/3'. "
                        "Run one shard per GPU via CUDA_VISIBLE_DEVICES.")
    p.add_argument("--candidates", default="all",
                   help="'all' (option B, score every epoch) or an int K "
                        "(option C, score only the top-K epochs by the free "
                        "0.3 ranking).")
    p.add_argument("--tune-workers", type=int, default=8,
                   help="CPU workers for the per-class threshold sweep. "
                        "1 = sequential (no fork). Keep "
                        "(GPU procs) x (this) <~ cores.")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Inference batch size (VRAM is ample; default 128).")
    p.add_argument("--no-copy-best", action="store_true",
                   help="Do not copy the winning epoch to paper_best.pt.")
    p.add_argument("--no-json", action="store_true",
                   help="Do not write rescore_summary.json.")
    p.add_argument("--no-cache-posteriors", action="store_true",
                   help="Do not cache the winner's val posteriors (Block D).")
    p.add_argument("--summarize", action="store_true",
                   help="Aggregate existing rescore_summary.json files and exit.")
    args = p.parse_args()
    if args.candidates != "all":
        args.candidates = int(args.candidates)
    return args


def main():
    args = parse_args()

    if args.summarize:
        summarize(args.runs_glob)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  "
          f"(visible: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    runs = apply_shard(discover_runs(args.runs_glob, args.only), args.shard)
    print(f"Processing {len(runs)} run(s):")
    for d in runs:
        print(f"  - {d.name}")
    if not runs:
        print("Nothing to do.")
        return

    # ----- build the shared val set ONCE (sites/geometry are fixed) -----
    val_sites = list(cfg.VAL_DATASETS)
    print(f"\nLoading val data for {val_sites} ...")
    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    val_segments = build_val_segments(val_manifest, val_annotations)
    gt_events = build_gt_events(val_annotations, file_start_dts)
    print(f"  {len(val_manifest)} files, {len(val_segments)} segments, "
          f"{len(gt_events)} GT events")

    ds = InMemVal(val_segments)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_infer, pin_memory=True)
    print(f"  preloaded {len(ds)} segments into RAM\n")

    t0 = time.time()
    results = []
    for run_dir in runs:
        print(f"{'=' * 70}\nRUN: {run_dir.name}\n{'=' * 70}")
        r = process_run(run_dir, loader, gt_events, args, device)
        if r:
            results.append(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Scored {len(results)} run(s).")
    for r in results:
        print(f"  {Path(r['run_dir']).name:45} best epoch {r['best_epoch']:>3}  "
              f"tuned paper-F1 {r['paper_f1']:.4f}")
    print("Run with --summarize (after all shards finish) for the aggregate table.")


if __name__ == "__main__":
    main()
