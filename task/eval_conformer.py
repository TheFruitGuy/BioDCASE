"""
eval_conformer
==============

Step-0 threshold tuning for a Conformer checkpoint, on the **final** stack.

Loads a checkpoint saved by ``train_conformer.py`` / ``conformer_core`` (both
store ``arch_kwargs``; ``num_classes`` and ``d_model`` are read straight off the
classifier weight so either checkpoint layout loads), reproduces
``train_final.validate``'s validation probabilities, then runs the per-class
coordinate-descent threshold sweep — the *same* search as ``validation_core``,
but against ``postprocess_final`` / ``config_final`` so the tuned numbers are
directly comparable to the Conformer's training-time ``macro_paper`` (your
0.400 / 0.388 fixed-0.3 figures).

Fast bits, matching ``ensemble_predict_cached``:
  * per-checkpoint probability cache (float16 pickle keyed by checkpoint
    identity + val construction) — first run does the GPU forward (~minutes),
    re-runs skip it and just re-tune (~seconds);
  * parallel within-class grid search via fork (``--workers``);
  * optional FP16 autocast on the forward pass (``--use-fp16``).

Reports fixed-0.3 vs tuned micro / macro / macro_paper F1 and per-class P/R/F1,
prints the chosen thresholds, and writes them to ``<ckpt_parent>_conformer_thr.json``.

Usage
-----
    python eval_conformer.py runs/conformer_3c_s42_.../conformer_best.pt
    python eval_conformer.py runs/phase13b_.../phase13b_best.pt --use-fp16
    python eval_conformer.py A/best.pt B/best.pt          # tune several at once
    python eval_conformer.py ckpt.pt --workers 1          # exactly reproducible
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import config_final as cfg
from postprocess_final import (
    Detection, collapse_probs_to_3class, compute_metrics, postprocess_predictions,
)

# ----------------------------------------------------------------------
# Per-class grids — identical to validation_core (single source of truth),
# copied here so the tuner stays on the final stack without importing the
# old-stack validation_core (which pulls in `config` / `postprocess`).
# ----------------------------------------------------------------------

CLASS_NAMES: list[str] = list(cfg.CALL_TYPES_3)
BMABZ_GRID = np.arange(0.20, 0.85, 0.05)
D_GRID = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.85, 0.10)])
BP_GRID = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.85, 0.10)])
THRESHOLD_GRIDS = [BMABZ_GRID, D_GRID, BP_GRID]
IOU_THRESHOLD = 0.3
CACHE_VERSION = "v1"


# ----------------------------------------------------------------------
# Threshold tuning (mirrors validation_core, final-stack postprocess)
# ----------------------------------------------------------------------

_WORKER_PROBS: dict | None = None
_WORKER_GT: list | None = None


def _grid_task(payload):
    c, t_val, trial = payload
    preds = postprocess_predictions(_WORKER_PROBS, trial)
    metrics = compute_metrics(preds, _WORKER_GT, iou_threshold=IOU_THRESHOLD)
    return c, float(t_val), float(metrics.get(CLASS_NAMES[c], {}).get("f1", 0.0))


def tune_thresholds_per_class(probs, gt_events, start=None, workers=1):
    """Per-class coordinate descent (BMABZ -> D -> BP). Parallel within a class."""
    thr = (np.array([0.5, 0.5, 0.5], dtype=np.float64) if start is None
           else np.asarray(start, dtype=np.float64).copy())

    if workers <= 1:
        for c, name in enumerate(CLASS_NAMES):
            best_f1, best_t = -1.0, thr[c]
            for t in THRESHOLD_GRIDS[c]:
                trial = thr.copy(); trial[c] = float(t)
                m = compute_metrics(postprocess_predictions(probs, trial),
                                    gt_events, iou_threshold=IOU_THRESHOLD)
                f1 = m.get(name, {}).get("f1", 0.0)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            thr[c] = best_t
        return thr

    ctx = mp.get_context("fork")
    global _WORKER_PROBS, _WORKER_GT
    _WORKER_PROBS, _WORKER_GT = probs, list(gt_events)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for c, name in enumerate(CLASS_NAMES):
                tasks = []
                for t in THRESHOLD_GRIDS[c]:
                    trial = thr.copy(); trial[c] = float(t)
                    tasks.append((c, float(t), trial))
                best_f1, best_t = -1.0, thr[c]
                for fut in as_completed([pool.submit(_grid_task, tk) for tk in tasks]):
                    _, t_val, f1 = fut.result()
                    if f1 > best_f1:
                        best_f1, best_t = f1, t_val
                thr[c] = best_t
    finally:
        _WORKER_PROBS = _WORKER_GT = None
    return thr


def evaluate_with_thresholds(probs, gt_events, thresholds):
    preds = postprocess_predictions(probs, np.asarray(thresholds, dtype=np.float64))
    return compute_metrics(preds, gt_events, iou_threshold=IOU_THRESHOLD)


def macro_f1(metrics) -> float:
    return float(np.mean([metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES]))


def macro_f1_paper(metrics) -> float:
    p = float(np.mean([metrics.get(c, {}).get("precision", 0.0) for c in CLASS_NAMES]))
    r = float(np.mean([metrics.get(c, {}).get("recall", 0.0) for c in CLASS_NAMES]))
    return 2.0 * p * r / (p + r + 1e-8)


# ----------------------------------------------------------------------
# Checkpoint loading + validation-prob collection (lazy heavy imports)
# ----------------------------------------------------------------------

def load_conformer(ckpt_path, device):
    """Rebuild WhaleVAD_Conformer from a checkpoint and set cfg.USE_3CLASS to
    match how it was trained. num_classes/d_model come off the classifier
    weight (layout-independent); nhead/layers/ffn_mult/conv_kernel from
    arch_kwargs (both checkpoint layouts store them, under `layers` or
    `num_layers`)."""
    from model_conformer import WhaleVAD_Conformer
    from spectrogram_final import SpectrogramExtractor

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    # The composed inner WhaleVAD also has a `classifier` (its dead BiLSTM head),
    # so its weight key ALSO ends in "classifier.weight" — but with a different
    # width (2*LSTM_HIDDEN). Match the Conformer's OWN top-level head, never
    # `_inner.*`, or d_model is read wrong and the load fails on a size mismatch.
    def _param(suffix):
        if suffix in sd:                       # exact top-level key
            return sd[suffix]
        cands = [k for k in sd if k.endswith("." + suffix) and "_inner" not in k]
        if not cands:
            raise KeyError(f"checkpoint has no top-level {suffix}")
        return sd[cands[0]]
    cls_w = _param("classifier.weight")        # [num_classes, d_model]
    num_classes, d_model = int(cls_w.shape[0]), int(cls_w.shape[1])

    a = ckpt.get("arch_kwargs", {})
    def pick(*names, default):
        for n in names:
            if a.get(n) is not None:
                return a[n]
        return default

    cfg.USE_3CLASS = (num_classes == 3)
    # Match the input channel count the checkpoint was trained with, so the
    # rebuilt filterbank and the extractor agree (else load/forward mismatch).
    cfg.DUAL_RESOLUTION = bool(a.get("dual_res", False))
    common = dict(
        num_classes=num_classes, d_model=d_model,
        nhead=pick("nhead", default=4),
        num_layers=pick("layers", "num_layers", default=4),
        ffn_mult=pick("ffn_mult", default=4),
        conv_kernel=pick("conv_kernel", default=31),
        dropout=pick("dropout", default=0.1),
    )
    # 13k FDY-CRNN checkpoints set backbone="crnn": rebuild the CNN-BiLSTM
    # (WhaleVAD_FDY), NOT a Conformer. d_model read above (=2*LSTM_HIDDEN) and the
    # `common` Conformer kwargs do not apply to the CRNN and are ignored here.
    if a.get("backbone") == "crnn":
        from model_crnn_fdy import WhaleVAD_FDY
        model = WhaleVAD_FDY(
            num_classes=num_classes,
            feat_channels=cfg.n_feat_channels(),
            fdy_targets=tuple(a.get("fdy_targets") or ("filterbank", "feat0")),
            fdy_basis=pick("fdy_basis", default=4),
            fdy_temp=pick("fdy_temp", default=1.0),
        ).to(device)
    # FDY checkpoints (phase 13d) carry `fdy_targets` in arch_kwargs and have a
    # different stem; build the matching subclass so the state_dict loads.
    elif a.get("fdy_targets"):
        from model_conformer_fdy import WhaleVAD_Conformer_FDY
        model = WhaleVAD_Conformer_FDY(
            fdy_targets=tuple(a["fdy_targets"]),
            fdy_basis=pick("fdy_basis", default=4),
            fdy_temp=pick("fdy_temp", default=1.0),
            **common,
        ).to(device)
    else:
        model = WhaleVAD_Conformer(**common).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():  # materialise the lazy projection before loading
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    model.load_state_dict(sd)
    model.eval()
    return model, spec, num_classes


def collect_val_probs(model, spec, device, use_fp16):
    """Reproduce train_final.validate's probability collection exactly, then
    collapse to 3 classes. Returns (probs3, gt_tuples)."""
    from torch.utils.data import DataLoader
    from dataset_final import (
        WhaleDataset, build_val_segments, collate_fn, get_file_manifest,
        load_annotations,
    )

    val_sites = list(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}

    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_fp16 and device.type == "cuda") else nullcontext())
    hop = spec.hop_length
    raw = {}
    for audio, _targets, _mask, metas in loader:
        audio = audio.to(device)
        with torch.no_grad(), autocast:
            logits = model(spec(audio))                 # no mask, exactly like validate
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            raw[key] = probs[j, :n_frames, :]

    probs3 = collapse_probs_to_3class(raw)              # no-op if already 3-class

    gt = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt.append((row["dataset"], row["filename"], row["label_3class"],
                   (row["start_datetime"] - fsd).total_seconds(),
                   (row["end_datetime"] - fsd).total_seconds()))
    return probs3, gt


# ----------------------------------------------------------------------
# Prob cache
# ----------------------------------------------------------------------

def _cache_key(ckpt_path: Path) -> str:
    st = ckpt_path.stat()
    ident = "|".join([
        str(ckpt_path.resolve()), str(st.st_size), str(st.st_mtime_ns),
        ",".join(sorted(cfg.VAL_DATASETS)),
        str(getattr(cfg, "EVAL_SEGMENT_S", "")),
        str(getattr(cfg, "EVAL_OVERLAP_S", "")),
        CACHE_VERSION,
    ])
    return hashlib.sha1(ident.encode()).hexdigest()


def get_probs(ckpt_path: Path, args, device):
    """Return (probs3, gt_tuples, num_classes), from cache if available."""
    cache_dir = Path(args.cache_dir)
    cache_file = cache_dir / f"{ckpt_path.parent.name}_{_cache_key(ckpt_path)[:16]}_probs.pkl"

    if not args.no_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            blob = pickle.load(f)
        cfg.USE_3CLASS = True  # cached probs are already collapsed to 3-class
        probs3 = {k: v.astype(np.float32) for k, v in blob["probs3"].items()}
        print(f"  [cache hit] {cache_file.name}  ({len(probs3)} segments)")
        return probs3, blob["gt"], blob["num_classes"]

    print("  [cache miss] running inference...")
    t0 = time.time()
    model, spec, num_classes = load_conformer(ckpt_path, device)
    probs3, gt = collect_val_probs(model, spec, device, args.use_fp16)
    cfg.USE_3CLASS = True  # probs3 is 3-class from here on
    print(f"  inference done in {time.time() - t0:.0f}s "
          f"({len(probs3)} segments, {num_classes}-class head)")

    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        blob = {"probs3": {k: v.astype(np.float16) for k, v in probs3.items()},
                "gt": gt, "num_classes": num_classes}
        with open(cache_file, "wb") as f:
            pickle.dump(blob, f)
        print(f"  cached probs -> {cache_file}")
    return probs3, gt, num_classes


# ----------------------------------------------------------------------
# Per-checkpoint evaluation + reporting
# ----------------------------------------------------------------------

def _fmt(tag, metrics):
    micro = metrics.get("overall", {}).get("f1", 0.0)
    lines = [f"  {tag:11} micro={micro:.3f}  macro={macro_f1(metrics):.3f}  "
             f"macro_paper={macro_f1_paper(metrics):.3f}"]
    for name in CLASS_NAMES:
        pc = metrics.get(name, {})
        lines.append(f"      {name.upper():6} P={pc.get('precision', 0):.3f} "
                     f"R={pc.get('recall', 0):.3f} F1={pc.get('f1', 0):.3f}")
    return "\n".join(lines)


def evaluate_checkpoint(ckpt_path: Path, args, device):
    print(f"\n=== {ckpt_path} ===")
    probs3, gt_tuples, num_classes = get_probs(ckpt_path, args, device)
    gt = [Detection(dataset=d, filename=f, label=lab, start_s=s, end_s=e)
          for (d, f, lab, s, e) in gt_tuples]

    fixed = evaluate_with_thresholds(probs3, gt, [0.3, 0.3, 0.3])
    print(_fmt("fixed@0.3", fixed))

    t0 = time.time()
    thr = tune_thresholds_per_class(probs3, gt, start=[0.5, 0.5, 0.5],
                                    workers=args.workers)
    tuned = evaluate_with_thresholds(probs3, gt, thr)
    print(_fmt("tuned", tuned))
    print(f"  thresholds  bmabz={thr[0]:.2f}  d={thr[1]:.2f}  bp={thr[2]:.2f}"
          f"   (tune {time.time() - t0:.0f}s, {args.workers} workers)")
    print(f"  delta macro_paper {macro_f1_paper(fixed):.3f} -> "
          f"{macro_f1_paper(tuned):.3f}  ({macro_f1_paper(tuned) - macro_f1_paper(fixed):+.3f})")

    out = ckpt_path.parent / f"{ckpt_path.parent.name}_conformer_thr.json"
    with open(out, "w") as f:
        json.dump({
            "checkpoint": str(ckpt_path), "num_classes": num_classes,
            "thresholds": {n: float(t) for n, t in zip(CLASS_NAMES, thr)},
            "tuned": {"micro": metrics_micro(tuned),
                      "macro": macro_f1(tuned), "macro_paper": macro_f1_paper(tuned),
                      "per_class": {n: tuned.get(n, {}) for n in CLASS_NAMES}},
        }, f, indent=2, default=float)
    print(f"  wrote {out}")
    return thr, tuned


def metrics_micro(metrics):
    return metrics.get("overall", {}).get("f1", 0.0)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=str,
                   help="One or more Conformer checkpoint .pt files to tune.")
    p.add_argument("--workers", type=int, default=min((os.cpu_count() or 1), 13),
                   help="Parallel grid workers (Linux fork). 1 = sequential / "
                        "exactly reproducible. Default min(cpu_count, 13).")
    p.add_argument("--cache-dir", type=str, default="runs/conformer_eval_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--use-fp16", action="store_true",
                   help="FP16 autocast on the forward pass (~1.5-2x faster).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  workers={args.workers}")
    results = []
    for cp in args.checkpoints:
        thr, tuned = evaluate_checkpoint(Path(cp), args, device)
        results.append((cp, macro_f1_paper(tuned), macro_f1(tuned)))
    if len(results) > 1:
        print("\n=== summary (tuned) ===")
        for cp, mp_paper, mf in sorted(results, key=lambda r: -r[1]):
            print(f"  macro_paper={mp_paper:.3f}  macro={mf:.3f}  {cp}")


if __name__ == "__main__":
    main()
