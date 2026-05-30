"""
Cached ensemble predict
=======================

Wraps ensemble_predict.py's pipeline with per-checkpoint probability
caching. First eval of a checkpoint runs full inference (~3 min) and
caches probs to disk. Subsequent ensemble evals that include the same
checkpoint skip inference entirely (~5 s for the whole pipeline).

Useful when the same checkpoints appear in multiple ensemble combinations
(A, B, C, D, leave-one-out ablations, etc.).

Optional FP16 autocast for ~1.5–2× speedup on the inference pass when
GPUs are free.

What gets cached (all under ``--cache_dir``)
--------------------------------------------
1. **Per-checkpoint probs** — float16 pickle keyed by ``(checkpoint,
   segment_s)``. The original cache; the big win when a checkpoint
   recurs across ensemble subsets.
2. **Per-model tuned thresholds** — JSON keyed by ``(checkpoint,
   segment_s)``, written only under ``--per_model_eval``. Skips the
   per-model threshold sweep on re-runs.
3. **Ensemble probs** — float16 pickle keyed by a hash of the full
   ensemble identity ``(checkpoints, weights, segment_s)``. On a hit
   (and without ``--per_model_eval``) the per-checkpoint loop AND the
   DataLoader build are skipped entirely.
4. **Ensemble tuned thresholds** — JSON keyed by the same ensemble
   identity. On a hit the (now-parallel) threshold sweep is skipped, so
   a repeated identical ensemble eval is near-instant.

Caching only helps repeated *identical* runs (same checkpoints, same
weights, same segment length). When sweeping subsets or weights, items
3–4 miss every time and the per-checkpoint cache (item 1) does the work.

Parallel threshold tuning
--------------------------
The per-class coordinate-descent sweep now runs through
``validation_core.tune_thresholds_per_class`` with ``parallel_workers``
controlled by ``--val-workers`` (default ``min(cpu_count, 13)``). The
sequential path (``--val-workers 1``) is bit-identical to the legacy
``ensemble_predict.tune_thresholds_on_probs``; the parallel path can
differ from it only on exact per-class-F1 ties (grid plateaus), where
the winning threshold is decided by worker completion order. Pass
``--val-workers 1`` for an exactly reproducible sweep.

Reported metrics
----------------
Every result line now prints **both** macro-F1 conventions:

- ``MACRO F1``       — mean of per-class F1s, the official BioDCASE
  Task 2 metric (the number tracked across runs).
- ``PAPER MACRO F1`` — F1 of mean-precision / mean-recall, the headline
  convention in Geldenhuys et al. (Whale-VAD, the 0.440 reference; the
  WhaleVAD-BPN comparison). This is report-only — thresholds are still
  tuned on per-class F1, so historical numbers stay comparable.

Variable evaluation segment length
----------------------------------
``--segment-s`` (default 30s) controls the tile length used for
validation. Different lengths produce different probability tensors, so
each cache file is keyed by ``(checkpoint, segment_s)``. 30s keeps the
legacy filename for backward compatibility; other lengths get a
``_seg<N>`` suffix.

Per-class weight vectors
------------------------
By default the ensemble is a uniform (or ``--weights``-weighted) average
of all checkpoints across all classes. For the per-class hybrid strategy
— where one ensemble subset is best at class A and a different subset
is best at class B — pass three per-class weight flags instead:

    --weights-bmabz w1 w2 ... wN
    --weights-d     w1 w2 ... wN
    --weights-bp    w1 w2 ... wN

Each vector has one weight per checkpoint (in the order they appear in
``--checkpoints``); ``0`` excludes that checkpoint from that class.
Vectors are normalized per class. The three flags must be given
together and are mutually exclusive with ``--weights``.

Example: PGI ensemble (5 ckpts) drives BMABZ + BP, AMSE MT ensemble
(5 ckpts) drives D — 10 checkpoints total::

    python ensemble_predict_cached.py --per_model_eval --use_fp16 \\
        --segment-s 60 \\
        --checkpoints \\
            runs/hnm_pgi_whalevad_<s1>/best_model.pt \\
            ... 4 more PGI ckpts ... \\
            runs/mt_hnm_pgi_amse_whalevad_<s1>/best_model.pt \\
            ... 4 more AMSE ckpts ... \\
        --weights-bmabz 1 1 1 1 1  0 0 0 0 0 \\
        --weights-d     0 0 0 0 0  1 1 1 1 1 \\
        --weights-bp    1 1 1 1 1  0 0 0 0 0
"""

from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from postprocess import Detection, collapse_probs_to_3class
from spectrogram import SpectrogramExtractor

# Inference / combination helpers stay in ensemble_predict; the threshold
# sweep + metrics now come from validation_core (single source of truth,
# parallel-capable, and the home of the paper-convention macro-F1).
from ensemble_predict import (
    average_prob_dicts, build_model_for_ckpt, predict_probabilities,
)
from validation_core import (
    tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper,
)


# Threshold caches are keyed (in part) by the tuning objective so that if
# a different objective is ever added, old caches won't be mistaken for it.
OBJECTIVE = "per_class_f1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights applied uniformly across "
                        "all classes. Mutually exclusive with the "
                        "--weights-<class> flags.")
    # Per-class weight vectors. If any is set, all three must be set, and
    # --weights must not be set. Each vector length must equal the number
    # of checkpoints.
    p.add_argument("--weights-bmabz", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for the BMABZ class. "
                        "Use 0 to exclude a checkpoint from this class.")
    p.add_argument("--weights-d", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for the D class.")
    p.add_argument("--weights-bp", nargs="+", type=float, default=None,
                   help="Per-checkpoint weights for the BP class.")

    p.add_argument("--per_model_eval", action="store_true")
    p.add_argument("--cache_dir", type=str, default="runs/prob_cache",
                   help="Directory for cached probs and tuned thresholds.")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable all caching (always recompute probs and "
                        "thresholds; nothing is read or written).")
    p.add_argument("--use_fp16", action="store_true",
                   help="FP16 autocast during inference. ~1.5–2× speedup, "
                        "negligible accuracy hit (<0.001 F1 typically).")
    p.add_argument("--val-workers", type=int,
                   default=min((os.cpu_count() or 1), 13),
                   help="Worker processes for the per-class threshold "
                        "sweep. Default min(cpu_count, 13). 1 = sequential "
                        "(exactly reproducible); >1 = parallel (Linux fork), "
                        "may differ only on exact F1 ties.")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Evaluation tile length in seconds. Cache files "
                        "are keyed by this so different lengths don't "
                        "collide. 30s keeps the legacy filename.")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S,
                   help="Overlap between consecutive tiles in seconds.")

    args = p.parse_args()

    # ---- Validate per-class weight flags ------------------------------
    per_class = [args.weights_bmabz, args.weights_d, args.weights_bp]
    any_pc = any(w is not None for w in per_class)
    all_pc = all(w is not None for w in per_class)
    if any_pc:
        if not all_pc:
            p.error("If any --weights-<class> flag is set, all three "
                    "(--weights-bmabz, --weights-d, --weights-bp) must "
                    "be set together.")
        if args.weights is not None:
            p.error("--weights is mutually exclusive with the "
                    "--weights-<class> flags. Use one or the other.")
        n = len(args.checkpoints)
        for cname, w in zip(("bmabz", "d", "bp"), per_class):
            if len(w) != n:
                p.error(f"--weights-{cname} has {len(w)} values, "
                        f"need {n} (one per checkpoint).")
    return args


# ----------------------------------------------------------------------
# Cache path / key helpers
# ----------------------------------------------------------------------

def _seg_tag(segment_s: float) -> str:
    """Empty for the legacy 30s cache, ``_seg<N>`` otherwise."""
    return "" if abs(segment_s - 30.0) < 1e-6 else f"_seg{int(segment_s)}"


def cache_path_for(ckpt_path: Path, cache_dir: Path,
                   segment_s: float = 30.0) -> Path:
    """One pickle per (checkpoint, segment_length) pair.

    30s keeps the legacy name (no suffix) so existing caches remain
    reusable. Other lengths get a ``_seg<N>`` suffix so they don't
    collide with the 30s cache or with each other.
    """
    return cache_dir / f"{ckpt_path.parent.name}{_seg_tag(segment_s)}_probs.pkl"


def per_model_thr_cache_path(ckpt_path, cache_dir: Path,
                             segment_s: float = 30.0) -> Path:
    """Tuned thresholds for a single checkpoint (``--per_model_eval``)."""
    name = Path(ckpt_path).parent.name
    return cache_dir / f"{name}{_seg_tag(segment_s)}_thr.json"


def ensemble_identity(ckpt_ids, weights_spec, segment_s, objective):
    """Canonical description of an ensemble — everything that determines
    its probabilities and tuned thresholds. Hashed for the cache key.
    """
    return {
        "ckpts": list(ckpt_ids),          # parent dir names, input order
        "weights": weights_spec,          # normalized; see build_weights_spec
        "segment_s": round(float(segment_s), 4),
        "objective": objective,
    }


def ensemble_key(identity) -> str:
    """Short stable hash of an ensemble identity dict."""
    payload = json.dumps(identity, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def ensemble_probs_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"ensemble_{key}_probs.pkl"


def ensemble_thr_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"ensemble_{key}_thr.json"


# ----------------------------------------------------------------------
# Prob + threshold (de)serialisation
# ----------------------------------------------------------------------

def load_cached_probs(path: Path):
    """Restore float32 from float16-on-disk cache."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {k: v.astype(np.float32) for k, v in d.items()}


def save_probs_to_cache(probs, path: Path):
    """Cast to float16 for ~50% disk savings (lossless at eval precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: v.astype(np.float16) for k, v in probs.items()}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_thresholds(thr, path: Path, meta: dict | None = None):
    """Persist tuned thresholds as small human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"thresholds": [float(t) for t in np.asarray(thr).ravel()]}
    if meta is not None:
        payload["meta"] = meta
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_thresholds(path: Path) -> np.ndarray:
    with open(path) as f:
        payload = json.load(f)
    return np.asarray(payload["thresholds"], dtype=np.float64)


@torch.no_grad()
def get_or_compute_probs(ckpt_path, spec_extractor, val_loader_factory, device,
                         cache_dir, no_cache=False, use_fp16=False,
                         segment_s=30.0):
    """Cache hit → load pickle. Cache miss → run inference, save, return.

    ``val_loader_factory`` is a zero-arg callable that builds (and memoises)
    the validation DataLoader. It's only invoked on a cache miss, so a
    full-cache run never pays the segment-build / loader cost.

    ``segment_s`` is used both to pick the cache filename and to size the
    dummy input that materialises the lazy CNN projection layer.
    """
    cp = cache_path_for(Path(ckpt_path), cache_dir, segment_s=segment_s)
    if not no_cache and cp.exists():
        print(f"  cache hit: {cp.name}")
        return load_cached_probs(cp)

    print("  cache miss, running inference...")
    t0 = time.time()
    val_loader = val_loader_factory()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, model_type = build_model_for_ckpt(ckpt, device)
    print(f"    type: {model_type}")

    # Materialize lazy projection layer before load_state_dict.
    with torch.no_grad():
        dummy = torch.randn(1, int(cfg.SAMPLE_RATE * segment_s), device=device)
        _ = model(spec_extractor(dummy))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            probs = predict_probabilities(
                model, model_type, spec_extractor, val_loader, device)
    else:
        probs = predict_probabilities(
            model, model_type, spec_extractor, val_loader, device)
    probs = collapse_probs_to_3class(probs)

    print(f"    computed in {time.time()-t0:.0f}s, "
          f"{len(probs)} segment prob arrays")
    if not no_cache:
        save_probs_to_cache(probs, cp)
        print(f"    cached → {cp}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def average_prob_dicts_per_class(prob_dicts, weights_per_class):
    """
    Per-class weighted average of probability dicts.

    Parameters
    ----------
    prob_dicts : list of dict[key -> ndarray (T, C)]
        One dict per checkpoint. Keys are segment identifiers, values are
        float arrays of shape ``(T_frames, n_classes)``. C is assumed to
        be 3 (i.e. ``collapse_probs_to_3class`` has already been applied
        upstream by ``get_or_compute_probs``).
    weights_per_class : ndarray of shape ``(n_classes, n_models)``
        Row c is the (already-normalized) weight vector across models
        for class c. Each row should sum to 1; a row of zeros is invalid
        and is caught upstream.

    Returns
    -------
    dict mapping each common key to ``(T, C)`` ndarray, where
    ``out[:, c] = sum_m weights_per_class[c, m] * prob_dicts[m][:, c]``,
    truncated to the shortest T across models for that key.

    This is the per-class analogue of ``average_prob_dicts``: instead of
    one weight vector applied uniformly to all classes, each class gets
    its own weight vector. A zero weight excludes a model from that
    class entirely, which is the mechanism behind the hybrid strategy
    (e.g. PGI ensemble for BMABZ+BP, AMSE MT ensemble for D).
    """
    if not prob_dicts:
        return {}
    n_models = len(prob_dicts)
    n_classes = weights_per_class.shape[0]
    assert weights_per_class.shape == (n_classes, n_models), (
        f"weights_per_class shape {weights_per_class.shape} does not "
        f"match (n_classes={n_classes}, n_models={n_models})"
    )

    # Intersect keys across models — defensive against minor mismatches.
    common = set(prob_dicts[0].keys())
    for pd in prob_dicts[1:]:
        common &= set(pd.keys())
    n_dropped = len(prob_dicts[0]) - len(common)
    if n_dropped > 0:
        print(f"  WARNING: {n_dropped} keys missing from at least one "
              f"model; dropping them from the per-class ensemble.")

    # Broadcasting shape: (M, 1, C) against stacked (M, T, C) → (M, T, C).
    w_bc = weights_per_class.T[:, None, :].astype(np.float32)

    out = {}
    for key in common:
        arrays = [pd[key] for pd in prob_dicts]
        min_T = min(a.shape[0] for a in arrays)
        # truncate to common T; class dim assumed aligned at n_classes
        stacked = np.stack(
            [a[:min_T, :n_classes].astype(np.float32) for a in arrays],
            axis=0,
        )  # (M, T, C)
        out[key] = (stacked * w_bc).sum(axis=0)  # (T, C)

    return out


def build_weights_spec(args, per_class_mode, n_ckpts):
    """Canonical, normalized weight description used for both combining
    and cache-keying. Normalizing here means two runs that differ only by
    unnormalized weight scale (e.g. ``1 1`` vs ``2 2``) hash to the same
    key and reuse the same cache.
    """
    if n_ckpts == 1:
        return {"mode": "single"}
    if per_class_mode:
        raw = np.array(
            [args.weights_bmabz, args.weights_d, args.weights_bp],
            dtype=np.float64,
        )
        row_sums = raw.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            bad = [c for c, s in zip(cfg.CALL_TYPES_3, row_sums.ravel())
                   if s <= 0]
            raise ValueError(
                f"Per-class weights for {bad} sum to zero — no models "
                f"would contribute to that class.")
        mat = raw / row_sums
        return {"mode": "per_class",
                "matrix": [[round(float(w), 6) for w in row] for row in mat]}
    if args.weights:
        total = sum(args.weights)
        return {"mode": "global",
                "weights": [round(float(w) / total, 6) for w in args.weights]}
    return {"mode": "uniform", "n": n_ckpts}


def combine_prob_dicts(all_prob_dicts, weights_spec):
    """Combine per-checkpoint prob dicts according to ``weights_spec``."""
    if len(all_prob_dicts) == 1:
        return all_prob_dicts[0]
    mode = weights_spec["mode"]
    if mode == "per_class":
        wpc = np.array(weights_spec["matrix"], dtype=np.float32)  # (3, N)
        return average_prob_dicts_per_class(all_prob_dicts, wpc)
    if mode == "global":
        return average_prob_dicts(all_prob_dicts,
                                  weights=weights_spec["weights"])
    # uniform
    return average_prob_dicts(all_prob_dicts, weights=None)


def per_class_print(metrics, thresholds, label="ENSEMBLE"):
    """Pretty-print the per-class table matching ensemble_predict's format,
    with both macro-F1 conventions."""
    print(f"\n  {label} RESULT:")
    for c in cfg.CALL_TYPES_3:
        m = metrics.get(c, {})
        idx = cfg.CALL_TYPES_3.index(c)
        print(f"    {c.upper():6} t={thresholds[idx]:.2f}  "
              f"TP={m.get('tp', 0):5} FP={m.get('fp', 0):5} "
              f"FN={m.get('fn', 0):5}  "
              f"P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} "
              f"F1={m.get('f1', 0):.3f}")
    overall = metrics.get("overall", {})
    macro = macro_f1(metrics)
    macro_p = macro_f1_paper(metrics)
    print(f"    OVERALL F1={overall.get('f1', 0):.3f}  "
          f"MACRO F1={macro:.3f}  PAPER MACRO F1={macro_p:.3f}")
    print("      (MACRO F1 = mean per-class F1, official BioDCASE Task 2; "
          "PAPER MACRO F1 = F1 of mean-P/mean-R, Geldenhuys convention)")
    print(f"    Tuned thresholds: "
          f"{['%.2f' % t for t in thresholds]}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fork is required for parallel tuning; downgrade gracefully off-Linux.
    if args.val_workers > 1:
        try:
            mp.get_context("fork")
        except (ValueError, RuntimeError):
            print("  'fork' start method unavailable; falling back to "
                  "--val-workers 1 (sequential).")
            args.val_workers = 1

    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache
    print(f"Cache dir: {cache_dir} (no_cache={args.no_cache}, "
          f"use_fp16={args.use_fp16})")
    print(f"Eval tiles: {args.segment_s:.0f}s segments, "
          f"{args.overlap_s:.1f}s overlap")
    print(f"Threshold tuning: {args.val_workers} worker(s) "
          f"({'parallel' if args.val_workers > 1 else 'sequential'})")

    n_ckpts = len(args.checkpoints)
    per_class_mode = args.weights_bmabz is not None  # all three guaranteed set
    ckpt_ids = [Path(c).parent.name for c in args.checkpoints]
    weights_spec = build_weights_spec(args, per_class_mode, n_ckpts)
    ident = ensemble_identity(ckpt_ids, weights_spec, args.segment_s, OBJECTIVE)
    ens_key = ensemble_key(ident)
    print(f"Ensemble key: {ens_key}  "
          f"(mode={weights_spec['mode']}, {n_ckpts} ckpt)")

    # ---- Ground-truth events: always needed, cheap to build. ----------
    print("\nLoading validation annotations...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    gt_events = []
    for _, row in val_anns.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    print(f"  {len(gt_events)} ground-truth events")

    spec_extractor = SpectrogramExtractor().to(device)

    # ---- Lazy validation loader: built only when inference is needed. -
    # A full-cache run (per-ckpt or ensemble-prob hit) never touches this,
    # so it skips segment building + DataLoader construction entirely.
    _loader_cache = {"obj": None}

    def val_loader_factory():
        if _loader_cache["obj"] is None:
            print("  building validation DataLoader (inference needed)...")
            val_segs = build_val_segments(
                val_manifest, val_anns,
                segment_s=args.segment_s, overlap_s=args.overlap_s,
            )
            _loader_cache["obj"] = DataLoader(
                WhaleDataset(val_segs), batch_size=args.batch_size,
                shuffle=False, num_workers=cfg.NUM_WORKERS,
                collate_fn=collate_fn, pin_memory=True,
            )
            print(f"  {len(val_segs)} {args.segment_s:.0f}s tiles "
                  f"({args.overlap_s:.1f}s overlap)")
        return _loader_cache["obj"]

    # ---- Ensemble probabilities: fast path or build from checkpoints. -
    ens_probs = None
    ens_prob_path = ensemble_probs_cache_path(cache_dir, ens_key)
    if use_cache and not args.per_model_eval and ens_prob_path.exists():
        print(f"\nEnsemble-prob cache hit: {ens_prob_path.name} "
              f"(skipping all per-checkpoint inference)")
        ens_probs = load_cached_probs(ens_prob_path)

    if ens_probs is None:
        all_prob_dicts = []
        for i, ckpt_path in enumerate(args.checkpoints):
            print(f"\n[{i+1}/{n_ckpts}] {ckpt_path}")
            probs = get_or_compute_probs(
                ckpt_path, spec_extractor, val_loader_factory, device,
                cache_dir, no_cache=args.no_cache, use_fp16=args.use_fp16,
                segment_s=args.segment_s)
            all_prob_dicts.append(probs)

            if args.per_model_eval:
                pm_path = per_model_thr_cache_path(
                    ckpt_path, cache_dir, args.segment_s)
                if use_cache and pm_path.exists():
                    thr = load_thresholds(pm_path)
                    print(f"  per-model threshold cache hit: {pm_path.name}")
                else:
                    thr = tune_thresholds_per_class(
                        probs, gt_events, parallel_workers=args.val_workers)
                    if use_cache:
                        save_thresholds(thr, pm_path, meta={
                            "ckpt": ckpt_ids[i],
                            "segment_s": args.segment_s,
                            "objective": OBJECTIVE,
                        })
                metrics = evaluate_with_thresholds(probs, gt_events, thr)
                macro = macro_f1(metrics)
                macro_p = macro_f1_paper(metrics)
                print(f"  individual: overall F1="
                      f"{metrics.get('overall', {}).get('f1', 0):.3f}, "
                      f"macro={macro:.3f}, macro_paper={macro_p:.3f}, "
                      f"thr={['%.2f' % t for t in thr]}")
                for c in cfg.CALL_TYPES_3:
                    m = metrics.get(c, {})
                    print(f"    {c.upper():6} F1={m.get('f1', 0):.3f} "
                          f"P={m.get('precision', 0):.3f} "
                          f"R={m.get('recall', 0):.3f}")

        # --------------------------------------------------------------
        # Combine → ensemble probabilities.
        # --------------------------------------------------------------
        print(f"\n{'='*64}\nENSEMBLE\n{'='*64}")
        if n_ckpts == 1 and (per_class_mode or args.weights is not None):
            print("Note: only one checkpoint provided; weight flags ignored.")
        if weights_spec["mode"] == "per_class":
            print("Per-class normalized weights (rows = classes, "
                  "cols = checkpoints in input order):")
            for c, row in zip(cfg.CALL_TYPES_3, weights_spec["matrix"]):
                print(f"  {c.upper():6}: [{', '.join('%.2f' % w for w in row)}]")
        elif weights_spec["mode"] == "global":
            print(f"Per-model weights (normalized): {weights_spec['weights']}")

        ens_probs = combine_prob_dicts(all_prob_dicts, weights_spec)
        if use_cache:
            save_probs_to_cache(ens_probs, ens_prob_path)
            print(f"  cached ensemble probs → {ens_prob_path.name}")
    else:
        print(f"\n{'='*64}\nENSEMBLE\n{'='*64}")

    # ---- Thresholds: cache hit or (parallel) tune. --------------------
    ens_thr_path = ensemble_thr_cache_path(cache_dir, ens_key)
    if use_cache and ens_thr_path.exists():
        ens_thr = load_thresholds(ens_thr_path)
        print(f"Ensemble-threshold cache hit: {ens_thr_path.name} "
              f"({['%.2f' % t for t in ens_thr]})")
    else:
        print("Tuning thresholds on ensemble probabilities...")
        ens_thr = tune_thresholds_per_class(
            ens_probs, gt_events, parallel_workers=args.val_workers)
        if use_cache:
            save_thresholds(ens_thr, ens_thr_path, meta={
                "ckpts": ckpt_ids,
                "weights": weights_spec,
                "segment_s": args.segment_s,
                "objective": OBJECTIVE,
            })
            print(f"  cached ensemble thresholds → {ens_thr_path.name}")

    metrics = evaluate_with_thresholds(ens_probs, gt_events, ens_thr)
    per_class_print(metrics, ens_thr, label="ENSEMBLE")


if __name__ == "__main__":
    main()
