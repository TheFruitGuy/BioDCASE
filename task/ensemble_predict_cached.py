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

from ensemble_predict import (
    average_prob_dicts, build_model_for_ckpt, predict_probabilities,
    tune_thresholds_on_probs, evaluate_with_thresholds,
)


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
                   help="Directory for per-checkpoint cached probs.")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable caching (always recompute).")
    p.add_argument("--use_fp16", action="store_true",
                   help="FP16 autocast during inference. ~1.5–2× speedup, "
                        "negligible accuracy hit (<0.001 F1 typically).")
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


def cache_path_for(ckpt_path: Path, cache_dir: Path,
                   segment_s: float = 30.0) -> Path:
    """One pickle per (checkpoint, segment_length) pair.

    30s keeps the legacy name (no suffix) so existing caches remain
    reusable. Other lengths get a ``_seg<N>`` suffix so they don't
    collide with the 30s cache or with each other.
    """
    seg_tag = "" if abs(segment_s - 30.0) < 1e-6 else f"_seg{int(segment_s)}"
    return cache_dir / f"{ckpt_path.parent.name}{seg_tag}_probs.pkl"


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


@torch.no_grad()
def get_or_compute_probs(ckpt_path, spec_extractor, val_loader, device,
                         cache_dir, no_cache=False, use_fp16=False,
                         segment_s=30.0):
    """Cache hit → load pickle. Cache miss → run inference, save, return.

    ``segment_s`` is used both to pick the cache filename and to size the
    dummy input that materialises the lazy CNN projection layer.
    """
    cp = cache_path_for(Path(ckpt_path), cache_dir, segment_s=segment_s)
    if not no_cache and cp.exists():
        print(f"  cache hit: {cp.name}")
        return load_cached_probs(cp)

    print(f"  cache miss, running inference...")
    t0 = time.time()
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


def per_class_print(metrics, thresholds, label="ENSEMBLE"):
    """Pretty-print the per-class table matching ensemble_predict's format."""
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
    macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                           for c in cfg.CALL_TYPES_3]))
    print(f"    OVERALL F1={overall.get('f1', 0):.3f}  "
          f"MACRO F1={macro:.3f}")
    print(f"    Tuned thresholds: "
          f"{['%.2f' % t for t in thresholds]}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir} (no_cache={args.no_cache}, "
          f"use_fp16={args.use_fp16})")
    print(f"Eval tiles: {args.segment_s:.0f}s segments, "
          f"{args.overlap_s:.1f}s overlap")

    # Validation loader is shared across all checkpoints.
    print("\nLoading validation data...")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(val_manifest, val_anns,
                                  segment_s=args.segment_s,
                                  overlap_s=args.overlap_s)
    val_loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  {len(val_segs)} {args.segment_s:.0f}s tiles "
          f"({args.overlap_s:.1f}s overlap)")

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

    # Get probs per checkpoint (cache hit or compute).
    all_prob_dicts = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        probs = get_or_compute_probs(
            ckpt_path, spec_extractor, val_loader, device,
            cache_dir, no_cache=args.no_cache, use_fp16=args.use_fp16,
            segment_s=args.segment_s)
        all_prob_dicts.append(probs)

        if args.per_model_eval:
            thr = tune_thresholds_on_probs(probs, gt_events)
            metrics = evaluate_with_thresholds(probs, gt_events, thr)
            macro = float(np.mean([metrics.get(c, {}).get("f1", 0.0)
                                   for c in cfg.CALL_TYPES_3]))
            print(f"  individual: overall F1="
                  f"{metrics.get('overall', {}).get('f1', 0):.3f}, "
                  f"macro={macro:.3f}, thr={['%.2f' % t for t in thr]}")
            for c in cfg.CALL_TYPES_3:
                m = metrics.get(c, {})
                print(f"    {c.upper():6} F1={m.get('f1', 0):.3f} "
                      f"P={m.get('precision', 0):.3f} "
                      f"R={m.get('recall', 0):.3f}")

    # ------------------------------------------------------------------
    # Combine → threshold tune → final metrics.
    # ------------------------------------------------------------------
    print(f"\n{'='*64}\nENSEMBLE\n{'='*64}")

    per_class_mode = args.weights_bmabz is not None  # all three guaranteed set

    if len(all_prob_dicts) == 1:
        # Single checkpoint: weights and per-class weights are no-ops.
        if per_class_mode or args.weights is not None:
            print("Note: only one checkpoint provided; weight flags ignored.")
        ens_probs = all_prob_dicts[0]

    elif per_class_mode:
        # Per-class weighted average. Build (3, N) weight matrix in
        # cfg.CALL_TYPES_3 order: (bmabz, d, bp).
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
        weights_per_class = (raw / row_sums).astype(np.float32)

        print("Per-class normalized weights (rows = classes, "
              "cols = checkpoints in input order):")
        for c, row in zip(cfg.CALL_TYPES_3, weights_per_class):
            print(f"  {c.upper():6}: [{', '.join('%.2f' % w for w in row)}]")

        ens_probs = average_prob_dicts_per_class(
            all_prob_dicts, weights_per_class)

    else:
        # Plain (uniform or globally weighted) average.
        weights = None
        if args.weights:
            total = sum(args.weights)
            weights = [w / total for w in args.weights]
            print(f"Per-model weights (normalized): {weights}")
        ens_probs = average_prob_dicts(all_prob_dicts, weights=weights)

    print("Tuning thresholds on ensemble probabilities...")
    ens_thr = tune_thresholds_on_probs(ens_probs, gt_events)
    metrics = evaluate_with_thresholds(ens_probs, gt_events, ens_thr)
    per_class_print(metrics, ens_thr, label="ENSEMBLE")


if __name__ == "__main__":
    main()
