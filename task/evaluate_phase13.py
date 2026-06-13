"""
evaluate_phase13.py -- rescore checkpoints in the OFFICIAL metric at any tile length.
=====================================================================================
Loads each checkpoint with the right builder -- native NAVE checkpoints via the NAVE
loader, any legacy phase variant (plain Conformer / FDY / FDY+PCEN / wide-kernel /
CRNN / LEAF / tfwSE / LPCEN / delta / dual-res) via eval_conformer.load_conformer --
evaluates it at the requested tile length (default 60 s), and reports the official
BioDCASE metric B = F1(mean-P, mean-R) per checkpoint, one row each.

    CUDA_VISIBLE_DEVICES=0 python evaluate_phase13.py --segment-s 60 --use_fp16 --val-workers 20 \
        --checkpoints \
          runs/phase13b_*/phase13b_best.pt \
          runs/phase13d_*/phase13d_best.pt \
          runs/phase13n_*/phase13n_best.pt \
          runs/phase13q_*/phase13q_best.pt \
          runs/phase13r_3c_s666_20260612_132012/phase13r_best.pt

Each checkpoint is evaluated INDEPENDENTLY (different architectures / seeds are not
ensembled here -- use evaluate_loso.py for the multi-seed ensemble). Probabilities
cache per (run, segment_s), shared with nave_eval / nave_seglen_sweep / evaluate_loso.

Thresholds: per-class F1 coordinate descent (default), or --tune-metric official to
refine for B. --official-scorer cross-checks the chosen row with evaluation_submissions.run.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_features import NAVEFeatureExtractor
from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn, get_file_manifest, load_annotations,
)
from postprocess_final import Detection, collapse_probs_to_3class, postprocess_predictions
from eval_conformer import (
    load_conformer, tune_thresholds_per_class, evaluate_with_thresholds,
    macro_f1, macro_f1_paper, CLASS_NAMES,
)
from nave_eval import (
    cache_path_for, load_cached_probs, save_probs_to_cache, build_nave_from_ckpt,
)
from nave_eval_official import (
    tune_for_official, write_submission_csvs, write_gt_csvs, official_B_from_confmatrix,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Native NAVE or any legacy phase checkpoint (mixable).")
    p.add_argument("--segment-s", type=float, default=60.0)
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--tune-metric", choices=["f1", "official"], default="f1")
    p.add_argument("--official-scorer", action="store_true",
                   help="Cross-check each row with evaluation_submissions.run().")
    p.add_argument("--out-dir", type=str, default="runs/phase13_submission")
    p.add_argument("--cache_dir", type=str, default="runs/nave_prob_cache")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--val-workers", type=int, default=min((os.cpu_count() or 1), 13))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


def load_model_spec(ckpt, device):
    """Native NAVE -> NAVE + NAVEFeatureExtractor; any legacy phase variant -> load_conformer."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    native = (isinstance(obj, dict) and obj.get("model") == "NAVE") \
        or any(k.startswith("stem.") for k in sd)
    if native:
        return build_nave_from_ckpt(ckpt, device), NAVEFeatureExtractor().to(device)
    model, spec, _ = load_conformer(ckpt, device)
    return model.to(device).eval(), spec.to(device)


@torch.no_grad()
def predict_probs(model, spec, loader, device, use_fp16):
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_fp16 and device.type == "cuda") else nullcontext())
    hop = getattr(spec, "hop_length", cfg.HOP_LENGTH)
    raw = {}
    for audio, _t, _m, metas in loader:
        audio = audio.to(device)
        with autocast:
            logits = model(spec(audio))
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n = min((meta["end_sample"] - meta["start_sample"]) // hop, probs[j].shape[0])
            raw[key] = probs[j, :n, :]
    return collapse_probs_to_3class(raw)


def get_or_compute(ckpt, loader_factory, device, cache_dir, seg, no_cache, use_fp16):
    cp = cache_path_for(Path(ckpt), cache_dir, seg)
    if not no_cache and cp.exists():
        print("    cache hit")
        return load_cached_probs(cp)
    print("    cache miss, loading architecture + running inference...")
    model, spec = load_model_spec(ckpt, device)
    probs = predict_probs(model, spec, loader_factory(), device, use_fp16)
    if not no_cache:
        save_probs_to_cache(probs, cp)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def main():
    args = parse_args()
    import config_final as _cf
    _cf.USE_3CLASS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.checkpoints)
    print(f"[phase13 rescore] {n} ckpt | {args.segment_s:.0f}s tiles | "
          f"tune={args.tune_metric} | official_scorer={args.official_scorer}")

    val_sites = list(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(val_sites)
    val_anns = load_annotations(val_sites, manifest=val_manifest)
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}
    gt = []
    for _, row in val_anns.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds()))
    print(f"  {len(gt)} ground-truth events")

    _loader = {"obj": None}

    def loader_factory():
        if _loader["obj"] is None:
            segs = build_val_segments(val_manifest, val_anns,
                                      segment_s=args.segment_s, overlap_s=args.overlap_s)
            _loader["obj"] = DataLoader(
                WhaleDataset(segs), batch_size=args.batch_size, shuffle=False,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
            print(f"  built loader: {len(segs)} tiles")
        return _loader["obj"]

    cache_dir = Path(args.cache_dir)
    rows = []
    for i, ckpt in enumerate(args.checkpoints):
        name = Path(ckpt).parent.name
        print(f"\n[{i+1}/{n}] {name}")
        _cf.USE_3CLASS = True
        probs = get_or_compute(ckpt, loader_factory, device, cache_dir,
                               args.segment_s, args.no_cache, args.use_fp16)

        thr = tune_thresholds_per_class(probs, gt, workers=args.val_workers)
        if args.tune_metric == "official":
            thr, _ = tune_for_official(probs, gt, start=thr)
        metrics = evaluate_with_thresholds(probs, gt, thr)
        B, A = macro_f1_paper(metrics), macro_f1(metrics)
        per = {c: metrics.get(c, {}).get("f1", 0.0) for c in CLASS_NAMES}
        rows.append((name, B, A, per, thr))
        print(f"  OFFICIAL B={B:.4f}  (A={A:.4f})  "
              f"{' '.join(f'{c.upper()}={per[c]:.3f}' for c in CLASS_NAMES)}  "
              f"thr={[round(float(t),2) for t in thr]}")

        if args.official_scorer:
            try:
                import evaluation_submissions as offsc
            except ImportError:
                print("    evaluation_submissions.py not found; skipping official scorer.")
            else:
                dets = postprocess_predictions(probs, np.asarray(thr))
                pred_dir, _ = write_submission_csvs(dets, file_start_dts,
                                                    Path(args.out_dir) / name / "pred")
                gt_dir = write_gt_csvs(val_anns, Path(args.out_dir) / name / "gt")
                conf = offsc.run(pred_dir, gt_dir, iou_threshold=cfg.IOU_THRESHOLD)
                Bo, P, R = official_B_from_confmatrix(conf)
                print(f"    official-scorer B={Bo:.4f} (meanP={P:.3f}, meanR={R:.3f})")

    # ---- summary (input order) ----
    print(f"\n{'='*78}\nPER-CHECKPOINT (official B, {args.segment_s:.0f}s tiles, tune={args.tune_metric})\n{'='*78}")
    hdr = f"{'checkpoint':38} {'B':>7} {'A':>7} " + " ".join(f"{c.upper():>6}" for c in CLASS_NAMES)
    print(hdr); print("-" * len(hdr))
    Bs = []
    for name, B, A, per, thr in rows:
        Bs.append(B)
        print(f"{name[:38]:38} {B:7.4f} {A:7.4f} "
              + " ".join(f"{per[c]:6.3f}" for c in CLASS_NAMES))
    if n > 1:
        Bs = np.array(Bs)
        print("-" * len(hdr))
        print(f"{'mean +/- std over checkpoints':38} {Bs.mean():7.4f} {Bs.std():7.4f}")
        print(f"{'best (reference only)':38} {Bs.max():7.4f}")


if __name__ == "__main__":
    main()
