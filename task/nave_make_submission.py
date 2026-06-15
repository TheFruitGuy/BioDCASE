"""
nave_make_submission.py -- run a trained NAVE / conformer model on the blind
EVALUATION set and write challenge-format submission CSVs.
==============================================================================
(Named nave_* to avoid colliding with the CNN-BiLSTM line's make_submission.py.)

Thresholds are tuned ONCE on the full development (validation) set at the chosen
tile length, then applied unchanged to the evaluation audio (never tuned on the
blind set). Output columns:

    dataset, filename, annotation, start_datetime, end_datetime

The annotation column carries the model's coarse label {bmabz, d, bp} as-is --
the challenge scorer relabels d -> fm internally, so we must submit 'd', not 'fm'.
Matching is by ISO8601 datetime within a +/-2 min window.

Per-checkpoint EVAL probabilities are cached under --eval-cache-dir (separate from
the dev cache), so an ensemble run is resumable: a seed already inferred is reused
instantly on a re-run, and a crash mid-ensemble does not throw away finished seeds.

    # single best seed
    CUDA_VISIBLE_DEVICES=0 python nave_make_submission.py \
        --checkpoints runs/phase13r_3c_s666_20260612_132012/phase13r_best.pt \
        --eval-root /home/matthias-nagl/BioDCASE/task/2026_BioDCASE_evaluation_set \
        --audio-subdir evaluation/audio \
        --segment-s 60 --use_fp16 --val-workers 20 --out-dir runs/submission_s666

    # ensemble (probabilities averaged; each seed cached on first pass)
    CUDA_VISIBLE_DEVICES=0 python nave_make_submission.py \
        --checkpoints <ckpt1> <ckpt2> ... <ckptN> \
        --eval-root .../2026_BioDCASE_evaluation_set --audio-subdir evaluation/audio \
        --segment-s 60 --use_fp16 --out-dir runs/submission_ens

ASSUMPTIONS (verify against your actual eval release):
  * eval audio laid out as  <eval-root>/<audio-subdir>/<site>/*.wav
    (or pass --flat if the wavs sit directly in <eval-root>/<audio-subdir>);
  * filenames encode the UTC start like 2014-06-29T23-00-00_000.wav
    (same convention as the dev set; parsed by dataset_final._parse_file_start_dt);
  * eval audio sample rate is 250 Hz (corpus standard).
Eyeball the first CSV before submitting.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import nave_config as cfg
from dataset_final import (
    WhaleDataset, build_val_segments, collate_fn, get_file_manifest, load_annotations,
    _parse_file_start_dt,
)
from postprocess_final import Detection, postprocess_predictions
from eval_conformer import (
    tune_thresholds_per_class, evaluate_with_thresholds, macro_f1_paper, CLASS_NAMES,
)
from nave_eval import combine_probs
from nave_eval_official import tune_for_official
from evaluate_phase13 import load_model_spec, predict_probs, get_or_compute


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One checkpoint (single model) or several (probability ensemble).")
    p.add_argument("--eval-root", required=True,
                   help="Root of the evaluation set (contains the audio subdir).")
    p.add_argument("--audio-subdir", default="audio",
                   help="Subdir under --eval-root holding the audio (e.g. 'evaluation/audio').")
    p.add_argument("--flat", action="store_true",
                   help="WAVs sit directly under the audio dir (one dataset).")
    p.add_argument("--dataset-name", default=None,
                   help="Dataset name to use in --flat mode (default: audio dir name).")
    p.add_argument("--segment-s", type=float, default=60.0)
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-checkpoint ensemble weights (default uniform).")
    p.add_argument("--tune-metric", choices=["f1", "official"], default="f1")
    p.add_argument("--out-dir", default="runs/submission")
    p.add_argument("--cache_dir", default="runs/nave_prob_cache",
                   help="DEV prob cache (reused from earlier dev evals).")
    p.add_argument("--eval-cache-dir", default="runs/nave_eval_prob_cache",
                   help="EVAL prob cache (separate from dev; keeps ensemble runs resumable).")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--val-workers", type=int, default=min((os.cpu_count() or 1), 13))
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        p.error(f"--weights has {len(args.weights)} values, need {len(args.checkpoints)}.")
    return args


def build_eval_manifest(eval_root, audio_subdir, flat, dataset_name):
    """Scan the evaluation audio into a manifest DataFrame mirroring
    dataset_final's columns: dataset, filename, path, duration_s, start_dt, end_dt."""
    base = Path(eval_root) / audio_subdir
    if not base.exists():
        base = Path(eval_root)
    if flat:
        groups = {(dataset_name or base.name): sorted(base.glob("*.wav"))}
    else:
        site_dirs = sorted(d for d in base.iterdir() if d.is_dir())
        if site_dirs:
            groups = {d.name: sorted(d.glob("*.wav")) for d in site_dirs}
        else:
            groups = {base.name: sorted(base.glob("*.wav"))}

    rows, skipped = [], 0
    for ds, wavs in groups.items():
        for wav in wavs:
            fsd = _parse_file_start_dt(wav.name)
            if fsd is None:
                skipped += 1
                continue
            info = sf.info(str(wav))
            dur = info.frames / float(info.samplerate)
            rows.append(dict(dataset=ds, filename=wav.name, path=str(wav),
                             duration_s=dur, start_dt=fsd,
                             end_dt=fsd + timedelta(seconds=dur)))
    if skipped:
        print(f"  WARNING: {skipped} file(s) skipped (unparseable start datetime).")
    if not rows:
        raise SystemExit(f"No usable .wav files found under {base}")
    df = pd.DataFrame(rows)
    print(f"  eval manifest: {len(df)} files across {df['dataset'].nunique()} dataset(s): "
          f"{sorted(df['dataset'].unique())}")
    return df


def main():
    args = parse_args()
    import config_final as _cf
    _cf.USE_3CLASS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.checkpoints)
    print(f"[submission] {n} checkpoint(s) | {args.segment_s:.0f}s tiles | tune={args.tune_metric}")

    # ---------- 1. tune thresholds on the FULL dev set ----------
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

    _dev_loader = {"obj": None}

    def dev_loader_factory():
        if _dev_loader["obj"] is None:
            segs = build_val_segments(val_manifest, val_anns,
                                      segment_s=args.segment_s, overlap_s=args.overlap_s)
            _dev_loader["obj"] = DataLoader(
                WhaleDataset(segs), batch_size=args.batch_size, shuffle=False,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
        return _dev_loader["obj"]

    print("  tuning thresholds on dev (uses 60s prob cache where available)...")
    dev_dicts = [get_or_compute(c, dev_loader_factory, device, Path(args.cache_dir),
                                args.segment_s, args.no_cache, args.use_fp16)
                 for c in args.checkpoints]
    dev_probs = combine_probs(dev_dicts, args.weights)
    thr = tune_thresholds_per_class(dev_probs, gt, workers=args.val_workers)
    if args.tune_metric == "official":
        thr, _ = tune_for_official(dev_probs, gt, start=thr)
    dev_B = macro_f1_paper(evaluate_with_thresholds(dev_probs, gt, thr))
    print(f"  dev B={dev_B:.4f}  thresholds={[round(float(t),3) for t in thr]}")

    # ---------- 2. inference on the EVALUATION audio (cached per checkpoint) ----------
    eval_manifest = build_eval_manifest(args.eval_root, args.audio_subdir,
                                        args.flat, args.dataset_name)
    eval_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in eval_manifest.iterrows()}
    empty_anns = val_anns.iloc[0:0]
    eval_segs = build_val_segments(eval_manifest, empty_anns,
                                   segment_s=args.segment_s, overlap_s=args.overlap_s)
    print(f"  eval tiles: {len(eval_segs)}")
    eval_loader = DataLoader(WhaleDataset(eval_segs), batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.NUM_WORKERS,
                             collate_fn=collate_fn, pin_memory=True)

    def eval_loader_factory():
        # fresh tqdm wrapper each call so the progress bar restarts per uncached seed
        return tqdm(eval_loader, total=len(eval_loader), desc="    eval", ncols=80)

    eval_dicts = []
    for i, ckpt in enumerate(args.checkpoints):
        print(f"    [{i+1}/{n}] {Path(ckpt).parent.name}")
        eval_dicts.append(get_or_compute(ckpt, eval_loader_factory, device,
                                         Path(args.eval_cache_dir), args.segment_s,
                                         args.no_cache, args.use_fp16))
    eval_probs = combine_probs(eval_dicts, args.weights)

    # ---------- 3. detections -> submission rows ----------
    # Submit the raw coarse label (bmabz / d / bp); the challenge scorer maps d -> fm.
    dets = postprocess_predictions(eval_probs, np.asarray(thr))
    rows = []
    for d in dets:
        fsd = eval_start_dts.get((d.dataset, d.filename))
        if fsd is None:
            continue
        rows.append(dict(
            dataset=d.dataset, filename=d.filename,
            annotation=d.label,
            start_datetime=(fsd + timedelta(seconds=float(d.start_s))).isoformat(),
            end_datetime=(fsd + timedelta(seconds=float(d.end_s))).isoformat()))
    sub = pd.DataFrame(rows, columns=["dataset", "filename", "annotation",
                                      "start_datetime", "end_datetime"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_dataset").mkdir(exist_ok=True)
    for ds, g in sub.groupby("dataset"):
        g.to_csv(out / "per_dataset" / f"{ds}.csv", index=False)
    sub.to_csv(out / "submission.csv", index=False)

    print(f"\n  wrote {len(sub)} detections to {out/'submission.csv'}")
    print(f"  per-dataset CSVs in {out/'per_dataset'}/")
    print("  detections per class:")
    for lbl, c in sub["annotation"].value_counts().items():
        print(f"    {lbl:6} {c}")
    print("  detections per dataset:")
    for ds, c in sub["dataset"].value_counts().items():
        print(f"    {ds:16} {c}")
    print("\n  Sanity-check the first rows before submitting:")
    print(sub.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
