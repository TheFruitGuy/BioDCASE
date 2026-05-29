"""
Hard-Negative Mining — _final-native, all sources, all targets
===============================================================

Turns the 10 re-scored base checkpoints (``paper_best.pt``) into the hard-
negative JSONs that the HNM fine-tuning runs (Block B) train against.

Why this exists instead of calling ``mine_hard_negatives.py`` 24x
-----------------------------------------------------------------
``mine_hard_negatives.py`` / ``mine_hard_negatives_7class.py`` are written
against the OLD module set (``config`` / ``dataset`` / ``postprocess`` /
``spectrogram`` / ``ensemble_predict``), not the ``_final`` modules the base
models were trained and re-scored with. That mismatch is exactly the
``USE_3CLASS``-dependent 7->3 collapse hazard we sidestepped in the rescore.
This script reuses the rescore plumbing (``build_model`` sets ``USE_3CLASS``
from the checkpoint's class count; ``infer`` returns probs keyed for
``stitch_segments``) and mirrors the mining LOGIC exactly:

  * proposal at a deliberately low target-class threshold (others 0.999)
  * stitch -> smooth -> threshold -> merge/filter
  * match to GT of the target class, keep events with max IoU < cutoff (FPs)
  * top-k by confidence, persist locations

3-class checkpoints mine the three group names (bmabz/d/bp) in 3-class space.
7-class checkpoints mine PER SUBCLASS (no collapse, GT from the fine-grained
``annotation`` field, a merge that skips the COLLAPSE_MAP rewrite), so the
7-class HNM trainer can do subclass-level PGI. The group structure is:
``bmabz={bma,bmb,bmz}``, ``d={bmd,bpd}``, ``bp={bp20,bp20plus}``.

Efficiency: each model's train-set probabilities are computed ONCE (10 passes
total). Single-best JSONs come straight from each model's probs; the per-head
ensemble JSONs come from the average of the 5 stitched file-prob streams
(stitch is linear, so averaging stitched files == stitching averaged windows).

Outputs (default --out-dir runs/hardnegs_final), with all three groups:
  3class/seed{S}/{bmabz,d,bp}.json          (5 seeds)        = 15
  3class/ensemble/{bmabz,d,bp}.json                          =  3
  7class/seed{S}/{7 subclasses}.json        (5 seeds)        = 35
  7class/ensemble/{7 subclasses}.json                        =  7
                                                       total = 60 JSONs

Usage
-----
::

    # full job, all heads + all targets, on one A6000 (skip the Blackwells)
    CUDA_VISIBLE_DEVICES=6 python mine_hard_negatives_final.py

    # split heads across two GPUs to halve wall time
    CUDA_VISIBLE_DEVICES=6 python mine_hard_negatives_final.py --heads 3class &
    CUDA_VISIBLE_DEVICES=8 python mine_hard_negatives_final.py --heads 7class &

    # see exactly what would be produced without running inference
    python mine_hard_negatives_final.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config_final as cfg
from dataset_final import build_val_segments, get_file_manifest, load_annotations
from model_final import WhaleVAD
from spectrogram_final import SpectrogramExtractor
import postprocess_final as pp
from postprocess_final import Detection

# wandb_utils is optional — never let a logging hiccup abort a long mining run.
try:
    import wandb_utils as wbu
except Exception:
    wbu = None


# ----------------------------------------------------------------------
# Locked mining hyperparameters (experiment plan section 3.3)
# ----------------------------------------------------------------------
PROPOSAL_THRESHOLD = 0.2     # target-class frame threshold (others 0.999)
MAX_IOU_FOR_FP = 0.1         # a proposal is an FP if max IoU with GT < this
TOP_K = 1500                 # keep this many FPs, highest-confidence first


# ----------------------------------------------------------------------
# Source checkpoints — real run dirs keyed by seed (symlink-independent).
# Edit here if your dirs differ. Checkpoint file = <dir>/<--ckpt-name>.
# ----------------------------------------------------------------------
SOURCES = {
    "3class": [
        (42,   "final_3c_s42_20260527_200054"),
        (2024, "final_3c_s2024_20260527_200309"),
        (7777, "final_3c_s7777_20260527_200401"),
        (9999, "final_3c_s9999_20260527_200852"),
        (1337, "final_3c_s1337_20260528_202023"),
    ],
    "7class": [
        (42,   "final_7c_s42_20260528_204414"),
        (7777, "final_20260526_074308"),
        (9999, "final_20260526_074410"),
        (2024, "final_20260527_060332"),
        (1337, "final_20260527_141710"),
    ],
}


def group_to_subclasses() -> dict[str, list[str]]:
    """Invert cfg.COLLAPSE_MAP over CALL_TYPES_7 -> {group: [subclasses...]}."""
    g2s: dict[str, list[str]] = {}
    for sub in cfg.CALL_TYPES_7:
        grp = cfg.COLLAPSE_MAP.get(sub, sub)
        g2s.setdefault(grp, []).append(sub)
    return g2s


# ======================================================================
# Inference (streaming train audio; SAMPLE_RATE=250 so reads are tiny and
# the OS page-cache keeps files hot across the 10 model passes)
# ======================================================================

class SegAudio(Dataset):
    def __init__(self, segments):
        self.segs = segments

    def __len__(self):
        return len(self.segs)

    def __getitem__(self, i):
        s = self.segs[i]
        audio, sr = sf.read(
            s.path, start=s.start_sample, stop=s.end_sample, dtype="float32")
        assert sr == cfg.SAMPLE_RATE, f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}"
        return (
            torch.from_numpy(audio),
            {"dataset": s.dataset, "filename": s.filename,
             "start_sample": s.start_sample, "end_sample": s.end_sample},
        )


def collate_infer(batch):
    audios, metas = zip(*batch)
    max_samp = max(a.size(0) for a in audios)
    out = torch.zeros(len(audios), max_samp)
    for i, a in enumerate(audios):
        out[i, : a.size(0)] = a
    return out, list(metas)


def build_model(num_classes: int, device):
    """Fresh model with USE_3CLASS set from the class count; lazy proj materialised."""
    cfg.USE_3CLASS = (num_classes == 3)
    model = WhaleVAD(num_classes=num_classes).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def detect_num_classes(ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt["model_state_dict"]["classifier.weight"].shape[0])


@torch.no_grad()
def infer(model, spec, loader, device) -> dict:
    """{(dataset, filename, start_sample): (n_frames, n_classes)} probabilities."""
    model.eval()
    hop = spec.hop_length
    probs = {}
    for audio, metas in tqdm(loader, desc="  infer", leave=False):
        audio = audio.to(device, non_blocking=True)
        p = torch.sigmoid(model(spec(audio))).cpu().numpy()
        for j, meta in enumerate(metas):
            n_samp = meta["end_sample"] - meta["start_sample"]
            nf = min(n_samp // hop, p[j].shape[0])
            probs[(meta["dataset"], meta["filename"], meta["start_sample"])] = p[j, :nf, :]
    return probs


def load_weights(model, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)


# ======================================================================
# Mining (mirrors mine_hard_negatives*.py exactly, on the _final stack)
# ======================================================================

def target_thresholds(n_classes: int, target_idx: int) -> np.ndarray:
    thr = np.full(n_classes, 0.999, dtype=np.float64)
    thr[target_idx] = PROPOSAL_THRESHOLD
    return thr


def merge_and_filter_7class(detections: list[Detection]) -> list[Detection]:
    """postprocess_final.merge_and_filter WITHOUT the COLLAPSE_MAP rewrite, so
    subclass labels survive (otherwise bma/bmb/bmz -> bmabz erases the target)."""
    groups: dict[tuple, list[Detection]] = {}
    for d in detections:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)
    final = []
    for _, events in groups.items():
        events.sort(key=lambda x: x.start_s)
        merged: list[Detection] = []
        for e in events:
            if not merged:
                merged.append(e)
            else:
                last = merged[-1]
                if e.start_s - last.end_s <= cfg.MERGE_GAP_S:
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(e)
        for m in merged:
            if cfg.POST_MIN_DUR_S <= (m.end_s - m.start_s) <= cfg.POST_MAX_DUR_S:
                final.append(m)
    return final


def build_gt_for_target(annotations, file_start_dts, target: str, is_3class: bool):
    field = "label_3class" if is_3class else "annotation"
    gt = []
    for _, row in annotations.iterrows():
        if row[field] != target:
            continue
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=target,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))
    return gt


def predictions_to_fps(predictions, gt, target: str, max_iou_for_fp: float):
    gt_by_file: dict[tuple, list[Detection]] = {}
    for g in gt:
        gt_by_file.setdefault((g.dataset, g.filename), []).append(g)
    fps = []
    for p in predictions:
        if p.label != target:
            continue
        cands = gt_by_file.get((p.dataset, p.filename), [])
        max_iou = max(
            (pp.compute_iou_1d(p.start_s, p.end_s, g.start_s, g.end_s) for g in cands),
            default=0.0,
        )
        if max_iou < max_iou_for_fp:
            fps.append(p)
    return fps


def mine_from_fileprobs(file_probs, target, is_3class, annotations, file_start_dts):
    """file_probs is the already-stitched {(ds,fn): (frames, nc)} stream.
    Returns (fps_topk, n_proposals, n_gt)."""
    cfg.USE_3CLASS = is_3class
    names = cfg.class_names()
    thr = target_thresholds(len(names), names.index(target))

    raw = []
    for (ds, fn), probs in file_probs.items():
        raw.extend(pp.threshold_to_detections(pp.smooth_probabilities(probs), thr, ds, fn))
    pred = pp.merge_and_filter(raw) if is_3class else merge_and_filter_7class(raw)

    gt = build_gt_for_target(annotations, file_start_dts, target, is_3class)
    fps = predictions_to_fps(pred, gt, target, MAX_IOU_FOR_FP)
    fps.sort(key=lambda d: d.confidence, reverse=True)
    if TOP_K > 0 and len(fps) > TOP_K:
        fps = fps[:TOP_K]
    return fps, len(pred), len(gt)


def write_json(out_path: Path, fps, target, checkpoints, datasets):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoints": [str(c) for c in checkpoints],
        "weights": None,
        "target_class": target,
        "threshold": PROPOSAL_THRESHOLD,
        "max_iou_for_fp": MAX_IOU_FOR_FP,
        "datasets": list(datasets),
        "n_fps": len(fps),
        "fps": [
            {"dataset": d.dataset, "filename": d.filename,
             "start_s": float(d.start_s), "end_s": float(d.end_s),
             "confidence": float(d.confidence), "target_class": target}
            for d in fps
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def add_into(running: dict, file_probs: dict):
    """Element-wise accumulate a stitched file-prob stream into a running sum."""
    for k, v in file_probs.items():
        running[k] = v.astype(np.float64) if k not in running else running[k] + v


# ======================================================================
# Driver
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--ckpt-name", default="paper_best.pt")
    p.add_argument("--out-dir", default="runs/hardnegs_final")
    p.add_argument("--heads", nargs="+", choices=["3class", "7class"],
                   default=["3class", "7class"])
    p.add_argument("--targets", nargs="+", choices=cfg.CALL_TYPES_3,
                   default=list(cfg.CALL_TYPES_3),
                   help="3-class group targets. 7-class head mines each group's "
                        "subclasses (bmabz->bma,bmb,bmz; d->bmd,bpd; bp->bp20,bp20plus).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Mining datasets (default: cfg.TRAIN_DATASETS).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (sources x targets -> output paths) and exit.")
    return p.parse_args()


def seed_all(seed):
    if wbu is not None and hasattr(wbu, "seed_everything"):
        wbu.seed_everything(seed, deterministic=False)
    else:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def resolve_targets(head, group_targets, g2s):
    """3-class head -> the group names; 7-class head -> their subclasses."""
    if head == "3class":
        return list(group_targets)
    out = []
    for g in group_targets:
        out.extend(g2s[g])
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    datasets = args.datasets or list(cfg.TRAIN_DATASETS)
    g2s = group_to_subclasses()

    # ---- plan + source existence check ------------------------------
    plan = []  # (head, label, [ (seed,dir) ... ], targets, is_ensemble)
    missing = []
    for head in args.heads:
        srcs = SOURCES[head]
        for seed, d in srcs:
            ck = Path(args.runs_root) / d / args.ckpt_name
            if not ck.exists():
                missing.append(str(ck))
        tgts = resolve_targets(head, args.targets, g2s)
        for seed, d in srcs:
            plan.append((head, f"seed{seed}", [(seed, d)], tgts, False))
        plan.append((head, "ensemble", list(srcs), tgts, True))

    n_jsons = sum(len(t) for *_, t, _ in plan)
    print(f"Heads: {args.heads} | group targets: {args.targets}")
    print(f"Mining datasets: {datasets}")
    print(f"Will write {n_jsons} JSON(s) under {out_dir}/")
    for head, label, srcs, tgts, is_ens in plan:
        tag = "ENSEMBLE(5)" if is_ens else srcs[0][1]
        for t in tgts:
            print(f"    {out_dir}/{head}/{label}/{t}.json   <- {tag}")

    if missing:
        print("\nERROR: missing checkpoints (run the rescore / create symlinks first):")
        for m in missing:
            print(f"    {m}")
        raise SystemExit(1)
    if args.dry_run:
        print("\n--dry-run: nothing inferred.")
        return

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- data (loaded once, shared by every model pass) -------------
    manifest = get_file_manifest(datasets)
    annotations = load_annotations(datasets, manifest=manifest)
    segments = build_val_segments(manifest, annotations)
    print(f"{len(manifest)} files, {len(annotations)} annotations, "
          f"{len(segments)} tiles to score")
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in manifest.iterrows()}
    loader = DataLoader(
        SegAudio(segments), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_infer, pin_memory=True)

    # ---- wandb (one run for the whole job; defensive) ---------------
    run = None
    if not args.no_wandb and wbu is not None:
        try:
            run = wbu.init_phase(
                "6", extra_tags=["mining", "hardneg_final", "all_targets"],
                job_type="mining",
                config={"proposal_threshold": PROPOSAL_THRESHOLD,
                        "max_iou_for_fp": MAX_IOU_FOR_FP, "top_k": TOP_K,
                        "heads": args.heads, "group_targets": args.targets,
                        "datasets": datasets, "out_dir": str(out_dir),
                        "n_jsons": n_jsons, "seed": args.seed})
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without logging.")
            run = None

    # ---- mine, head by head -----------------------------------------
    summary: dict[str, int] = {}
    gt_counts: dict[str, int] = {}
    t_start = time.time()

    for head in args.heads:
        is_3class = (head == "3class")
        n_cls = 3 if is_3class else 7
        srcs = SOURCES[head]
        tgts = resolve_targets(head, args.targets, g2s)
        ens_sum: dict = {}
        ckpts = []
        print(f"\n{'='*66}\nHEAD: {head}  ({len(srcs)} models, targets={tgts})\n{'='*66}")

        for seed, d in srcs:
            ck = Path(args.runs_root) / d / args.ckpt_name
            got = detect_num_classes(ck)
            assert got == n_cls, f"{ck} has {got}-class head, expected {n_cls} for {head}"
            model, spec = build_model(n_cls, device)
            load_weights(model, ck, device)
            t0 = time.time()
            file_probs = pp.stitch_segments(infer(model, spec, loader, device))
            print(f"  [seed {seed}] {d}  inference+stitch {time.time()-t0:.0f}s")

            for t in tgts:
                fps, n_prop, n_gt = mine_from_fileprobs(
                    file_probs, t, is_3class, annotations, file_start_dts)
                write_json(out_dir / head / f"seed{seed}" / f"{t}.json",
                           fps, t, [ck], datasets)
                summary[f"{head}/s{seed}/{t}"] = len(fps)
                gt_counts[f"{head}/{t}"] = n_gt
                print(f"      {t:<9} proposals={n_prop:<6} gt={n_gt:<6} "
                      f"FPs_kept={len(fps)}")

            add_into(ens_sum, file_probs)
            ckpts.append(ck)
            del model, file_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ensemble = average of the stitched file-prob streams
        ens = {k: (v / len(srcs)).astype(np.float32) for k, v in ens_sum.items()}
        del ens_sum
        print(f"  [ensemble of {len(srcs)}]")
        for t in tgts:
            fps, n_prop, n_gt = mine_from_fileprobs(
                ens, t, is_3class, annotations, file_start_dts)
            write_json(out_dir / head / "ensemble" / f"{t}.json",
                       fps, t, ckpts, datasets)
            summary[f"{head}/ensemble/{t}"] = len(fps)
            print(f"      {t:<9} proposals={n_prop:<6} gt={n_gt:<6} FPs_kept={len(fps)}")
        del ens

    # ---- report -----------------------------------------------------
    print(f"\n{'='*66}\nDONE in {(time.time()-t_start)/60:.1f} min — "
          f"{len(summary)} JSON(s) under {out_dir}/\n{'='*66}")
    for k in sorted(summary):
        print(f"  {k:<32} {summary[k]:>6} FPs")

    # ---- wandb finalize (standard Run API only; defensive) ----------
    if run is not None:
        try:
            for k, v in summary.items():
                run.summary[f"n_fps/{k}"] = v
            for k, v in gt_counts.items():
                run.summary[f"n_gt/{k}"] = v
            run.summary["n_jsons"] = len(summary)
            run.summary["total_fps"] = int(sum(summary.values()))
            run.summary["verdict"] = (
                f"Mined {len(summary)} hard-neg JSONs (heads={args.heads}, "
                f"targets={args.targets}) at threshold {PROPOSAL_THRESHOLD}, "
                f"max_iou_for_fp={MAX_IOU_FOR_FP}, top_k={TOP_K}. "
                f"Files under {out_dir}/.")
            run.finish()
        except Exception as e:
            print(f"[wandb] finalize failed ({e}).")


if __name__ == "__main__":
    main()
