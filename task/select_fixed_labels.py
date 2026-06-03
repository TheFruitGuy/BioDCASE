"""
Select fixed pseudo-labels for FlatMatch (Fix label) — ENSEMBLE-teacher variant
================================================================================

Algorithm 2 of Huang et al. (NeurIPS 2023) stabilises cross-sharpness by
pseudo-labeling the most-confident unlabeled clips with FIXED hard labels and
adding them to the data that computes the worst-case perturbation eps*. The
paper SSL-pretrains to build confidence; here we instead pseudo-label with the
strongest model we have.

This variant pseudo-labels with an ENSEMBLE of checkpoints (averaged sigmoid
posteriors) rather than a single base model. The intended teacher is the full
10-model MT ensemble (5 seeds x {solo, ens}), whose D-class F1 (~0.317) far
exceeds any base model (~0.155) — so its high-confidence D detections are much
cleaner pseudo-labels, and you can afford a larger D quota. Because the teacher
is the ensemble (not a per-seed base), the resulting fixed set is SEED-AGNOSTIC:
select once, reuse for every stack seed.

NOTE this turns the experiment into ensemble distillation — the stack is taught
by the MT ensemble. A *positive* result is then confounded with "you distilled
MT"; a *negative* one is clean and strong ("even handed the MT ensemble's own D
detections as labels, FlatMatch couldn't surpass MT"). Distillation also caps
the ceiling at ~the teacher, so expect the gap to shrink, not flip.

This script:
  1. loads N checkpoints (each ckpt["model_state_dict"] is a plain WhaleVAD;
     for MT runs that is the EMA teacher — EMATeacher.state_dict() returns the
     inner model, so it loads exactly like a base model),
  2. scores a random sample of AADC clips by the ENSEMBLE's per-class peak
     detection confidence (max over frames of the mean sigmoid prob),
  3. selects the top --n-fix-per-class clips for EACH class (union, deduped),
  4. freezes a hard per-frame multi-label pseudo-target via --thresholds
     (the MT-ensemble's tuned thresholds, or 0.5 if omitted),
  5. caches audio + targets + masks to an .npz that train_flatmatch_final.py
     loads via --fixed-labels.

On thresholds: pseudo-labels want PRECISION, so a uniform 0.5 (the default) is
usually cleaner than F1-tuned thresholds, which are recall-favouring for some
classes. Passing the MT-ensemble's tuned thresholds (from the MT_ENSEMBLE row
of aggregate_numbers' CSV) matches the eval operating point but, for D (~0.6),
can zero out borderline D clips. Check the per-class summary either way.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 python select_fixed_labels.py \\
        --checkpoints runs/mtfb_3class_s*/best_model.pt \\
        --aadc-root /home/matthias-nagl/BioDCASE/task/data_pretrain/audio \\
        --aadc-sites Casey2018 DDU2018 Kerguelen2018 Kerguelen2021 Kerguelen2023 \\
        --sample-clips 20000 --n-fix-per-class 600 \\
        --out runs/fixed_labels/fixed_3c_mtens.npz
        # optional: --thresholds 0.30 0.60 0.35   (MT-ensemble tuned thresholds)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from rescore_base_epochs import build_model
from ssl_dataset_final import build_pretrain_manifest, SSLClipDataset, collate_ssl


def quarantine_check(aadc_sites):
    overlap = set(aadc_sites) & set(cfg.VAL_DATASETS)
    if overlap:
        raise SystemExit(f"AADC sites overlap VAL sites {sorted(overlap)} — refusing.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoints to ensemble as the teacher (averaged "
                        "sigmoid posteriors). Pass the full MT ensemble, e.g. "
                        "runs/mtfb_3class_s*/best_model.pt (shell-expanded).")
    p.add_argument("--aadc-root", required=True)
    p.add_argument("--aadc-sites", nargs="+", required=True)
    p.add_argument("--sample-clips", type=int, default=20000,
                   help="Random AADC clips to score before selecting (bounds cost). "
                        "Larger -> more candidate confident D clips for the teacher.")
    p.add_argument("--n-fix-per-class", type=int, default=600,
                   help="Top-confidence clips kept PER class (union -> <= n_classes*this). "
                        "A strong ensemble teacher supports a larger D quota.")
    p.add_argument("--thresholds", nargs="+", type=float, default=None,
                   help="Per-class thresholds for binarising pseudo-labels (e.g. the "
                        "MT-ensemble's tuned thresholds). Omit -> 0.5 (precision-favouring).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output .npz path.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    quarantine_check(args.aadc_sites)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load the ensemble teacher (each model_state_dict is a plain WhaleVAD) ----
    models = []
    spec = None
    n_classes = None
    for cp in args.checkpoints:
        ck = torch.load(cp, map_location=device, weights_only=False)
        nc = int(ck["model_state_dict"]["classifier.weight"].shape[0])
        if n_classes is None:
            n_classes = nc
        elif nc != n_classes:
            raise SystemExit(f"Checkpoint {cp} is {nc}-class but the first is "
                             f"{n_classes}-class — all teacher members must match.")
        m, sp = build_model(n_classes, device)
        if spec is None:
            spec = sp                                  # identical extractor; build once
        m.load_state_dict(ck["model_state_dict"], strict=False)
        m.eval()
        models.append(m)
    is_3class = (n_classes == 3)
    cfg.USE_3CLASS = is_3class
    head = "3class" if is_3class else "7class"
    classes = cfg.CALL_TYPES_3 if is_3class else cfg.CALL_TYPES_7
    print(f"Device: {device} | teacher: {len(models)}-model ensemble | head: "
          f"{head} ({n_classes}-class)")

    # ---- binarisation thresholds ------------------------------------
    if args.thresholds is not None:
        thr_np = np.asarray(args.thresholds, dtype=np.float32).reshape(-1)
        if thr_np.shape[0] != n_classes:
            raise SystemExit(f"--thresholds needs {n_classes} values, got {thr_np.shape[0]}.")
        print(f"  binarising pseudo-labels with provided thresholds: "
              f"{np.array2string(thr_np, precision=3)}")
    else:
        thr_np = np.full(n_classes, 0.5, dtype=np.float32)
        print("  no --thresholds given -> binarising at 0.5 (precision-favouring default)")

    manifest = build_pretrain_manifest(train_datasets=None,
                                       aadc_sites=list(args.aadc_sites),
                                       aadc_root=args.aadc_root)
    ds = SSLClipDataset(manifest, clip_seconds=cfg.TRAIN_SEGMENT_S,
                        sample_rate=cfg.SAMPLE_RATE, epoch_clips=args.sample_clips)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_ssl,
                        pin_memory=True)

    # ---- scan: store audio + ENSEMBLE-MEAN per-frame probs + peak conf ----
    audios: list[np.ndarray] = []
    probs_full: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    sites: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"scoring AADC ({len(models)}-model ens)"):
            audio = batch["audio"].to(device, non_blocking=True)
            s = spec(audio)                            # shared across members
            p = None
            for m in models:
                pi = torch.sigmoid(m(s))               # (B, T, C)
                p = pi if p is None else p + pi
            p = p / len(models)                        # mean posterior over the ensemble
            peak = p.amax(dim=1)                       # (B, C)
            audio_np = audio.cpu().numpy().astype(np.float32)
            p_np = p.cpu().numpy().astype(np.float32)
            peak_np = peak.cpu().numpy().astype(np.float32)
            for b in range(audio.size(0)):
                audios.append(audio_np[b]); probs_full.append(p_np[b])
                peaks.append(peak_np[b]); sites.append(batch["sites"][b])
    peak_arr = np.stack(peaks)                          # (N, C)
    N = peak_arr.shape[0]
    print(f"  scored {N} clips with the {len(models)}-model ensemble "
          f"in {(time.time()-t0)/60:.1f} min")

    # ---- per-class top-k selection (union, deduped) ------------------
    selected, sel_for = set(), {}
    for c in range(n_classes):
        order = np.argsort(-peak_arr[:, c])[:args.n_fix_per_class]
        for i in order:
            selected.add(int(i))
            sel_for.setdefault(int(i), []).append(classes[c])
    selected = sorted(selected)

    # ---- freeze hard pseudo-labels on the selected clips -------------
    out_audio, out_tgt, out_mask, out_site, out_sel, out_peak = [], [], [], [], [], []
    for i in selected:
        pf = probs_full[i]                              # (T, C)
        tgt = (pf >= thr_np[None, :]).astype(np.float32)  # (T, C) hard multi-label
        out_audio.append(audios[i])
        out_tgt.append(tgt)
        out_mask.append(np.ones(pf.shape[0], dtype=np.uint8))
        out_site.append(sites[i])
        out_sel.append("|".join(sel_for[i]))
        out_peak.append(peak_arr[i])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        audios=np.stack(out_audio),                     # (M, n_samples) f32
        targets=np.stack(out_tgt),                      # (M, T, C) f32 {0,1}
        masks=np.stack(out_mask),                       # (M, T) uint8
        peak=np.stack(out_peak),                        # (M, C) f32
        selected_for=np.array(out_sel),                 # (M,) str
        site=np.array(out_site),                        # (M,) str
        n_classes=np.int64(n_classes),
        sample_rate=np.int64(cfg.SAMPLE_RATE),
        thresholds=thr_np,
        teacher_checkpoints=np.array([str(c) for c in args.checkpoints]),
        n_teacher_members=np.int64(len(models)),
    )

    # ---- summary: confidence per class + frame coverage --------------
    M = len(selected)
    print(f"\n{'='*68}\nFixed set: {M} unique clips "
          f"(<= {n_classes} x {args.n_fix_per_class} = {n_classes*args.n_fix_per_class}) "
          f"| teacher = {len(models)}-model ensemble\n{'='*68}")
    tgt_arr = np.stack(out_tgt)                          # (M, T, C)
    for c in range(n_classes):
        topk = np.sort(peak_arr[:, c])[::-1][:args.n_fix_per_class]
        frac_pos = float((tgt_arr[:, :, c].sum(axis=1) > 0).mean())
        print(f"  {classes[c].upper():6}  selected peak-conf "
              f"min={topk.min():.3f} mean={topk.mean():.3f} max={topk.max():.3f}  | "
              f"{frac_pos*100:4.1f}% of fixed clips carry a {classes[c].upper()} frame")
    print(f"\nSaved -> {args.out}")
    print("Check D's min peak-conf: the MT-ensemble teacher should push it well above "
          "the base model's (~0.53) — if D is now confidently high, the fixed set is the "
          "richer/cleaner D signal you were after.")


if __name__ == "__main__":
    main()
