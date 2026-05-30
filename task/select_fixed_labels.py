"""
Select fixed pseudo-labels for FlatMatch (Fix label) — from-base adaptation
===========================================================================

Algorithm 2 of Huang et al. (NeurIPS 2023) stabilises cross-sharpness in the
scarce-label regime by pseudo-labeling the most-confident unlabeled clips with
FIXED hard labels and adding them to the data that computes the worst-case
perturbation eps*. The paper first SSL-pretrains 16 epochs to build confidence;
here we warm-start from a converged base model, so the base IS that confident
model and the pretrain phase is skipped.

This script:
  1. loads a 3c/7c base checkpoint (the same paper_best.pt the FlatMatch trainer
     warm-starts from) and its tuned per-class thresholds,
  2. scores a random sample of AADC unlabeled clips by PER-CLASS peak detection
     confidence (max over frames of the sigmoid prob for that class),
  3. selects the top --n-fix-per-class clips for EACH class (union, deduped) so
     every class — including the scarce D — is represented in the fixed set,
  4. freezes a hard per-frame multi-label pseudo-target on each selected clip
     by thresholding the base model's own prediction (the "fixed label"),
  5. caches audio + targets + masks to an .npz that train_flatmatch_final.py
     loads via --fixed-labels.

The pseudo-labels are only as good as the base model's confidence on the
(cross-domain) AADC clips; high-confidence false positives would inject noise
into eps*, which is exactly why #fix is kept small. Inspect the per-class
confidence summary this prints before trusting the set.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 python select_fixed_labels.py \\
        --checkpoint runs/final_3c_s42_20260527_200054/paper_best.pt \\
        --aadc-root /home/matthias-nagl/BioDCASE/task/data_pretrain/audio \\
        --aadc-sites Casey2018 DDU2018 DDU2019 Kerguelen2018 Kerguelen2019 \\
        --n-fix-per-class 200 --sample-clips 5000 \\
        --out runs/fixed_labels/fixed_3c_s42.npz
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
    p.add_argument("--checkpoint", required=True,
                   help="Base paper_best.pt to pseudo-label from (its model_state_dict).")
    p.add_argument("--aadc-root", required=True)
    p.add_argument("--aadc-sites", nargs="+", required=True)
    p.add_argument("--sample-clips", type=int, default=5000,
                   help="Random AADC clips to score before selecting (bounds cost).")
    p.add_argument("--n-fix-per-class", type=int, default=200,
                   help="Top-confidence clips kept PER class (union -> <= n_classes*this).")
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
    is_3class = (n_classes == 3)
    cfg.USE_3CLASS = is_3class
    head = "3class" if is_3class else "7class"
    print(f"Device: {device} | base checkpoint head: {head} ({n_classes}-class)")

    model, spec = build_model(n_classes, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # tuned per-class thresholds (fall back to 0.5 if the base didn't store them)
    thr = ckpt.get("thresholds")
    if thr is None:
        thr_np = np.full(n_classes, 0.5, dtype=np.float32)
        print("  no thresholds in checkpoint -> binarising pseudo-labels at 0.5")
    else:
        thr_np = (thr.detach().cpu().numpy() if torch.is_tensor(thr)
                  else np.asarray(thr, dtype=np.float32)).astype(np.float32).reshape(-1)
        print(f"  pseudo-label thresholds (per class): "
              f"{np.array2string(thr_np, precision=3)}")

    manifest = build_pretrain_manifest(train_datasets=None,
                                       aadc_sites=list(args.aadc_sites),
                                       aadc_root=args.aadc_root)
    ds = SSLClipDataset(manifest, clip_seconds=cfg.TRAIN_SEGMENT_S,
                        sample_rate=cfg.SAMPLE_RATE, epoch_clips=args.sample_clips)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_ssl,
                        pin_memory=True)

    # --- scan: store audio + full per-frame probs + per-class peak conf ----
    audios: list[np.ndarray] = []
    probs_full: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    sites: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring AADC"):
            audio = batch["audio"].to(device, non_blocking=True)
            p = torch.sigmoid(model(spec(audio)))           # (B, T, C)
            peak = p.amax(dim=1)                            # (B, C)
            audio_np = audio.cpu().numpy().astype(np.float32)
            p_np = p.cpu().numpy().astype(np.float32)
            peak_np = peak.cpu().numpy().astype(np.float32)
            for b in range(audio.size(0)):
                audios.append(audio_np[b]); probs_full.append(p_np[b])
                peaks.append(peak_np[b]); sites.append(batch["sites"][b])
    peak_arr = np.stack(peaks)                              # (N, C)
    N = peak_arr.shape[0]
    print(f"  scored {N} clips in {(time.time()-t0)/60:.1f} min")

    # --- per-class top-k selection (union, deduped) ------------------------
    classes = cfg.CALL_TYPES_3 if is_3class else cfg.CALL_TYPES_7
    selected, sel_for = set(), {}
    for c in range(n_classes):
        order = np.argsort(-peak_arr[:, c])[:args.n_fix_per_class]
        for i in order:
            selected.add(int(i))
            sel_for.setdefault(int(i), []).append(classes[c])
    selected = sorted(selected)

    # --- freeze hard pseudo-labels on the selected clips -------------------
    out_audio, out_tgt, out_mask, out_site, out_sel, out_peak = [], [], [], [], [], []
    for i in selected:
        pf = probs_full[i]                                  # (T, C)
        tgt = (pf >= thr_np[None, :]).astype(np.float32)    # (T, C) hard multi-label
        out_audio.append(audios[i])
        out_tgt.append(tgt)
        out_mask.append(np.ones(pf.shape[0], dtype=np.uint8))
        out_site.append(sites[i])
        out_sel.append("|".join(sel_for[i]))
        out_peak.append(peak_arr[i])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        audios=np.stack(out_audio),                         # (M, n_samples) f32
        targets=np.stack(out_tgt),                          # (M, T, C) f32 {0,1}
        masks=np.stack(out_mask),                           # (M, T) uint8
        peak=np.stack(out_peak),                            # (M, C) f32
        selected_for=np.array(out_sel),                     # (M,) str
        site=np.array(out_site),                            # (M,) str
        n_classes=np.int64(n_classes),
        sample_rate=np.int64(cfg.SAMPLE_RATE),
        thresholds=thr_np,
        source_checkpoint=np.array(str(args.checkpoint)),
    )

    # --- summary: confidence per class + frame coverage --------------------
    M = len(selected)
    print(f"\n{'='*64}\nFixed set: {M} unique clips "
          f"(<= {n_classes} x {args.n_fix_per_class} = {n_classes*args.n_fix_per_class})\n{'='*64}")
    tgt_arr = np.stack(out_tgt)                              # (M, T, C)
    for c in range(n_classes):
        topk = np.sort(peak_arr[:, c])[::-1][:args.n_fix_per_class]
        frac_pos = float((tgt_arr[:, :, c].sum(axis=1) > 0).mean())   # clips with any c frame
        print(f"  {classes[c].upper():6}  selected peak-conf "
              f"min={topk.min():.3f} mean={topk.mean():.3f} max={topk.max():.3f}  | "
              f"{frac_pos*100:4.1f}% of fixed clips carry a {classes[c].upper()} frame")
    print(f"\nSaved -> {args.out}")
    print("Inspect the min peak-conf per class: if it's low (e.g. < ~0.5 for D), "
          "those pseudo-positives are weak and likely to inject noise into eps*.")


if __name__ == "__main__":
    main()
