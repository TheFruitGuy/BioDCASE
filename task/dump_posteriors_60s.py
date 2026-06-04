"""
dump_posteriors_60s.py — save longer-tile _final val posteriors per member
==========================================================================

``best_posteriors.npz`` in each run dir was produced by the _final training at
``cfg.EVAL_SEGMENT_S`` (30 s) tiling. To evaluate the ensemble at 60 s without
mixing in the old (config/dataset/postprocess) stack, this script re-runs each
member's *_final* validation inference at a longer tile length and writes the
result in the **same** ``{"ds|fn": (T,3)}`` float16 format, to a new filename
(default ``best_posteriors_60s.npz``).

It stays entirely in the _final stack: it builds the model exactly as the _final
trainers do (``build_model`` from ``rescore_base_epochs``, load the checkpoint's
``model_state_dict`` with ``strict=False``), runs the same forward pass as
``train_hnm_final.validate`` (``sigmoid(model(spec(audio)))``, segments keyed
``(ds, fn, start_sample)``), collapses to 3-class with the trainer's own
``collapse_to_3class``, and stitches + saves with the trainer's own
``save_posteriors`` (``pp.stitch_segments`` -> ``"ds|fn"`` keys). The *only*
difference from the saved 30 s posteriors is ``--segment-s``.

After running this, point the LOSO / search scripts at the new file::

    python loso_eval.py      --checkpoints ... --weights-... --posterior-name best_posteriors_60s.npz
    python postproc_search.py --checkpoints ... --weights-... --posterior-name best_posteriors_60s.npz

Usage
-----
::

    # pick a NON-Blackwell GPU (the sm_120 cards crash the cu118 build at the LSTM)
    CUDA_VISIBLE_DEVICES=3 python dump_posteriors_60s.py \
        --checkpoints runs/hnm_3class_s{42,2024,7777,9999,1337}_ens_nopgi/best_model.pt \
                      runs/mtfb_3class_s{42,2024,7777,9999,1337}_ens_asymmetric_mse/best_model.pt \
        --segment-s 60

The 60 s val DataLoader is built once and shared across all members. Each
member's per-model 60 s all-val paper-F1 is printed as a free sanity check
(it should be in the same ballpark as its 30 s number).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="member best_model.pt paths (HNM and/or MT; any 3- or 7-class)")
    p.add_argument("--segment-s", type=float, default=60.0,
                   help="evaluation tile length in seconds (saved posteriors use 30)")
    p.add_argument("--overlap-s", type=float, default=None,
                   help="tile overlap in seconds (default cfg.EVAL_OVERLAP_S, = the 30 s setting)")
    p.add_argument("--out-name", default=None,
                   help="output npz filename per run dir (default best_posteriors_<seg>s.npz)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="inference batch size (60 s tiles are ~2x the memory of 30 s; lower if OOM)")
    p.add_argument("--overwrite", action="store_true", help="recompute even if the output exists")
    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import config_final as cfg
    from dataset_final import (
        load_annotations, get_file_manifest, build_val_segments, WhaleDataset, collate_fn,
    )
    from rescore_base_epochs import build_model, build_gt_events
    # reuse the trainer's exact collapse + stitch/save so output == best_posteriors.npz format
    from train_hnm_final import collapse_to_3class, save_posteriors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overlap_s = args.overlap_s if args.overlap_s is not None else cfg.EVAL_OVERLAP_S
    out_name = args.out_name or f"best_posteriors_{int(args.segment_s)}s.npz"
    print(f"Device: {device}   segment {args.segment_s:.0f}s   overlap {overlap_s:.1f}s   -> {out_name}")

    # ---- ground truth (only needed for the per-model sanity paper-F1) --------
    cfg.USE_3CLASS = True
    anns = load_annotations(list(cfg.VAL_DATASETS))
    manifest = get_file_manifest(list(cfg.VAL_DATASETS))
    fsd = {(r["dataset"], r["filename"]): r["start_dt"] for _, r in manifest.iterrows()}
    gt_events = build_gt_events(anns, fsd)

    # ---- 60 s val loader, built ONCE and shared across members ---------------
    val_segs = build_val_segments(manifest, anns, segment_s=args.segment_s, overlap_s=overlap_s)
    loader = DataLoader(WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    print(f"{len(val_segs)} tiles of {args.segment_s:.0f}s ({overlap_s:.1f}s overlap)\n")

    # optional: tuned per-model paper-F1 (sanity only); skip if rescore lacks it
    try:
        from rescore_base_epochs import tune_thresholds, evaluate, paper_f1
        have_scorer = True
    except Exception:
        have_scorer = False

    for ck in args.checkpoints:
        rd = Path(ck).parent
        out = rd / out_name
        if out.exists() and not args.overwrite:
            print(f"  [skip] {rd.name}: {out_name} exists (use --overwrite)")
            continue

        ckpt = torch.load(ck, map_location=device, weights_only=False)
        n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
        is_3class = (n_classes == 3)
        model, spec = build_model(n_classes, device)            # _final model + spectrogram
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        hop = spec.hop_length

        probs = {}
        with torch.no_grad():
            for audio, targets, mask, metas in tqdm(loader, desc=f"infer {rd.name}", leave=False):
                audio = audio.to(device, non_blocking=True)
                logits = model(spec(audio))
                p = torch.sigmoid(logits).cpu().numpy()
                for j, m in enumerate(metas):
                    nf = min((m["end_sample"] - m["start_sample"]) // hop, p[j].shape[0])
                    probs[(m["dataset"], m["filename"], m["start_sample"])] = p[j, :nf, :]

        cfg.USE_3CLASS = is_3class
        probs3 = collapse_to_3class(probs, n_classes)            # flips flag; ends 3-class
        save_posteriors(probs3, out)                             # stitch -> "ds|fn" -> npz

        tag = ""
        if have_scorer:
            cfg.USE_3CLASS = True
            thr = tune_thresholds(probs3, gt_events, 1)
            tag = f"  all-val paper-F1={paper_f1(evaluate(probs3, gt_events, thr)):.4f}"
        n_files = len({(ds, fn) for (ds, fn, _) in probs3})
        print(f"  [done] {rd.name:42s} {n_files} files -> {out_name}{tag}")

        del model, probs, probs3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nWrote {out_name} into each run dir. Now run e.g.:")
    print(f"  python loso_eval.py --checkpoints <same> --weights-... --posterior-name {out_name}")


if __name__ == "__main__":
    main()
