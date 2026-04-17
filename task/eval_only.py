"""
Quick script to evaluate a trained checkpoint on validation set
with tuned thresholds. Prints the paper-comparable F1.

    python eval_only.py --checkpoint runs/whalevad_XXXX/final_model.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss
from dataset import build_dataloaders, load_annotations, get_file_manifest
from train import validate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build val loader
    _, _, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest    = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    thresholds = ckpt["thresholds"]
    if torch.is_tensor(thresholds):
        thresholds = thresholds.cpu()
    thresholds_t = torch.tensor(thresholds, device=device, dtype=torch.float32)
    print(f"Using thresholds: {thresholds_t.tolist()}")

    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    # Init lazy layer
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = WhaleVADLoss().to(device)

    # Overall
    print("\n=== OVERALL (all validation sets combined) ===")
    val = validate(
        model, spec_extractor, val_loader, criterion, device,
        thresholds_t, val_annotations, file_start_dts,
    )
    print(f"Final Overall F1 (paper-comparable): {val['mean_f1']:.3f}")

    # Per-dataset breakdown (paper Table 4)
    print("\n=== PER-DATASET BREAKDOWN (paper Table 4 style) ===")
    from postprocess import (
        postprocess_predictions, compute_metrics, Detection,
    )
    from tqdm import tqdm

    all_probs = {}
    hop = spec_extractor.hop_length
    with torch.no_grad():
        for audio, _, _, metas in tqdm(val_loader, desc="Inference"):
            audio = audio.to(device)
            logits = model(spec_extractor(audio))
            probs = torch.sigmoid(logits).cpu().numpy()
            for j, meta in enumerate(metas):
                key = (meta["dataset"], meta["filename"], meta["start_sample"])
                n_samp = meta["end_sample"] - meta["start_sample"]
                n_frames = min(n_samp // hop, probs[j].shape[0])
                all_probs[key] = probs[j, :n_frames, :]

    pred_events = postprocess_predictions(all_probs, thresholds_t.cpu().numpy())
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        label = row["label_3class"] if cfg.USE_3CLASS else row["annotation"]
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"], label=label,
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"]   - fsd).total_seconds(),
        ))

    for ds_name in cfg.VAL_DATASETS:
        ds_preds = [d for d in pred_events if d.dataset == ds_name]
        ds_gts   = [d for d in gt_events   if d.dataset == ds_name]
        m = compute_metrics(ds_preds, ds_gts, iou_threshold=0.3)
        print(f"\n  {ds_name}:")
        for cls in cfg.class_names():
            if cls in m:
                r = m[cls]
                print(f"    {cls:6} TP={r['tp']:5} FP={r['fp']:6} FN={r['fn']:6}  "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
        print(f"    OVERALL F1: {m['overall']['f1']:.3f}")


if __name__ == "__main__":
    main()
