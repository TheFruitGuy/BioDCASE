"""
Conformer-SED Training Driver (final pipeline)
==============================================

Trains ``WhaleVAD_Conformer`` using the *exact* recipe, data pipeline,
validation and checkpoint-selection logic as the final CNN-BiLSTM
baseline (``train_final.py``), so the two are a fair head-to-head.

What is reused verbatim from ``train_final`` (identical recipe/measurement):
  - ``validate``                     — 7→3 collapse, fixed-threshold (0.3)
                                        event-level scoring, returns macro_paper
  - ``compute_pos_weight``           — segment-count-normalised weighted-BCE weights
  - ``resample_negatives_for_epoch`` — per-epoch fresh negative subset
  - ``seed_everything`` / ``seeded_dataloader_kwargs``
  - the loss (plain weighted ``BCEWithLogitsLoss``, masked manually — NO focal),
    optimiser, fixed LR, 30 epochs, no scheduler, no early stopping
  - checkpoint selection on ``macro_paper`` (= F1 of mean-P/mean-R, the
    Geldenhuys/paper convention — the metric the final pipeline already uses)

What differs (and only this):
  1. The model is ``WhaleVAD_Conformer`` (7-class head, collapsed to 3 at
     eval) instead of ``model_final.WhaleVAD``.
  2. The training step passes a per-frame ``key_padding_mask`` to the model
     so attention ignores zero-padded frames in variable-length batches
     (``train_final`` masks padding only in the loss; for global attention
     masking it in the model matters). Validation is left unmasked, exactly
     as ``train_final.validate`` does it — keeping measurement identical.

Usage
-----
    CUDA_VISIBLE_DEVICES=5 python train_conformer.py
    CUDA_VISIBLE_DEVICES=5 python train_conformer.py --wandb
    CUDA_VISIBLE_DEVICES=5 python train_conformer.py --d-model 144 --layers 4 --wandb
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
from dataset_final import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_val_segments,
    extend_all_segments, WhaleDataset, collate_fn,
)
from spectrogram_final import SpectrogramExtractor
from model_conformer import WhaleVAD_Conformer

# Reuse the final pipeline's recipe/measurement primitives verbatim.
from train_final import (
    seed_everything, seeded_dataloader_kwargs, compute_pos_weight,
    validate, resample_negatives_for_epoch,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Conformer-SED Whale-VAD model (phase 13) on the final pipeline.")
    # Conformer hyperparameters
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--conv-kernel", type=int, default=31)
    p.add_argument("--dropout", type=float, default=0.1)
    # Recipe / bookkeeping — mirror train_final.py
    p.add_argument("--wandb", action="store_true",
                   help="Log this run to Weights & Biases (phase 13). Off by default.")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                   help=f"Number of training epochs (default {cfg.EPOCHS}).")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help=f"Master random seed (default {cfg.SEED}).")
    p.add_argument("--num-classes", type=int, default=7, choices=[3, 7],
                   help="Train as 3-class (coarse) or 7-class (fine, default; "
                        "collapsed to 3 at eval). Matches train_final.py.")
    return p.parse_args()


# ======================================================================
# Model construction
# ======================================================================

def build_conformer(device: torch.device, args: argparse.Namespace):
    """Build the Conformer + spectrogram extractor; one dummy forward to
    materialise the lazy projection (so the optimiser captures it)."""
    model = WhaleVAD_Conformer(
        num_classes=cfg.n_classes(),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
        ffn_mult=args.ffn_mult, conv_kernel=args.conv_kernel, dropout=args.dropout,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


# ======================================================================
# Mask-aware training step (mirrors train_final.train_one_epoch otherwise)
# ======================================================================

def train_one_epoch_masked(model, spec_extractor, loader, criterion, optimizer, device):
    """One training pass; attention is masked over padded frames. Loss
    masking/normalisation is identical to train_final.train_one_epoch."""
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)                      # True = valid frame

        # key_padding_mask: True = padded. The model reconciles any off-by-one
        # between this length and its internal frame count.
        logits = model(spec_extractor(audio), key_padding_mask=~mask.bool())

        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
        optimizer.step()

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    # Set the class-mode flag BEFORE any dataset/model is constructed.
    cfg.USE_3CLASS = (args.num_classes == 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Head: {cfg.n_classes()}-class "
          f"({'coarse' if cfg.USE_3CLASS else 'fine'}); Conformer "
          f"d_model={args.d_model}, layers={args.layers}, heads={args.nhead}")

    seed = seed_everything(args.seed, deterministic=False)

    train_sites = list(cfg.TRAIN_DATASETS)
    val_sites = list(cfg.VAL_DATASETS)

    # --- pos_weight from annotations, before any dataloader exists ---
    print(f"\nComputing {cfg.n_classes()}-class pos_weight over "
          f"{len(train_sites)} sites...")
    pos_weight, weight_info = compute_pos_weight(train_sites, device, verbose=True)

    # --- optional W&B run (phase 13) ---
    run = None
    if args.wandb:
        import wandb_utils as wbu
        import wandb
        run = wbu.init_phase("13", config={
            "arch": "conformer",
            "d_model": args.d_model, "nhead": args.nhead, "layers": args.layers,
            "ffn_mult": args.ffn_mult, "conv_kernel": args.conv_kernel,
            "dropout": args.dropout,
            "lr": cfg.LR, "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE, "threshold": cfg.THRESHOLD,
            "seed": seed, "num_classes": cfg.n_classes(), "use_3class": cfg.USE_3CLASS,
            "neg_ratio": cfg.NEG_RATIO, "neg_resample_each_epoch": True,
            "segment_s": cfg.TRAIN_SEGMENT_S, "epochs": args.epochs,
            "train_sites": train_sites, "val_sites": val_sites,
            "lstm_hidden": cfg.LSTM_HIDDEN, "lstm_layers": cfg.LSTM_LAYERS,
            "pos_weight": weight_info["pos_weight"],
            "pos_weight_counts": weight_info["annotation_counts"],
            "pos_weight_ratio": weight_info["weight_ratio"],
        }, mode=args.wandb_mode, extra_tags=["from_scratch", "conformer"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"conformer_{cfg.n_classes()}c_s{seed}"
    run_dir = Path(cfg.OUTPUT_DIR) / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration:")
    print(f"  Training sites:   {train_sites}  ({len(train_sites)})")
    print(f"  Validation sites: {val_sites}")
    print(f"  Output:           {cfg.n_classes()}-class training, collapsed to 3 at eval")
    print(f"  Loss:             weighted BCE (segment-count normalised), no focal")
    print(f"  Negatives:        resampled at the start of every epoch")
    print(f"  Sequence model:   Conformer ({args.layers}× blocks, d_model={args.d_model})")
    print(f"  LR={cfg.LR}, batch={cfg.BATCH_SIZE}, epochs={args.epochs}")

    # --- fixed positives + static validation set (identical to train_final) ---
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    print(f"  {len(train_manifest)} files, {len(train_annotations)} annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    pos_segs = extend_all_segments(pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    print(f"  Positive segments (fixed): {len(pos_segs)}")
    print(f"  Negative segments per epoch: {n_neg}")

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, {len(val_annotations)} "
          f"annotations, {len(val_segments)} segments")

    # --- model, loss, optimiser ---
    model, spec_extractor = build_conformer(device, args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    if run is not None:
        run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # --- training loop with per-epoch negative resampling ---
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {args.epochs} epochs (per-epoch negative resampling)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_segments = resample_negatives_for_epoch(
            pos_segs_extended=pos_segs,
            train_annotations=train_annotations,
            train_manifest=train_manifest,
            n_neg=n_neg,
            segment_s=cfg.TRAIN_SEGMENT_S,
            epoch=epoch,
            verbose=True,
        )
        train_loader = DataLoader(
            WhaleDataset(train_segments), batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
            **seeded_dataloader_kwargs(seed),
        )

        train_loss = train_one_epoch_masked(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=cfg.THRESHOLD,
        )
        epoch_time = time.time() - t0

        # Checkpoint selection on paper-convention macro-F1 (same as train_final).
        improved = val["macro_paper"] > best_f1
        if improved:
            best_f1 = val["macro_paper"]
        marker = " *** new best" if improved else ""

        print(f"\nEpoch {epoch:2d}/{args.epochs}  ({epoch_time:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        macro = sum(val["per_class"][n]["f1"] for n in cfg.CALL_TYPES_3) / 3
        print(f"    OVERALL F1={val['f1']:.3f}  MACRO F1={macro:.3f}  "
              f"MACRO_PAPER F1={val['macro_paper']:.3f}")

        if run is not None:
            wbu.log_epoch_3class(epoch, train_loss, val)
            # log the actual selection metric explicitly (log_epoch_3class
            # labels micro F1 as "f1_macro", so surface macro_paper too).
            wandb.log({
                "val/f1_micro": val["f1"],
                "val/f1_macro_mean": macro,
                "val/f1_macro_paper": val["macro_paper"],
            }, step=epoch)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "f1": val["f1"],
            "macro_f1": macro,
            "macro_paper_f1": val["macro_paper"],
            "per_class": val["per_class"],
        })

        ckpt = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "f1": val["f1"], "macro_paper_f1": val["macro_paper"],
            "history": history,
            "pos_weight": pos_weight.detach().cpu().tolist(),
            "arch_kwargs": {
                "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.layers,
                "ffn_mult": args.ffn_mult, "conv_kernel": args.conv_kernel,
                "dropout": args.dropout, "num_classes": cfg.n_classes(),
            },
        }
        torch.save(ckpt, run_dir / f"conformer_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "conformer_best.pt")

    # --- summary (mirrors train_final) ---
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    macro_papers = [h["macro_paper_f1"] for h in history]
    print(f"\nMicro F1 by epoch:       {[f'{f:.3f}' for f in f1s]}")
    print(f"Macro F1 by epoch:       {[f'{m:.3f}' for m in macros]}")
    print(f"Macro_paper F1 by epoch: {[f'{m:.3f}' for m in macro_papers]}")
    print(f"\nBest micro F1:       {max(f1s):.3f}  (epoch {f1s.index(max(f1s)) + 1})")
    print(f"Best macro F1:       {max(macros):.3f}")
    print(f"Best macro_paper F1: {max(macro_papers):.3f}  "
          f"(epoch {macro_papers.index(max(macro_papers)) + 1})")
    for name in cfg.CALL_TYPES_3:
        best = max(h["per_class"][name]["f1"] for h in history)
        print(f"  best {name}: {best:.3f}")

    second_half = macro_papers[len(macro_papers) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability (macro_paper): mean swing {mean_swing:.3f}, "
          f"max swing {max_swing:.3f}")

    verdict = (f"Conformer (d_model={args.d_model}, {args.layers}L): best "
               f"macro_paper F1 {max(macro_papers):.3f} (micro {max(f1s):.3f}, "
               f"macro {max(macros):.3f}); second-half mean swing {mean_swing:.3f}.")
    if run is not None:
        wbu.finalize_phase(history, verdict=verdict,
                           best_ckpt=run_dir / "conformer_best.pt")


if __name__ == "__main__":
    main()
