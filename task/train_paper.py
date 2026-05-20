"""
Faithful Baseline Training Script (train_paper.py)
==================================================
Strict reproduction of the DCASE 2025 Whale-VAD baseline.
Features:
  - Dynamically overwrites config.py values (no manual edits needed).
  - No early stopping (trains unconditionally for defined epochs).
  - Cosine Annealing Learning Rate Scheduler.
  - Negative resampling every epoch (paper-faithful; production uses 5).
  - Dynamically collapses 7-class predictions to 3-class during validation.
  - Saves best_model.pt strictly based on the 3-Class Macro F1 score.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ====================================================================
# 1. RUNTIME CONFIG OVERRIDES (MUST HAPPEN BEFORE PROJECT IMPORTS)
# ====================================================================
import config as cfg

print("\n" + "=" * 50)
print(">>> FORCING PAPER-FAITHFUL CONFIG OVERRIDES <<<")
cfg.USE_3CLASS = False
cfg.LR = 1e-5
# Paper resamples negatives every epoch (Section 5.5). Production
# train.py uses 5 to save compute; we override to 1 here.
RESAMPLE_EVERY = 1
print(f"  USE_3CLASS set to:     {cfg.USE_3CLASS}")
print(f"  LR set to:             {cfg.LR}")
print(f"  RESAMPLE_EVERY set to: {RESAMPLE_EVERY}")
print("=" * 50 + "\n")

# ====================================================================
# 2. PROJECT IMPORTS
# ====================================================================
import wandb_utils as wbu
from dataset import build_dataloaders, load_annotations, collate_fn
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from spectrogram import SpectrogramExtractor
from postprocess import (
    collapse_probs_to_3class,
    postprocess_predictions,
    compute_metrics,
    Detection,
    _parse_filename_dt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SOTA Baseline (paper-faithful)")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run folder")
    parser.add_argument("--epochs", type=int, default=80, help="Total epochs to train")
    parser.add_argument("--seed", type=int, default=cfg.SEED,
                        help="Master RNG seed for Python/NumPy/Torch/CUDA and "
                             "for DataLoader shuffle order. Different seeds let "
                             "you run multi-seed reproductions.")
    return parser.parse_args()


def align_lengths(logits, targets, mask):
    """
    Reconcile length mismatches between logits and targets.

    The CNN front end can produce a different frame count than the
    naive ``n_samples // hop_length`` count the dataset uses to paint
    targets, due to padding and pooling boundary effects. Truncate or
    zero-pad ``targets``/``mask`` to match ``logits.size(1)``.

    Copied verbatim from production ``train.py``.
    """
    T_m, T_t = logits.size(1), targets.size(1)
    if T_m < T_t:
        targets = targets[:, :T_m, :]
        mask = mask[:, :T_m]
    elif T_m > T_t:
        pad_t = torch.zeros(targets.size(0), T_m - T_t, targets.size(2),
                            device=targets.device)
        targets = torch.cat([targets, pad_t], dim=1)
        pad_m = torch.zeros(mask.size(0), T_m - T_t, dtype=torch.bool,
                            device=mask.device)
        mask = torch.cat([mask, pad_m], dim=1)
    return targets, mask


def main():
    args = parse_args()

    # 1. Setup Environment
    # Seed BEFORE build_dataloaders -- TrainingDatasetWithResample draws
    # the initial negative pool at construction time, so seeding here
    # makes those negatives reproducible across runs.
    wbu.seed_everything(args.seed, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.OUTPUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting {args.run_name} on {device} (seed={args.seed})")
    print(f"Targeting {args.epochs} epochs with LR={cfg.LR} + Cosine Scheduler")
    print(f"Output dir: {out_dir}")

    # 2. Data & Ground Truth Setup
    # build_dataloaders returns (train_ds, train_loader, val_loader); we
    # keep train_ds so we can call resample_negatives() between epochs.
    train_ds, train_loader, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)

    # Pre-build validation ground-truth events (in 3-class space).
    file_start_dts = {}
    for ds, fn in val_annotations[["dataset", "filename"]].drop_duplicates().values:
        dt = _parse_filename_dt(fn)
        if dt:
            file_start_dts[(ds, fn)] = dt

    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if not fsd:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],  # ALWAYS evaluate against 3-class labels
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    # 3. Model, Optimizer, Loss, Scheduler
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    pw = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
    criterion = WhaleVADLoss(pos_weight=pw).to(device)

    best_macro_f1 = 0.0
    best_epoch = 0
    best_components = [0.0, 0.0, 0.0]
    best_thresh_at_best = np.full(3, 0.5)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        # --- NEGATIVE RESAMPLING (paper Section 5.5) -----------------
        # train_ds.segments is updated in place, but the live DataLoader
        # captured its segment list at __iter__ time. To pick up the new
        # negatives we have to rebuild the loader. (Same pattern as
        # production train.py line 553.)
        # The seeded_dataloader_kwargs derive the loader's RNG from
        # (master_seed + epoch), so shuffle order is reproducible across
        # reruns of the same --seed value.
        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print(f"  Resampling negatives for epoch {epoch}")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **wbu.seeded_dataloader_kwargs(args.seed + epoch),
            )

        # --- TRAIN ---------------------------------------------------
        model.train()
        train_loss = 0.0
        n_batches = 0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch}/{args.epochs}] LR: {current_lr:.2e}")

        for audio, targets, mask, metas in tqdm(train_loader, desc="Train", leave=False):
            audio, targets, mask = audio.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()
            spec = spec_extractor(audio)
            logits = model(spec)
            targets, mask = align_lengths(logits, targets, mask)
            loss = criterion(logits, targets, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # --- STEP SCHEDULER -----------------------------------------
        scheduler.step()

        # --- VALIDATION ---------------------------------------------
        model.eval()
        all_probs = {}
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for audio, targets, mask, metas in tqdm(val_loader, desc="Val", leave=False):
                audio, targets, mask = audio.to(device), targets.to(device), mask.to(device)

                spec = spec_extractor(audio)
                logits = model(spec)
                targets, mask = align_lengths(logits, targets, mask)
                val_loss += criterion(logits, targets, mask).item()
                n_val += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                hop = spec_extractor.hop_length

                for j, meta in enumerate(metas):
                    key = (meta["dataset"], meta["filename"], meta["start_sample"])
                    n_samp = meta["end_sample"] - meta["start_sample"]
                    n_frames = min(n_samp // hop, probs[j].shape[0])
                    all_probs[key] = probs[j, :n_frames, :]

        val_loss /= max(n_val, 1)

        # --- 7 -> 3 COLLAPSE & PER-CLASS THRESHOLD SWEEP ------------
        # 1. Map 7-channel probs to 3-channel (max over subclasses).
        all_probs_3c = collapse_probs_to_3class(all_probs)

        # 2. CRITICAL: postprocess_predictions labels its outputs from
        #    cfg.class_names(), which returns the 7-class list while
        #    USE_3CLASS=False. After our collapse the arrays are 3-channel,
        #    so without flipping the flag the 3 channels would be labelled
        #    [bma, bmb, bmz] and merge_and_filter would collapse all three
        #    to "bmabz" via COLLAPSE_MAP -- every D and BP detection gets
        #    dropped and macro F1 falls to BMABZ_F1 / 3. We toggle inside
        #    a try/finally so the next epoch's training still sees 7-class.
        candidates = np.linspace(0.1, 0.9, 9)
        best_thresholds = np.full(3, 0.5)
        macro_f1_components = []

        cfg.USE_3CLASS = True
        try:
            for c, cls_name in enumerate(cfg.CALL_TYPES_3):
                best_cls_f1 = 0.0
                for t_try in candidates:
                    threshs = best_thresholds.copy()
                    threshs[c] = t_try
                    preds = postprocess_predictions(all_probs_3c, threshs)
                    metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
                    f1 = metrics.get(cls_name, {}).get("f1", 0.0)

                    if f1 > best_cls_f1:
                        best_cls_f1 = f1
                        best_thresholds[c] = t_try

                macro_f1_components.append(best_cls_f1)
        finally:
            cfg.USE_3CLASS = False  # restore for next epoch's training/dataset

        epoch_macro_f1 = float(np.mean(macro_f1_components))

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Collapsed Macro F1: {epoch_macro_f1:.4f} "
            f"(BMABZ: {macro_f1_components[0]:.3f}, "
            f"D: {macro_f1_components[1]:.3f}, "
            f"BP: {macro_f1_components[2]:.3f})"
        )

        # --- SAVING LOGIC -------------------------------------------
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_f1": best_macro_f1,
            "thresholds": torch.tensor(best_thresholds),
        }
        torch.save(state, out_dir / "latest_model.pt")

        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            best_epoch = epoch
            best_components = list(macro_f1_components)
            best_thresh_at_best = best_thresholds.copy()
            state["best_f1"] = best_macro_f1
            torch.save(state, out_dir / "best_model.pt")
            print(f"  *** New Best Collapsed Macro F1 saved! ***")


    # --- FINAL SUMMARY ----------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run name:         {args.run_name}")
    print(f"Seed:             {args.seed}")
    print(f"Epochs trained:   {args.epochs}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best Macro F1:    {best_macro_f1:.4f} "
          f"(BMABZ: {best_components[0]:.3f}, "
          f"D: {best_components[1]:.3f}, "
          f"BP: {best_components[2]:.3f})")
    print(f"Tuned thresholds: bmabz={best_thresh_at_best[0]:.2f}, "
          f"d={best_thresh_at_best[1]:.2f}, "
          f"bp={best_thresh_at_best[2]:.2f}")
    print(f"Best checkpoint:  {out_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
