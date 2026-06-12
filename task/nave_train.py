"""
NAVE training entry point.
==========================
Trains a single NAVE model (the locked recipe) and checkpoints the EMA weights
selected by tuned macro F1. Reuses the verified data pipeline, negative
resampling, per-epoch threshold tuning, EMA and optimiser exactly as the recipe
used them; the only differences from the old ``train_phase13r`` harness are that
it builds the clean ``NAVE`` model and logs to a fresh Weights & Biases project
(``cfg.WANDB_PROJECT``) so later experiments stay separate.

    CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed 42
    CUDA_VISIBLE_DEVICES=1 python nave_train.py --seed 2024 --tune-workers 20

Checkpoints (native NAVE format) land in ``runs/nave_s<seed>_<timestamp>/`` as
``nave_best.pt`` (best tuned macro) and ``nave_epoch_NN.pt``. They load with
``nave_evaluate.py`` / ``nave_ensemble.py`` or ``NAVE().load_checkpoint(...)``.

NOTE: the data/validation helpers imported below currently read ``config_final``;
its shared values (sites, batch size, neg ratio, segment length, workers, post-
processing) are identical to ``nave_config``. At new-git consolidation, point
them at ``nave_config`` and this module is fully self-contained.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nave_config as cfg
from nave_model import NAVE
from nave_features import NAVEFeatureExtractor

# verified, reused (see NOTE above)
from dataset_final import (
    load_annotations, get_file_manifest, build_positive_segments,
    build_val_segments, extend_all_segments, WhaleDataset, collate_fn,
)
from train_final import (
    seed_everything, seeded_dataloader_kwargs, compute_pos_weight,
    resample_negatives_for_epoch, validate,
)
from conformer_core import EMA, build_optim_sched, MaskedBCELoss, _train_epoch
from tuned_val import validate_tuned


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help="Run seed (vary it to build the ensemble).")
    p.add_argument("--tune-workers", type=int, default=8,
                   help="Parallel workers for the per-epoch threshold tuner.")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    return p.parse_args()


def main():
    args = parse_args()
    seed = seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.EPOCHS
    print(f"[NAVE] device={device}  seed={seed}  epochs={epochs}  k={cfg.CONV_KERNEL}")

    train_sites, val_sites = list(cfg.TRAIN_DATASETS), list(cfg.VAL_DATASETS)
    pos_weight, weight_info = compute_pos_weight(train_sites, device, verbose=True)

    # --- W&B (fresh NAVE project; not the old phase registry) ---
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            entity=cfg.WANDB_ENTITY, project=cfg.WANDB_PROJECT, mode=cfg.WANDB_MODE,
            name=f"nave_s{seed}", group="nave_final",
            tags=["nave", "conformer", "fdy", "pcen", f"k{cfg.CONV_KERNEL}"],
            config={
                "model": "NAVE", "seed": seed, "epochs": epochs,
                "d_model": cfg.D_MODEL, "nhead": cfg.NHEAD, "layers": cfg.NUM_LAYERS,
                "ffn_mult": cfg.FFN_MULT, "conv_kernel": cfg.CONV_KERNEL,
                "dropout": cfg.DROPOUT, "fdy_basis": cfg.FDY_BASIS,
                "optimizer": cfg.OPTIMIZER, "lr": cfg.LR, "weight_decay": cfg.WEIGHT_DECAY,
                "ema_decay": cfg.EMA_DECAY, "batch_size": cfg.BATCH_SIZE,
                "neg_ratio": cfg.NEG_RATIO, "pos_weight": weight_info["pos_weight"],
            })

    run_dir = Path(cfg.OUTPUT_DIR) / f"nave_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[NAVE] run dir: {run_dir}")

    # --- data ---
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    pos_segs = build_positive_segments(train_annotations, train_manifest)
    pos_segs = extend_all_segments(pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}

    # --- model / loss / optim ---
    model = NAVE().to(device)
    spec_extractor = NAVEFeatureExtractor().to(device)
    print(f"[NAVE] parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_criterion = MaskedBCELoss(pos_weight=pos_weight).to(device)
    val_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight).to(device)

    opt_kwargs = dict(optimizer=cfg.OPTIMIZER, peak_lr=cfg.LR, warmup_epochs=0.0,
                      schedule="const", weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = max(1, (len(pos_segs) + n_neg) // cfg.BATCH_SIZE)
    optimizer, scheduler, is_sf = build_optim_sched(model, opt_kwargs, steps_per_epoch, epochs)
    ema = EMA(model, decay=cfg.EMA_DECAY)

    # --- loop ---
    best_macro, last_thr = 0.0, None
    train_loader = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if train_loader is None or (epoch - 1) % cfg.RESAMPLE_EVERY == 0:
            train_segments = resample_negatives_for_epoch(
                pos_segs_extended=pos_segs, train_annotations=train_annotations,
                train_manifest=train_manifest, n_neg=n_neg,
                segment_s=cfg.TRAIN_SEGMENT_S, epoch=epoch, verbose=True)
            train_loader = DataLoader(
                WhaleDataset(train_segments), batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
                **seeded_dataloader_kwargs(seed))

        train_loss = _train_epoch(model, spec_extractor, train_loader, train_criterion,
                                  optimizer, scheduler, device, ema)

        # select / checkpoint on the EMA weights
        ema.store_and_copy(model)
        try:
            val = validate_tuned(model, spec_extractor, val_loader, val_criterion, device,
                                 val_annotations, file_start_dts,
                                 workers=args.tune_workers, start_thr=last_thr)
            last_thr = val.get("thresholds", last_thr)
        except Exception as e:
            print(f"  [tune] failed ({type(e).__name__}: {e}); fixed-{cfg.THRESHOLD} fallback")
            val = validate(model, spec_extractor, val_loader, val_criterion, device,
                           val_annotations, file_start_dts, threshold=cfg.THRESHOLD)
        eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(model)

        macro = sum(val["per_class"][c]["f1"] for c in cfg.CALL_TYPES_3) / 3
        improved = macro > best_macro
        if improved:
            best_macro = macro

        print(f"[NAVE] epoch {epoch:2d}/{epochs} ({time.time()-t0:.0f}s)"
              f"{' *** new best' if improved else ''}")
        print(f"  train {train_loss:.4f}  val {val['loss']:.4f}  "
              f"MACRO {macro:.3f}  THR {[round(t,3) for t in val.get('thresholds', [])]}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

        if run is not None:
            import wandb
            wandb.log({"train/loss": train_loss, "val/loss": val["loss"],
                       "val/macro_f1": macro, "val/f1_micro": val["f1"],
                       "val/macro_paper": val["macro_paper"],
                       **{f"val/f1_{c}": val["per_class"][c]["f1"] for c in cfg.CALL_TYPES_3}},
                      step=epoch)

        ckpt = {"model_state_dict": eval_state, "epoch": epoch, "seed": seed,
                "macro_f1": macro, "thresholds": val.get("thresholds"),
                "model": "NAVE", "ema_decay": cfg.EMA_DECAY}
        torch.save(ckpt, run_dir / f"nave_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "nave_best.pt")

    print(f"\n[NAVE] best tuned macro F1 = {best_macro:.3f}   ->  {run_dir/'nave_best.pt'}")
    if run is not None:
        run.summary["best_macro_f1"] = best_macro
        run.finish()


if __name__ == "__main__":
    main()
