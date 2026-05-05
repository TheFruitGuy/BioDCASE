"""
Phase 3α: Standard Contrastive Pretraining (Data Diversity Only)
================================================================

Tests whether SSL pretraining on more diverse Antarctic PAM data helps
the supervised model generalize across sites.

- Data: cfg.TRAIN_DATASETS (8 BioDCASE training sites) + 4 AADC sites
        (DDU2018, DDU2019, Kerguelen2018, Kerguelen2019)
- Augmentations per view: random 30s window (handled by dataset),
        volume scaling (audio), narrowband freq mask outside [13, 53] (spec)
- NO cross-site noise mixing (that's 3β)

Usage
-----
::

    python pretrain_phase3a.py --aadc-root /path/to/aadc/audio \\
                               --output-dir runs_pretrain/3a_seed42 \\
                               [--seed 42] [--no-wandb]

After completion, fine-tune via the existing supervised pipeline::

    python train.py --pretrained runs_pretrain/3a_seed42/encoder_best.pt \\
                    --freeze_epochs 10
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor

from ssl_dataset import build_pretrain_manifest, SSLClipDataset, collate_ssl
from ssl_augmentations import make_view
from pretrain_core import PretrainConfig, pretrain
from linear_probe import build_probe_loaders


# Default 4 AADC sites — Path B from the handoff.
DEFAULT_AADC_SITES = ["DDU2018", "DDU2019", "Kerguelen2018", "Kerguelen2019"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aadc-root", type=Path, required=True,
                   help="Root dir containing AADC site subfolders "
                        "(e.g. /var/.../ssl_pretrain/audio)")
    p.add_argument("--aadc-sites", nargs="+", default=DEFAULT_AADC_SITES,
                   help=f"AADC sites to include (default: {DEFAULT_AADC_SITES})")
    p.add_argument("--include-train-sites", action="store_true", default=True,
                   help="Include cfg.TRAIN_DATASETS in pretraining (default: True)")
    p.add_argument("--no-train-sites", dest="include_train_sites",
                   action="store_false",
                   help="Exclude cfg.TRAIN_DATASETS (AADC only)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save encoder_best.pt / encoder_last.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epoch-clips", type=int, default=50_000,
                   help="Random clips drawn per epoch (virtual epoch length)")
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--probe-patience", type=int, default=3)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def set_seed(seed: int):
    """
    Seed all stochastic sources for reproducibility. Routes through
    ``wandb_utils.seed_everything`` so the same seeding semantics
    (PYTHONHASHSEED + Python + NumPy + torch CPU/CUDA) apply across
    every phase across the project.
    """
    wbu.seed_everything(seed, deterministic=False)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3a] device={device}  seed={args.seed}  "
          f"output={args.output_dir.resolve()}")

    # ------- Wandb -------
    wandb_log_fn = None
    if not args.no_wandb:
        try:
            import wandb
            run = wbu.init_phase(
                "3a",
                config={
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "epoch_clips": args.epoch_clips,
                    "lr": args.lr,
                    "temperature": args.temperature,
                    "include_train_sites": args.include_train_sites,
                    "aadc_sites": args.aadc_sites,
                },
                mode=args.wandb_mode,
            )
            wandb_log_fn = lambda m, step: wandb.log(m, step=step)
        except KeyError:
            print("[3a] WARNING: phase '3a' not in PHASE_REGISTRY. "
                  "Add it to wandb_utils.py or pass --no-wandb.")
            raise
        except ImportError:
            print("[3a] wandb not installed; running with --no-wandb")

    # ------- Manifest -------
    manifest = build_pretrain_manifest(
        train_datasets=(cfg.TRAIN_DATASETS if args.include_train_sites else None),
        aadc_sites=args.aadc_sites,
        aadc_root=args.aadc_root,
        expected_sample_rate=cfg.SAMPLE_RATE,
    )
    # Persist for inspection / reproducibility
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)

    # ------- Dataset / loader -------
    dataset = SSLClipDataset(
        manifest=manifest,
        clip_seconds=30.0,
        sample_rate=cfg.SAMPLE_RATE,
        epoch_clips=args.epoch_clips,
    )
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_ssl,
        pin_memory=True, drop_last=True,
        # Random sampling happens inside __getitem__ — workers seeded
        # via torch's default + `idx` in SSLClipDataset for diversity.
    )

    # ------- Probe loaders -------
    probe_train_loader, probe_val_loader = build_probe_loaders(
        seed=args.seed,
        num_workers=min(args.num_workers, 4),
    )

    # ------- View pipeline (no cross-site mix in 3α) -------
    spec_extractor = SpectrogramExtractor().to(device)

    def view_fn(audio, sites, sx):
        return make_view(
            audio=audio, sites=sites, spec_extractor=sx,
            use_volume=True, use_freq_mask=True, use_cross_site=False,
            volume_p=0.5, freq_mask_p=0.5,
        )

    # ------- Pretrain -------
    pre_cfg = PretrainConfig(
        train_loader=train_loader,
        spec_extractor=spec_extractor,
        view_fn=view_fn,
        n_epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        probe_every=args.probe_every,
        probe_patience=args.probe_patience,
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader,
        output_dir=args.output_dir,
        wandb_log_fn=wandb_log_fn,
        seed=args.seed,
    )
    summary = pretrain(pre_cfg, device=device)
    print(f"[3a] done: best probe F1 {summary['best_probe_f1']:.4f} "
          f"at epoch {summary['best_epoch']}")
    print(f"[3a] best ckpt: {summary['best_ckpt']}")
    print(f"[3a] last ckpt: {summary['last_ckpt']}")

    # ----- Wandb finalize: stamp summary + log encoder as artifact -----
    if wandb_log_fn is not None:
        verdict = (
            f"Phase 3a: best probe F1 {summary['best_probe_f1']:.4f} "
            f"at epoch {summary['best_epoch']} (final epoch "
            f"{summary['final_epoch']}). Encoder ready for fine-tuning "
            f"via train.py --pretrained."
        )
        wbu.finalize_eval_phase(
            {
                "best_probe_f1": float(summary["best_probe_f1"]),
                "best_epoch":    int(summary["best_epoch"]),
                "final_epoch":   int(summary["final_epoch"]),
            },
            verdict=verdict,
            artifact_path=summary["best_ckpt"],
            artifact_type="encoder",
            artifact_metadata={
                "phase":          "3a",
                "best_probe_f1":  float(summary["best_probe_f1"]),
                "best_epoch":     int(summary["best_epoch"]),
                "aadc_sites":     args.aadc_sites,
                "include_train_sites": args.include_train_sites,
            },
        )


if __name__ == "__main__":
    main()
