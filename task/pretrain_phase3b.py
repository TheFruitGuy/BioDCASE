"""
Phase 3β: Contrastive Pretraining + Cross-Site Noise Mixing
===========================================================

Same data as 3α, but adds cross-site noise mixing to one of the two
augmented views per pair. The hypothesis: forcing the encoder to map
together two clips that differ in their site-specific noise floor
pushes it toward site-invariant representations — which is exactly
what we want for cross-site test generalization (DDU2021, Kerguelen2020).

Cross-site mix uses an *extended* no-call pool that includes random
windows from the AADC sites alongside the labeled training-site no-call
clips. This gives 12-site noise diversity (vs 8-site for the supervised
Phase 1e pool), at the cost of a small amount of label noise — a small
fraction of "no-call" AADC windows may contain calls. Given the ~5%
frame-level call prevalence reported in the literature, this contamination
is acceptable.

Usage
-----
::

    python pretrain_phase3b.py --aadc-root /path/to/aadc/audio \\
                               --output-dir runs_pretrain/3b_seed42 \\
                               [--existing-pool no_call_pool.pt] \\
                               [--seed 42] [--no-wandb]

After completion, fine-tune via the existing supervised pipeline::

    python train.py --pretrained runs_pretrain/3b_seed42/encoder_best.pt \\
                    --freeze_epochs 10
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor

from ssl_dataset import build_pretrain_manifest, SSLClipDataset, collate_ssl
from ssl_augmentations import make_view
from pretrain_core import PretrainConfig, pretrain
from linear_probe import build_probe_loaders


DEFAULT_AADC_SITES = ["DDU2018", "DDU2019", "Kerguelen2018", "Kerguelen2019"]


# ======================================================================
# Extended no-call pool (training sites + random AADC windows)
# ======================================================================

@dataclass
class ExtendedNoCallPool:
    """
    In-RAM pool of cross-site no-call clips for SSL.

    Compatible API with phase1_baseline.NoCallPool — a single
    ``sample(exclude_site, n_samples, device)`` method returning a
    1-D float32 tensor on ``device``.

    Built from two sources:
    - ``existing_pool``: the labeled training-site clips precomputed
      by ``precompute_no_call_pool.py``. These are guaranteed no-call.
    - AADC random windows: per-site samples from the SSL pretrain
      manifest. Treated as no-call (calls are rare ~5% of frames).
    """
    by_site: dict[str, torch.Tensor]   # site → (n_clips, n_samples) int16
    sample_rate: int
    clip_samples: int

    def sample(
        self, *, exclude_site: str, n_samples: int, device: torch.device,
    ) -> torch.Tensor:
        candidates = [s for s in self.by_site.keys() if s != exclude_site]
        if not candidates:
            raise RuntimeError(
                f"ExtendedNoCallPool has no sites other than {exclude_site}"
            )
        site = candidates[int(torch.randint(0, len(candidates), (1,)).item())]
        clips = self.by_site[site]
        n_clips, clip_samples = clips.shape
        if n_samples > clip_samples:
            raise RuntimeError(
                f"Pool clip length ({clip_samples}) < requested {n_samples}"
            )
        ci = int(torch.randint(0, n_clips, (1,)).item())
        if n_samples < clip_samples:
            offset = int(torch.randint(0, clip_samples - n_samples + 1, (1,)).item())
        else:
            offset = 0
        chunk = clips[ci, offset:offset + n_samples].to(device).float() / 32768.0
        return chunk


def build_extended_pool(
    manifest: pd.DataFrame,
    existing_pool_path: Path | None,
    aadc_sites: list[str],
    n_per_site: int = 50,
    clip_samples: int = 7500,
    sample_rate: int = 250,
    seed: int = 1337,
) -> ExtendedNoCallPool:
    """
    Build the extended pool by combining an existing training-site pool
    with random AADC-site windows.

    The AADC windows are sampled uniformly at random from each site's
    files — no annotation filtering, since AADC has no labels. With a
    base call prevalence of ~5% of frames, ~95% of random 30s windows
    are call-free; the ~5% that contain calls add a small amount of
    augmentation noise but don't compromise the contrastive signal.
    """
    rng = np.random.default_rng(seed)
    by_site: dict[str, torch.Tensor] = {}

    # 1) Load existing training-site pool if provided
    if existing_pool_path is not None and Path(existing_pool_path).exists():
        data = torch.load(existing_pool_path, map_location="cpu", weights_only=False)
        if data.get("version") != 1:
            raise ValueError(f"Unsupported pool version: {data.get('version')}")
        if data["sample_rate"] != sample_rate:
            raise ValueError(
                f"Existing pool SR {data['sample_rate']} != {sample_rate}"
            )
        if data["clip_samples"] != clip_samples:
            raise ValueError(
                f"Existing pool clip len {data['clip_samples']} != {clip_samples}"
            )
        for site, tensor in data["clips"].items():
            by_site[site] = tensor
        print(f"[3b pool] loaded {len(by_site)} training sites from "
              f"{existing_pool_path}")
    else:
        if existing_pool_path is not None:
            print(f"[3b pool] WARNING: existing pool {existing_pool_path} "
                  "not found — pool will only contain AADC sites")

    # 2) Sample n_per_site random windows from each AADC site
    for site in aadc_sites:
        site_files = manifest[manifest["site"] == site]
        if site_files.empty:
            print(f"[3b pool] WARNING: no files for {site}, skipping")
            continue

        durations = site_files["duration_s"].to_numpy()
        weights = durations / durations.sum()
        clips = []
        attempts = 0
        max_attempts = n_per_site * 5
        while len(clips) < n_per_site and attempts < max_attempts:
            attempts += 1
            row = site_files.iloc[int(rng.choice(len(site_files), p=weights))]
            n_avail = int(row["n_samples"])
            if n_avail < clip_samples:
                continue
            offset = int(rng.integers(0, n_avail - clip_samples + 1))
            try:
                audio, _ = sf.read(
                    row["path"], start=offset, frames=clip_samples,
                    dtype="float32", always_2d=False,
                )
            except Exception as e:
                print(f"[3b pool] read error {row['path']}: {e}")
                continue
            if audio.ndim > 1:
                audio = audio[:, 0]
            if len(audio) != clip_samples:
                continue
            # Convert float32 [-1, 1] → int16 to match existing pool format
            clip_i16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
            clips.append(torch.from_numpy(clip_i16))

        if clips:
            by_site[site] = torch.stack(clips, dim=0)
            print(f"[3b pool] {site}: {len(clips)} clips")
        else:
            print(f"[3b pool] WARNING: failed to collect clips for {site}")

    if not by_site:
        raise RuntimeError("Extended pool is empty")
    print(f"[3b pool] total: {len(by_site)} sites, "
          f"{sum(t.shape[0] for t in by_site.values())} clips")
    return ExtendedNoCallPool(
        by_site=by_site,
        sample_rate=sample_rate,
        clip_samples=clip_samples,
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aadc-root", type=Path, required=True)
    p.add_argument("--aadc-sites", nargs="+", default=DEFAULT_AADC_SITES)
    p.add_argument("--include-train-sites", action="store_true", default=True)
    p.add_argument("--no-train-sites", dest="include_train_sites",
                   action="store_false")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--existing-pool", type=Path, default=Path("no_call_pool.pt"),
                   help="Path to precomputed training-site no-call pool "
                        "(default: ./no_call_pool.pt)")
    p.add_argument("--n-per-aadc-site", type=int, default=50,
                   help="Random no-call windows per AADC site")
    p.add_argument("--cross-site-prob", type=float, default=0.5,
                   help="Probability of mixing per sample per view")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epoch-clips", type=int, default=50_000)
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
    apply across every phase across the project.
    """
    wbu.seed_everything(seed, deterministic=False)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3b] device={device}  seed={args.seed}  "
          f"output={args.output_dir.resolve()}")

    # ------- Wandb -------
    wandb_log_fn = None
    if not args.no_wandb:
        try:
            import wandb
            run = wbu.init_phase(
                "3b",
                config={
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "epoch_clips": args.epoch_clips,
                    "lr": args.lr,
                    "temperature": args.temperature,
                    "cross_site_prob": args.cross_site_prob,
                    "include_train_sites": args.include_train_sites,
                    "aadc_sites": args.aadc_sites,
                },
                mode=args.wandb_mode,
            )
            wandb_log_fn = lambda m, step: wandb.log(m, step=step)
        except KeyError:
            print("[3b] WARNING: phase '3b' not in PHASE_REGISTRY. "
                  "Add it to wandb_utils.py or pass --no-wandb.")
            raise
        except ImportError:
            print("[3b] wandb not installed; running with --no-wandb")

    # ------- Manifest -------
    manifest = build_pretrain_manifest(
        train_datasets=(cfg.TRAIN_DATASETS if args.include_train_sites else None),
        aadc_sites=args.aadc_sites,
        aadc_root=args.aadc_root,
        expected_sample_rate=cfg.SAMPLE_RATE,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)

    # ------- Build extended no-call pool (training + AADC) -------
    pool = build_extended_pool(
        manifest=manifest,
        existing_pool_path=(args.existing_pool
                            if args.existing_pool and args.existing_pool.exists()
                            else None),
        aadc_sites=args.aadc_sites,
        n_per_site=args.n_per_aadc_site,
        clip_samples=int(30.0 * cfg.SAMPLE_RATE),
        sample_rate=cfg.SAMPLE_RATE,
        seed=args.seed,
    )

    # ------- Dataset / loader -------
    dataset = SSLClipDataset(
        manifest=manifest, clip_seconds=30.0,
        sample_rate=cfg.SAMPLE_RATE, epoch_clips=args.epoch_clips,
    )
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_ssl,
        pin_memory=True, drop_last=True,
    )

    # ------- Probe loaders -------
    probe_train_loader, probe_val_loader = build_probe_loaders(
        seed=args.seed, num_workers=min(args.num_workers, 4),
    )

    # ------- View pipeline WITH cross-site mix -------
    spec_extractor = SpectrogramExtractor().to(device)
    cross_site_p = args.cross_site_prob

    def view_fn(audio, sites, sx):
        return make_view(
            audio=audio, sites=sites, spec_extractor=sx,
            use_volume=True, use_freq_mask=True,
            use_cross_site=True, no_call_pool=pool,
            volume_p=0.5, freq_mask_p=0.5, cross_site_p=cross_site_p,
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
    print(f"[3b] done: best probe F1 {summary['best_probe_f1']:.4f} "
          f"at epoch {summary['best_epoch']}")
    print(f"[3b] best ckpt: {summary['best_ckpt']}")
    print(f"[3b] last ckpt: {summary['last_ckpt']}")

    # ----- Wandb finalize: stamp summary + log encoder as artifact -----
    if wandb_log_fn is not None:
        verdict = (
            f"Phase 3b: best probe F1 {summary['best_probe_f1']:.4f} "
            f"at epoch {summary['best_epoch']} (final epoch "
            f"{summary['final_epoch']}). Cross-site mix "
            f"p={args.cross_site_prob}. Encoder ready for fine-tuning "
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
                "phase":          "3b",
                "best_probe_f1":  float(summary["best_probe_f1"]),
                "best_epoch":     int(summary["best_epoch"]),
                "aadc_sites":     args.aadc_sites,
                "cross_site_prob": args.cross_site_prob,
                "include_train_sites": args.include_train_sites,
            },
        )


if __name__ == "__main__":
    main()
