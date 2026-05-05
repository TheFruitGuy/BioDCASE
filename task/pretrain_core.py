"""
SSL Pretraining Core
====================

Shared SimCLR-style contrastive pretraining loop used by both
``pretrain_phase3a.py`` (data diversity only) and ``pretrain_phase3b.py``
(data diversity + cross-site noise mixing).

Architecture
------------
::

    audio → SpectrogramExtractor → spec_aug → WhaleVAD encoder
                                                    │
                                              (B, T_frames, 64)
                                                    │
                                          mean pool over time
                                                    │
                                               (B, 64)
                                                    │
                                          ProjectionHead (64 → 128 → 128)
                                                    │
                                            L2-normalized z
                                                    │
                                               InfoNCE loss

Two views per batch — same source clip, independent augmentation
realisations — give the contrastive pair (z_a, z_b).

Encoder = ``filterbank``, ``feat_extractor``, ``residual_stack``,
``feat_proj`` (lazy). The projection head is discarded at fine-tuning
time; only the encoder weights are saved into ``encoder_state_dict``,
which ``train.py --pretrained`` loads with ``strict=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from model import WhaleVAD
from spectrogram import SpectrogramExtractor


# ======================================================================
# Projection head
# ======================================================================

class ProjectionHead(nn.Module):
    """SimCLR-style 2-layer MLP, L2-normalized output."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ======================================================================
# Forward pass: spec → encoder → mean pool → projection
# ======================================================================

def encode_batch(
    model: WhaleVAD,
    projection: ProjectionHead,
    spec: torch.Tensor,                 # (B, 3, F, T_spec)
) -> torch.Tensor:
    """Encoder + mean pool + projection. Returns L2-normalised ``(B, D_proj)``."""
    x = model.filterbank(spec)
    x = model.feat_extractor(x)
    x = model.residual_stack(x)
    B, C, Fr, T = x.shape
    x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)
    model._init_projection(C * Fr, x.device)
    x = model.feat_proj(x)              # (B, T_feat, 64)
    h = x.mean(dim=1)                   # (B, 64)
    return projection(h)


# ======================================================================
# InfoNCE (SimCLR symmetric)
# ======================================================================

def info_nce_loss(
    z_a: torch.Tensor,                  # (B, D), L2-normalized
    z_b: torch.Tensor,                  # (B, D)
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    SimCLR symmetric InfoNCE.

    For batch size B we have 2B embeddings. The positive of position
    i ∈ [0, B) is at i + B; for i ∈ [B, 2B) the positive is at i - B.
    All other 2B - 2 entries are negatives.
    """
    B, D = z_a.shape
    z = torch.cat([z_a, z_b], dim=0)             # (2B, D)
    sim = z @ z.T / temperature                  # (2B, 2B)
    # Mask out self-similarity along the diagonal.
    self_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(self_mask, float("-inf"))
    targets = torch.cat(
        [torch.arange(B, 2 * B), torch.arange(0, B)], dim=0,
    ).to(z.device)
    loss = F.cross_entropy(sim, targets)

    with torch.no_grad():
        # Top-1 retrieval accuracy: did the model put the true positive
        # at the highest-similarity spot among the 2B - 1 candidates?
        top1 = (sim.argmax(dim=1) == targets).float().mean().item()

    return loss, {"info_nce_loss": loss.item(), "info_nce_top1": top1}


# ======================================================================
# Encoder state-dict extraction
# ======================================================================

ENCODER_PREFIXES = ("filterbank", "feat_extractor", "residual_stack", "feat_proj")


def get_encoder_state_dict(model: WhaleVAD) -> dict[str, torch.Tensor]:
    """
    Filter ``model.state_dict()`` to encoder modules only.

    The classifier and BiLSTM are not pretrained; we omit them from the
    saved checkpoint so ``train.py --pretrained`` doesn't see stale
    randomly-initialized weights for those. ``strict=False`` in train.py
    means the missing keys are tolerated.
    """
    full = model.state_dict()
    return {k: v for k, v in full.items() if k.startswith(ENCODER_PREFIXES)}


# ======================================================================
# Pretraining configuration
# ======================================================================

@dataclass
class PretrainConfig:
    # Data
    train_loader: DataLoader = None         # required at runtime
    spec_extractor: SpectrogramExtractor = None
    view_fn: Callable = None                # (audio, sites, spec_extractor) → spec
    # Training
    n_epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    temperature: float = 0.1
    proj_hidden_dim: int = 128
    proj_out_dim: int = 128
    # Probe
    probe_every: int = 10                   # epochs
    probe_n_epochs: int = 5
    probe_train_loader: Optional[DataLoader] = None
    probe_val_loader: Optional[DataLoader] = None
    # Early stopping (on probe F1)
    probe_patience: int = 3                 # no-improvement probe calls before stop
    # Output
    output_dir: Path = Path("./runs_pretrain")
    save_every: int = 10                    # always-saved "latest" checkpoint
    # Logging
    log_every: int = 50                     # batches
    wandb_log_fn: Optional[Callable] = None # called as wandb_log_fn(metrics, step)
    # Reproducibility
    seed: int = 42


# ======================================================================
# Main pretraining loop
# ======================================================================

def pretrain(
    cfg_pre: PretrainConfig,
    device: torch.device,
) -> dict:
    """
    Run SSL pretraining and return a result summary dict.

    Side effects:
      - Writes ``encoder_best.pt`` (best probe F1) and ``encoder_last.pt``
        to ``cfg_pre.output_dir``.
      - Calls ``cfg_pre.wandb_log_fn(metrics, step)`` after every batch
        if provided.
    """
    cfg_pre.output_dir.mkdir(parents=True, exist_ok=True)

    # ------- Model -------
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)
    # Trigger feat_proj lazy init (so its weights show up in state_dict
    # and so its weights get optimised in this run).
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(cfg_pre.spec_extractor(dummy))

    projection = ProjectionHead(
        in_dim=cfg.PROJECTION_DIM,
        hidden_dim=cfg_pre.proj_hidden_dim,
        out_dim=cfg_pre.proj_out_dim,
    ).to(device)

    # The supervised classifier and BiLSTM are part of `model` but not
    # used in pretraining. We exclude them from the optimiser so they
    # don't drift away from their initialisation; train.py loads the
    # encoder weights and re-initialises lstm/classifier from scratch.
    encoder_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            encoder_params.append(param)
        else:
            param.requires_grad = False
    proj_params = list(projection.parameters())

    n_enc = sum(p.numel() for p in encoder_params)
    n_proj = sum(p.numel() for p in proj_params)
    print(f"[pretrain] encoder params: {n_enc:,}  projection params: {n_proj:,}")

    optimizer = torch.optim.AdamW(
        encoder_params + proj_params,
        lr=cfg_pre.lr,
        weight_decay=cfg_pre.weight_decay,
    )

    # Cosine schedule over total epoch count — typical for SimCLR-style.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_pre.n_epochs, eta_min=cfg_pre.lr * 0.01,
    )

    # ------- State -------
    best_probe_f1 = -1.0
    best_epoch = -1
    no_improve_probes = 0
    history: list[dict] = []
    global_step = 0

    print(f"[pretrain] starting: {cfg_pre.n_epochs} epochs, "
          f"{len(cfg_pre.train_loader)} batches/epoch, "
          f"temperature={cfg_pre.temperature}")

    for epoch in range(1, cfg_pre.n_epochs + 1):
        epoch_t0 = time.monotonic()
        model.train()
        projection.train()

        ep_loss = 0.0
        ep_top1 = 0.0
        ep_n = 0

        for bi, batch in enumerate(cfg_pre.train_loader):
            audio = batch["audio"].to(device, non_blocking=True)
            sites = batch["sites"]

            # Two independent augmentation realisations of the same
            # source audio. The view_fn handles spec extraction.
            spec_a = cfg_pre.view_fn(audio, sites, cfg_pre.spec_extractor)
            spec_b = cfg_pre.view_fn(audio, sites, cfg_pre.spec_extractor)

            z_a = encode_batch(model, projection, spec_a)
            z_b = encode_batch(model, projection, spec_b)

            loss, info = info_nce_loss(z_a, z_b, temperature=cfg_pre.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                encoder_params + proj_params, cfg_pre.grad_clip,
            )
            optimizer.step()

            ep_loss += info["info_nce_loss"]
            ep_top1 += info["info_nce_top1"]
            ep_n += 1
            global_step += 1

            if cfg_pre.wandb_log_fn is not None and (bi % cfg_pre.log_every == 0):
                cfg_pre.wandb_log_fn(
                    {
                        "train/loss": info["info_nce_loss"],
                        "train/top1": info["info_nce_top1"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        scheduler.step()
        epoch_dt = time.monotonic() - epoch_t0
        avg_loss = ep_loss / max(ep_n, 1)
        avg_top1 = ep_top1 / max(ep_n, 1)
        print(f"[pretrain] epoch {epoch:3d}: "
              f"loss={avg_loss:.4f}  top1={avg_top1:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({epoch_dt:.0f}s)")

        rec = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_top1": avg_top1,
            "epoch_time_s": epoch_dt,
        }

        # ------- Linear probe -------
        if (epoch % cfg_pre.probe_every == 0
                and cfg_pre.probe_train_loader is not None
                and cfg_pre.probe_val_loader is not None):
            from linear_probe import run_linear_probe
            probe_t0 = time.monotonic()
            probe_metrics = run_linear_probe(
                model=model,
                spec_extractor=cfg_pre.spec_extractor,
                probe_train_loader=cfg_pre.probe_train_loader,
                probe_val_loader=cfg_pre.probe_val_loader,
                device=device,
                n_epochs=cfg_pre.probe_n_epochs,
            )
            probe_dt = time.monotonic() - probe_t0
            rec.update(probe_metrics)
            rec["probe_time_s"] = probe_dt
            f1 = probe_metrics["probe_macro_f1"]
            per_class = probe_metrics["probe_per_class_f1"]
            print(f"[pretrain]   probe: macro_f1={f1:.4f} "
                  f"per_class={[f'{x:.3f}' for x in per_class]} "
                  f"({probe_dt:.0f}s)")

            if cfg_pre.wandb_log_fn is not None:
                cfg_pre.wandb_log_fn(
                    {
                        "probe/macro_f1": f1,
                        **{f"probe/f1_class{i}": v
                           for i, v in enumerate(per_class)},
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if f1 > best_probe_f1:
                best_probe_f1 = f1
                best_epoch = epoch
                no_improve_probes = 0
                # Save best checkpoint
                ckpt_path = cfg_pre.output_dir / "encoder_best.pt"
                torch.save({
                    "encoder_state_dict": get_encoder_state_dict(model),
                    "projection_state_dict": projection.state_dict(),
                    "epoch": epoch,
                    "probe_macro_f1": f1,
                    "probe_per_class_f1": per_class,
                    "config": vars(cfg_pre) if False else {  # avoid loader dump
                        k: v for k, v in vars(cfg_pre).items()
                        if k not in ("train_loader", "spec_extractor",
                                     "probe_train_loader", "probe_val_loader",
                                     "view_fn", "wandb_log_fn")
                    },
                }, ckpt_path)
                print(f"[pretrain]   new best — saved {ckpt_path}")
            else:
                no_improve_probes += 1
                print(f"[pretrain]   no improvement "
                      f"({no_improve_probes}/{cfg_pre.probe_patience})")
                if no_improve_probes >= cfg_pre.probe_patience:
                    print(f"[pretrain] early stop at epoch {epoch}")
                    history.append(rec)
                    break

        history.append(rec)

        # ------- Periodic "last" checkpoint -------
        if epoch % cfg_pre.save_every == 0:
            ckpt_path = cfg_pre.output_dir / "encoder_last.pt"
            torch.save({
                "encoder_state_dict": get_encoder_state_dict(model),
                "projection_state_dict": projection.state_dict(),
                "epoch": epoch,
            }, ckpt_path)

    # Final "last" save regardless of where the loop ended.
    last_path = cfg_pre.output_dir / "encoder_last.pt"
    torch.save({
        "encoder_state_dict": get_encoder_state_dict(model),
        "projection_state_dict": projection.state_dict(),
        "epoch": history[-1]["epoch"] if history else 0,
    }, last_path)

    summary = {
        "best_probe_f1": best_probe_f1,
        "best_epoch": best_epoch,
        "final_epoch": history[-1]["epoch"] if history else 0,
        "history": history,
        "best_ckpt": str(cfg_pre.output_dir / "encoder_best.pt"),
        "last_ckpt": str(last_path),
    }
    print(f"[pretrain] done. best probe F1 {best_probe_f1:.4f} at epoch {best_epoch}")
    return summary
