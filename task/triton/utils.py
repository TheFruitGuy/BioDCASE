"""
Triton — Pipeline utilities
===========================

Small helpers used by ``train.py`` and the model files. Deliberately kept
free of wandb dependencies — wandb glue lives in ``wandb_utils.py``.

Contents
--------
- ``seed_everything``: deterministic seeding across Python/NumPy/PyTorch
- ``seeded_dataloader_kwargs``: reproducible DataLoader shuffle/workers
- ``align_lengths``: reconcile small (B, T_m) vs (B, T_t) mismatches
  between model output and frame-level targets
- ``log_param_count``: pretty-print trainable parameter count
- ``unwrap``: get the underlying ``nn.Module`` from a ``nn.DataParallel`` wrap
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA) so a training script
    produces the same F1 trajectory across runs.

    Call this once at the top of ``main()``, before any model, dataset,
    or DataLoader is constructed — DataLoader workers inherit RNG state
    at construction time, so order matters.

    Parameters
    ----------
    seed : int
        Master seed.
    deterministic : bool
        If True, forces cuDNN into deterministic mode. Makes runs
        bit-identical at the cost of ~10–30% throughput. Leave False
        for development; turn on for the final-report run.

    Returns
    -------
    int
        The seed that was set, so it can be stuffed straight into a
        wandb config: ``seed = seed_everything(42)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seeded_dataloader_kwargs(seed: int) -> dict:
    """
    Return kwargs for ``DataLoader(...)`` that make shuffle order and
    worker RNG state reproducible.

    Usage
    -----
    ::

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True,
            **seeded_dataloader_kwargs(seed),
        )
    """
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {"generator": g, "worker_init_fn": _worker_init}


# ----------------------------------------------------------------------
# Shape reconciliation
# ----------------------------------------------------------------------

def align_lengths(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconcile small length mismatches between model output and targets.

    The CNN front end can produce one or two fewer/more frames than the
    naive ``n_samples // hop_length`` count, depending on padding and
    pooling boundary effects. This helper truncates or zero-pads
    ``targets`` and ``mask`` to match ``logits`` so the loss is
    computed unchanged.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, T_m, C)
    targets : torch.Tensor, shape (B, T_t, C)
    mask : torch.Tensor, shape (B, T_t), dtype=bool

    Returns
    -------
    targets, mask : torch.Tensor
        Both with time dim equal to ``T_m``.
    """
    T_m, T_t = logits.size(1), targets.size(1)
    if T_m < T_t:
        targets = targets[:, :T_m, :]
        mask = mask[:, :T_m]
    elif T_m > T_t:
        pad_t = torch.zeros(
            targets.size(0), T_m - T_t, targets.size(2),
            device=targets.device,
        )
        targets = torch.cat([targets, pad_t], dim=1)
        pad_m = torch.zeros(
            mask.size(0), T_m - T_t, dtype=torch.bool, device=mask.device,
        )
        mask = torch.cat([mask, pad_m], dim=1)
    return targets, mask


# ----------------------------------------------------------------------
# Small conveniences
# ----------------------------------------------------------------------

def log_param_count(model: nn.Module, prefix: str = "") -> int:
    """
    Print the trainable parameter count and return it.

    Useful for sanity-checking that pretrained-freeze schedules and
    optimizer-only-some-params tricks are actually doing what they
    claim.
    """
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"{prefix}Parameters" if prefix else "Parameters"
    print(f"{label}: {n:,}")
    return n


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped in ``nn.DataParallel``."""
    return model.module if isinstance(model, nn.DataParallel) else model
