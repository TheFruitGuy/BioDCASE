"""
Linear Probe for SSL Encoder Quality
====================================

Periodic feature-quality evaluator called every ``probe_every`` epochs
during pretraining. The probe:

    1. Freezes the encoder
    2. Trains a single linear layer (encoder_dim → n_classes) on a small
       subset of labeled training segments for a few epochs
    3. Reports macro frame-level F1 on a disjoint held-out subset

Frame-level F1 (not event-level) is intentional: it's an order of
magnitude faster, doesn't require post-processing or threshold tuning,
and is sufficient as a "is the encoder learning anything useful" signal.
The honest cross-site generalization signal comes later, during
fine-tuning via ``train.py --pretrained``.

Probe data
----------
Disjoint random split of files from ``cfg.TRAIN_DATASETS``:
    - probe-train: 8% of training files
    - probe-val:   2% of training files

Both subsets are sampled once at start of pretraining and reused for
every probe call, so probe F1 across epochs is comparable.

Why probe data overlaps with pretraining data
---------------------------------------------
The probe asks "are the encoder's features linearly separable for the
supervised task?" — a question about feature geometry, not
generalization. Using overlap data is correct for that question.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config as cfg


# ======================================================================
# Probe data construction
# ======================================================================

def build_probe_loaders(
    probe_train_frac: float = 0.08,
    probe_val_frac: float = 0.02,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build small probe-train / probe-val DataLoaders from labeled
    training data.

    We reuse the existing supervised data pipeline (``dataset.py``) for
    its segment construction, then take a small disjoint random subset
    of the produced segment list. This way the probe sees the same
    spectrogram/target shapes as the supervised model.
    """
    # Local imports to keep this module's import cost low when not in use.
    from dataset import (
        get_file_manifest, load_annotations,
        build_val_segments, WhaleDataset, collate_fn,
    )

    print("[probe] building probe loaders")
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    annotations = load_annotations(cfg.TRAIN_DATASETS, manifest=manifest)

    # We use *validation-style* fixed-length segments for both probe
    # train and val so behaviour is deterministic and comparable across
    # probe calls. The supervised positive/negative builder samples
    # collars stochastically per epoch, which would add noise we don't
    # want in the probe signal.
    segments = build_val_segments(manifest, annotations)
    print(f"[probe]   {len(segments)} candidate segments")

    rng = random.Random(seed)
    indices = list(range(len(segments)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * probe_train_frac)
    n_val = int(n_total * probe_val_frac)
    if n_train < 50 or n_val < 20:
        raise RuntimeError(
            f"Probe split too small: n_train={n_train}, n_val={n_val}. "
            "Increase probe_train_frac / probe_val_frac."
        )

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    full_ds = WhaleDataset(segments)
    probe_train_ds = Subset(full_ds, train_idx)
    probe_val_ds = Subset(full_ds, val_idx)

    probe_train_loader = DataLoader(
        probe_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    probe_val_loader = DataLoader(
        probe_val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"[probe]   probe_train: {len(probe_train_ds)}, "
          f"probe_val: {len(probe_val_ds)}")
    return probe_train_loader, probe_val_loader


# ======================================================================
# Encoder feature extraction (matches the SSL forward pipeline exactly)
# ======================================================================

@torch.no_grad()
def encoder_forward(model, spec_extractor, audio: torch.Tensor) -> torch.Tensor:
    """
    Run audio → encoder → per-frame features ``(B, T_frames, 64)``.

    Matches the SSL forward path used by ``pretrain_core``, so probe
    features come from exactly the representation being trained.
    """
    spec = spec_extractor(audio)
    x = model.filterbank(spec)
    x = model.feat_extractor(x)
    x = model.residual_stack(x)
    B, C, Fr, T = x.shape
    x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)
    model._init_projection(C * Fr, x.device)
    x = model.feat_proj(x)             # (B, T, 64)
    return x


# ======================================================================
# Probe training and evaluation
# ======================================================================

def _frame_macro_f1(
    probs: torch.Tensor,                 # (N_frames, C), float
    targets: torch.Tensor,               # (N_frames, C), 0/1
    threshold: float = 0.5,
) -> tuple[float, list[float]]:
    """Macro-averaged frame-level F1 across classes."""
    preds = (probs >= threshold).float()
    f1s: list[float] = []
    eps = 1e-8
    for c in range(targets.size(-1)):
        p, t = preds[:, c], targets[:, c]
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(float(f1))
    return float(np.mean(f1s)), f1s


def run_linear_probe(
    model,
    spec_extractor,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 5,
    lr: float = 1e-3,
    n_classes: Optional[int] = None,
) -> dict:
    """
    Train a fresh linear head on top of the (frozen) encoder for
    ``n_epochs`` epochs, then return macro frame-F1 on probe-val.

    The encoder is set to eval mode (frozen BatchNorm stats) for the
    probe so consecutive probes are directly comparable.
    """
    if n_classes is None:
        n_classes = cfg.n_classes()

    # Freeze encoder, eval mode (stops BN running stats from drifting).
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    head = nn.Linear(cfg.PROJECTION_DIM, n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    # ----------- train --------------
    head.train()
    for epoch in range(1, n_epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        for audio, targets, mask, _ in probe_train_loader:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)            # (B, T_target, C)
            mask = mask.to(device, non_blocking=True)                  # (B, T_target)

            with torch.no_grad():
                feats = encoder_forward(model, spec_extractor, audio)  # (B, T_feat, 64)

            logits = head(feats)                                       # (B, T_feat, C)

            # Align T_feat vs T_target: take min and crop both.
            T = min(logits.size(1), targets.size(1))
            logits = logits[:, :T, :]
            tgt = targets[:, :T, :]
            m = mask[:, :T].unsqueeze(-1).float()

            loss_el = F.binary_cross_entropy_with_logits(
                logits, tgt, reduction="none",
            )
            loss = (loss_el * m).sum() / (m.sum() * n_classes + 1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            n_batches += 1
        # Optional: print probe-train loss for visibility
        # print(f"[probe] epoch {epoch}: loss={ep_loss / max(n_batches, 1):.4f}")

    # ----------- eval --------------
    head.eval()
    all_probs, all_tgts = [], []
    with torch.no_grad():
        for audio, targets, mask, _ in probe_val_loader:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            feats = encoder_forward(model, spec_extractor, audio)
            logits = head(feats)

            T = min(logits.size(1), targets.size(1))
            logits = logits[:, :T, :]
            tgt = targets[:, :T, :]
            m = mask[:, :T]

            # Flatten valid frames only
            valid = m.bool().reshape(-1)
            probs = torch.sigmoid(logits).reshape(-1, n_classes)[valid].cpu()
            tgt_flat = tgt.reshape(-1, n_classes)[valid].cpu()

            all_probs.append(probs)
            all_tgts.append(tgt_flat)

    probs_cat = torch.cat(all_probs, dim=0)
    tgts_cat = torch.cat(all_tgts, dim=0)
    macro_f1, per_class = _frame_macro_f1(probs_cat, tgts_cat, threshold=0.5)

    # Re-enable grads on encoder for the next pretraining step.
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    return {
        "probe_macro_f1": macro_f1,
        "probe_per_class_f1": per_class,
        "probe_n_val_frames": int(probs_cat.size(0)),
    }
