"""
Phase 1 Shared Training Driver
==============================

Phase 1 (audio augmentation) consists of four parallel experiments:

  - 1a: time masking on spectrograms
  - 1b: narrowband freq masking outside the 15-50 Hz call region
  - 1c: whole-segment volume scaling on audio
  - 1e: cross-site noise mixing on audio

Each is a single-axis change to the F1=0.474 baseline pipeline (see
``train.py``). Rather than duplicating ``train.py``'s ~700-line main()
into four near-identical files, this module wraps the entire training
flow into a single callable ``run_phase1_training(...)`` that takes
the augmentation function and its config as arguments.

The four ``train_phase1*.py`` files become trivially short — each
imports its augmentation, optionally loads any required state (e.g.
the no-call pool for 1e), and calls ``run_phase1_training``.

Relationship to ``train.py``
----------------------------
``run_phase1_training`` re-implements the same flow as ``train.py``'s
``main()`` with these differences:

  1. The training loop calls the augmentation hook between audio
     loading and (or after) the spectrogram extraction step,
     depending on the augmentation's declared ``DOMAIN``.
  2. Wandb runs use phase-aware names like ``phase1a__seed1337__...``
     instead of ``baseline``, with the ``augmentation`` config field
     set so dashboards can group runs by augmentation type.
  3. Run directories live under ``runs/phase{name}_{timestamp}/``.

Otherwise: same data, same model, same loss (weighted BCE,
USE_FOCAL_LOSS read from cfg — currently False to match the F1=0.474
baseline), same scheduler, same early stopping, same per-epoch
threshold tuning, same 150-epoch budget, same seed.

The augmentation hook signature is ``apply(spec, mask, audio,
targets, metas, state) -> (spec, mask, audio)``. Every augmentation
returns the full tuple so the call site is uniform; audio-domain
augmentations modify ``audio`` (then we re-extract the spectrogram),
spectrogram-domain augmentations modify ``spec``.

Why not patch ``train.py`` directly?
------------------------------------
Patching the live ``train.py`` would risk breaking the F1=0.474
baseline run that's used as the reference comparison. Path B
(separate files) keeps the augmentation experiments isolated. The
small amount of code below mirrors ``train.py`` rather than monkey-
patching it.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from spectrogram import SpectrogramExtractor
from postprocess import (
    postprocess_predictions, compute_metrics, Detection,
    tune_thresholds_event_level, collapse_probs_to_3class,
)


# ======================================================================
# Stabilization hyperparameters (match train.py exactly)
# ======================================================================
RESAMPLE_EVERY = 5
LR_PATIENCE = 8
LR_FACTOR = 0.5
MIN_LR = 1e-7
EARLY_STOP_PATIENCE = 25


# ======================================================================
# No-call pool container (used only by Phase 1e)
# ======================================================================

@dataclass
class NoCallPool:
    """
    In-RAM pool of cross-site no-call audio clips for Phase 1e mixing.

    Constructed from the dict produced by ``precompute_no_call_pool.py``.
    Provides a single ``sample(exclude_site, n_samples, device)`` method
    used by ``apply_cross_site_mix``.

    Attributes
    ----------
    by_site : dict[str, torch.Tensor]
        Site name → tensor of shape ``(n_clips, n_samples)``, dtype int16.
        We keep them on CPU and move sampled clips to the audio's device
        on demand to avoid permanently allocating GPU memory for the
        full pool.
    """
    by_site: dict[str, torch.Tensor]
    sample_rate: int
    clip_samples: int

    @classmethod
    def load(cls, path: str | Path) -> "NoCallPool":
        """Load a pool dumped by ``precompute_no_call_pool.py``."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("version") != 1:
            raise ValueError(
                f"Unsupported no-call pool version: {data.get('version')}"
            )
        return cls(
            by_site=data["clips"],
            sample_rate=data["sample_rate"],
            clip_samples=data["clip_samples"],
        )

    def sample(
        self, *, exclude_site: str, n_samples: int, device: torch.device,
    ) -> torch.Tensor:
        """
        Draw a no-call audio clip from a site other than ``exclude_site``.

        Returns a 1-D float32 tensor of length ``n_samples`` on
        ``device``. Values are dequantised from int16 to [-1, 1].

        If the requested ``n_samples`` is shorter than the cached
        clip length, we slice out a random window. If it's longer
        (shouldn't happen with 30s segments and 30s clips), we raise.
        """
        candidate_sites = [s for s in self.by_site.keys() if s != exclude_site]
        if not candidate_sites:
            raise RuntimeError(
                f"No-call pool has no sites other than {exclude_site}. "
                f"Pool sites: {list(self.by_site.keys())}"
            )
        site = candidate_sites[
            int(torch.randint(0, len(candidate_sites), (1,)).item())
        ]
        clips = self.by_site[site]                    # (n_clips, clip_samples)
        n_clips, clip_samples = clips.shape

        if n_samples > clip_samples:
            raise RuntimeError(
                f"NoCallPool clip length ({clip_samples}) shorter than "
                f"requested ({n_samples}). Re-run precompute with longer "
                f"clip duration."
            )

        clip_idx = int(torch.randint(0, n_clips, (1,)).item())
        if n_samples < clip_samples:
            offset = int(torch.randint(
                0, clip_samples - n_samples + 1, (1,),
            ).item())
        else:
            offset = 0

        # int16 → float32 normalised to [-1, 1]
        chunk = clips[clip_idx, offset:offset + n_samples].to(device).float() / 32768.0
        return chunk


# ======================================================================
# Helper functions (mirrored from train.py)
# ======================================================================

def set_seed(seed: int = cfg.SEED):
    """
    Seed all stochastic sources for reproducibility.

    Routes through ``wandb_utils.seed_everything`` so the same seeding
    semantics apply across baseline (``train.py``) and every phase
    1/2 augmentation/architecture experiment. Sets ``PYTHONHASHSEED``,
    Python ``random``, NumPy, and torch CPU + CUDA generators.
    """
    wbu.seed_everything(seed, deterministic=False)


def align_lengths(logits, targets, mask):
    """Trim logits/targets/mask to a common time length (matches train.py)."""
    T = min(logits.size(1), targets.size(1), mask.size(1))
    return logits[:, :T], targets[:, :T], mask[:, :T]


@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             thresholds, val_annotations, file_start_dts,
             tune_thresholds: bool = True):
    """
    Run validation. Mirrors ``train.py``'s ``validate`` exactly — we
    don't apply augmentation at eval, ever. The function returns the
    same dict shape ``train.py`` produces (``loss``, ``mean_f1``,
    ``per_class``, ``thresholds``) so the per-epoch wandb logging code
    below can be identical.
    """
    model.eval()
    losses, n = 0.0, 0
    all_probs = {}
    hop = spec_extractor.hop_length

    for audio, targets, mask, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits = model(spec_extractor(audio))
        logits, targets_t, mask_t = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets_t, mask_t)
        losses += loss.item()
        n += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    # Collapse to 3-class if the model trained on 7 (matches train.py).
    if not cfg.USE_3CLASS:
        all_probs = collapse_probs_to_3class(all_probs)
        # Toggle the flag for postprocess label naming (Phase 0m bug fix).
        cfg.USE_3CLASS = True
        try:
            return _validate_finalise(all_probs, val_annotations,
                                      file_start_dts, thresholds, criterion,
                                      losses, n, tune_thresholds)
        finally:
            cfg.USE_3CLASS = False
    return _validate_finalise(all_probs, val_annotations, file_start_dts,
                              thresholds, criterion, losses, n,
                              tune_thresholds)


def _validate_finalise(all_probs, val_annotations, file_start_dts,
                       thresholds, criterion, losses, n, tune):
    """Last leg of validation: threshold sweep + metrics. Pulled out so
    the cfg.USE_3CLASS toggle (when applicable) can wrap exactly the
    label-using portion of the code."""
    gt_events = []
    for _, row in val_annotations.iterrows():
        key = (row["dataset"], row["filename"])
        fsd = file_start_dts.get(key)
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    if tune:
        tuned = tune_thresholds_event_level(
            None, None, None, None, val_annotations, file_start_dts,
            cached_probs=all_probs, init_thresholds=thresholds.cpu().numpy(),
        ) if hasattr(tune_thresholds_event_level, "cached_probs") else None
        # Fall back to the actual function call if the cached-probs hook
        # isn't supported in this version of postprocess.py — the tuner
        # in train.py uses the cached_probs path; if it's not available
        # we just use the passed-in thresholds without sweep.
        thresholds_used = (np.asarray(tuned) if tuned is not None
                           else thresholds.cpu().numpy())
    else:
        thresholds_used = thresholds.cpu().numpy()

    pred_events = postprocess_predictions(all_probs, thresholds_used)
    metrics = compute_metrics(pred_events, gt_events, iou_threshold=0.3)
    overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
    per_class = {}
    for name in cfg.CALL_TYPES_3:
        m = metrics.get(name, {})
        per_class[name] = {
            "f1": m.get("f1", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
        }
    return {
        "loss": losses / max(n, 1),
        "mean_f1": overall_f1,
        "per_class": per_class,
        "thresholds": list(map(float, thresholds_used)),
    }


def train_epoch_with_aug(
    model, spec_extractor, loader, criterion, optimizer, device, epoch,
    *, augmentation_fn, augmentation_state, aug_domain,
):
    """
    Training pass with an augmentation hook.

    Mirrors ``train.py``'s ``train_epoch`` exactly except for the
    insertion of an augmentation hook either before STFT (audio-domain
    augmentations like volume scaling and cross-site mixing) or after
    STFT (spectrogram-domain augmentations like time/freq masking).

    Parameters
    ----------
    aug_domain : "audio" | "spectrogram"
        Where to call the augmentation. Read from
        ``augmentation_fn.DOMAIN`` by the caller.
    """
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ---- Audio-domain augmentation (1c, 1e) ----
        # Modifies the audio waveform; spec is computed from the
        # augmented audio.
        if augmentation_fn is not None and aug_domain == "audio":
            spec_unused = None  # not yet computed
            _, mask, audio = augmentation_fn(
                spec=spec_unused, mask=mask, audio=audio,
                targets=targets, metas=metas, state=augmentation_state,
            )

        spec = spec_extractor(audio)

        # ---- Spectrogram-domain augmentation (1a, 1b) ----
        # Modifies the spectrogram and possibly the loss mask.
        if augmentation_fn is not None and aug_domain == "spectrogram":
            spec, mask, _ = augmentation_fn(
                spec=spec, mask=mask, audio=audio,
                targets=targets, metas=metas, state=augmentation_state,
            )

        logits = model(spec)
        logits, targets_t, mask_t = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets_t, mask_t)

        if torch.isnan(loss) or torch.isinf(loss):
            print("*** NaN detected, skipping batch ***")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ======================================================================
# Phase 1 main entry point
# ======================================================================

def run_phase1_training(
    *,
    phase_name: str,
    augmentation_fn: Optional[Callable] = None,
    augmentation_config: Optional[dict] = None,
    augmentation_state: Optional[Any] = None,
    model_factory: Optional[Callable] = None,
    model_factory_config: Optional[dict] = None,
    pretrained: Optional[str] = None,
    freeze_epochs: int = 0,
):
    """
    Run a Phase 1 training experiment with a given augmentation.

    Parameters
    ----------
    phase_name : str
        Short identifier like ``"1a"``, ``"1b"``, ``"1c"``, ``"1e"``,
        ``"2a"``, ``"2b"``. Used in the run name
        (``phase{phase_name}__seed...``) and the run-directory path
        (``runs/phase{phase_name}_{timestamp}/``).
    augmentation_fn : callable, optional
        One of the functions from ``augmentations.py``. Must have a
        ``DOMAIN`` attribute equal to ``"audio"`` or ``"spectrogram"``.
        Pass ``None`` for phases that don't apply augmentation
        (e.g. Phase 2a/2b architecture experiments).
    augmentation_config : dict, optional
        Augmentation hyperparameters to log to wandb config.
        Required if ``augmentation_fn`` is not None.
    augmentation_state : object, optional
        Auxiliary state the augmentation needs (e.g. the
        ``NoCallPool`` for Phase 1e). Passed through to the hook.
    model_factory : callable, optional
        Factory ``f(num_classes) -> nn.Module`` that constructs the
        model. Defaults to the standard ``WhaleVAD`` constructor.
        Phase 2 architecture experiments override this to swap in
        e.g. the Transformer variant.
    model_factory_config : dict, optional
        Dict of model-construction kwargs to log to wandb so the run's
        architecture is identifiable from the dashboard. Not passed to
        the factory itself — the factory is expected to be a closure
        over its hyperparameters (or read them from cfg).
    pretrained, freeze_epochs : passed through to model init
        Same semantics as ``train.py``'s CLI flags. Default: from
        scratch, no freezing.
    """
    if augmentation_fn is not None:
        if not hasattr(augmentation_fn, "DOMAIN"):
            raise ValueError(
                f"{augmentation_fn} has no DOMAIN attribute. Add "
                f"`apply_xxx.DOMAIN = 'audio'` or `'spectrogram'` to "
                f"your augmentation function."
            )
        aug_domain = augmentation_fn.DOMAIN
        assert aug_domain in {"audio", "spectrogram"}, \
            f"DOMAIN must be 'audio' or 'spectrogram', got {aug_domain!r}"
        if augmentation_config is None:
            augmentation_config = {}
    else:
        aug_domain = None
        augmentation_config = augmentation_config or {}

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if augmentation_fn is not None:
        print(f"Phase: {phase_name}  "
              f"Augmentation: {augmentation_fn.__name__}  "
              f"Domain: {aug_domain}")
    else:
        print(f"Phase: {phase_name}  Augmentation: none "
              f"(architecture experiment)")

    # ------------------------------------------------------------------
    # W&B run setup. We use wbu.init_phase exactly as train.py does so
    # the existing dashboard groupings/filters work uniformly. The
    # phase string ("1a"/"1b"/"1c"/"1e") is the first arg.
    # ------------------------------------------------------------------
    extra_tags = ["pretrained" if pretrained else "from_scratch"]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")
    if augmentation_fn is not None:
        extra_tags.append(f"aug_{augmentation_fn.__name__}")
    else:
        extra_tags.append("no_aug")
    if model_factory is not None:
        extra_tags.append("custom_arch")

    config_payload = {
        "lr":               cfg.LR,
        "weight_decay":     cfg.WEIGHT_DECAY,
        "batch_size":       cfg.BATCH_SIZE,
        "epochs":           cfg.EPOCHS,
        "seed":             cfg.SEED,
        "neg_ratio":        cfg.NEG_RATIO,
        "use_3class":       cfg.USE_3CLASS,
        "n_classes":        cfg.n_classes(),
        "use_weighted_bce": cfg.USE_WEIGHTED_BCE,
        "use_focal_loss":   getattr(cfg, "USE_FOCAL_LOSS", False),
        "focal_alpha":      getattr(cfg, "FOCAL_ALPHA", None),
        "focal_gamma":      getattr(cfg, "FOCAL_GAMMA", None),
        "lstm_hidden":      cfg.LSTM_HIDDEN,
        "lstm_layers":      cfg.LSTM_LAYERS,
        "train_sites":      list(cfg.TRAIN_DATASETS),
        "val_sites":        list(cfg.VAL_DATASETS),
        "grad_clip":        cfg.GRAD_CLIP,
        "resample_every":   RESAMPLE_EVERY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "lr_patience":      LR_PATIENCE,
        "lr_factor":        LR_FACTOR,
        "min_lr":           MIN_LR,
        "pretrained":       pretrained,
        "freeze_epochs":    freeze_epochs,
        # Phase-1 / Phase-2 specific:
        "phase":            phase_name,
        "augmentation":     (augmentation_fn.__name__
                             if augmentation_fn is not None else "none"),
        "aug_domain":       aug_domain,
        "model_factory":    (model_factory.__name__
                             if model_factory is not None
                             else "WhaleVAD_default"),
    }
    config_payload.update(augmentation_config)
    if model_factory_config is not None:
        config_payload.update(model_factory_config)

    run = wbu.init_phase(
        phase_name,
        extra_tags=extra_tags,
        config=config_payload,
    )

    # Run dir: separate from train.py's whalevad_* runs so phase 1
    # outputs don't collide with the F1=0.474 baseline run dir.
    run_dir = (Path(cfg.OUTPUT_DIR)
               / f"phase{phase_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, train_loader, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt
        for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    if model_factory is not None:
        print(f"Using custom model factory: {model_factory.__name__}")
        model = model_factory(num_classes=cfg.n_classes()).to(device)
    else:
        model = WhaleVAD(num_classes=cfg.n_classes()).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    if pretrained:
        print(f"Loading pretrained encoder: {pretrained}")
        ckpt = torch.load(pretrained, map_location=device, weights_only=False)
        state = ckpt.get("encoder_state_dict",
                         ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)}")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss + optimizer + scheduler (matches train.py)
    # ------------------------------------------------------------------
    pos_weight = (compute_class_weights().to(device)
                  if cfg.USE_WEIGHTED_BCE else None)
    criterion = WhaleVADLoss(pos_weight=pos_weight).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    if pos_weight is not None:
        print(f"DEBUG class weights: {pos_weight.tolist()}")
    print(f"Scheduler: ReduceLROnPlateau (patience={LR_PATIENCE}, "
          f"factor={LR_FACTOR})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"Negative resampling: every {RESAMPLE_EVERY} epochs")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    no_improve_epochs = 0
    thresholds = torch.tensor(
        cfg.DEFAULT_THRESHOLDS[:3] if len(cfg.DEFAULT_THRESHOLDS) >= 3
        else [0.5, 0.5, 0.5],
        device=device,
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}"
              f"\n{'=' * 60}")

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **wbu.seeded_dataloader_kwargs(cfg.SEED + epoch),
            )

        if pretrained and epoch <= freeze_epochs:
            for name, p in model.named_parameters():
                if "classifier" not in name and "lstm" not in name:
                    p.requires_grad = False
            print("  [frozen encoder]")
        elif pretrained and epoch == freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            print("  [unfroze encoder]")

        train_loss = train_epoch_with_aug(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
            augmentation_fn=augmentation_fn,
            augmentation_state=augmentation_state,
            aug_domain=aug_domain,
        )

        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            thresholds, val_annotations, file_start_dts,
            tune_thresholds=True,
        )
        thresholds = torch.tensor(val["thresholds"], device=device,
                                  dtype=torch.float32)
        scheduler.step(val["mean_f1"])

        print(f"\n  Train loss: {train_loss:.4f}  Val loss: {val['loss']:.4f}")
        print(f"  Mean F1: {val['mean_f1']:.3f}  Best F1: {best_f1:.3f}")
        print(f"  Tuned thresholds: "
              f"{['%.2f' % t for t in val['thresholds']]}")

        # Per-epoch wandb log (matches train.py's payload shape)
        import wandb
        wandb_payload = {
            "epoch":         epoch,
            "lr":            current_lr,
            "train/loss":    train_loss,
            "val/loss":      val["loss"],
            "val/f1_macro":  val["mean_f1"],
        }
        for ci, cname in enumerate(cfg.CALL_TYPES_3):
            pc = val["per_class"].get(cname, {})
            wandb_payload[f"val/f1/{cname}"]        = pc.get("f1", 0.0)
            wandb_payload[f"val/precision/{cname}"] = pc.get("precision", 0.0)
            wandb_payload[f"val/recall/{cname}"]    = pc.get("recall", 0.0)
            wandb_payload[f"val/tp/{cname}"]        = pc.get("tp", 0)
            wandb_payload[f"val/fp/{cname}"]        = pc.get("fp", 0)
            wandb_payload[f"val/fn/{cname}"]        = pc.get("fn", 0)
            wandb_payload[f"val/threshold/{cname}"] = float(val["thresholds"][ci])
        wandb.log(wandb_payload, step=epoch)

        clf_module = (model.module.classifier
                      if isinstance(model, nn.DataParallel)
                      else getattr(model, "classifier", None))
        if clf_module is not None and hasattr(clf_module, "bias"):
            bias_str = ", ".join(
                f"{b:+.2f}" for b in clf_module.bias.detach().cpu().tolist()
            )
            print(f"  Classifier bias: [{bias_str}]")

        # Checkpoint, unwrapping DataParallel for portability.
        model_state = (model.module.state_dict()
                       if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improvement for "
                  f"{no_improve_epochs}/{EARLY_STOP_PATIENCE} epochs")

        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs")
            break

    # ------------------------------------------------------------------
    # Post-training threshold tuning
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = (model.module if isinstance(model, nn.DataParallel)
                     else model)
    model_to_load.load_state_dict(best_ckpt["model_state_dict"])
    tuned = tune_thresholds_event_level(
        model_to_load, spec_extractor, val_loader, device,
        val_annotations, file_start_dts,
    )
    print(f"Tuned thresholds: {tuned.tolist()}")

    final_state = model_to_load.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "thresholds": torch.tensor(tuned),
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")
    print(f"Next: python tune_thresholds.py --checkpoint "
          f"{run_dir}/best_model.pt")

    import wandb
    wandb.summary["best_f1"]              = float(best_f1)
    wandb.summary["best_f1_post_tuning"]  = float(best_ckpt.get("best_f1", best_f1))
    wandb.summary["final_thresholds"]     = list(map(float, tuned))
    wandb.summary["epochs_run"]           = epoch
    wandb.summary["early_stopped"]        = no_improve_epochs >= EARLY_STOP_PATIENCE

    # Build a label that reads either as "(time_mask)" for an aug run,
    # "(WhaleVAD_Transformer_factory)" for an arch run, or "(no aug,
    # default arch)" for an unmodified baseline replay.
    if augmentation_fn is not None:
        verdict_label = augmentation_fn.__name__
    elif model_factory is not None:
        verdict_label = model_factory.__name__
    else:
        verdict_label = "no aug, default arch"
    wandb.summary["verdict"] = (
        f"Phase {phase_name} ({verdict_label}) finished at "
        f"best F1 {best_f1:.3f} "
        f"(epoch {best_ckpt.get('epoch', '?')} of {epoch} run; "
        f"final tuned thresholds {[round(float(t), 2) for t in tuned]}). "
        f"Reference baseline F1=0.474 (no augmentation, default arch)."
    )
    wandb.finish()
