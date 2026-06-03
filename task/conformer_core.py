"""
conformer_core
==============

Shared training engine for the Phase 13 Conformer ladder. Each
``train_phase13X.py`` is a thin entry point that calls :func:`run_training`
with its phase-specific knobs (optimiser/schedule, EMA on/off, model builder,
augmentation hook).

Everything about the *recipe and measurement* is reused verbatim from
``train_final`` — data pipeline, per-epoch negative resampling, fixed-threshold
validation, ``macro_paper`` selection, segment-count weighted-BCE, seeding — so
every rung stays a fair, apples-to-apples comparison against the CNN-BiLSTM
final baseline. The only things a rung is allowed to change are the optimiser
schedule (13a), an EMA of the weights (13b), the architecture hyperparameters
(13c), the frontend (13d), and augmentation (13f).

Heavy imports (train_final, dataset_final, wandb) are deferred into
``run_training`` so the lightweight pieces (``EMA``, ``build_optim_sched``) can
be imported and unit-tested without the full data/runtime stack.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

import config_final as cfg
from model_conformer import WhaleVAD_Conformer
from spectrogram_final import SpectrogramExtractor


def build_conformer(device, a):
    """Standard Conformer builder (shared by 13a/13b/13c). ``a`` carries the
    arch hyperparameters (d_model, nhead, layers, ffn_mult, conv_kernel, dropout).
    Runs one dummy forward to materialise the lazy projection."""
    model = WhaleVAD_Conformer(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def add_arch_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Conformer architecture + run bookkeeping args (shared across rungs)."""
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--conv-kernel", type=int, default=31)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40,
                   help="Conformers train longer than the 30-epoch BiLSTM recipe.")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--num-classes", type=int, default=3, choices=[3, 7],
                   help="3-class direct (default, your better setting) or 7-class.")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--dual-res", action="store_true",
                   help="Append a short-window magnitude channel (4-ch input); "
                        "sharpens the D-call downsweep. Off = the 3-ch baseline.")
    p.add_argument("--ema", action="store_true",
                   help="EMA weight averaging (DESED-standard stabiliser): the "
                        "averaged weights are validated and checkpointed. Averages "
                        "over the per-epoch negative-resampling noise.")
    p.add_argument("--filteraug", action="store_true",
                   help="FilterAugment + frequency warping (phase 13f): "
                        "frequency-only augmentation on the magnitude channels; "
                        "leaves phase and the time axis (call structure) untouched.")
    p.add_argument("--fa-db", type=float, default=4.5,
                   help="FilterAugment max per-band gain in dB (default +/-4.5).")
    p.add_argument("--no-freq-warp", action="store_true",
                   help="With --filteraug, disable the frequency-warp component.")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast on the model forward (the STFT stays fp32). "
                        "Halves activation memory so 60 s segments fit at batch 32, "
                        "and speeds training; bf16 needs no GradScaler. Off = fp32.")
    return p


def add_opt_args(p: argparse.ArgumentParser,
                 default_optimizer: str = "sf-radam") -> argparse.ArgumentParser:
    """Optimiser + warmup/decay args (phase 13a lever; inherited by 13b/13c)."""
    p.add_argument("--optimizer", choices=["sf-radam", "radam", "adamw"],
                   default=default_optimizer,
                   help="sf-radam = Schedule-Free RAdam (folds warmup + weight "
                        "averaging; constant LR; ignores --warmup-epochs/--lr-schedule). "
                        "radam / adamw use --warmup-epochs + --lr-schedule (the "
                        "lower-variance Miyazaki-style fallback).")
    p.add_argument("--peak-lr", type=float, default=1e-3,
                   help="Peak/constant LR (vs the BiLSTM recipe's fixed 5e-5). "
                        "Schedule-free wants this set a bit higher than a scheduled "
                        "run; tune up if it underperforms the Step-0 baseline.")
    p.add_argument("--warmup-epochs", type=float, default=1.0,
                   help="Linear warmup epochs (radam/adamw only; ignored by sf-radam).")
    p.add_argument("--lr-schedule", choices=["cosine", "step", "const"], default="cosine",
                   help="Decay shape after warmup (radam/adamw only; ignored by sf-radam).")
    return p


def opt_kwargs_from(a) -> dict:
    return dict(optimizer=a.optimizer, peak_lr=a.peak_lr,
                warmup_epochs=a.warmup_epochs, schedule=a.lr_schedule,
                weight_decay=cfg.WEIGHT_DECAY)


# ======================================================================
# Exponential moving average of weights (eval-time teacher; phase 13b)
# ======================================================================

class EMA:
    """EMA over the model's float tensors (params + BN running stats)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store_and_copy(self, model: nn.Module):
        """Stash current weights and load the EMA weights for evaluation."""
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k in self.shadow:
            msd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Put the stashed training weights back."""
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}


# ======================================================================
# Optimiser + warmup/decay scheduler (phase 13a)
# ======================================================================

def build_optim_sched(model: nn.Module, opt_kwargs: dict,
                      steps_per_epoch: int, total_epochs: int):
    """
    Build ``(optimizer, scheduler, is_schedule_free)`` from ``opt_kwargs``:
      optimizer    : "sf-radam" | "radam" | "adamw"
      peak_lr      : float
      weight_decay : float
      warmup_epochs: float   (linear warmup 0 -> peak; radam/adamw only)
      schedule     : "cosine" | "step" | "const"  (decay after warmup; radam/adamw only)

    For "sf-radam" the scheduler is None (constant LR, averaging instead of a
    schedule) and is_schedule_free is True; the caller must flip the optimizer
    .train()/.eval() around training/validation. For radam/adamw the scheduler is
    a per-step LambdaLR (warmup -> decay) and is_schedule_free is False.
    """
    name = opt_kwargs.get("optimizer", "adamw").lower()
    peak_lr = float(opt_kwargs.get("peak_lr", cfg.LR))
    wd = float(opt_kwargs.get("weight_decay", cfg.WEIGHT_DECAY))
    betas = (cfg.BETA1, cfg.BETA2)

    # Schedule-Free RAdam (phase 13a default): iterate averaging replaces BOTH
    # warmup and the decay schedule. RAdam's variance rectification *is* the
    # warmup, so RAdamScheduleFree has no warmup_steps arg and runs at constant
    # LR. The optimizer must be flipped .train()/.eval() around train/val (the
    # averaged iterate only materialises in eval mode). Returns is_sf=True.
    if name in ("sf-radam", "radam-sf", "schedulefree-radam"):
        from schedulefree import RAdamScheduleFree
        optimizer = RAdamScheduleFree(model.parameters(), lr=peak_lr,
                                      betas=betas, weight_decay=wd)
        return optimizer, None, True

    if name == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=peak_lr,
                                      weight_decay=wd, betas=betas)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                      weight_decay=wd, betas=betas)

    warmup_epochs = float(opt_kwargs.get("warmup_epochs", 0.0))
    schedule = opt_kwargs.get("schedule", "const").lower()
    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        if schedule == "step":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 ** int(min(0.999, progress) * 3)   # x0.1 each third
        return 1.0                                          # const

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler, False


# ======================================================================
# Mask-aware training step (shared by all ladder rungs)
# ======================================================================

def _train_epoch(model, spec_extractor, loader, criterion, optimizer,
                 scheduler, device, ema=None, augment_fn=None, amp=False):
    from tqdm import tqdm

    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        if augment_fn is not None:
            spec = augment_fn(spec)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(spec, key_padding_mask=~mask.bool())
            T = min(logits.size(1), targets.size(1))
            logits, targets_t, mask_t = logits[:, :T], targets[:, :T], mask[:, :T]

            valid = mask_t.unsqueeze(-1).float()
            per_frame = criterion(logits, targets_t) * valid
            loss = per_frame.sum() / (valid.sum() * targets_t.size(-1)).clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


# ======================================================================
# Full training run (parameterised; mirrors train_final's loop)
# ======================================================================

def run_training(phase: str, *, build_model, arch_kwargs, opt_kwargs,
                 epochs: int, seed: int, wandb_mode: str = "online",
                 use_ema: bool = False, ema_decay: float = 0.999,
                 augment_fn=None, extra_tags=()):
    """
    Train one ladder rung end-to-end.

    Parameters
    ----------
    phase : str                  registry key for W&B (e.g. "13a")
    build_model : callable       (device, arch_kwargs) -> (model, spec_extractor)
    arch_kwargs : Namespace/obj  passed through to build_model
    opt_kwargs : dict            see build_optim_sched
    epochs, seed : int
    wandb_mode : str             online | offline | disabled (logging always on)
    use_ema : bool               evaluate/checkpoint an EMA of the weights (13b)
    ema_decay : float
    augment_fn : callable | None spectrogram augmentation hook (13f)
    extra_tags : iterable        extra W&B tags
    """
    # --- deferred heavy imports (kept out of module scope for testability) ---
    import wandb
    import wandb_utils as wbu
    from torch.utils.data import DataLoader
    from dataset_final import (
        load_annotations, get_file_manifest, build_positive_segments,
        build_val_segments, extend_all_segments, WhaleDataset, collate_fn,
    )
    from train_final import (
        seed_everything, seeded_dataloader_kwargs, compute_pos_weight,
        validate, resample_negatives_for_epoch,
    )

    # Dual-resolution input is a global feature flag: set it before building the
    # model or ANY spectrogram extractor so the filterbank's in-channels and the
    # extractor's output channels stay consistent everywhere downstream.
    cfg.DUAL_RESOLUTION = bool(getattr(arch_kwargs, "dual_res", False))

    # EMA + FilterAugment can be driven by the shared add_arch_args flags so any
    # rung enables them without bespoke wiring; an explicitly-passed value wins.
    use_ema = bool(use_ema or getattr(arch_kwargs, "ema", False))
    ema_decay = float(getattr(arch_kwargs, "ema_decay", ema_decay))
    if augment_fn is None and getattr(arch_kwargs, "filteraug", False):
        from filter_augment import build_spec_augment
        augment_fn = build_spec_augment(
            db_range=getattr(arch_kwargs, "fa_db", 4.5),
            freq_warp=not getattr(arch_kwargs, "no_freq_warp", False),
        )
    amp = bool(getattr(arch_kwargs, "amp", False))

    seed = seed_everything(seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase {phase}] device={device}  head={cfg.n_classes()}-class  "
          f"ema={use_ema}  amp={amp}  opt={opt_kwargs}")

    train_sites = list(cfg.TRAIN_DATASETS)
    val_sites = list(cfg.VAL_DATASETS)

    pos_weight, weight_info = compute_pos_weight(train_sites, device, verbose=True)

    # --- W&B (always on) ---
    cfg_log = {
        "phase": phase, "use_ema": use_ema, "ema_decay": ema_decay if use_ema else None,
        "augment": augment_fn is not None, "epochs": epochs, "seed": seed,
        "num_classes": cfg.n_classes(), "use_3class": cfg.USE_3CLASS,
        "batch_size": cfg.BATCH_SIZE, "neg_ratio": cfg.NEG_RATIO,
        "segment_s": cfg.TRAIN_SEGMENT_S, "train_sites": train_sites,
        "val_sites": val_sites, "pos_weight": weight_info["pos_weight"],
        "pos_weight_ratio": weight_info["weight_ratio"],
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        **{f"arch_{k}": v for k, v in vars(arch_kwargs).items()},
    }
    run = wbu.init_phase(phase, config=cfg_log, mode=wandb_mode,
                         extra_tags=list(extra_tags))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase{phase}_{cfg.n_classes()}c_s{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase {phase}] run dir: {run_dir}")

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
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}

    # --- model / loss / optim ---
    model, spec_extractor = build_model(device, arch_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[phase {phase}] parameters: {n_params:,}")
    run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight).to(device)

    steps_per_epoch = max(1, (len(pos_segs) + n_neg) // cfg.BATCH_SIZE)
    optimizer, scheduler, is_sf = build_optim_sched(model, opt_kwargs, steps_per_epoch, epochs)
    if use_ema and is_sf:
        print(f"[phase {phase}] schedule-free optimiser already averages weights; "
              f"ignoring EMA (subsumed).")
        use_ema = False
    ema = EMA(model, decay=ema_decay) if use_ema else None

    # --- training loop ---
    history, best_f1 = [], 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_segments = resample_negatives_for_epoch(
            pos_segs_extended=pos_segs, train_annotations=train_annotations,
            train_manifest=train_manifest, n_neg=n_neg,
            segment_s=cfg.TRAIN_SEGMENT_S, epoch=epoch, verbose=True,
        )
        train_loader = DataLoader(
            WhaleDataset(train_segments), batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
            **seeded_dataloader_kwargs(seed),
        )

        if is_sf:
            optimizer.train()                 # gradient-evaluation point
        train_loss = _train_epoch(model, spec_extractor, train_loader, criterion,
                                  optimizer, scheduler, device, ema, augment_fn, amp)

        # Switch to the weights we select/checkpoint on: the schedule-free
        # averaged iterate, or the EMA shadow (these are mutually exclusive).
        if is_sf:
            optimizer.eval()                  # materialise the averaged iterate
        if ema is not None:
            ema.store_and_copy(model)
        val = validate(model, spec_extractor, val_loader, criterion, device,
                       val_annotations, file_start_dts, threshold=cfg.THRESHOLD)
        eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            ema.restore(model)
        # (schedule-free stays in eval; next epoch's optimizer.train() restores
        #  the gradient-evaluation weights.)

        macro = sum(val["per_class"][c]["f1"] for c in cfg.CALL_TYPES_3) / 3
        improved = val["macro_paper"] > best_f1
        if improved:
            best_f1 = val["macro_paper"]
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"\n[{phase}] Epoch {epoch:2d}/{epochs} ({time.time()-t0:.0f}s)"
              f"{' *** new best' if improved else ''}  lr={cur_lr:.2e}")
        print(f"  Train {train_loss:.4f}  Val {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                  f"F1={pc['f1']:.3f}")
        print(f"    OVERALL {val['f1']:.3f}  MACRO {macro:.3f}  "
              f"MACRO_PAPER {val['macro_paper']:.3f}")

        wbu.log_epoch_3class(epoch, train_loss, val)
        wandb.log({"val/f1_micro": val["f1"], "val/f1_macro_mean": macro,
                   "val/f1_macro_paper": val["macro_paper"], "lr": cur_lr}, step=epoch)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val["loss"], "f1": val["f1"], "macro_f1": macro,
                        "macro_paper_f1": val["macro_paper"],
                        "per_class": val["per_class"]})

        ckpt = {"epoch": epoch, "model_state_dict": eval_state,
                "f1": val["f1"], "macro_paper_f1": val["macro_paper"],
                "ema": use_ema, "history": history,
                "pos_weight": pos_weight.detach().cpu().tolist(),
                "arch_kwargs": vars(arch_kwargs), "opt_kwargs": opt_kwargs}
        torch.save(ckpt, run_dir / f"phase{phase}_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / f"phase{phase}_best.pt")

    # --- summary ---
    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    papers = [h["macro_paper_f1"] for h in history]
    second_half = papers[len(papers) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1]) for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    print(f"\n[{phase}] best macro_paper {max(papers):.3f} "
          f"(micro {max(f1s):.3f}, macro {max(macros):.3f}); "
          f"second-half mean swing {mean_swing:.3f}")

    verdict = (f"Phase {phase}: best macro_paper {max(papers):.3f} "
               f"(micro {max(f1s):.3f}, macro {max(macros):.3f}); "
               f"second-half swing {mean_swing:.3f}.")
    wbu.finalize_phase(history, verdict=verdict, best_ckpt=run_dir / f"phase{phase}_best.pt")
    return history
