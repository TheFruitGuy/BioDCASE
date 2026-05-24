"""
Phase 11a: Two-Stream PCEN Frontend (de-risk before LEAF)
=========================================================

Single-axis change to the F1=0.474 baseline (``train.py``):
replace the magnitude channel ``mag = |STFT|`` with TWO PCEN streams at
different time constants. Phase channels (cos/sin) are unchanged.

    Baseline input  (B, 3, F, T)  = [|STFT|,     cos_ph, sin_ph]
    Phase 11a input (B, 4, F, T)  = [PCEN_fast,  PCEN_slow, cos_ph, sin_ph]

What this tests
---------------
Per-band AGC as a frontend prior. Magnitude-STFT hands the CNN raw energy
with sustained ambient (ship, ice, swell) often 20-40 dB above call
energy, forcing the learned frontend to implicitly solve the noise-
normalization problem with conv layers. PCEN solves it analytically via
the IIR-smoothed division:

    PCEN(t,f) = ( mag(t,f) / (eps + M(t,f))^alpha + delta )^r - delta^r

where M is a 1st-order IIR smoothing of mag with time constant T.

Why two streams
---------------
Per-class duration P95 in our training data:
    BMABZ  12.89 s   (long tonal calls)
    D       4.18 s   (short downsweeps)
    BP      4.04 s   (short pulsed)

A single T has to satisfy the longest class without ducking it. T_slow=25s
preserves BMABZ (~D/T=0.3 at median). T_fast=5s preserves D and BP and
sharpens their attack against shorter-timescale ambient. Concatenating
both as input channels lets the CNN learn which stream matters per class.

Implementation notes
--------------------
- IIR uses ``torchaudio.functional.lfilter`` with an analytical warm-start
  correction: ``M_correct[n] = M_lfilter[n] + y0 * (1-s)^(n+1)`` where
  ``y0`` is the per-band mean of the first 1 second. Without this,
  lfilter's implicit zero initial state means T_slow=25s on a 30s segment
  only reaches 70% of true equilibrium by the last frame.
- PCEN params (alpha=0.98, delta=2.0, r=0.5) from Wang et al. 2017 are
  fixed in this experiment. Pure prior test: if PCEN moves F1 outside the
  seed band, this is evidence for the frontend-normalization bottleneck;
  follow-up phase can make these learnable.
- Model is constructed with ``feat_channels=4`` to accept the wider input.
- Demean step (``stft - stft.mean(time)``) is preserved for parity with
  the baseline's phase channels; PCEN runs on the demeaned magnitude,
  which is mathematically redundant but harmless.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase11a.py
    # or with custom T:
    CUDA_VISIBLE_DEVICES=<gpu> python train_phase11a.py --T_fast 5 --T_slow 25
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.functional import lfilter
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest, collate_fn,
)
from postprocess import (
    tune_thresholds_event_level,
)

# Reuse helpers and constants from train.py so this script tracks any
# future changes to the baseline training recipe automatically.
from train import (
    align_lengths, validate, train_epoch,
    RESAMPLE_EVERY, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, MIN_LR,
)


# ======================================================================
# PCEN math
# ======================================================================

def _pcen_smooth(mag: torch.Tensor, s: float) -> torch.Tensor:
    """
    1st-order IIR smoothing along the time axis with warm-start init.

    Implements ``M[n] = s * mag[n] + (1-s) * M[n-1]`` with ``M[-1]`` set
    to the per-band mean of the first ~1 second of input. This is
    important: torchaudio's ``lfilter`` starts from implicit zero state,
    which for slow IIRs (T_slow=25s on a 30s segment) means ``M`` is
    still climbing toward equilibrium for the entire window — leaving
    the PCEN output artificially boosted and time-position-dependent.

    Trick: the IIR is linear, so the effect of nonzero initial state
    ``y0`` adds independently. With ``M[-1] = y0`` instead of 0,

        M_correct[n] = M_lfilter_zero[n] + y0 * (1 - s)^(n + 1)

    which is a fully vectorized closed-form correction.

    Parameters
    ----------
    mag : torch.Tensor
        Magnitude spectrogram, shape ``(B, F, T)``, last axis is time.
    s : float
        Smoothing coefficient. ``s = 1 - exp(-hop_dur / T_seconds)``.
        Small ``s`` corresponds to slow tracking (long ``T``).

    Returns
    -------
    torch.Tensor
        Smoothed magnitude, same shape as ``mag``.
    """
    device, dtype = mag.device, mag.dtype
    T = mag.size(-1)

    # IIR coefficients for y[n] = s * x[n] + (1 - s) * y[n-1].
    # In lfilter convention: a = [1, -(1-s)], b = [s, 0].
    a = torch.tensor([1.0, -(1.0 - s)], device=device, dtype=dtype)
    b = torch.tensor([s, 0.0], device=device, dtype=dtype)
    # clamp=False is essential: clamping would clip the IIR output to
    # [-1, 1] and destroy the magnitude scale we depend on.
    M_zero_init = lfilter(mag, a, b, clamp=False)

    # Warm-start: estimate equilibrium from the first 1s of input
    # (50 frames at 20 ms/frame). 1s is long enough to average over any
    # single short call onset, short enough that local stationarity is
    # a fair assumption.
    n_init = min(50, T)
    y0 = mag[..., :n_init].mean(dim=-1, keepdim=True)  # (B, F, 1)

    # Analytical correction. ``factor`` has shape (T,) and broadcasts.
    n_idx = torch.arange(T, device=device, dtype=dtype)
    factor = (1.0 - s) ** (n_idx + 1)  # (T,)

    return M_zero_init + y0 * factor


def _pcen(mag: torch.Tensor, s: float,
          alpha: float, delta: float, r: float, eps: float) -> torch.Tensor:
    """
    Per-Channel Energy Normalization (Wang et al. 2017).

    Returns ``((mag / (eps + M)^alpha) + delta)^r - delta^r`` where ``M``
    is the IIR-smoothed magnitude. The trailing ``- delta^r`` ensures
    PCEN(0) = 0, which keeps the silence floor at zero rather than at a
    constant offset.

    Parameters
    ----------
    mag : torch.Tensor, shape (B, F, T)
    s : float
        IIR smoothing coefficient.
    alpha : float
        AGC exponent. Default 0.98 from the paper; closer to 1 = stronger
        normalization.
    delta : float
        Bias before root compression. Default 2.0.
    r : float
        Root-compression exponent. Default 0.5 (square root).
    eps : float
        Small constant to keep the divisor strictly positive.
    """
    M = _pcen_smooth(mag, s)
    return (mag / (eps + M).pow(alpha) + delta).pow(r) - delta ** r


# ======================================================================
# Frontend extractor
# ======================================================================

class TwoStreamPCENExtractor(nn.Module):
    """
    Replaces ``|STFT|`` with two PCEN streams at different time constants,
    concatenated with the existing cos/sin phase channels.

    Output: ``(B, 4, F, T)`` with channels
    ``[pcen_fast, pcen_slow, cos_phase, sin_phase]``.

    Note
    ----
    PCEN parameters (alpha, delta, r, eps) are stored as plain Python
    floats — not ``nn.Parameter``s — because this experiment treats PCEN
    as a fixed prior. A follow-up phase that makes them learnable should
    register them via ``nn.Parameter(torch.tensor(...))`` and pass
    ``requires_grad=True`` selectively. The time constants ``T_fast`` /
    ``T_slow`` are likewise fixed; ``s_fast`` / ``s_slow`` are derived
    once at construction time.
    """

    def __init__(self,
                 T_fast: float = 5.0,
                 T_slow: float = 25.0,
                 alpha: float = 0.98,
                 delta: float = 2.0,
                 r: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.hop_length = cfg.HOP_LENGTH

        # Convert T (seconds) → s (per-frame smoothing coefficient).
        # s = 1 - exp(-hop_dur / T). Derivation: discretizing the
        # continuous-time first-order lowpass y' = (x - y) / T with
        # zero-order-hold sampling at period hop_dur.
        hop_dur = self.hop_length / cfg.SAMPLE_RATE
        self.T_fast = T_fast
        self.T_slow = T_slow
        self.s_fast = 1.0 - math.exp(-hop_dur / T_fast)
        self.s_slow = 1.0 - math.exp(-hop_dur / T_slow)

        # PCEN scalar params (frozen prior).
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps

        # Hann window as buffer (moves with .to(device), not a parameter).
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Complex STFT, same parameters as baseline ``SpectrogramExtractor``.
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )  # (B, F, T) complex

        # Preserve baseline's per-frequency demean (acts on complex STFT
        # before mag/phase split). Keeps cos/sin channels comparable to
        # the baseline; the slight extra DC suppression on the magnitude
        # is harmless given PCEN's own normalization.
        if cfg.NORM_FEATURES == "demean":
            stft = stft - stft.mean(dim=-1, keepdim=True)

        mag = stft.abs()
        angle = stft.angle()

        # Two PCEN streams at different time constants.
        pcen_fast = _pcen(mag, self.s_fast,
                          self.alpha, self.delta, self.r, self.eps)
        pcen_slow = _pcen(mag, self.s_slow,
                          self.alpha, self.delta, self.r, self.eps)

        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        # (B, 4, F, T) — stack on a new channel dimension.
        feat = torch.stack([pcen_fast, pcen_slow, cos_ph, sin_ph], dim=1)
        return feat


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--T_fast", type=float, default=5.0,
                   help="Fast-stream PCEN time constant in seconds. "
                        "Should be > P95(D) ≈ 4s to preserve short calls.")
    p.add_argument("--T_slow", type=float, default=25.0,
                   help="Slow-stream PCEN time constant in seconds. "
                        "Should be ≳ 2 × P95(BMABZ) ≈ 25s to preserve "
                        "long tonal calls.")
    p.add_argument("--alpha", type=float, default=0.98,
                   help="PCEN AGC exponent (Wang et al. 2017 default).")
    p.add_argument("--delta", type=float, default=2.0,
                   help="PCEN compression bias.")
    p.add_argument("--r", type=float, default=0.5,
                   help="PCEN compression exponent.")
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    wbu.seed_everything(cfg.SEED, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PCEN: T_fast={args.T_fast}s  T_slow={args.T_slow}s  "
          f"alpha={args.alpha}  delta={args.delta}  r={args.r}")

    # ------------------------------------------------------------------
    # Wandb. Mirrors train.py's tag conventions so the new run files in
    # alongside baseline runs in the same dashboard, distinguishable by
    # the ``pcen_frontend`` tag and the ``phase=11a`` config field.
    # ------------------------------------------------------------------
    extra_tags = ["from_scratch", "pcen_frontend", "two_stream_pcen"]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if not cfg.USE_WEIGHTED_BCE and not getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("plain_bce")

    run = wbu.init_phase(
        "11a",
        extra_tags=extra_tags,
        config={
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
            # Phase 11a-specific
            "frontend":         "two_stream_pcen",
            "feat_channels":    4,
            "pcen_T_fast":      args.T_fast,
            "pcen_T_slow":      args.T_slow,
            "pcen_alpha":       args.alpha,
            "pcen_delta":       args.delta,
            "pcen_r":           args.r,
            "pcen_trainable":   False,
        },
    )

    run_dir = Path(cfg.OUTPUT_DIR) / f"phase11a_{time.strftime('%Y%m%d_%H%M%S')}"
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
    # Model: 4-channel input
    # ------------------------------------------------------------------
    spec_extractor = TwoStreamPCENExtractor(
        T_fast=args.T_fast,
        T_slow=args.T_slow,
        alpha=args.alpha,
        delta=args.delta,
        r=args.r,
    ).to(device)
    # feat_channels=4 -> filterbank's first Conv2d takes 4 input channels
    # instead of 3. Everything downstream is unchanged.
    model = WhaleVAD(num_classes=cfg.n_classes(), feat_channels=4).to(device)

    # Initialize the lazy projection layer with a dummy forward.
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        feat = spec_extractor(dummy)
        print(f"Feature shape: {feat.shape}  (expected (1, 4, F, T))")
        model(feat)

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"PCEN s_fast={spec_extractor.s_fast:.5f}  "
          f"s_slow={spec_extractor.s_slow:.5f}")

    # ------------------------------------------------------------------
    # Loss / optimizer / scheduler
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
    # Training loop. Identical structure to train.py — the only
    # difference is the spec_extractor and model.feat_channels above.
    # ------------------------------------------------------------------
    best_f1 = 0.0
    no_improve_epochs = 0
    thresholds = torch.tensor(
        cfg.DEFAULT_THRESHOLDS[:3] if len(cfg.DEFAULT_THRESHOLDS) >= 3
        else [0.5, 0.5, 0.5],
        device=device,
    )

    epoch = 0
    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}\n"
              f"{'=' * 60}")

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **wbu.seeded_dataloader_kwargs(cfg.SEED + epoch),
            )

        train_loss = train_epoch(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch,
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

        # Wandb per-epoch payload (same shape as train.py).
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
                      else model.classifier)
        bias_str = ", ".join(f"{b:+.2f}"
                             for b in clf_module.bias.detach().cpu().tolist())
        print(f"  Classifier bias: [{bias_str}]")

        model_state = (model.module.state_dict()
                       if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model_state,
            "best_f1":          best_f1,
            "thresholds":       thresholds.cpu(),
            "feat_channels":    4,
            "pcen_config": {
                "T_fast": args.T_fast, "T_slow": args.T_slow,
                "alpha":  args.alpha,  "delta":  args.delta,
                "r":      args.r,
            },
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
    # Post-training threshold tuning on the best checkpoint
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
        "thresholds":       torch.tensor(tuned),
        "feat_channels":    4,
        "pcen_config": {
            "T_fast": args.T_fast, "T_slow": args.T_slow,
            "alpha":  args.alpha,  "delta":  args.delta,
            "r":      args.r,
        },
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Wandb summary + artifact
    # ------------------------------------------------------------------
    import wandb
    wandb.summary["best_f1"]              = float(best_f1)
    wandb.summary["best_f1_post_tuning"]  = float(best_ckpt.get("best_f1", best_f1))
    wandb.summary["final_thresholds"]     = list(map(float, tuned))
    wandb.summary["epochs_run"]           = epoch
    wandb.summary["early_stopped"]        = no_improve_epochs >= EARLY_STOP_PATIENCE
    wandb.summary["pcen_T_fast"]          = float(args.T_fast)
    wandb.summary["pcen_T_slow"]          = float(args.T_slow)
    wandb.summary["verdict"] = (
        f"Phase 11a (two-stream PCEN T={args.T_fast}/{args.T_slow}s) "
        f"finished at best F1 {best_f1:.3f} "
        f"(epoch {best_ckpt.get('epoch', '?')} of {epoch} run; "
        f"final tuned thresholds {[round(float(t), 2) for t in tuned]})."
    )

    art = wandb.Artifact(
        f"model-{run.name}", type="model",
        metadata={
            "best_f1":          float(best_f1),
            "best_epoch":       int(best_ckpt.get("epoch", 0)),
            "epochs_run":       int(epoch),
            "tuned_thresholds": list(map(float, tuned)),
            "frontend":         "two_stream_pcen",
            "feat_channels":    4,
            "pcen_T_fast":      args.T_fast,
            "pcen_T_slow":      args.T_slow,
            "pcen_alpha":       args.alpha,
            "pcen_delta":       args.delta,
            "pcen_r":           args.r,
        },
    )
    art.add_file(str(run_dir / "best_model.pt"))
    art.add_file(str(run_dir / "final_model.pt"))
    run.log_artifact(art, aliases=["best", "phase11a"])

    wandb.finish()


if __name__ == "__main__":
    main()
