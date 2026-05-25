"""
Phase 11a: Two-Stream PCEN Frontend (de-risk before LEAF)
=========================================================

Single-axis change to the final recipe (``train_final.py``): replace the
magnitude channel ``mag = |STFT|`` with TWO PCEN streams at different
time constants. Phase channels (cos/sin) are unchanged. Everything else
about the recipe is held fixed: 8-site training, 7 fine-grained classes
collapsed to 3 at evaluation, segment-count weighted BCE, per-epoch
negative resampling, paper BiLSTM (hidden 128, 2 layers).

    Final input  (B, 3, F, T)  = [|STFT|,    cos_ph, sin_ph]
    Phase 11a    (B, 4, F, T)  = [PCEN_fast, PCEN_slow, cos_ph, sin_ph]

What this tests
---------------
Per-band AGC as a frontend prior. Magnitude-STFT hands the CNN raw
energy with sustained ambient (ship, ice, swell) often 20-40 dB above
call energy, forcing the learned frontend to implicitly solve the
noise-normalisation problem with conv layers. PCEN solves it
analytically:

    PCEN(t,f) = ( mag(t,f) / (eps + M(t,f))^alpha + delta )^r - delta^r

where ``M`` is a 1st-order IIR smoothing of the magnitude with time
constant ``T``.

Why two streams
---------------
Per-class duration P95 in the training data:
    BMABZ  12.89 s   (long tonal calls)
    D       4.18 s   (short downsweeps)
    BP      4.04 s   (short pulsed)

A single ``T`` can't preserve both ends. ``T_slow=25 s`` keeps BMABZ
(D/T ~ 0.3 at median) almost untouched; ``T_fast=5 s`` preserves D and
BP while sharpening their attack against shorter-timescale ambient.
Concatenating both as input channels lets the CNN learn which stream
matters per class.

Implementation notes
--------------------
- IIR uses ``torchaudio.functional.lfilter`` with an analytical
  warm-start correction: ``M_correct[n] = M_lfilter[n] + y0 * (1-s)^(n+1)``
  where ``y0 = mean(mag[..., :n_init])``. Without this, T_slow=25s on a
  30s segment only reaches 70% of true equilibrium by the last frame.
- PCEN params (alpha=0.98, delta=2.0, r=0.5) from Wang et al. 2017 are
  fixed in this experiment. Pure prior test; a follow-up phase can make
  them learnable.
- ``WhaleVAD(num_classes=7, feat_channels=4)`` -- 7-class training is
  preserved exactly as in the final recipe.

Usage
-----
::

    python train_phase11a.py                          # online W&B (default)
    python train_phase11a.py --wandb-mode offline     # offline (no network)
    python train_phase11a.py --T_fast 5 --T_slow 25   # sweep PCEN T

W&B logging is always on; ``--wandb-mode`` selects online/offline/disabled.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchaudio.functional import lfilter

import config_final as cfg
from dataset_final import (
    load_annotations, get_file_manifest,
    build_positive_segments, build_val_segments,
    extend_all_segments, WhaleDataset, collate_fn,
)
from model_final import WhaleVAD

# Helpers reused from train_final.py so any future tweak to the recipe
# propagates without duplication.
from train_final import (
    seed_everything,
    seeded_dataloader_kwargs,
    compute_pos_weight,
    train_one_epoch,
    validate,
    resample_negatives_for_epoch,
)


# ======================================================================
# PCEN math
# ======================================================================

def _pcen_smooth(mag: torch.Tensor, s: float) -> torch.Tensor:
    """
    1st-order IIR smoothing along the time axis with analytical
    warm-start initialisation.

    Implements ``M[n] = s * mag[n] + (1-s) * M[n-1]`` with ``M[-1]`` set
    to the per-band mean of the first ~1 s of input.
    ``torchaudio.functional.lfilter`` starts from implicit zero state;
    for slow IIRs (T_slow=25s on a 30s segment) that means ``M`` is
    still climbing toward equilibrium for the entire window, leaving the
    PCEN output time-position-dependent.

    The IIR is linear, so the contribution of a nonzero initial state
    ``y0`` adds independently of the zero-init response:

        M_correct[n] = M_lfilter_zero[n] + y0 * (1 - s)^(n + 1)

    which is a fully vectorised closed-form correction. Verified against
    an explicit-init Python loop to within 2e-4 absolute error on a
    30 s / T=25 s test signal.

    Parameters
    ----------
    mag : torch.Tensor, shape ``(B, F, T)``
    s : float
        Smoothing coefficient. ``s = 1 - exp(-hop_dur / T_seconds)``.
        Small ``s`` corresponds to slow tracking (long ``T``).
    """
    device, dtype = mag.device, mag.dtype
    T = mag.size(-1)

    # lfilter convention: y[n] = s*x[n] + (1-s)*y[n-1] => a = [1, -(1-s)],
    # b = [s, 0]. ``clamp=False`` is essential -- clamping would clip the
    # IIR output to [-1, 1] and destroy the magnitude scale.
    a = torch.tensor([1.0, -(1.0 - s)], device=device, dtype=dtype)
    b = torch.tensor([s, 0.0], device=device, dtype=dtype)
    M_zero_init = lfilter(mag, a, b, clamp=False)

    # Warm-start: equilibrium estimate from the first 1 s of input
    # (50 frames at 20 ms/frame). 1 s is long enough to average over
    # short-call onsets, short enough that local stationarity is fair.
    n_init = min(50, T)
    y0 = mag[..., :n_init].mean(dim=-1, keepdim=True)  # (B, F, 1)

    n_idx = torch.arange(T, device=device, dtype=dtype)
    factor = (1.0 - s) ** (n_idx + 1)  # (T,)

    return M_zero_init + y0 * factor


def _pcen(mag: torch.Tensor, s: float, alpha: float, delta: float,
          r: float, eps: float) -> torch.Tensor:
    """
    Per-Channel Energy Normalisation (Wang et al. 2017).

    Returns ``((mag / (eps + M)^alpha) + delta)^r - delta^r`` where
    ``M`` is the IIR-smoothed magnitude. The trailing ``- delta^r``
    keeps the silence floor at zero rather than at a constant offset.
    """
    M = _pcen_smooth(mag, s)
    return (mag / (eps + M).pow(alpha) + delta).pow(r) - delta ** r


# ======================================================================
# Frontend extractor (replaces SpectrogramExtractor)
# ======================================================================

class TwoStreamPCENExtractor(nn.Module):
    """
    Drop-in replacement for ``spectrogram_final.SpectrogramExtractor``.
    Swaps the magnitude channel for two PCEN streams at different time
    constants, keeping the cos/sin phase channels unchanged.

    Output shape ``(B, 4, F, T)``:
    ``[pcen_fast, pcen_slow, cos_phase, sin_phase]``.
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

        # T (seconds) -> s (per-frame smoothing coefficient).
        # Discretise the continuous-time first-order lowpass
        # y' = (x - y) / T with zero-order-hold sampling at hop_dur:
        #     s = 1 - exp(-hop_dur / T)
        hop_dur = self.hop_length / cfg.SAMPLE_RATE
        self.T_fast = T_fast
        self.T_slow = T_slow
        self.s_fast = 1.0 - math.exp(-hop_dur / T_fast)
        self.s_slow = 1.0 - math.exp(-hop_dur / T_slow)

        # PCEN scalar params (frozen prior in this experiment).
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps

        # Hann window as buffer (moves with .to(device); not a parameter).
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )  # (B, F, T) complex

        # Preserve the final pipeline's complex-demean step. Keeps cos/sin
        # comparable to the baseline; the slight extra DC suppression on
        # the magnitude before PCEN is harmless given PCEN's own
        # normalisation.
        if cfg.NORM_FEATURES == "demean":
            stft = stft - stft.mean(dim=-1, keepdim=True)

        mag = stft.abs()
        angle = stft.angle()

        pcen_fast = _pcen(mag, self.s_fast,
                          self.alpha, self.delta, self.r, self.eps)
        pcen_slow = _pcen(mag, self.s_slow,
                          self.alpha, self.delta, self.r, self.eps)

        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        return torch.stack([pcen_fast, pcen_slow, cos_ph, sin_ph], dim=1)


def build_model(device: torch.device, extractor: nn.Module):
    """
    Build the 7-class WhaleVAD with ``feat_channels=4`` and run one
    dummy forward pass to materialise the lazy projection layer.
    Mirrors ``train_final.build_model`` exactly, only differing in the
    filterbank's input-channel count.
    """
    model = WhaleVAD(num_classes=7, feat_channels=4).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(extractor(dummy))
    return model


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 11a: train the final recipe with a two-stream "
                    "PCEN frontend."
    )
    # W&B logging is always on for phase-ladder runs so every rung lands in
    # the dashboard; only the run mode is configurable (offline if no
    # network, disabled to suppress entirely for smoke tests).
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B run mode (default online).")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                   help=f"Number of training epochs (default {cfg.EPOCHS}).")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help=f"Master random seed (default {cfg.SEED}).")
    # PCEN knobs
    p.add_argument("--T_fast", type=float, default=5.0,
                   help="Fast-stream PCEN time constant in seconds. Should "
                        "be > P95(D) ~ 4 s to preserve short calls.")
    p.add_argument("--T_slow", type=float, default=25.0,
                   help="Slow-stream PCEN time constant in seconds. Should "
                        "be >~ 2 x P95(BMABZ) ~ 25 s to preserve long calls.")
    p.add_argument("--alpha", type=float, default=0.98,
                   help="PCEN AGC exponent.")
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

    # 7-class targets, exactly as in train_final.py. The validate()
    # helper from train_final toggles cfg.USE_3CLASS internally for the
    # duration of post-processing and restores it afterwards.
    cfg.USE_3CLASS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PCEN: T_fast={args.T_fast}s  T_slow={args.T_slow}s  "
          f"alpha={args.alpha}  delta={args.delta}  r={args.r}")

    seed = seed_everything(args.seed, deterministic=False)
    sampling_rng = random.Random(seed)

    train_sites = list(cfg.TRAIN_DATASETS)
    val_sites = list(cfg.VAL_DATASETS)

    # --- pos_weight from annotations, before any dataloader exists ---
    print(f"\nComputing 7-class pos_weight over {len(train_sites)} sites...")
    pos_weight, weight_info = compute_pos_weight(train_sites, device, verbose=True)

    # --- W&B run (always on for phase-ladder tracking) ---
    import wandb_utils as wbu
    run = wbu.init_phase("11a", config={
        "lr":             cfg.LR,
        "weight_decay":   cfg.WEIGHT_DECAY,
        "batch_size":     cfg.BATCH_SIZE,
        "threshold":      cfg.THRESHOLD,
        "seed":           seed,
        "neg_ratio":      cfg.NEG_RATIO,
        "neg_resample_each_epoch": True,
        "segment_s":      cfg.TRAIN_SEGMENT_S,
        "epochs":         args.epochs,
        "train_sites":    train_sites,
        "val_sites":      val_sites,
        "lstm_hidden":    cfg.LSTM_HIDDEN,
        "lstm_layers":    cfg.LSTM_LAYERS,
        "pos_weight":     weight_info["pos_weight"],
        "pos_weight_counts": weight_info["annotation_counts"],
        "pos_weight_ratio": weight_info["weight_ratio"],
        # Phase 11a specific
        "frontend":       "two_stream_pcen",
        "feat_channels":  4,
        "pcen_T_fast":    args.T_fast,
        "pcen_T_slow":    args.T_slow,
        "pcen_alpha":     args.alpha,
        "pcen_delta":     args.delta,
        "pcen_r":         args.r,
        "pcen_trainable": False,
    }, mode=args.wandb_mode)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase11a_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration:")
    print(f"  Training sites:   {train_sites}  ({len(train_sites)})")
    print(f"  Validation sites: {val_sites}")
    print(f"  Output:           7-class training, collapsed to 3-class at eval")
    print(f"  Loss:             weighted BCE (segment-count normalised)")
    print(f"  Negatives:        resampled at the start of every epoch")
    print(f"  Frontend:         two-stream PCEN -> 4-channel CNN input")
    print(f"  LSTM:             hidden={cfg.LSTM_HIDDEN}, layers={cfg.LSTM_LAYERS}")
    print(f"  LR={cfg.LR}, batch={cfg.BATCH_SIZE}, epochs={args.epochs}")

    # --- fixed positives + static validation set ---
    print(f"\nLoading training data...")
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    print(f"  {len(train_manifest)} files, {len(train_annotations)} annotations")

    pos_segs = build_positive_segments(
        train_annotations, train_manifest, rng=sampling_rng,
    )
    pos_segs = extend_all_segments(pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)
    print(f"  Positive segments (fixed): {len(pos_segs)}")
    print(f"  Negative segments per epoch: {n_neg}")

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    file_start_dts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in val_manifest.iterrows()
    }
    print(f"  Val: {len(val_manifest)} files, {len(val_annotations)} "
          f"annotations, {len(val_segments)} segments")

    # --- model, loss, optimizer ---
    # Re-seed torch immediately before weight init so the model is
    # identical regardless of any earlier torch RNG consumption
    # (e.g. by wandb.init). No-op without W&B; makes both paths agree.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    spec_extractor = TwoStreamPCENExtractor(
        T_fast=args.T_fast,
        T_slow=args.T_slow,
        alpha=args.alpha,
        delta=args.delta,
        r=args.r,
    ).to(device)
    model = build_model(device, spec_extractor)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"PCEN s_fast={spec_extractor.s_fast:.5f}  "
          f"s_slow={spec_extractor.s_slow:.5f}")
    run.config.update({"n_params": n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weight,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.BETA1, cfg.BETA2),
    )

    # --- training loop with per-epoch negative resampling ---
    history = []
    print(f"\n{'=' * 60}")
    print(f"Training {args.epochs} epochs (per-epoch negative resampling)")
    print(f"{'=' * 60}")

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_segments = resample_negatives_for_epoch(
            pos_segs_extended=pos_segs,
            train_annotations=train_annotations,
            train_manifest=train_manifest,
            n_neg=n_neg,
            segment_s=cfg.TRAIN_SEGMENT_S,
            epoch=epoch,
            rng=sampling_rng,
            verbose=True,
        )
        train_loader = DataLoader(
            WhaleDataset(train_segments), batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
            **seeded_dataloader_kwargs(seed),
        )

        train_loss = train_one_epoch(
            model, spec_extractor, train_loader, criterion, optimizer, device,
        )
        val = validate(
            model, spec_extractor, val_loader, criterion, device,
            val_annotations, file_start_dts, threshold=cfg.THRESHOLD,
        )
        epoch_time = time.time() - t0

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]
        marker = " *** new best" if improved else ""

        print(f"\nEpoch {epoch:2d}/{args.epochs}  ({epoch_time:.0f}s){marker}")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val['loss']:.4f}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} TP={pc['tp']:5} FP={pc['fp']:6} "
                  f"FN={pc['fn']:5}  P={pc['precision']:.3f} "
                  f"R={pc['recall']:.3f} F1={pc['f1']:.3f}")
        macro = sum(val["per_class"][n]["f1"] for n in cfg.CALL_TYPES_3) / 3
        print(f"    OVERALL F1={val['f1']:.3f}  MACRO F1={macro:.3f}")

        wbu.log_epoch_3class(epoch, train_loss, val)

        history.append({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val["loss"],
            "f1":          val["f1"],
            "macro_f1":    macro,
            "per_class":   val["per_class"],
        })

        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "f1":               val["f1"],
            "history":          history,
            "pos_weight":       pos_weight.detach().cpu().tolist(),
            # Frontend metadata so inference can rebuild the same extractor.
            "frontend":         "two_stream_pcen",
            "feat_channels":    4,
            "pcen_config": {
                "T_fast": args.T_fast, "T_slow": args.T_slow,
                "alpha":  args.alpha,  "delta":  args.delta,
                "r":      args.r,
            },
        }
        torch.save(ckpt, run_dir / f"phase11a_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "phase11a_best.pt")

    # --- summary ---
    print(f"\n{'=' * 60}")
    print("PHASE 11A SUMMARY")
    print(f"{'=' * 60}")

    f1s = [h["f1"] for h in history]
    macros = [h["macro_f1"] for h in history]
    print(f"\nMicro F1 by epoch: {[f'{f:.3f}' for f in f1s]}")
    print(f"Macro F1 by epoch: {[f'{m:.3f}' for m in macros]}")
    print(f"\nBest micro F1: {max(f1s):.3f}  (epoch {f1s.index(max(f1s)) + 1})")
    print(f"Best macro F1: {max(macros):.3f}")
    for name in cfg.CALL_TYPES_3:
        best = max(h["per_class"][name]["f1"] for h in history)
        print(f"  best {name}: {best:.3f}")

    second_half = f1s[len(f1s) // 2:]
    swings = [abs(second_half[i] - second_half[i - 1])
              for i in range(1, len(second_half))]
    mean_swing = sum(swings) / max(len(swings), 1)
    max_swing = max(swings) if swings else 0.0
    print(f"\nSecond-half stability: mean swing {mean_swing:.3f}, "
          f"max swing {max_swing:.3f}")

    verdict = (
        f"Phase 11a (two-stream PCEN T={args.T_fast}/{args.T_slow}s): "
        f"best micro F1 {max(f1s):.3f}, best macro F1 {max(macros):.3f}; "
        f"second-half mean F1 swing {mean_swing:.3f}."
    )
    wbu.finalize_phase(history, verdict=verdict,
                       best_ckpt=run_dir / "phase11a_best.pt")


if __name__ == "__main__":
    main()
