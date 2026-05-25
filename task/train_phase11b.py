"""
Phase 11b: Stacked PCEN Frontend (5-channel, mag + 2xPCEN + cos/sin)
====================================================================

Where 11a *replaced* the magnitude channel with two PCEN streams, 11b
*stacks* them: keep the working magnitude channel intact, append two
PCEN streams as additional input. The model's first conv layer
(filterbank Conv2d) takes 5 channels instead of 3 and can learn to
weight raw magnitude vs PCEN-normalised streams per output kernel.

    Final baseline (B, 3, F, T) = [mag,    cos_ph, sin_ph]
    Phase 11a      (B, 4, F, T) = [pcen_f, pcen_s, cos_ph, sin_ph]
    Phase 11b      (B, 5, F, T) = [mag, pcen_f, pcen_s, cos_ph, sin_ph]

Why stack instead of replace
----------------------------
11a's pure-replacement test landed well below baseline (macro F1 0.313
vs ~0.45 baseline). Stacking is a strictly more conservative test:
information-theoretically, the network can always learn to zero out the
PCEN kernels and recover baseline behavior, so a from-scratch stacked
run should not underperform baseline in expectation if optimization
converges. Empirically it still can; the lower bound is just tighter.

Two variants (selected via --variant)
-------------------------------------
The complex-domain demean step is a per-band time-mean subtraction
applied to the complex STFT before the magnitude/phase split. PCEN's
IIR is a separate magnitude-domain time-averaging normalisation. They
overlap; doing both means PCEN sees a magnitude that already had its
persistent component reduced.

  --variant sym:  demean is applied once to the complex STFT. Both the
                  magnitude channel and the PCEN streams compute on the
                  resulting demeaned magnitude. Matches the baseline's
                  pre-normalisation exactly; PCEN's IIR sees diminished
                  background to track.
  --variant asym: demean is applied for the magnitude and phase channels
                  (matching the baseline), but PCEN streams compute on
                  raw (non-demeaned) magnitude. Each pre-normalisation
                  runs on the input it was designed for.

Together, 11b-sym vs 11b-asym isolates the demean-PCEN interaction from
the stacking question.

Training recipe is identical to 11a (same seed, 30 epochs, fixed
cfg.THRESHOLD=0.3, per-epoch neg resampling, segment-count WBCE, from
scratch). Only the frontend differs. W&B logging is always on; the
phase tag is "11b-sym" or "11b-asym" depending on --variant, so the
two runs land as separate phases in the dashboard.

Launch
------
::

    python train_phase11b.py --variant sym
    python train_phase11b.py --variant asym

Both runs are independent; launch in parallel on separate GPUs if
available.
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

# Helpers reused from train_final.py so any tweak to the recipe
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
# PCEN math (verbatim from train_phase11a.py)
# ======================================================================

def _pcen_smooth(mag: torch.Tensor, s: float) -> torch.Tensor:
    """
    1st-order IIR smoothing along the time axis with analytical
    warm-start initialisation.

    Implements ``M[n] = s * mag[n] + (1-s) * M[n-1]`` with ``M[-1]`` set
    to the per-band mean of the first ~1 s of input.
    ``torchaudio.functional.lfilter`` starts from implicit zero state;
    the closed-form correction ``y0 * (1-s)^(n+1)`` recovers a nonzero
    initial state without an explicit loop.
    """
    device, dtype = mag.device, mag.dtype
    T = mag.size(-1)

    a = torch.tensor([1.0, -(1.0 - s)], device=device, dtype=dtype)
    b = torch.tensor([s, 0.0], device=device, dtype=dtype)
    M_zero_init = lfilter(mag, a, b, clamp=False)

    n_init = min(50, T)
    y0 = mag[..., :n_init].mean(dim=-1, keepdim=True)
    n_idx = torch.arange(T, device=device, dtype=dtype)
    factor = (1.0 - s) ** (n_idx + 1)
    return M_zero_init + y0 * factor


def _pcen(mag: torch.Tensor, s: float, alpha: float, delta: float,
          r: float, eps: float) -> torch.Tensor:
    """
    Per-Channel Energy Normalisation (Wang et al. 2017).
    ``((mag / (eps + M)^alpha) + delta)^r - delta^r`` with M = IIR smoothed mag.
    """
    M = _pcen_smooth(mag, s)
    return (mag / (eps + M).pow(alpha) + delta).pow(r) - delta ** r


# ======================================================================
# Extractor base class
# ======================================================================

class _StackedPCENBase(nn.Module):
    """
    Shared setup for both 11b variants. Subclasses implement ``forward``
    to differ only in how demean interacts with the PCEN inputs.

    Output: ``(B, 5, F, T)`` with channels
    ``[mag, pcen_fast, pcen_slow, cos_phase, sin_phase]``.
    """

    variant: str = ""  # set by subclasses for logging / debugging

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

        hop_dur = self.hop_length / cfg.SAMPLE_RATE
        self.T_fast = T_fast
        self.T_slow = T_slow
        self.s_fast = 1.0 - math.exp(-hop_dur / T_fast)
        self.s_slow = 1.0 - math.exp(-hop_dur / T_slow)

        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps

        self.register_buffer("window", torch.hann_window(self.win_length))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )


# ======================================================================
# Variant: symmetric demean
# ======================================================================

class StackedPCENSymmetric(_StackedPCENBase):
    """
    11b-sym: demean once to the complex STFT (matching baseline), then
    derive both the magnitude channel and the PCEN streams from the
    same demeaned magnitude.

    PCEN's IIR sees diminished persistent background here, because
    demean has already subtracted the per-band complex time-mean. This
    is the simpler structure: one STFT, one demean, one magnitude.
    """

    variant = "sym"

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        stft = self._stft(audio)  # (B, F, T) complex

        # Single demean, applied to both mag and PCEN paths.
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

        return torch.stack([mag, pcen_fast, pcen_slow, cos_ph, sin_ph], dim=1)


# ======================================================================
# Variant: asymmetric demean
# ======================================================================

class StackedPCENAsymmetric(_StackedPCENBase):
    """
    11b-asym: keep demean for the mag and phase channels (so the
    baseline-style triplet [mag, cos_ph, sin_ph] is bit-for-bit identical
    to what the final baseline computes), but feed the PCEN streams with
    RAW (non-demeaned) magnitude so PCEN's IIR sees the persistent
    background it is supposed to track.

    Each pre-normalisation runs on the input it was designed for:
    - magnitude channel: baseline's per-band complex demean
    - PCEN streams:      PCEN's IIR-based per-band AGC
    - phase channels:    from the demeaned STFT (paired with mag)

    Cost: two ``.abs()`` calls (one on demeaned, one on raw STFT).
    """

    variant = "asym"

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        stft = self._stft(audio)  # (B, F, T) complex

        # Demeaned branch -> baseline-style mag + phase channels.
        if cfg.NORM_FEATURES == "demean":
            stft_demeaned = stft - stft.mean(dim=-1, keepdim=True)
        else:
            stft_demeaned = stft

        mag = stft_demeaned.abs()
        angle = stft_demeaned.angle()
        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        # Raw branch -> PCEN streams.
        mag_raw = stft.abs()
        pcen_fast = _pcen(mag_raw, self.s_fast,
                          self.alpha, self.delta, self.r, self.eps)
        pcen_slow = _pcen(mag_raw, self.s_slow,
                          self.alpha, self.delta, self.r, self.eps)

        return torch.stack([mag, pcen_fast, pcen_slow, cos_ph, sin_ph], dim=1)


# ======================================================================
# Model builder
# ======================================================================

def build_model(device: torch.device, extractor: nn.Module):
    """
    7-class WhaleVAD with ``feat_channels=5`` and a dummy forward pass
    to materialise the lazy projection layer.
    """
    model = WhaleVAD(num_classes=7, feat_channels=5).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(extractor(dummy))
    return model


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 11b: train the final recipe with a stacked "
                    "5-channel PCEN frontend."
    )
    p.add_argument("--variant", required=True, choices=["sym", "asym"],
                   help="Demean handling. 'sym': demean once, both mag and "
                        "PCEN see demeaned magnitude. 'asym': demean for mag "
                        "and phase only; PCEN computes on raw magnitude.")
    # W&B logging is always on; only the run mode is configurable.
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B run mode (default online).")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                   help=f"Number of training epochs (default {cfg.EPOCHS}).")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help=f"Master random seed (default {cfg.SEED}).")
    # PCEN knobs (defaults match 11a for a clean comparison)
    p.add_argument("--T_fast", type=float, default=5.0,
                   help="Fast-stream PCEN time constant in seconds.")
    p.add_argument("--T_slow", type=float, default=25.0,
                   help="Slow-stream PCEN time constant in seconds.")
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

    phase_tag = f"11b-{args.variant}"   # "11b-sym" or "11b-asym"

    # 7-class targets, exactly as in train_final.py and 11a.
    cfg.USE_3CLASS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase: {phase_tag}  ({args.variant.upper()} demean variant)")
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
    run = wbu.init_phase(phase_tag, config={
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
        # Phase 11b specific
        "frontend":       "stacked_pcen",
        "feat_channels":  5,
        "channel_order":  ["mag", "pcen_fast", "pcen_slow", "cos_ph", "sin_ph"],
        "variant":        args.variant,
        "demean_for_mag_phase": True,
        "demean_for_pcen": (args.variant == "sym"),
        "pcen_T_fast":    args.T_fast,
        "pcen_T_slow":    args.T_slow,
        "pcen_alpha":     args.alpha,
        "pcen_delta":     args.delta,
        "pcen_r":         args.r,
        "pcen_trainable": False,
    }, mode=args.wandb_mode)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase11b_{args.variant}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("\nConfiguration:")
    print(f"  Training sites:   {train_sites}  ({len(train_sites)})")
    print(f"  Validation sites: {val_sites}")
    print(f"  Output:           7-class training, collapsed to 3-class at eval")
    print(f"  Loss:             weighted BCE (segment-count normalised)")
    print(f"  Negatives:        resampled at the start of every epoch")
    print(f"  Frontend:         stacked PCEN ({args.variant}) -> 5-channel CNN input")
    print(f"  Channel order:    [mag, pcen_fast, pcen_slow, cos_ph, sin_ph]")
    if args.variant == "sym":
        print(f"  Demean:           applied once to complex STFT (mag + PCEN)")
    else:
        print(f"  Demean:           applied for mag/phase only; PCEN on raw mag")
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
    # identical regardless of any earlier torch RNG consumption.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.variant == "sym":
        spec_extractor = StackedPCENSymmetric(
            T_fast=args.T_fast, T_slow=args.T_slow,
            alpha=args.alpha, delta=args.delta, r=args.r,
        ).to(device)
    else:
        spec_extractor = StackedPCENAsymmetric(
            T_fast=args.T_fast, T_slow=args.T_slow,
            alpha=args.alpha, delta=args.delta, r=args.r,
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
            "frontend":         "stacked_pcen",
            "variant":          args.variant,
            "feat_channels":    5,
            "channel_order":    ["mag", "pcen_fast", "pcen_slow", "cos_ph", "sin_ph"],
            "pcen_config": {
                "T_fast": args.T_fast, "T_slow": args.T_slow,
                "alpha":  args.alpha,  "delta":  args.delta,
                "r":      args.r,
            },
        }
        torch.save(ckpt, run_dir / f"phase11b_{args.variant}_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / f"phase11b_{args.variant}_best.pt")

    # --- summary ---
    print(f"\n{'=' * 60}")
    print(f"PHASE 11B-{args.variant.upper()} SUMMARY")
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
        f"Phase 11b-{args.variant} (stacked PCEN, T={args.T_fast}/{args.T_slow}s, "
        f"demean={'sym' if args.variant == 'sym' else 'asym'}): "
        f"best micro F1 {max(f1s):.3f}, best macro F1 {max(macros):.3f}; "
        f"second-half mean F1 swing {mean_swing:.3f}."
    )
    wbu.finalize_phase(history, verdict=verdict,
                       best_ckpt=run_dir / f"phase11b_{args.variant}_best.pt")


if __name__ == "__main__":
    main()
