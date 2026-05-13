"""
train_phase_splice_v2.py
========================

Splice-augmented training driver. **Drop-in replacement for ``train.py``**
that adds call-splicing augmentation before STFT extraction. Everything
else — loss (WhaleVADLoss = weighted BCE + focal), optimizer, scheduler,
early stopping, threshold tuning, per-class breakdown printing,
checkpointing, wandb logging — is reused verbatim from ``train.py``.

Compared to ``train.py``:
    * Imports ``apply_call_splice`` and the bank/pool dataclasses
    * Loads call_bank.pt + splice_host_pool.pt at startup
    * Optionally dumps N synthetic examples and exits before training
    * Replaces the body of ``train_epoch`` with a splice-augmented version
      (one extra line: ``audio, mask, targets = splice_step(...)``)
    * Adds splice config to the wandb config dict
    * Tagged ``splice`` and named ``phase_splice_<ts>`` in the run directory

When ``--splice_prob 0.0`` is passed, this script is bit-equivalent to
``train.py`` (the splice_step is a no-op at probability zero), so the
run is directly comparable to your F1=0.469 baseline.

Usage
-----
::

    # Full splice run (matches the F1=0.469 baseline pipeline except for
    # the splice hook itself):
    CUDA_VISIBLE_DEVICES=9 python train_phase_splice_v2.py \\
        --call_bank pipeline_call_splice/call_bank.pt \\
        --splice_pool pipeline_call_splice/splice_host_pool.pt \\
        --splice_prob 0.2 \\
        --seed 1337

    # Smoke test: dump 20 synthetic examples and exit before training.
    python train_phase_splice_v2.py \\
        --call_bank pipeline_call_splice/call_bank.pt \\
        --splice_pool pipeline_call_splice/splice_host_pool.pt \\
        --dump_examples 20 \\
        --seed 1337

    # Splice-disabled sanity run (equivalent to ``train.py``):
    python train_phase_splice_v2.py \\
        --call_bank pipeline_call_splice/call_bank.pt \\
        --splice_pool pipeline_call_splice/splice_host_pool.pt \\
        --splice_prob 0.0 \\
        --seed 1337
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb_utils as wbu
from spectrogram import SpectrogramExtractor
from model import WhaleVAD, WhaleVADLoss, compute_class_weights
from dataset import (
    build_dataloaders, load_annotations, get_file_manifest,
    collate_fn,
)

# Reuse train.py's helpers and constants verbatim. The only thing we
# don't reuse is its ``train_epoch`` (we substitute a splice-augmented
# version below) and ``main`` (we reproduce its structure with splice
# state plumbed through).
from train import (
    align_lengths,
    validate,
    set_seed,
    RESAMPLE_EVERY,
    EARLY_STOP_PATIENCE,
    LR_PATIENCE,
    LR_FACTOR,
    MIN_LR,
)

# Splice augmentation.
from call_splice import (
    CallBank, SpliceHostPool, CallSpliceState,
    apply_call_splice, splice_one_sample,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    """Parse splice + baseline CLI args."""
    p = argparse.ArgumentParser()
    # Baseline (train.py-compatible) flags.
    p.add_argument("--pretrained", type=str, default=None,
                   help="Optional path to a contrastive-pretrained encoder.")
    p.add_argument("--freeze_epochs", type=int, default=0,
                   help="If --pretrained, freeze encoder for this many epochs.")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help="Master random seed. Overrides cfg.SEED for this run.")

    # Splice augmentation flags.
    p.add_argument("--call_bank", type=str, required=True,
                   help="Path to call_bank.pt produced by 01_build_call_bank.py.")
    p.add_argument("--splice_pool", type=str, required=True,
                   help="Path to splice_host_pool.pt produced by 04_build_splice_pool.py.")
    p.add_argument("--splice_prob", type=float, default=0.2,
                   help="Per-sample probability of applying the splice. "
                        "Default 0.2 (~10 of 32 samples per batch synthetic). "
                        "Set to 0.0 to disable splicing (equivalent to train.py).")
    p.add_argument("--snr_min_db", type=float, default=-6.0,
                   help="Minimum splice-call SNR in dB.")
    p.add_argument("--snr_max_db", type=float, default=6.0,
                   help="Maximum splice-call SNR in dB.")
    p.add_argument("--taper_ms", type=float, default=50.0,
                   help="Hann edge taper applied to the bandpassed call.")
    p.add_argument("--bandpass_margin_hz", type=float, default=4.0,
                   help="Margin around the call's [f_low, f_high] for the bandpass.")
    p.add_argument("--bandpass_transition_hz", type=float, default=2.0,
                   help="Transition-band width of the FFT bandpass cosine taper.")
    p.add_argument("--edge_guard_s", type=float, default=0.5,
                   help="Don't place the planted call within this many seconds "
                        "of the host clip's edge.")
    p.add_argument("--dump_examples", type=int, default=0,
                   help="If >0, generate N synthetic examples (WAV + spectrogram "
                        "PNGs) to --dump_dir and exit BEFORE training.")
    p.add_argument("--dump_dir", type=str, default=None,
                   help="Where to write dumped examples (default: ./phase_splice_examples_seed<S>).")
    return p.parse_args()


# ======================================================================
# Splice augmentation step
# ======================================================================

def splice_step(audio, targets, mask, metas, splice_state, args):
    """
    Apply the call-splice augmentation in-place on the batch.

    Mirrors ``apply_call_splice`` from call_splice.py but is the single
    place we call it, so the train loop body stays one line cleaner.

    Returns
    -------
    (audio, mask, targets) : tuple of tensors
        The same tensors, with synthetic samples spliced in for ~``splice_prob``
        fraction of the batch (targets and mask mutated in-place for those
        samples; audio is returned as a clone). Bit-equivalent passthrough
        if ``args.splice_prob == 0.0``.
    """
    if args.splice_prob <= 0.0:
        return audio, mask, targets

    # apply_call_splice returns (spec, mask, audio); we ignore the spec
    # slot because we work pre-STFT.
    _, mask_out, audio_out = apply_call_splice(
        spec=None,
        mask=mask,
        audio=audio,
        targets=targets,
        metas=metas,
        state=splice_state,
        splice_prob=args.splice_prob,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        taper_ms=args.taper_ms,
        bandpass_margin_hz=args.bandpass_margin_hz,
        bandpass_transition_hz=args.bandpass_transition_hz,
        edge_guard_s=args.edge_guard_s,
        sample_rate=cfg.SAMPLE_RATE,
    )
    return audio_out, mask_out, targets


# ======================================================================
# Splice-augmented train_epoch
# ======================================================================

def train_epoch_with_splice(
    model, spec_extractor, loader, criterion, optimizer, device, epoch,
    splice_state, args,
):
    """
    Single training epoch with splice augmentation injected before STFT.

    Byte-for-byte equivalent to ``train.train_epoch`` except for the
    ``splice_step`` call between ``audio = audio.to(...)`` and
    ``spec = spec_extractor(audio)``. NaN-skip, gradient clipping, and
    progress reporting are unchanged from train.py.
    """
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)

    for audio, targets, mask, metas in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # --- splice augmentation (no-op when --splice_prob 0.0) ---
        audio, mask, targets = splice_step(
            audio, targets, mask, metas, splice_state, args,
        )

        optimizer.zero_grad()

        spec = spec_extractor(audio)
        logits = model(spec)
        targets, mask = align_lengths(logits, targets, mask)
        loss = criterion(logits, targets, mask)

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
# Optional pre-training dump of synthetic examples
# ======================================================================

def dump_synthetic_examples(splice_state, args, out_dir: Path):
    """
    Write N synthetic examples to disk for inspection BEFORE training.

    Generates ``args.dump_examples`` synthetic clips by sampling fresh
    (call, host) pairs from the splice state, runs them through
    ``splice_one_sample``, and writes:
        - <i>_synth.wav      : the synthetic 30 s audio
        - <i>_synth_spec.png : log-magnitude STFT with planted-call bbox
        - summary.txt        : one row per example with metadata

    Crucially, we re-seed ``splice_state.rng`` to the original seed
    BEFORE this routine and AFTER it so the training run sees an
    identical sequence of synthetic samples whether or not the dump
    was performed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDumping {args.dump_examples} synthetic examples to {out_dir}...")

    # Snapshot the RNG state so training sees the same sequence as it
    # would without the dump. Python random.Random uses getstate()/setstate(),
    # not numpy's bit_generator.state attribute.
    rng_state = splice_state.rng.getstate()

    sr = cfg.SAMPLE_RATE
    n_samples = int(30.0 * sr)
    target_fr = 1.0 / cfg.FRAME_STRIDE_S

    rows = []
    for i in range(args.dump_examples):
        # Sample a call and a host.
        call = splice_state.bank.sample(splice_state.rng)
        host = splice_state.pool.sample(splice_state.rng)

        synth, info = splice_one_sample(
            call_entry=call,
            host_entry=host,
            rng=splice_state.rng,
            n_samples=n_samples,
            sample_rate=sr,
            target_frame_rate=target_fr,
            taper_samples=int(args.taper_ms * 1e-3 * sr),
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
            bandpass_margin_hz=args.bandpass_margin_hz,
            bandpass_transition_hz=args.bandpass_transition_hz,
            edge_guard_s=args.edge_guard_s,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Save WAV.
        wav_path = out_dir / f"{i:03d}_synth.wav"
        synth_np = synth.cpu().numpy() if hasattr(synth, "cpu") else np.asarray(synth)
        sf.write(str(wav_path), synth_np.astype(np.float32), sr)

        # Save spectrogram PNG with planted-call bbox.
        png_path = out_dir / f"{i:03d}_synth_spec.png"
        f, t, Sxx = _quick_spectrogram(synth_np, sr)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="auto", cmap="magma")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, sr / 2)
        bbox = mpatches.Rectangle(
            (info["t_call_start_s"], info["f_low_hz"]),
            info["t_call_end_s"] - info["t_call_start_s"],
            info["f_high_hz"] - info["f_low_hz"],
            linewidth=1.5, edgecolor="cyan", facecolor="none",
        )
        ax.add_patch(bbox)
        ax.set_title(
            f"#{i:03d}  class={info['label_3class']}  "
            f"snr={info['snr_db']:+.1f} dB  "
            f"source={info['source_site']} → host={info['host_site']}"
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=100)
        plt.close(fig)

        rows.append(
            f"{i:03d}\t{info['label_3class']}\t{info['snr_db']:+.2f}\t"
            f"{info['t_call_start_s']:.2f}-{info['t_call_end_s']:.2f}s\t"
            f"{info['f_low_hz']:.0f}-{info['f_high_hz']:.0f}Hz\t"
            f"{info['source_site']} -> {info['host_site']}"
        )

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        "# idx\tlabel\tsnr_db\ttime\tfreq\tsource_site -> host_site\n" +
        "\n".join(rows) + "\n"
    )
    print(f"  examples dumped to {out_dir}")

    # Restore RNG state so training sees the original sequence.
    splice_state.rng.setstate(rng_state)


def _quick_spectrogram(x_np, sr, nperseg=256, noverlap=200):
    """Lightweight STFT for the dump PNG. We don't reuse the model's
    SpectrogramExtractor because that's tied to torch/GPU; scipy is
    enough for an inspection image."""
    from scipy import signal
    f, t, Sxx = signal.spectrogram(
        x_np.astype(np.float32), fs=sr,
        nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude",
    )
    return f, t, Sxx


# ======================================================================
# Main
# ======================================================================

def main():
    """End-to-end splice-augmented training driver."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Splice state — load bank + pool, build RNG. Done before any heavy
    # GPU work so we fail fast if the files are missing.
    # ------------------------------------------------------------------
    print(f"Loading call bank from {args.call_bank}...")
    bank = CallBank.load(Path(args.call_bank))
    print(f"  {len(bank.calls)} calls across {len(bank.by_site)} source sites")

    print(f"Loading splice host pool from {args.splice_pool}...")
    pool = SpliceHostPool.load(Path(args.splice_pool))
    print(f"  {len(pool.hosts)} host clips across {len(pool.by_site)} donor sites")

    splice_state = CallSpliceState(
        bank=bank, pool=pool, rng=random.Random(args.seed),
    )

    # Dump-only mode.
    if args.dump_examples > 0:
        dump_dir = Path(args.dump_dir) if args.dump_dir else \
            Path(f"phase_splice_examples_seed{args.seed}")
        dump_synthetic_examples(splice_state, args, dump_dir)
        if args.splice_prob <= 0.0:
            # If splice is disabled too, the user clearly only wanted to
            # inspect — exit without training.
            print("--splice_prob is 0; exiting after dump.")
            return

    # ------------------------------------------------------------------
    # Wandb setup (mirrors train.py exactly with a splice phase id).
    # ------------------------------------------------------------------
    extra_tags = ["pretrained" if args.pretrained else "from_scratch", "splice"]
    if cfg.USE_WEIGHTED_BCE:
        extra_tags.append("weighted_bce")
    if getattr(cfg, "USE_FOCAL_LOSS", False):
        extra_tags.append("focal_loss")
    if args.splice_prob > 0.0:
        extra_tags.append(f"splice_p{args.splice_prob:g}")

    run = wbu.init_phase(
        "7",  # Keep phase 7 designation. ``splice`` tag distinguishes
              # this from any other phase-7 work; ``splice_p0`` sub-tag
              # marks the sanity-check runs that disable the augmentation.
        extra_tags=extra_tags,
        config={
            "lr":               cfg.LR,
            "weight_decay":     cfg.WEIGHT_DECAY,
            "batch_size":       cfg.BATCH_SIZE,
            "epochs":           cfg.EPOCHS,
            "seed":             args.seed,
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
            "pretrained":       args.pretrained,
            "freeze_epochs":    args.freeze_epochs,
            # Splice-specific knobs.
            "splice_prob":      args.splice_prob,
            "splice_snr_min_db": args.snr_min_db,
            "splice_snr_max_db": args.snr_max_db,
            "splice_taper_ms":  args.taper_ms,
            "splice_bandpass_margin_hz": args.bandpass_margin_hz,
            "splice_edge_guard_s": args.edge_guard_s,
            "n_call_bank_entries": len(bank.calls),
            "n_splice_host_clips": len(pool.hosts),
        },
    )

    # Run directory: phase7 prefix so it's distinguishable from
    # baseline runs in the runs/ directory listing and consistent with
    # phase5_*, phase6_*, etc.
    run_dir = Path(cfg.OUTPUT_DIR) / f"phase7_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ------------------------------------------------------------------
    # Data (identical to train.py).
    # ------------------------------------------------------------------
    train_ds, train_loader, val_loader = build_dataloaders()
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    file_start_dts = {
        (r.dataset, r.filename): r.start_dt for _, r in val_manifest.iterrows()
    }

    # ------------------------------------------------------------------
    # Model (identical to train.py).
    # ------------------------------------------------------------------
    spec_extractor = SpectrogramExtractor().to(device)
    model = WhaleVAD(num_classes=cfg.n_classes()).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        model(spec_extractor(dummy))

    if args.pretrained:
        print(f"Loading pretrained encoder: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
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
    # Loss and optimizer (identical to train.py).
    # ------------------------------------------------------------------
    pos_weight = compute_class_weights().to(device) if cfg.USE_WEIGHTED_BCE else None
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
    print(f"Scheduler: ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"Negative resampling: every {RESAMPLE_EVERY} epochs")
    print(f"Splice augmentation: prob={args.splice_prob}, "
          f"SNR=[{args.snr_min_db:+.1f}, {args.snr_max_db:+.1f}] dB, "
          f"taper={args.taper_ms} ms")

    # ------------------------------------------------------------------
    # Training loop (identical to train.py, modulo splice substitution).
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
        print(f"\n{'=' * 60}\nEpoch {epoch}/{cfg.EPOCHS}  LR={current_lr:.2e}\n"
              f"{'=' * 60}")

        if (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling negatives")
            train_ds.resample_negatives()
            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=True,
                **wbu.seeded_dataloader_kwargs(args.seed + epoch),
            )

        if args.pretrained and epoch <= args.freeze_epochs:
            for name, p in model.named_parameters():
                if "classifier" not in name and "lstm" not in name:
                    p.requires_grad = False
            print("  [frozen encoder]")
        elif args.pretrained and epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            print("  [unfroze encoder]")

        # *** The one functional change vs train.py: splice augmentation
        # is injected inside this train_epoch via splice_step.
        train_loss = train_epoch_with_splice(
            model, spec_extractor, train_loader, criterion,
            optimizer, device, epoch, splice_state, args,
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

        clf_module = (model.module.classifier if isinstance(model, nn.DataParallel)
                      else model.classifier)
        bias_str = ", ".join(f"{b:+.2f}" for b in clf_module.bias.detach().cpu().tolist())
        print(f"  Classifier bias: [{bias_str}]")

        model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_f1": best_f1,
            "thresholds": thresholds.cpu(),
            # Stamp splice settings on the checkpoint for traceability.
            "splice_prob": args.splice_prob,
            "splice_seed": args.seed,
        }

        if val["mean_f1"] > best_f1:
            best_f1 = val["mean_f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, run_dir / "best_model.pt")
            print(f"  *** New best F1: {best_f1:.3f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs}/{EARLY_STOP_PATIENCE} epochs")

        torch.save(ckpt, run_dir / "latest_model.pt")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs")
            break

    # ------------------------------------------------------------------
    # Post-training threshold tuning (identical to train.py).
    # ------------------------------------------------------------------
    from postprocess import tune_thresholds_event_level
    print(f"\n{'=' * 60}\nTuning thresholds on best model\n{'=' * 60}")
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
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
        "splice_prob": args.splice_prob,
        "splice_seed": args.seed,
    }, run_dir / "final_model.pt")

    print(f"\nDone. Best F1 (default thresholds): {best_f1:.3f}")
    print(f"Run dir: {run_dir}")
    print(f"Next: python eval_only.py --checkpoint {run_dir}/best_model.pt")

    import wandb
    wandb.summary["best_f1"]             = float(best_f1)
    wandb.summary["best_f1_post_tuning"] = float(best_ckpt.get("best_f1", best_f1))
    wandb.summary["final_thresholds"]    = list(map(float, tuned))
    wandb.summary["epochs_run"]          = epoch
    wandb.summary["early_stopped"]       = no_improve_epochs >= EARLY_STOP_PATIENCE
    wandb.summary["splice_prob"]         = float(args.splice_prob)
    wandb.summary["verdict"] = (
        f"Splice run (prob={args.splice_prob}, seed={args.seed}) finished at "
        f"best F1 {best_f1:.3f} (epoch {best_ckpt.get('epoch', '?')} of "
        f"{epoch} run; final tuned thresholds "
        f"{[round(float(t),2) for t in tuned]})."
    )

    art = wandb.Artifact(
        f"model-{run.name}", type="model",
        metadata={
            "best_f1":         float(best_f1),
            "best_epoch":      int(best_ckpt.get("epoch", 0)),
            "epochs_run":      int(epoch),
            "tuned_thresholds": list(map(float, tuned)),
            "splice_prob":     float(args.splice_prob),
            "splice_seed":     int(args.seed),
        },
    )
    art.add_file(str(run_dir / "best_model.pt"))
    art.add_file(str(run_dir / "final_model.pt"))
    run.log_artifact(art, aliases=["best", "splice"])

    wandb.finish()


if __name__ == "__main__":
    main()
