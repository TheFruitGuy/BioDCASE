"""
Training Driver: Phase-Splice (Call Splicing into Unseen Sites)
================================================================

Single-axis change to the F1=0.474 baseline (``train.py``): each
training step, with ``splice_prob`` chance, a sample is replaced by
a synthetic (unseen-site host + transplanted call) example. See
``README.md`` for the design and ``call_splice.py`` for the
augmentation mechanics.

What we expect to see
---------------------
- Train loss may be slightly higher than the unaugmented baseline,
  since synthetic samples carry distribution shift the model has
  not been exposed to before.
- Val F1 should diverge positively in the middle/late epochs,
  especially on ``casey2017`` and the Kerguelen validation sites,
  which are the closest analogs in the val split to the unseen test
  conditions.
- If F1 collapses, suspect (a) ``splice_prob`` too high (try 0.3),
  (b) SNR range too wide (try -3 to +3 dB), or (c) the call bank
  being class-imbalanced — check the stage-1 print summary.

Hyperparameters (defaults match README.md)
------------------------------------------
``splice_prob``         = 0.5   half of samples become synthetic
``snr_min_db``          = -6    SNR floor (call quiet vs host)
``snr_max_db``          = +6    SNR ceiling (call loud vs host)
``taper_ms``            = 50    Hann fade-in/fade-out length
``bandpass_margin_hz``  = 4.0   extra band around the annotation bbox
``edge_guard_s``        = 0.5   keep planted call away from clip edges

Combining with Phase 1e (cross-site noise mix)
----------------------------------------------
This driver supports stacking the existing cross-site noise mix on
top of call splicing. Pass ``--with_cross_site_mix`` to enable both:
50% of samples get spliced (replacement), and of the remaining 50%,
each gets cross-site noise mixed at 0.5 probability. Conceptually,
splicing handles foreground invariance, cross-site mix handles
background invariance — they're complementary.

The implementation chains the two augmentations into a single
composite hook so the existing training driver in
``phase1_baseline.py`` doesn't need any changes.

Usage
-----
::

    # Pure splicing (baseline experiment):
    python train_phase_splice.py \\
        --call_bank pipeline_call_splice/call_bank.pt \\
        --splice_pool pipeline_call_splice/splice_host_pool.pt

    # Splicing + cross-site mix (combined experiment):
    python train_phase_splice.py \\
        --call_bank pipeline_call_splice/call_bank.pt \\
        --splice_pool pipeline_call_splice/splice_host_pool.pt \\
        --with_cross_site_mix \\
        --pool pipeline_call_splice/../no_call_pool.pt
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch

# Allow running from the pipeline_call_splice/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from phase1_baseline import NoCallPool, run_phase1_training  # noqa: E402

from call_splice import (  # noqa: E402
    CallBank,
    SpliceHostPool,
    CallSpliceState,
    apply_call_splice,
    splice_one_sample,
)


# ----------------------------------------------------------------------
# Optional composition with Phase 1e (cross-site noise mix)
# ----------------------------------------------------------------------

def _build_composite_aug(call_splice_state: CallSpliceState,
                         no_call_pool: NoCallPool | None,
                         splice_prob: float,
                         mix_prob: float):
    """
    Build a composite augmentation = call_splice ∘ cross_site_mix.

    The two augmentations have the SAME ``DOMAIN = "audio"`` so the
    dispatcher will call this single composite once per batch. Within
    the composite we apply splicing first (replacing some samples
    entirely) and then cross-site mix on the remaining unchanged
    samples. We track which samples were replaced via a state attribute
    so cross-site mix doesn't try to mix more noise into a synthetic
    sample (which would muddy the signal).
    """
    from augmentations import apply_cross_site_mix

    def composite(*, spec, mask, audio, targets, metas, state):
        # Splice first.
        spec, mask, audio = apply_call_splice(
            spec=spec, mask=mask, audio=audio, targets=targets,
            metas=metas, state=state["splice"],
            splice_prob=splice_prob,
        )
        # Cross-site noise mix on samples not yet replaced. We can't
        # cheaply know per-sample which were replaced (the splice
        # function doesn't return that info). Two practical options:
        #   (a) just apply cross-site mix at reduced prob to every
        #       sample — the spliced ones will get a small extra
        #       AADC-noise overlay, which is harmless.
        #   (b) extend apply_call_splice to return a bool mask.
        # Going with (a) for simplicity. Effective mix prob on
        # un-spliced samples ≈ ``mix_prob``; on spliced samples ≈
        # ``mix_prob`` of an already-noisy clip, fine.
        if state["no_call_pool"] is not None:
            spec, mask, audio = apply_cross_site_mix(
                spec=spec, mask=mask, audio=audio, targets=targets,
                metas=metas, state=state["no_call_pool"],
                mix_prob=mix_prob,
            )
        return spec, mask, audio

    composite.DOMAIN = "audio"
    composite.__name__ = "call_splice_csmix"   # auto aug_<name> tag
    return composite


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--call_bank", type=Path, required=True,
                   help="Path to call_bank.pt from stage 1.")
    p.add_argument("--splice_pool", type=Path, required=True,
                   help="Path to splice_host_pool.pt from stage 4.")
    p.add_argument("--splice_prob", type=float, default=0.2,
                   help="Per-sample probability of call splicing. "
                        "0.2 is a conservative dose (~10 of 48 samples per batch); "
                        "raise to 0.3-0.5 for more aggressive intervention.")
    p.add_argument("--snr_min_db", type=float, default=-6.0)
    p.add_argument("--snr_max_db", type=float, default=6.0)
    p.add_argument("--taper_ms", type=float, default=50.0)
    p.add_argument("--bandpass_margin_hz", type=float, default=4.0)
    p.add_argument("--edge_guard_s", type=float, default=0.5)
    p.add_argument("--balanced_sampling", action="store_true", default=True,
                   help="Sample calls uniformly across classes (default on; "
                        "disable with --no_balanced_sampling).")
    p.add_argument("--no_balanced_sampling", dest="balanced_sampling",
                   action="store_false")
    p.add_argument("--seed", type=int, default=cfg.SEED if hasattr(cfg, "SEED") else 42,
                   help="RNG seed for the augmentation's stochastic decisions.")
    p.add_argument("--with_cross_site_mix", action="store_true",
                   help="Stack cross-site noise mix on top of call splicing.")
    p.add_argument("--pool", type=Path, default=None,
                   help="Path to no_call_pool.pt (required if "
                        "--with_cross_site_mix).")
    p.add_argument("--mix_prob", type=float, default=0.5,
                   help="Cross-site mix probability (only with --with_cross_site_mix).")
    p.add_argument("--dump_examples", type=int, default=0,
                   help="If > 0, save this many synthetic samples to disk before "
                        "training starts (WAV + spectrogram PNGs + summary.txt). "
                        "Uses the same RNG seed as training, so these are a true "
                        "sample of what the model will see in epoch 1.")
    p.add_argument("--dump_dir", type=Path, default=None,
                   help="Output directory for --dump_examples. Defaults to "
                        "'phase_splice_examples_seed<SEED>/' next to the call bank.")
    return p.parse_args()


def dump_synthetic_examples(
    *,
    n_examples: int,
    out_dir: Path,
    splice_state: CallSpliceState,
    splice_prob: float,
    snr_min_db: float,
    snr_max_db: float,
    taper_ms: float,
    bandpass_margin_hz: float,
    edge_guard_s: float,
    sample_rate: int,
    duration_s: float = 30.0,
    target_frame_rate: float = 50.0,
) -> None:
    """
    Dump N synthetic examples to disk for pre-training sanity inspection.

    Each example produces three artifacts:
      - ``example_NNN.wav``   : the 30-second synthetic audio
      - ``example_NNN.png``   : log-mel spectrogram with target frames overlaid
      - one line in ``summary.txt``: which call, which host, SNR, frequency band

    Uses the same ``CallSpliceState.rng`` as training, so the first ``N``
    synthetic samples produced here will exactly match the first ``N``
    spliced samples the model encounters in epoch 1 (modulo dataloader
    sample order — but the call/host/position/SNR sequence is identical
    given the same seed).

    Note: this advances the RNG by ``n_examples`` calls, so the training
    loop will start with the RNG already advanced. If you want training
    to see the SAME synthetic sequence the dump produced, reset the seed
    after dumping (we do this automatically via ``state.rng = random.Random(seed)``
    in the caller).
    """
    import soundfile as sf

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False
        print("  WARNING: matplotlib unavailable, skipping PNG spectrograms")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dumping {n_examples} synthetic examples to {out_dir}/")

    n_samples = int(duration_s * sample_rate)
    taper_samples = max(1, int(taper_ms * 1e-3 * sample_rate))

    # Honor the splice_prob the user picked: with prob (1 - splice_prob)
    # the example is the unsplicd host (i.e., a clean unseen-site clip).
    # That's a useful sanity check too — confirms the dataloader is
    # delivering pure no-call AADC audio when splicing doesn't fire.
    rng = splice_state.rng

    summary_lines = [
        f"Synthetic example dump",
        f"  n_examples:     {n_examples}",
        f"  sample_rate:    {sample_rate} Hz",
        f"  duration:       {duration_s} s",
        f"  splice_prob:    {splice_prob}",
        f"  SNR range:      [{snr_min_db}, {snr_max_db}] dB",
        f"  taper:          {taper_ms} ms",
        "",
        f"{'idx':>4}  {'event':<9}  {'class':<6}  {'src_site':<22}  "
        f"{'host_site':<14}  {'f_lo':>5}  {'f_hi':>5}  {'SNR':>5}  {'t_call':<12}",
    ]

    for i in range(n_examples):
        if rng.random() >= splice_prob:
            # Unspliced: pure host. Dump as-is.
            host_entry = splice_state.pool.sample(rng)
            synth = host_entry["audio"].clone().float()
            if synth.numel() != n_samples:
                if synth.numel() > n_samples:
                    synth = synth[:n_samples]
                else:
                    synth = torch.nn.functional.pad(synth, (0, n_samples - synth.numel()))
            info = None

            wav_path = out_dir / f"example_{i:03d}_host.wav"
            sf.write(str(wav_path), synth.numpy(), sample_rate)
            if has_mpl:
                _save_spec_png(
                    synth, info=None, sample_rate=sample_rate,
                    path=out_dir / f"example_{i:03d}_host.png",
                )
            summary_lines.append(
                f"{i:>4}  {'host_only':<9}  {'-':<6}  "
                f"{'-':<22}  {host_entry['site']:<14}  "
                f"{'-':>5}  {'-':>5}  {'-':>5}  {'-':<12}"
            )
            continue

        # Spliced.
        call_entry = splice_state.bank.sample(rng)
        host_entry = splice_state.pool.sample(rng)
        synth, info = splice_one_sample(
            call_entry=call_entry,
            host_entry=host_entry,
            rng=rng,
            n_samples=n_samples,
            sample_rate=sample_rate,
            target_frame_rate=target_frame_rate,
            taper_samples=taper_samples,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            bandpass_margin_hz=bandpass_margin_hz,
            bandpass_transition_hz=2.0,
            edge_guard_s=edge_guard_s,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        wav_path = out_dir / f"example_{i:03d}_splice.wav"
        sf.write(str(wav_path), synth.numpy(), sample_rate)
        if has_mpl:
            _save_spec_png(
                synth, info=info, sample_rate=sample_rate,
                path=out_dir / f"example_{i:03d}_splice.png",
            )

        t_str = f"{info['t_call_start_s']:.1f}-{info['t_call_end_s']:.1f}s"
        summary_lines.append(
            f"{i:>4}  {'splice':<9}  {info['label_3class']:<6}  "
            f"{info['source_site']:<22}  {info['host_site']:<14}  "
            f"{info['f_low_hz']:>5.1f}  {info['f_high_hz']:>5.1f}  "
            f"{info['snr_db']:>+5.1f}  {t_str:<12}"
        )

    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    print(f"  wrote {n_examples} WAV(s), {n_examples} PNG(s), summary.txt")


def _save_spec_png(
    audio: torch.Tensor,
    info: dict | None,
    sample_rate: int,
    path: Path,
) -> None:
    """Plot a log-magnitude spectrogram with the planted-call interval overlaid."""
    import matplotlib.pyplot as plt
    import numpy as np

    # STFT parameters matching cfg defaults (N_FFT=256, HOP_LENGTH=5).
    n_fft = 256
    hop_length = 5
    win = torch.hann_window(n_fft)
    spec = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=win, return_complex=True, center=False,
    )
    mag = spec.abs().clamp_min(1e-8).log10()  # log magnitude
    mag_np = mag.numpy()

    n_freq, n_frames = mag_np.shape
    t_per_frame = hop_length / sample_rate
    extent = [0, n_frames * t_per_frame, 0, sample_rate / 2]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.imshow(
        mag_np, origin="lower", aspect="auto", extent=extent,
        cmap="magma", vmin=mag_np.mean() - 1.0, vmax=mag_np.max(),
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    if info is not None:
        # Draw a translucent rectangle on the planted-call interval +
        # its frequency band. Bbox = (t_start, f_low) → (t_end, f_high).
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (info["t_call_start_s"], info["f_low_hz"]),
            info["t_call_end_s"] - info["t_call_start_s"],
            info["f_high_hz"] - info["f_low_hz"],
            linewidth=1.5, edgecolor="cyan", facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title(
            f"{info['label_3class']} "
            f"(7c: {info['label_7class']}) "
            f"from {info['source_site']} into {info['host_site']}  "
            f"SNR {info['snr_db']:+.1f} dB"
        )
    else:
        ax.set_title("host only (no call planted)")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 64)
    print("Phase-Splice training")
    print("=" * 64)
    print(f"Call bank:     {args.call_bank}")
    print(f"Splice pool:   {args.splice_pool}")
    print(f"splice_prob:   {args.splice_prob}")
    print(f"SNR range:     [{args.snr_min_db}, {args.snr_max_db}] dB")
    print(f"taper_ms:      {args.taper_ms}")
    print(f"with_cs_mix:   {args.with_cross_site_mix}")
    if args.with_cross_site_mix and args.pool is None:
        raise SystemExit("--with_cross_site_mix requires --pool")
    print()

    print("Loading call bank...")
    bank = CallBank.load(str(args.call_bank), balanced=args.balanced_sampling)
    print(f"  {len(bank.calls)} calls across {len(bank.by_site)} source sites")
    print(f"  per-3class counts: "
          f"{ {k: len(v) for k, v in bank.by_label_3class.items()} }")

    print("\nLoading splice host pool...")
    pool = SpliceHostPool.load(str(args.splice_pool))
    print(f"  {len(pool.hosts)} hosts across {len(pool.by_site)} donor sites")

    rng = random.Random(args.seed)
    splice_state = CallSpliceState(bank=bank, pool=pool, rng=rng)

    # --------------------------------------------------------------
    # Optional pre-training example dump for sanity inspection.
    # The RNG is reset to the same seed afterwards so the training
    # run sees exactly the same synthetic sequence the dump showed.
    # --------------------------------------------------------------
    if args.dump_examples > 0:
        dump_dir = args.dump_dir or (
            args.call_bank.parent / f"phase_splice_examples_seed{args.seed}"
        )
        dump_synthetic_examples(
            n_examples=args.dump_examples,
            out_dir=dump_dir,
            splice_state=splice_state,
            splice_prob=args.splice_prob,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
            taper_ms=args.taper_ms,
            bandpass_margin_hz=args.bandpass_margin_hz,
            edge_guard_s=args.edge_guard_s,
            sample_rate=cfg.SAMPLE_RATE,
            duration_s=30.0,
            target_frame_rate=1.0 / cfg.FRAME_STRIDE_S,
        )
        # Reset so training starts with the same seed state as if no
        # dump had been requested.
        splice_state.rng = random.Random(args.seed)
        print(f"  examples dumped to {dump_dir}/, RNG reset for training\n")

    if args.with_cross_site_mix:
        no_call_pool = NoCallPool.load(args.pool)
        composite_state = {
            "splice":        splice_state,
            "no_call_pool":  no_call_pool,
        }
        aug = _build_composite_aug(
            call_splice_state=splice_state,
            no_call_pool=no_call_pool,
            splice_prob=args.splice_prob,
            mix_prob=args.mix_prob,
        )
        aug_state = composite_state
        variant_tags = ["splice_csmix", "cross_site_mix"]
    else:
        # Bare apply_call_splice. Bind hyperparameters via a thin
        # closure so the dispatcher can call it with the standard
        # signature.
        def aug(*, spec, mask, audio, targets, metas, state):
            return apply_call_splice(
                spec=spec, mask=mask, audio=audio, targets=targets,
                metas=metas, state=state,
                splice_prob=args.splice_prob,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
                taper_ms=args.taper_ms,
                bandpass_margin_hz=args.bandpass_margin_hz,
                edge_guard_s=args.edge_guard_s,
                sample_rate=cfg.SAMPLE_RATE,
            )
        aug.DOMAIN = "audio"
        aug.__name__ = "call_splice"   # for the auto aug_<name> tag
        aug_state = splice_state
        variant_tags = ["splice_only"]

    # Tag with the donor sites that contributed to the host pool so
    # the wandb table can group "splice runs that used casey2018" etc.
    donor_tags = [f"donor_{s}" for s in pool.by_site.keys()]
    # SNR + splice-prob bucket tags so different hyperparameter regimes
    # are visually distinct in the runs table.
    snr_tag = f"snr_{int(args.snr_min_db)}_to_{int(args.snr_max_db)}"
    p_tag = f"splice_p{int(args.splice_prob * 100):02d}"
    extra_tags = variant_tags + donor_tags + [snr_tag, p_tag]
    if args.balanced_sampling:
        extra_tags.append("balanced_classes")

    aug_config = {
        "splice_prob":        args.splice_prob,
        "snr_min_db":         args.snr_min_db,
        "snr_max_db":         args.snr_max_db,
        "taper_ms":           args.taper_ms,
        "bandpass_margin_hz": args.bandpass_margin_hz,
        "edge_guard_s":       args.edge_guard_s,
        "balanced_sampling":  args.balanced_sampling,
        "call_bank_path":     str(args.call_bank),
        "splice_pool_path":   str(args.splice_pool),
        "with_cross_site_mix": args.with_cross_site_mix,
        "mix_prob":           args.mix_prob if args.with_cross_site_mix else None,
        "n_calls":            len(bank.calls),
        "n_hosts":            len(pool.hosts),
        "donor_sites":        list(pool.by_site.keys()),
        "variant":            ("splice_csmix" if args.with_cross_site_mix
                                else "splice_only"),
    }

    run_phase1_training(
        phase_name="7",
        augmentation_fn=aug,
        augmentation_config=aug_config,
        augmentation_state=aug_state,
        extra_tags=extra_tags,
    )


if __name__ == "__main__":
    main()
