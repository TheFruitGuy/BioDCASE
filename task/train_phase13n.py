"""
Phase 13n -- FDY-Conformer + PCEN extra channel  (a D-targeted base lever)
==========================================================================

13d (FDY-Conformer) with ONE extra input channel: fixed PCEN on the raw STFT
magnitude (see ``spectrogram_pcen``). Clean A/B -- channels 0-2 are 13d's exact
phase-aware STFT; channel 3 is the transient-enhanced PCEN view that should let
the weak, background-buried D downsweeps stand out. The stem's filterbank just
takes 4 input channels instead of 3 (FDY-swapped as in 13d).

Reuses ``conformer_core`` with 13d's recipe (RAdam, constant LR 5e-5, no warmup,
weighted BCE, per-epoch negative resampling, fixed-0.3 validation, macro_paper
selection, W&B), so versus 13d the ONLY change is the PCEN channel. ::

    CUDA_VISIBLE_DEVICES=<nonBlackwell> \
        /home/matthias-nagl/.conda/envs/bio/bin/python train_phase13n.py

Baseline: 13d at macro_paper 0.424 / MACRO 0.390 / D ceiling 0.20. The question
is whether channel 3 lifts D; watch D F1 in the first ~10 epochs.

Levers
------
``--pcen-smooth``  fixed PCEN EMA smoothing s (default 0.025 ~ 0.8 s background
                   tracking). Smaller = slower background, so longer D calls pop
                   more; larger = faster, risks normalising the call itself.
``--fdy-targets`` / ``--fdy-basis`` / ``--fdy-temp``  the 13d FDY stem (defaults).

Do NOT pass --filteraug or --amp (clean A/B vs 13d; both were off there).
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen(device, a):
    """FDY-Conformer with feat_channels=4 (3 STFT + 1 PCEN) + the PCEN extractor.
    One dummy forward materialises the lazy projection before training."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                 # store for eval rebuild
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        feat_channels=spec.n_out_channels,                # 4 = 3 STFT + 1 PCEN
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13n: FDY-Conformer + PCEN extra channel")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Identical recipe to 13d so the ONLY difference is the PCEN channel.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs are FDY (default both, as 13d).")
    p.add_argument("--fdy-basis", type=int, default=4,
                   help="K basis kernels per FDY conv (13d default 4).")
    p.add_argument("--fdy-temp", type=float, default=1.0,
                   help="FDY attention softmax temperature (1.0 = plain).")
    p.add_argument("--pcen-smooth", type=float, default=0.025,
                   help="Fixed PCEN EMA smoothing s (default 0.025).")
    a = p.parse_args()

    # Eval-loader discriminator: a 13n checkpoint carries fdy_targets (like 13d)
    # PLUS pcen_channel, so eval_conformer swaps in the PCEN extractor and builds
    # the FDY-Conformer with feat_channels read from arch_kwargs.
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13n", build_model=build_conformer_fdy_pcen, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", "pcen_channel", "13n"],
    )


if __name__ == "__main__":
    main()
