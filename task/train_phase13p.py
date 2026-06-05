"""
Phase 13p -- FDY-Conformer + PCEN channel + dual-resolution (5 input channels)
==============================================================================

13n appended a fixed PCEN channel but ran with DUAL_RESOLUTION=off, so the PCEN
channel effectively REPLACED the short-window-magnitude channel 13d carried --
both are 4-channel, the 4th just differs. This rung turns DUAL_RESOLUTION on so
the model gets BOTH:

    ch 0..2  phase-aware STFT       (mag, cos phi, sin phi)
    ch 3     short-window magnitude (fine time resolution -> sharp BMABZ/BP onsets)
    ch 4     fixed PCEN             (transient AGC -> D sensitivity)

Hypothesis: short-window-mag and PCEN are complementary (time resolution vs
transient AGC), so having both lifts all three classes; if redundant, we learn
that cheaply at ~zero extra cost (the filterbank just takes 5 input channels).

Plumbing: ``run_training`` reads ``a.dual_res`` and sets ``cfg.DUAL_RESOLUTION``
BEFORE ``build_model`` runs, so ``PCENChannelExtractor`` returns 5 channels and
the FDY filterbank sizes to 5 automatically. ``eval_conformer`` already restores
``cfg.DUAL_RESOLUTION`` from the checkpoint and reads ``feat_channels``, so this
evaluates through the existing pcen_channel branch with NO eval change.

Reuses ``conformer_core`` with 13d's recipe; versus 13n the ONLY change is the
extra short-window-mag channel. ::

    CUDA_VISIBLE_DEVICES=<nonBlackwell> \
        /home/matthias-nagl/.conda/envs/bio/bin/python train_phase13p.py

Baseline: 13n at tuned D 0.336 / MACRO 0.463 (4-channel). Do 5 channels beat it?
``--pcen-smooth`` as in 13n (default 0.025). Do NOT pass --filteraug or --amp.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen5(device, a):
    """Identical to 13n's builder; with cfg.DUAL_RESOLUTION on (set by
    run_training from a.dual_res before this runs) the PCEN extractor yields 5
    channels and the FDY filterbank sizes to 5."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                  # 5 with dual_res on
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        feat_channels=spec.n_out_channels,                 # 5 = 3 STFT + short-mag + PCEN
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13p: FDY-Conformer + PCEN channel + dual resolution")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Identical recipe to 13d/13n so the ONLY difference is the extra channel.
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

    a.dual_res = True        # run_training sets cfg.DUAL_RESOLUTION from this -> 5 ch
    a.pcen_channel = True     # eval swaps in the PCEN extractor (now 5-channel)

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13p", build_model=build_conformer_fdy_pcen5, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,
        extra_tags=["conformer", "fdy", "pcen_channel", "dual_res", "13p"],
    )


if __name__ == "__main__":
    main()
