"""
Phase 13m -- LEAF learnable frontend + Conformer  (the last base swing)
=======================================================================

LEAF replaces the fixed log-mel/STFT frontend with a learnable Gabor
filterbank + Gaussian lowpass + sPCEN (see ``model_leaf`` for the
paper-accurate implementation, ported from CPJKU/EfficientLEAF). Unlike every
prior 13x lever, which reweights a fixed STFT, this changes *what is measured*.
It is the BioDCASE winner's load-bearing frontend (their LEAF 0.64 vs log-mel
0.44).

Clean A/B
---------
LEAF sits on the PLAIN Conformer (no FDY): the comparison is a frontend swap,
LEAF-vs-STFT, on the same backbone. Reuses ``conformer_core`` with the base
recipe -- RAdam, constant LR 5e-5, no warmup, weighted BCE, per-epoch negative
resampling, fixed-0.3 validation, macro_paper selection, W&B. ::

    CUDA_VISIBLE_DEVICES=<nonBlackwell> \
        /home/matthias-nagl/.conda/envs/bio/bin/python train_phase13m.py

Baselines: plain Conformer (STFT) ~0.404, and 13d (STFT+FDY) 0.429/0.419/D 0.204.

Do NOT pass --filteraug or --amp: LEAF consumes raw audio (spectrogram augment
would corrupt the waveform), and the complex Gabor filters must not run under
bf16 autocast.

Levers
------
``--leaf-init`` mel | linear   filterbank init. mel (paper default) is ~linear
                               below 1 kHz, so it matches the narrow whale band.
``--leaf-compression`` pcen | log   pcen is the paper default (learnable sPCEN).
                               If PCEN's AGC flattens the output and the model
                               collapses (the prior PCEN-as-frontend failure
                               mode), ``log`` is the validated whale fallback.
``--leaf-n-filters`` (default 128)   matches the STFT's post-stem F.
``--leaf-min-freq`` (default 5.0 Hz)
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_leaf(device, a):
    """LEAF-Conformer (feat_channels forced to 1 inside the model) + a passthrough
    spec_extractor that hands the raw waveform to the model. One dummy forward
    materialises the lazy projection before training."""
    from model_leaf import WhaleVAD_Conformer_LEAF, LeafPassthrough

    model = WhaleVAD_Conformer_LEAF(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        leaf_n_filters=a.leaf_n_filters, leaf_init=a.leaf_init,
        leaf_compression=a.leaf_compression, leaf_min_freq=a.leaf_min_freq,
    ).to(device)
    spec = LeafPassthrough().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(description="Phase 13m: LEAF frontend + Conformer")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Base recipe (same as the plain Conformer / 13d) so the only change is LEAF.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const")
    p.add_argument("--leaf-n-filters", type=int, default=128,
                   help="LEAF Gabor filters (default 128, ~matches STFT post-stem F).")
    p.add_argument("--leaf-init", choices=["mel", "linear"], default="mel",
                   help="Gabor init. mel = paper default (~linear below 1 kHz).")
    p.add_argument("--leaf-compression", choices=["pcen", "log"], default="pcen",
                   help="pcen = paper default (sPCEN); log = whale fallback.")
    p.add_argument("--leaf-min-freq", type=float, default=5.0,
                   help="Lowest filter centre frequency in Hz (default 5).")
    a = p.parse_args()

    # Eval-loader discriminator: a 13m checkpoint carries leaf=True and NO
    # fdy_targets/tfwse, so eval_conformer rebuilds the LEAF subclass and swaps
    # in the passthrough extractor.
    a.leaf = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13m", build_model=build_leaf, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,
        extra_tags=["conformer", "leaf", "13m", a.leaf_init, a.leaf_compression],
    )


if __name__ == "__main__":
    main()
