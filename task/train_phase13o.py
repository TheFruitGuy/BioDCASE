"""
Phase 13o -- FDY-Conformer + LEARNABLE per-band PCEN channel  (ambitious 13n)
=============================================================================

13n with the PCEN channel's alpha / delta / power / smooth made LEARNABLE per
frequency bin (see ``model_lpcen``). The extractor
(``spectrogram_lpcen.RawMagChannelExtractor``) hands the model the raw STFT
magnitude as the last channel; the model's ``LearnablePCEN`` turns it into a
trainable PCEN view, so the optimiser -- which only touches
``model.parameters()`` -- tunes the per-band AGC to whatever surfaces D. Every
band inits at 13n's fixed values, so at step 0 this is byte-for-byte 13n.

Channels 0..2 are 13d/13n's exact phase-aware STFT -- whatever the PCEN params
learn, the loud BMABZ/BP information survives, so the LEAF / PCEN-as-frontend
flattening that killed D (11a/11b/13m) cannot recur here.

Reuses ``conformer_core`` with 13d's recipe (RAdam, constant LR 5e-5, no warmup,
weighted BCE, per-epoch negative resampling, fixed-0.3 validation, macro_paper
selection, W&B), so versus 13n the ONLY change is fixed -> learnable PCEN. ::

    CUDA_VISIBLE_DEVICES=<nonBlackwell> \
        /home/matthias-nagl/.conda/envs/bio/bin/python train_phase13o.py

Baseline: 13n at tuned D 0.336 / MACRO 0.463 (seed 42, fixed PCEN). Question:
does learning the per-band AGC beat the fixed one on tuned D? Eval is the
verdict -- the fixed-0.3 validation log will look like 13n until the thresholds
are tuned.

Levers
------
``--lpcen-global``  one shared PCEN (4 scalars) instead of per-band (4x129).
``--fdy-targets`` / ``--fdy-basis`` / ``--fdy-temp``  the 13d FDY stem (defaults).
Do NOT pass --filteraug or --amp (clean lineage vs 13n; both were off there).
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_lpcen(device, a):
    """FDY-Conformer (feat_channels=4: 3 STFT + 1 raw-mag) with a learnable
    per-band PCEN inside the model, plus the raw-magnitude extractor. One dummy
    forward materialises the lazy projection before training."""
    from model_lpcen import WhaleVAD_Conformer_FDY_LPCEN
    from spectrogram_lpcen import RawMagChannelExtractor

    spec = RawMagChannelExtractor().to(device)             # last channel = raw |STFT|
    a.feat_channels = spec.n_out_channels                  # store for eval rebuild
    model = WhaleVAD_Conformer_FDY_LPCEN(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        feat_channels=spec.n_out_channels,                 # 4 = 3 STFT + 1 raw-mag
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
        lpcen_global=a.lpcen_global,
    ).to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13o: FDY-Conformer + learnable per-band PCEN channel")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Identical recipe to 13d/13n so the ONLY difference is the learnable PCEN.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs are FDY (default both, as 13d).")
    p.add_argument("--fdy-basis", type=int, default=4,
                   help="K basis kernels per FDY conv (13d default 4).")
    p.add_argument("--fdy-temp", type=float, default=1.0,
                   help="FDY attention softmax temperature (1.0 = plain).")
    p.add_argument("--lpcen-global", action="store_true",
                   help="Share one PCEN across all bins (4 scalars) vs per-band.")
    a = p.parse_args()

    # Eval discriminator: a 13o checkpoint carries fdy_targets PLUS lpcen (NOT
    # pcen_channel), so eval_conformer builds the LPCEN subclass and swaps in the
    # raw-magnitude extractor. lpcen_global is stored so the param shape matches.
    a.lpcen = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13o", build_model=build_conformer_fdy_lpcen, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", "lpcen", "13o"],
    )


if __name__ == "__main__":
    main()
