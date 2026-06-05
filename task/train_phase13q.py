"""
Phase 13q -- FDY-Conformer + PCEN channel + wide conv kernel (k=65)
===================================================================

13n with the Conformer conv module's depthwise kernel widened 31 -> 65, and
nothing else changed. At hop=5 / 250 Hz a frame is 20 ms, so k=31 spans only
~0.62 s of LOCAL context, but a D downsweep is 1-2 s -- the conv module cannot
see a whole D call, only the global self-attention does. k=65 (~1.30 s) lets the
local feature extractor track the full downsweep. The kernel is depthwise, so
the cost is negligible (+34 taps x d_model). This targets D's temporal extent,
not generic capacity.

``eval_conformer`` reads conv_kernel straight off the checkpoint's arch_kwargs,
so this evaluates through the existing pcen_channel branch with NO eval change.

Reuses ``conformer_core`` with 13d's recipe; versus 13n the ONLY change is the
conv kernel width. ::

    CUDA_VISIBLE_DEVICES=<nonBlackwell> \
        /home/matthias-nagl/.conda/envs/bio/bin/python train_phase13q.py

Baseline: 13n at tuned D 0.336 / MACRO 0.463 (k=31). Does a D-spanning local
kernel lift D? ``--conv-kernel`` overrides (default 65 here; MUST stay odd so the
depthwise padding preserves length). ``--pcen-smooth`` as in 13n (default 0.025).
Do NOT pass --filteraug or --amp.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen(device, a):
    """13n's builder verbatim (FDY-Conformer, feat_channels=4: 3 STFT + 1 PCEN,
    + the PCEN extractor); conv_kernel comes from args (default widened to 65)."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                  # store for eval rebuild
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        feat_channels=spec.n_out_channels,                 # 4 = 3 STFT + 1 PCEN
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13q: FDY-Conformer + PCEN channel + wide conv kernel")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # 13d/13n recipe, but widen the depthwise conv kernel to span a D downsweep.
    # conv_kernel=65 overrides add_arch_args' default of 31 (must be odd).
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   conv_kernel=65)
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

    if a.conv_kernel % 2 == 0:
        raise SystemExit(f"--conv-kernel must be odd (got {a.conv_kernel}); "
                         "the depthwise conv pads (k-1)//2 each side.")

    # Eval discriminator: a 13q checkpoint carries fdy_targets PLUS pcen_channel
    # and feat_channels=4 (like 13n), and conv_kernel=65 is read off arch_kwargs,
    # so eval rebuilds the FDY-Conformer at k=65 with the PCEN extractor.
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13q", build_model=build_conformer_fdy_pcen, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,
        extra_tags=["conformer", "fdy", "pcen_channel", "wide_kernel", "13q"],
    )


if __name__ == "__main__":
    main()
