"""
Phase 13t -- multi-scale depthwise FDY-Conformer (BP-recovery probe)
====================================================================

13r/EMA converged the conformer line at ~0.485 tuned MACRO with a *single*
depthwise kernel (conv_kernel=65 = 1.30 s). Going k=31 -> 65 bought D (+0.054)
but cost BP (-0.006): a 1.3 s kernel is too long for BP's short pulses. 13t
tests whether a **multi-scale** depthwise module recovers that BP loss while
keeping D's gain, by running several depthwise kernels in parallel inside the
conv module and summing them (short kernel for the brief BP pulse, long kernel
for the D downsweep). D's gain lives in the long branch, which stays, so the
intended outcome is "keep D + small BP".

This is 13r verbatim (FDY stem, PCEN 4th channel, tuned per-epoch eval, macro
selection, all stability levers as flags) with ONE addition:

    --conv-kernels K [K ...]   odd depthwise kernels run in parallel & summed
                               (default 15 65: ~0.30 s for BP, ~1.30 s for D).
                               When set, it replaces the single --conv-kernel
                               depthwise; conv_kernels=None (not passed) is the
                               plain single-kernel module, byte-identical to 13r.

The checkpoint carries pcen_channel + feat_channels=4 + conv_kernels, so eval
rebuilds through the existing pcen_channel branch (conv_kernels flows via the
shared `common` kwargs) with no eval change. ::

    # the BP-recovery probe, on the EMA recipe:
    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13t.py \
        --ema --conv-kernels 15 65 --tune-workers 20
    # single-kernel control == 13r (sanity; should match 13r EMA 0.485):
    ... train_phase13t.py --ema --conv-kernel 65 --tune-workers 20   # omit --conv-kernels? no:
    #   to reproduce 13r exactly, run train_phase13r.py; 13t always multi-scales.

Do NOT pass --filteraug or --amp. Every --conv-kernels entry MUST be odd.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen_ms(device, a):
    """13r builder + multi-scale depthwise conv module (parallel kernels summed).

    Identical to ``train_phase13r.build_conformer_fdy_pcen`` except it threads
    ``conv_kernels`` into the model; ``conv_kernel`` is kept in arch_kwargs for
    the eval rebuild plumbing but is unused when ``conv_kernels`` is set."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                  # store for eval rebuild
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult,
        conv_kernel=a.conv_kernel,                         # fallback; unused here
        conv_kernels=a.conv_kernels,                       # multi-scale parallel kernels
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
        description="Phase 13t: multi-scale depthwise FDY-Conformer")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Match 13r's measurement upgrades and 13q recipe defaults; stability levers
    # (--ema etc.) are inherited from add_arch_args and stay additive.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   conv_kernel=65, eval_mode="tuned", select_by="macro")
    p.add_argument("--conv-kernels", nargs="+", type=int, default=[15, 65],
                   help="Odd depthwise kernel sizes run in parallel and summed "
                        "(default 15 65: short for BP pulses, long for D sweep).")
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

    bad = [k for k in a.conv_kernels if k % 2 == 0]
    if bad:
        raise SystemExit(f"--conv-kernels must all be odd (got even {bad}); "
                         "each depthwise conv pads (k-1)//2 per side.")

    # Eval rebuild: pcen_channel + feat_channels=4 + conv_kernels carried in the
    # checkpoint -> eval reuses the existing pcen_channel branch unchanged.
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13t", build_model=build_conformer_fdy_pcen_ms, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,                                     # --ema flag drives it
        extra_tags=["conformer", "fdy", "pcen_channel", "multiscale", "13t"],
    )


if __name__ == "__main__":
    main()
