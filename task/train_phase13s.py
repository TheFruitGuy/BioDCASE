"""
Phase 13s -- wide-kernel FDY-Conformer + delta-channel front-end
================================================================

Stacks temporal-derivative channels on the 13q/13r winner. The PCEN-channel
extractor (mag, cos, sin, PCEN) is extended with delta and delta-delta channels
computed along time from the PCEN channel -> 6-channel input:

    [mag, cos, sin, PCEN, delta(PCEN), delta-delta(PCEN)]

Why deltas, for D specifically: the D-call is a 90->25 Hz downsweep, 1-4 s, with
high FM variability, and ~71.5% of D misses in this project are background-
sensitivity (not inter-class confusion). A stationary noise floor has a near-zero
temporal derivative while a moving FM contour does not, so a delta channel is an
intrinsic high-pass-in-time filter -- it suppresses the stationary background and
emphasises the moving downsweep. This is the cheap, parameter-less version of the
pitch-tracking cue classic low-frequency detectors relied on; the deltas are a
fixed linear transform (no learnable params), so channels 0..3 stay byte-identical
to 13q and only the front-end information content changes.

Inherits 13r's stability harness + tuned-per-epoch eval unchanged (--ema,
--lr-schedule, --resample-every, --eval-mode, --select-by, --tune-workers).
Front-end knobs: --delta-order {1,2} (default 2), --delta-source {pcen,mag}
(default pcen). conv_kernel default 65 (wide kernel kept; deltas stack on the
winner). Do NOT pass --filteraug or --amp. ::

    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13s.py --tune-workers 8
    # delta of the magnitude channel instead of PCEN, first-order only:
    ... train_phase13s.py --tune-workers 8 --delta-source mag --delta-order 1
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_delta(device, a):
    """FDY-Conformer with the delta-channel front-end (feat_channels = 4 + order).
    Mirrors 13q's builder; only the extractor (and hence in-channels) changes."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from delta_channels import DeltaChannelExtractor

    spec = DeltaChannelExtractor(smooth=a.pcen_smooth, order=a.delta_order,
                                 source=a.delta_source).to(device)
    a.feat_channels = spec.n_out_channels                  # 4 + order; store for eval
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        feat_channels=spec.n_out_channels,                 # 4 STFT+PCEN + delta(s)
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13s: wide-kernel FDY-Conformer + delta-channel front-end")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   conv_kernel=65, eval_mode="tuned", select_by="macro")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs are FDY (default both, as 13d).")
    p.add_argument("--fdy-basis", type=int, default=4,
                   help="K basis kernels per FDY conv (13d default 4).")
    p.add_argument("--fdy-temp", type=float, default=1.0,
                   help="FDY attention softmax temperature (1.0 = plain).")
    p.add_argument("--pcen-smooth", type=float, default=0.025,
                   help="Fixed PCEN EMA smoothing s (default 0.025).")
    p.add_argument("--delta-order", type=int, default=2, choices=[1, 2],
                   help="Append delta only (1) or delta + delta-delta (2).")
    p.add_argument("--delta-source", choices=["pcen", "mag"], default="pcen",
                   help="Differentiate the PCEN channel (default) or the demeaned "
                        "magnitude channel.")
    a = p.parse_args()

    if a.conv_kernel % 2 == 0:
        raise SystemExit(f"--conv-kernel must be odd (got {a.conv_kernel}); "
                         "the depthwise conv pads (k-1)//2 each side.")

    # Eval rebuild: a 13s checkpoint carries delta=True + delta_order/delta_source
    # + feat_channels (4+order) + conv_kernel, so eval_conformer's delta branch
    # reconstructs the DeltaChannelExtractor and the matching FDY-Conformer.
    a.delta = True
    a.pcen_channel = False

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13s", build_model=build_conformer_fdy_delta, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,                                     # --ema flag drives it
        extra_tags=["conformer", "fdy", "wide_kernel", "delta",
                    "stability", "13s"],
    )


if __name__ == "__main__":
    main()
