"""
Phase 13u -- concat-then-project multi-scale FDY-Conformer (BP-recovery, take 2)
================================================================================

13t ran several depthwise kernels in parallel inside the conv module and
**summed** them. Every multi-scale-by-sum variant lost to a single wide kernel
(k=129, 0.495): summing forces one channel to serve every scale, so the scales
entangle and the module reverts to an averaged-kernel behaviour.

13u keeps the kernels parallel but fuses by **concat-then-project** instead:
concatenate the per-kernel depthwise outputs on the channel axis, then a 1x1
pointwise conv projects len(ks)*d_model -> d_model. The scales stay *separable*
and the network *learns* the mixing weights (per output channel) instead of
being forced to add them. This directly attacks the mechanism that killed 13t,
and is the natural home for recovering the -0.006 BP that the single long kernel
cost (short branch for the brief BP pulse, long branch for the D downsweep /
BMABZ moan, projection learns to route).

This is the 13t harness verbatim (FDY stem, PCEN 4th channel, tuned per-epoch
eval, macro selection, --ema etc. as flags) with ONE change: ``--ms-fuse``
defaults to ``concat`` and the default kernel pair is ``65 129`` (BP-scale and
the D/BMABZ ceiling kernel) rather than 13t's ``15 65``. ::

    # the probe, on the EMA recipe (judge on D-F1 AND BP-F1 vs single k=129 0.495):
    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13u.py \
        --ema --conv-kernels 65 129 --ms-fuse concat --tune-workers 20

    # sum control (== 13t mechanism on the same pair, for the ablation):
    ... python train_phase13u.py --ema --conv-kernels 65 129 --ms-fuse sum --tune-workers 20

The checkpoint carries pcen_channel + feat_channels=4 + conv_kernels + ms_fuse,
so eval rebuilds through the existing pcen_channel branch (ms_fuse flows via the
shared `common` kwargs) with no eval change. A "concat" checkpoint adds the
``...conv.ms_proj.*`` keys; "sum"/single-kernel checkpoints do not.

Honest caveat: the 1x1 projection can also learn to *ignore* one branch and
collapse to single-kernel behaviour, in which case 13u ties 0.495 rather than
beating it. Cheap to find out. Do NOT pass --filteraug or --amp. Every
--conv-kernels entry MUST be odd.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen_ms_concat(device, a):
    """13t builder + a fusion mode for the multi-scale depthwise conv module.

    Threads both ``conv_kernels`` and ``ms_fuse`` into the model; ``conv_kernel``
    is kept in arch_kwargs for the eval rebuild plumbing but is unused when
    ``conv_kernels`` is set."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                  # store for eval rebuild
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult,
        conv_kernel=a.conv_kernel,                         # fallback; unused here
        conv_kernels=a.conv_kernels,                       # multi-scale parallel kernels
        ms_fuse=a.ms_fuse,                                 # "concat" (default) or "sum"
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
        description="Phase 13u: concat-then-project multi-scale FDY-Conformer")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   conv_kernel=65, eval_mode="tuned", select_by="macro")
    p.add_argument("--conv-kernels", nargs="+", type=int, default=[65, 129],
                   help="Odd depthwise kernel sizes run in parallel (default "
                        "65 129: BP-scale + the D/BMABZ ceiling kernel).")
    p.add_argument("--ms-fuse", choices=["concat", "sum"], default="concat",
                   help="How parallel-kernel outputs are fused. concat (default) "
                        "= cat on channels then 1x1 project (scales separable, "
                        "learned mixing); sum = 13t mechanism (entangles).")
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

    # Eval rebuild: pcen_channel + feat_channels=4 + conv_kernels + ms_fuse
    # carried in the checkpoint -> eval reuses the pcen_channel branch unchanged.
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13u", build_model=build_conformer_fdy_pcen_ms_concat, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,                                     # --ema flag drives it
        extra_tags=["conformer", "fdy", "pcen_channel", "multiscale",
                    "concat_project", "13u"],
    )


if __name__ == "__main__":
    main()
