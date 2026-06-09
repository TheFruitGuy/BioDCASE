"""
Phase 13v -- thin temporal decoder head on the FDY-Conformer recipe
===================================================================

The conformer line removed WhaleVAD's BiLSTM *backbone* wholesale. A light
decoder *head* on top of the per-frame features is a different thing: one
bidirectional GRU layer, or a 2-3 layer dilated TCN, between the Conformer
blocks and the classifier (still before the sigmoid).

Why this targets D specifically: eval is event-level (IoU >= 0.3), and D's
dominant failure under background is *fragmented / dropped events* -- the
per-frame head flickers across the 90->25 Hz downsweep. A smoothing decoder
attacks event-level temporal coherence directly, and it is orthogonal to every
lever in the ledger (front-end, kernel width, depth, optimisation).

This is the locked recipe -- single wide kernel **k=129** + PCEN-4ch + FDY stem
+ tuned per-epoch eval + macro selection + (--ema) -- with ONE addition:

    --decoder {gru,tcn}     a residual decoder head over the d_model features,
                            applied after the blocks and before the classifier.
                            none/omitted == the plain recipe (byte-identical).
    --decoder-layers N      GRU depth (default 1) or number of dilated TCN
                            blocks (dilation 1,2,4,...; use 3 for ~k=15 eff. RF).

Placement note: the head runs in **feature space** (d_model), not on the 3-class
logits. A 3-channel logit head is ~a learnable median filter, which would just
relearn the post-hoc min-duration / merge-gap smoothing in postprocess_final.py
(D merge-gap ~3 s already validated) -- the failure mode to avoid. Feature-space
gives the head real context to close gaps the post-proc cannot. ::

    # GRU head on the recipe (judge on D event-F1 vs single k=129 0.495):
    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13v.py \
        --ema --conv-kernel 129 --decoder gru --decoder-layers 1 --tune-workers 20

    # dilated-TCN head (closest to a learnable post-proc smoother):
    ... python train_phase13v.py --ema --conv-kernel 129 --decoder tcn --decoder-layers 3 --tune-workers 20

The checkpoint carries pcen_channel + feat_channels=4 + decoder + decoder_layers,
so eval rebuilds through the existing pcen_channel branch (decoder flows via the
shared `common` kwargs) with no eval change; the head adds ``decoder.*`` keys.

Honest falsification: if the head does not beat the tuned post-processed recipe
on D, it has only relearned the median filter -> drop it. Do NOT pass
--filteraug or --amp.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen_decoder(device, a):
    """Recipe builder (single k=129, PCEN-4ch, FDY) + an optional decoder head.

    Threads ``decoder`` / ``decoder_layers`` into the model. ``conv_kernels`` is
    left None (this is the single-wide-kernel recipe, not multi-scale)."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                  # store for eval rebuild
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult,
        conv_kernel=a.conv_kernel,                         # single wide kernel (129)
        dropout=a.dropout,
        decoder=a.decoder,                                 # "gru" | "tcn"
        decoder_layers=a.decoder_layers,
        feat_channels=spec.n_out_channels,                 # 4 = 3 STFT + 1 PCEN
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13v: temporal decoder head on the FDY-Conformer recipe")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Locked recipe defaults: single wide kernel k=129 (not 13q's 65).
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   conv_kernel=129, eval_mode="tuned", select_by="macro")
    p.add_argument("--decoder", choices=["gru", "tcn"], default="gru",
                   help="Decoder head over d_model frame features before the "
                        "classifier. gru = 1+ layer BiGRU; tcn = dilated "
                        "depthwise-separable Conv1d stack.")
    p.add_argument("--decoder-layers", type=int, default=1,
                   help="GRU depth or number of dilated TCN blocks (default 1; "
                        "use 3 for the TCN).")
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
        raise SystemExit(f"--conv-kernel must be odd (got {a.conv_kernel}).")

    # Eval rebuild: pcen_channel + feat_channels=4 + decoder(+layers) carried in
    # the checkpoint -> eval reuses the pcen_channel branch unchanged.
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13v", build_model=build_conformer_fdy_pcen_decoder, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,                                     # --ema flag drives it
        extra_tags=["conformer", "fdy", "pcen_channel", "decoder_head",
                    a.decoder, "13v"],
    )


if __name__ == "__main__":
    main()
