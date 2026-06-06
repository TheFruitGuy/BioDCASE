"""
Phase 13r -- wide-kernel FDY-Conformer stability harness + tuned per-epoch eval
===============================================================================

13q (FDY-Conformer + PCEN channel + conv_kernel=65) is the current conformer-line
winner (tuned D 0.362 / MACRO 0.465), but its training is *unstable*: D collapses
intermittently (val-loss spikes to ~0.24 with D F1 -> ~0.01 every 5-7 epochs),
so the "best" epoch is partly luck and multi-seed will be high-variance. 13r is
the same wide-kernel recipe with three INDEPENDENT stability levers exposed as
flags so each can be ablated, plus two measurement upgrades baked in:

  measurement (default ON):
    --eval-mode tuned   per-epoch validation runs per-class threshold tuning
                        instead of fixed-0.3 (fixed-0.3 has mis-ranked epochs
                        repeatedly in this project). --tune-workers parallelises it.
    --select-by macro   the best checkpoint is chosen on tuned mean-per-class F1
                        (the official BioDCASE metric), not fixed-0.3 macro_paper.

  stability levers (default OFF / 13q-equivalent -- add to ablate one at a time):
    --ema               EMA-of-weights; the averaged weights never contain a
                        spike-epoch's parameters -> smooths the D collapse.
    --lr-schedule cosine  anneal the LR to ~0 (default here is 'const' = 13q);
                        a constant LR over 40 epochs keeps perturbing late.
    --resample-every N  hold the negative set fixed for N epochs instead of
                        redrawing 58k negatives every epoch (default 1 = every
                        epoch); a fresh D-confusable draw lurches the boundary.

Default 13r (no extra flags) = the 13q recipe re-measured under tuned-eval +
macro-selection -> a clean anchor. Each lever is purely additive on top. ::

    # anchor (13q recipe, tuned eval, macro selection):
    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13r.py --tune-workers 8
    # ablate one lever:
    ... train_phase13r.py --tune-workers 8 --ema
    ... train_phase13r.py --tune-workers 8 --lr-schedule cosine
    ... train_phase13r.py --tune-workers 8 --resample-every 3
    # full stack:
    ... train_phase13r.py --tune-workers 8 --ema --lr-schedule cosine --resample-every 3

conv_kernel default 65 (MUST stay odd). --pcen-smooth as in 13n (0.025). Do NOT
pass --filteraug or --amp.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen(device, a):
    """13n/13q builder verbatim (FDY-Conformer, feat_channels=4: 3 STFT + 1 PCEN,
    + the PCEN extractor); conv_kernel from args (default widened to 65)."""
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
        description="Phase 13r: wide-kernel FDY-Conformer stability harness")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Default = 13q recipe (const LR, no EMA, resample every epoch) so each
    # stability lever is a pure addition; but turn the measurement upgrades ON.
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
    a = p.parse_args()

    if a.conv_kernel % 2 == 0:
        raise SystemExit(f"--conv-kernel must be odd (got {a.conv_kernel}); "
                         "the depthwise conv pads (k-1)//2 each side.")

    # Eval rebuild: like 13q, the checkpoint carries pcen_channel + feat_channels=4
    # + conv_kernel, so eval reuses the existing pcen_channel branch unchanged.
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13r", build_model=build_conformer_fdy_pcen, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,                                     # --ema flag drives it
        extra_tags=["conformer", "fdy", "pcen_channel", "wide_kernel",
                    "stability", "13r"],
    )


if __name__ == "__main__":
    main()
