"""
Phase 13d — Conformer with a Frequency-Dynamic Conv stem
========================================================

The D-class lever. 13a–13c established that the Conformer caps at ~0.40
macro_paper with D-F1 ~0.15 regardless of optimiser, weight-averaging, or
threshold tuning — D is a *frontend* limitation. This rung replaces the two
early stem convs (the learnable filterbank and the first feature-extractor
conv, the only ones with enough frequency resolution to matter) with
**Frequency-Dynamic Convolutions** (Nam et al. 2022), whose kernel varies per
frequency bin. See ``model_conformer_fdy`` for the architecture and placement
rationale. ~+0.64 M params (2.0 M → 2.6 M), all from the 5×5 feat0 conv.

Clean A/B
---------
This defaults to the *proven-stable near-base* recipe — RAdam, constant LR
5e-5, no warmup — i.e. the recipe under which the plain Conformer reached its
best tuned 0.404. So a bare ::

    CUDA_VISIBLE_DEVICES=0 python train_phase13d.py

changes exactly one thing versus that baseline — the FDY stem — and any
movement (especially in D) is attributable to it. The optimiser flags from
``add_opt_args`` are still available to override (e.g. ``--optimizer sf-radam``
to stack FDY on the schedule-free recipe once 13a's verdict is in).

Levers
------
``--fdy-targets`` {filterbank,feat0}  which early convs become FDY (default both).
                                      ``--fdy-targets filterbank`` is the near-free
                                      variant (~2.0 M params) if 2.6 M is too heavy.
``--fdy-basis``    K basis kernels per FDY conv (default 4, the paper's value).
``--fdy-temp``     attention softmax temperature (default 1.0 = plain softmax).

Everything else (data, weighted-BCE, per-epoch resampling, fixed-0.3 validation,
macro_paper selection, W&B) is the final pipeline via conformer_core. Evaluate
the saved checkpoint with ``eval_conformer.py`` — it detects the FDY stem from
the checkpoint's arch_kwargs and rebuilds the right class automatically.
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy(device, a):
    """FDY-stem Conformer builder (mirrors conformer_core.build_conformer but
    swaps the early convs for FDY). One dummy forward materialises the lazy
    projection before training."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_final import SpectrogramExtractor

    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(description="Phase 13d: Conformer + FDY stem")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Default to the stable near-base recipe so FDY is the only change vs 0.404.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4,
                   help="K basis kernels per FDY conv (paper default 4).")
    p.add_argument("--fdy-temp", type=float, default=1.0,
                   help="Attention softmax temperature (1.0 = plain softmax).")
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13d", build_model=build_conformer_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
