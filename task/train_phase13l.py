"""
Phase 13l -- FDY-Conformer + tfwSE  (frequency lever for D)
===========================================================

Single axis on top of 13d: add one time-frame frequency-wise SE after the
(FDY) filterbank, at F=41. 13k showed the backbone is not the gap (CNN-BiLSTM
ties the Conformer under a matched recipe); D stalls at ~0.20 everywhere, and
the diagnosis is a frontend-sensitivity miss. tfwSE re-weights frequency
*per time frame*, so the emphasised band can track a D downsweep. See
``model_tfwse`` for the mechanism and the caveats (paper's best is SE+tfwSE;
tfwSE may be redundant with FDY).

Clean A/B
---------
Reuses ``conformer_core`` with exactly 13d's recipe -- RAdam, constant LR 5e-5,
no warmup, weighted BCE, per-epoch negative resampling, fixed-0.3 validation,
macro_paper selection, W&B -- so versus 13d the *only* change is the tfwSE
block (+871 params). Any movement, especially in D, is attributable to it. ::

    CUDA_VISIBLE_DEVICES=0 python train_phase13l.py

is the comparison run against 13d's 0.429 / 0.419 / D 0.204.

Levers
------
``--fdy-targets`` / ``--fdy-basis`` / ``--fdy-temp``  the 13d FDY stem (defaults).
``--tfwse-reduction``  frequency bottleneck ratio in tfwSE (default 4 -> 41->10->41).
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_tfwse(device, a):
    """FDY-Conformer with a tfwSE block after the filterbank. Mirrors 13d's
    builder; one dummy forward materialises the lazy projection AND the lazy
    tfwSE frequency conv before training."""
    from model_tfwse import WhaleVAD_Conformer_FDY_tfwSE
    from spectrogram_final import SpectrogramExtractor

    model = WhaleVAD_Conformer_FDY_tfwSE(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult, conv_kernel=a.conv_kernel,
        dropout=a.dropout,
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp, tfwse_reduction=a.tfwse_reduction,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13l: FDY-Conformer + tfwSE")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Identical recipe to 13d so the ONLY difference is tfwSE.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs are FDY (default both, as 13d).")
    p.add_argument("--fdy-basis", type=int, default=4,
                   help="K basis kernels per FDY conv (13d default 4).")
    p.add_argument("--fdy-temp", type=float, default=1.0,
                   help="FDY attention softmax temperature (1.0 = plain).")
    p.add_argument("--tfwse-reduction", type=int, default=4,
                   help="Frequency bottleneck ratio in tfwSE (default 4).")
    a = p.parse_args()

    # Eval-loader discriminator: the 13l checkpoint carries fdy_targets (like
    # 13d) PLUS this flag, so eval_conformer rebuilds the tfwSE subclass instead
    # of the plain FDY-Conformer (checked before the fdy_targets branch).
    a.tfwse = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13l", build_model=build_conformer_fdy_tfwse, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["conformer", "fdy", "tfwse", "13l"],
    )


if __name__ == "__main__":
    main()
