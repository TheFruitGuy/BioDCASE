"""
Phase 13k -- FDY-CRNN  (backbone test: BiLSTM vs Conformer under the FDY stem)
==============================================================================

The recheck confirmed WhaleVAD is a CNN-BiLSTM (CRNN), reproduced at ~0.443
macro_paper -- above the FDY-Conformer's 0.429 on the same dev split -- and the
SED literature reports Conformers over-smooth in time relative to CRNNs. This
rung isolates the backbone: it takes 13d's FDY stem (the one frontend lever that
moved the needle) and puts it on the proven-stronger BiLSTM instead of the
Conformer body. See ``model_crnn_fdy`` for the architecture.

Clean A/B
---------
This reuses ``conformer_core`` with exactly 13d's near-base recipe -- RAdam,
constant LR 5e-5, no warmup, weighted BCE, per-epoch negative resampling,
fixed-0.3 validation, macro_paper selection, W&B -- so versus the FDY-Conformer
(13d) the *only* thing that changes is the sequence model (BiLSTM vs Conformer).
Any movement, especially in D, is attributable to the backbone. A bare ::

    CUDA_VISIBLE_DEVICES=0 python train_phase13k.py

is the comparison run. The Conformer-specific arch flags from ``add_arch_args``
(``--d-model`` etc.) exist but are ignored by the CRNN builder.

Note on AMP
-----------
``--amp`` defaults OFF here. The CRNN's BiLSTM is O(T), not the Conformer's
O(T^2) attention, so 30 s / batch 32 fits comfortably in fp32 -- which also
matches 13d's fp32 precision, keeping the A/B clean.

Levers
------
``--fdy-targets`` {filterbank,feat0}  which early convs become FDY (default both).
``--fdy-basis``    K basis kernels per FDY conv (default 4, the paper's value).
``--fdy-temp``     attention softmax temperature (default 1.0 = plain softmax).

Evaluate the saved checkpoint with ``eval_conformer.py``; verify it rebuilds the
CRNN class from the checkpoint's arch_kwargs (it was written for Conformer
checkpoints -- a CRNN-load branch may be needed).
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_crnn_fdy(device, a):
    """FDY-stem CNN-BiLSTM builder. Mirrors ``build_conformer_fdy`` but builds
    the WhaleVAD CRNN; one dummy forward materialises the lazy projection."""
    from model_crnn_fdy import WhaleVAD_FDY
    from spectrogram_final import SpectrogramExtractor

    model = WhaleVAD_FDY(
        num_classes=cfg.n_classes(), feat_channels=cfg.n_feat_channels(),
        fdy_targets=tuple(a.fdy_targets), fdy_basis=a.fdy_basis,
        fdy_temp=a.fdy_temp,
    ).to(device)
    spec = SpectrogramExtractor().to(device)
    with torch.no_grad():
        model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))
    return model, spec


def main():
    p = argparse.ArgumentParser(
        description="Phase 13k: FDY-CRNN (CNN-BiLSTM + FDY stem)")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    # Same near-base recipe as 13d so the ONLY difference vs the FDY-Conformer is
    # the backbone. AMP off: the BiLSTM is O(T), fp32 fits and matches 13d.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4,
                   help="K basis kernels per FDY conv (paper default 4).")
    p.add_argument("--fdy-temp", type=float, default=1.0,
                   help="Attention softmax temperature (1.0 = plain softmax).")
    a = p.parse_args()

    # Eval-time discriminator. 13k's arch_kwargs carry fdy_targets exactly like
    # the FDY-Conformer (13d), so on its own eval_conformer would rebuild
    # WhaleVAD_Conformer_FDY and fail the strict load. This flag (saved into the
    # checkpoint's arch_kwargs) tells the loader to rebuild WhaleVAD_FDY instead.
    a.backbone = "crnn"

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13k", build_model=build_crnn_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False, extra_tags=["crnn", "fdy", "13k", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
