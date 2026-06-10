"""
Phase 13w -- loss-function swap on the FDY-Conformer recipe (D-recall probe)
============================================================================

The whole conformer line trained with one objective: masked, segment-count-
weighted BCE. The optimisation ablations (EMA / cosine / resample) never touched
the *loss shape*. D's bottleneck is sensitivity — its positive frames are a tiny
minority drowned by the negative flood — which is exactly what a recall-oriented
loss attacks. 13w swaps the training loss while holding the locked recipe fixed
(single k=129 + PCEN-4ch + FDY stem + tuned per-epoch eval + macro select +
--ema). Architecture, data, and optimiser are unchanged; only the loss differs.

    --loss {bce,asl,tversky}     (default asl)
      bce      original masked weighted BCE — the control; reproduces 0.495.
      asl      Asymmetric Loss (Ridnik/Ben-Baruch ICCV'21): down-weights easy
               negatives so rare D positives keep gradient mass. NOT a per-class
               focal gamma, and trained FROM SCRATCH (not the failed Phase-8
               focal fine-tune). gamma_pos=0, gamma_neg=4, clip=0.05.
      tversky  Focal-Tversky (Abraham&Khan'18): beta>alpha up-weights false
               negatives -> recall on D; gamma>1 focuses hard class-segments.
               alpha=0.3, beta=0.7, gamma=4/3.

The loss is a training-only concern — the checkpoint's architecture is identical
to the recipe, so eval rebuilds through the existing pcen_channel branch with no
eval change. Validation loss is always reported as BCE (comparable across runs);
selection is on tuned macro F1 as always. ::

    # the two recall-loss probes, judged on D-F1 (and macro) vs single k=129 0.495:
    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13w.py \
        --ema --conv-kernel 129 --loss asl --tune-workers 20
    CUDA_VISIBLE_DEVICES=<nonBlackwell> python train_phase13w.py \
        --ema --conv-kernel 129 --loss tversky --tune-workers 20
    # BCE control (should reproduce the recipe 0.495 within seed noise):
    ... python train_phase13w.py --ema --conv-kernel 129 --loss bce --tune-workers 20

Do NOT pass --filteraug or --amp. ASL/Tversky get NO pos_weight by design
(their imbalance handling replaces it; the recipe pos_weight still applies to bce).
"""

import argparse

import torch

import config_final as cfg
from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)


def build_conformer_fdy_pcen(device, a):
    """The locked recipe builder: single wide-kernel FDY-Conformer + PCEN 4th
    channel. Loss selection happens in run_training (training-only), so the
    model here is identical to 13r/13v's recipe model."""
    from model_conformer_fdy import WhaleVAD_Conformer_FDY
    from spectrogram_pcen import PCENChannelExtractor

    spec = PCENChannelExtractor(smooth=a.pcen_smooth).to(device)
    a.feat_channels = spec.n_out_channels                  # store for eval rebuild
    model = WhaleVAD_Conformer_FDY(
        num_classes=cfg.n_classes(), d_model=a.d_model, nhead=a.nhead,
        num_layers=a.layers, ffn_mult=a.ffn_mult,
        conv_kernel=a.conv_kernel,                         # single wide kernel (129)
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
        description="Phase 13w: loss-function swap on the FDY-Conformer recipe")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   conv_kernel=129, eval_mode="tuned", select_by="macro")
    # --- loss selection (read by conformer_core.make_train_criterion) ---
    p.add_argument("--loss", choices=["bce", "asl", "tversky"], default="asl",
                   help="Training loss. bce=original (control); asl=Asymmetric "
                        "Loss; tversky=Focal-Tversky. Default asl.")
    p.add_argument("--asl-gamma-neg", type=float, default=4.0,
                   help="ASL focusing on negatives (down-weights easy negs).")
    p.add_argument("--asl-gamma-pos", type=float, default=0.0,
                   help="ASL focusing on positives (0 = no down-weighting).")
    p.add_argument("--asl-clip", type=float, default=0.05,
                   help="ASL probability shift on negatives (hard-thresholds easy negs).")
    p.add_argument("--tversky-alpha", type=float, default=0.3,
                   help="Tversky FP weight (alpha < beta tilts toward recall).")
    p.add_argument("--tversky-beta", type=float, default=0.7,
                   help="Tversky FN weight (higher = more recall on rare D).")
    p.add_argument("--tversky-gamma", type=float, default=1.3333,
                   help="Focal-Tversky focusing exponent (>1 focuses hard segments).")
    # --- recipe FDY/PCEN knobs ---
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

    # Eval rebuild: pcen_channel + feat_channels=4 carried in the checkpoint ->
    # eval reuses the pcen_channel branch unchanged (loss args are ignored there).
    a.pcen_channel = True

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13w", build_model=build_conformer_fdy_pcen, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=False,                                     # --ema flag drives it
        extra_tags=["conformer", "fdy", "pcen_channel", "loss_swap", a.loss, "13w"],
    )


if __name__ == "__main__":
    main()
