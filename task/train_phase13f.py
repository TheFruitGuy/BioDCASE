"""
Phase 13f — consolidated strong base: FDY + FilterAugment + EMA
==========================================================================

The "better base model and training" rung, aimed at clearing ~0.45 macro_paper
(WhaleVAD's 0.440 reference). It stacks the validated frequency-adaptive frontend
with the proven, structure-safe training recipe:

  * FDY stem (13d)                           — the validated frequency-adaptive
    frontend; the one lever that moved D (tuned D ~0.15 -> ~0.20).
  * FilterAugment + frequency warping       — proven SED regularisation that is
    frequency-only, so the temporal call structure (Z-call unit order, fin pulse
    rhythm) is preserved — unlike the call-splice augmentation that failed.
  * EMA weight averaging                     — DESED-standard stabiliser; averages
    over the per-epoch negative-resampling noise that drives the swings.

This is just the 13d FDY trainer with `--ema --filteraug` defaulted on, logged
under its own phase label. dual-res is OFF by default — the 13d A/B found it ties
tuned D (0.206 vs 0.204) while regressing BMABZ/BP/micro, so it was dropped; it is
still available via `--dual-res` for the ablation row under 13f. A bare ::

    CUDA_VISIBLE_DEVICES=<gpu> python train_phase13f.py

runs the recipe. For the attribution ablations, run the 13d trainer with the flags
off/subset (e.g. `train_phase13d.py --ema` to drop FilterAugment), which logs under
13d so the pair is easy to compare. Multi-seed the winner.

The saved checkpoint is evaluated with `eval_conformer.py` exactly like 13d — it
auto-detects the FDY stem from arch_kwargs (and the 4-channel input only if you opt
into --dual-res); EMA and FilterAugment are training-only and need no eval-side
handling.
"""

import argparse

from conformer_core import (run_training, add_arch_args, add_opt_args,
                            opt_kwargs_from)
from train_phase13d import build_conformer_fdy
import config_final as cfg


def main():
    p = argparse.ArgumentParser(
        description="Phase 13f: FDY + dual-res + FilterAugment + EMA")
    add_arch_args(p)
    add_opt_args(p, default_optimizer="radam")
    p.add_argument("--fdy-targets", nargs="+", default=["filterbank", "feat0"],
                   choices=["filterbank", "feat0"],
                   help="Which early stem convs become FDY (default both).")
    p.add_argument("--fdy-basis", type=int, default=4)
    p.add_argument("--fdy-temp", type=float, default=1.0)
    # The 13f recipe: near-base optimiser + FilterAugment + EMA on.
    # dual_res is intentionally NOT defaulted on — the 13d A/B found it ties tuned
    # D (0.206 vs 0.204) while regressing BMABZ/BP/micro, so it was dropped. It is
    # still reachable via the --dual-res flag for the ablation row under 13f.
    p.set_defaults(peak_lr=5e-5, warmup_epochs=0.0, lr_schedule="const",
                   ema=True, filteraug=True)
    a = p.parse_args()

    cfg.USE_3CLASS = (a.num_classes == 3)
    run_training(
        "13f", build_model=build_conformer_fdy, arch_kwargs=a,
        opt_kwargs=opt_kwargs_from(a),
        epochs=a.epochs, seed=a.seed, wandb_mode=a.wandb_mode,
        use_ema=a.ema, extra_tags=["conformer", "fdy", "13f", *a.fdy_targets],
    )


if __name__ == "__main__":
    main()
