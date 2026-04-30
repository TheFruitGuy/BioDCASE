"""
wandb_utils.py — Phase-aware wandb integration for BioDCASE Task 2.

Each phase in the ladder (0, 0c, 0d, 0e, 0f, 0g, 0h, ...) tests one
intervention vs. its parent. This module makes that structure
first-class on every run so the wandb UI shows the ladder as a
filterable evolution rather than a pile of independent runs.

What lands on each run
----------------------
* ``config.phase``                e.g. "0h"
* ``config.parent_phase``         e.g. "0g"
* ``config.hypothesis``           one-line statement of what's being tested
* ``config.new_interventions``    what changed vs. parent
* ``config.cumulative_interventions``
                                  full list of interventions active in
                                  this run (walked via parent chain)
* ``tags``                        ``[phase0h, weighted_bce, 3class_output,
                                     multi_site_train, fixed_30s_train_segments,
                                     same_site_split]``
* ``notes``                       hypothesis text (visible in run header)
* ``summary.best_f1`` / ``best_epoch`` / ``final_f1``
* ``summary.second_half_{mean,max}_swing``  (the stability numbers your
                                  VERDICT block already computes)
* ``summary.verdict``             plain-English outcome
* artifact ``model-<run.name>``   best checkpoint, aliased ``best`` and
                                  ``phase0h``

Usage
-----
::

    import wandb_utils as wbu

    run = wbu.init_phase("0h", config={
        "lr": PHASE0_LR, "batch_size": PHASE0_BATCH_SIZE,
        "epochs": PHASE0F_EPOCHS, "seed": 42,
        "train_sites": PHASE0F_TRAIN_SITES,
        "val_sites":   PHASE0F_VAL_SITES,
        "segment_s":   PHASE0E_SEGMENT_S,
        "pos_weight":  pos_weight.tolist(),
    })

    for epoch in range(1, PHASE0F_EPOCHS + 1):
        train_loss = train_one_epoch_3class(...)
        val        = validate_3class(...)
        wbu.log_epoch_3class(epoch, train_loss, val)
        # your existing torch.save() lines stay as they are

    wbu.finalize_phase(history, verdict=verdict_text,
                       best_ckpt=run_dir / "phase0h_best.pt")
"""

from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
from typing import Optional

import wandb


# ---------------------------------------------------------------------------
# Project-level config — set these once.
# ---------------------------------------------------------------------------

WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  "your-entity")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "biodcase26-task2-whale-sed")
WANDB_GROUP   = "phase0_ladder"


# ---------------------------------------------------------------------------
# The ladder. Add a new entry every time you start a new phase script.
# ``parent`` should match the phase you forked from; ``interventions``
# are the *new* changes vs. that parent. Cumulative interventions are
# derived automatically.
# ---------------------------------------------------------------------------

PHASE_REGISTRY: dict[str, dict] = {
    "0": dict(
        parent=None,
        hypothesis=("Initial baseline: kerguelen2005 → casey2017. "
                    "Cross-site, single-class. Diagnostic: F1 stays 0."),
        interventions=["single_site_train", "cross_site_val", "single_class"],
    ),
    "0c": dict(
        parent="0",
        hypothesis=("Eliminate cross-site shift via same-site 80/20 file "
                    "split. Tests whether the pipeline learns at all."),
        interventions=["same_site_split"],
    ),
    "0d": dict(
        parent="0c",
        hypothesis=("Reduce BatchNorm momentum 0.1 → 0.01 to stabilise "
                    "val feature distribution and damp F1 oscillation."),
        interventions=["bn_momentum_0.01"],
    ),
    "0e": dict(
        parent="0c",
        hypothesis=("Force training segments to 30s to match validation "
                    "tile length. Removes train/val context-length gap."),
        interventions=["fixed_30s_train_segments"],
    ),
    "0f": dict(
        parent="0e",
        hypothesis=("Scale to 4-site Antarctic training set with the "
                    "official val split. First real-task baseline."),
        interventions=["multi_site_train", "official_val_split"],
    ),
    "0g": dict(
        parent="0f",
        hypothesis=("Add 3-class output (bmabz, d, bp) with plain BCE. "
                    "Headline: per-class F1 trajectory."),
        interventions=["3class_output"],
    ),
    "0h": dict(
        parent="0g",
        hypothesis=("Weighted BCE (wc = N/Pc on the training subset) to "
                    "push rare-class F1. Test for destabilisation."),
        interventions=["weighted_bce"],
    ),
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def cumulative_interventions(phase: str) -> list[str]:
    """Walk the parent chain and collect every intervention up to ``phase``."""
    out: list[str] = []
    p: Optional[str] = phase
    while p is not None:
        out = PHASE_REGISTRY[p]["interventions"] + out
        p = PHASE_REGISTRY[p]["parent"]
    return out


def _git_sha() -> str:
    try:
        return subprocess.getoutput("git rev-parse HEAD")[:10]
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.getoutput("git status --porcelain"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_phase(phase: str, config: dict,
               name_suffix: str = "",
               extra_tags: Optional[list[str]] = None,
               mode: str = "online"):
    """
    Start a wandb run for one phase of the ablation ladder.

    Call this once at the top of ``main()``. Everything the prof needs
    to understand the run — phase id, parent, hypothesis, full
    intervention chain, git sha — lands on the run automatically.
    """
    if phase not in PHASE_REGISTRY:
        raise KeyError(
            f"Phase {phase!r} not in PHASE_REGISTRY. "
            f"Add it to wandb_utils.py before launching the run."
        )

    meta = PHASE_REGISTRY[phase]
    cumulative = cumulative_interventions(phase)

    seed = config.get("seed", "noseed")
    timestamp = time.strftime("%m%d-%H%M")
    name = f"phase{phase}__seed{seed}__{timestamp}"
    if name_suffix:
        name = f"{name}__{name_suffix}"

    full_config = {
        **config,
        "phase":                    phase,
        "parent_phase":             meta["parent"],
        "hypothesis":               meta["hypothesis"],
        "new_interventions":        meta["interventions"],
        "cumulative_interventions": cumulative,
        "git_sha":                  _git_sha(),
        "git_dirty":                _git_dirty(),
    }

    tags = [f"phase{phase}", *cumulative]
    if extra_tags:
        tags.extend(extra_tags)

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type=f"phase{phase}",
        name=name,
        tags=tags,
        notes=meta["hypothesis"],
        config=full_config,
        mode=mode,
    )


def log_epoch(epoch: int, train_loss: float, val: dict) -> None:
    """Single-class per-epoch log (phases 0, 0c, 0d, 0e, 0f)."""
    wandb.log({
        "epoch":          epoch,
        "train/loss":     train_loss,
        "val/loss":       val["loss"],
        "val/f1":         val["f1"],
        "val/precision":  val["precision"],
        "val/recall":     val["recall"],
        "val/tp":         val["tp"],
        "val/fp":         val["fp"],
        "val/fn":         val["fn"],
    }, step=epoch)


def log_epoch_3class(epoch: int, train_loss: float, val: dict) -> None:
    """Multi-class per-epoch log (phases 0g, 0h, ...)."""
    payload = {
        "epoch":         epoch,
        "train/loss":    train_loss,
        "val/loss":      val["loss"],
        "val/f1_macro":  val["f1"],
    }
    for cname, pc in val["per_class"].items():
        payload[f"val/f1/{cname}"]        = pc["f1"]
        payload[f"val/precision/{cname}"] = pc["precision"]
        payload[f"val/recall/{cname}"]    = pc["recall"]
        payload[f"val/tp/{cname}"]        = pc["tp"]
        payload[f"val/fp/{cname}"]        = pc["fp"]
        payload[f"val/fn/{cname}"]        = pc["fn"]
    wandb.log(payload, step=epoch)


def stability_metrics(history: list[dict], key: str = "f1") -> dict:
    """
    Mean and max epoch-to-epoch swing of ``key`` over the second half
    of ``history``. Same numbers your VERDICT blocks already print.
    """
    series = [h[key] for h in history]
    second = series[len(series) // 2:]
    swings = [abs(second[i] - second[i - 1]) for i in range(1, len(second))]
    return {
        "second_half_mean_swing": sum(swings) / max(len(swings), 1),
        "second_half_max_swing":  max(swings) if swings else 0.0,
    }


def finalize_phase(history: list[dict],
                   verdict: str = "",
                   best_ckpt: Optional[Path | str] = None) -> None:
    """
    Stamp summary metrics + verdict on the run, log best checkpoint
    as an artifact, and finish the run.
    """
    if not history:
        wandb.finish()
        return

    f1s = [h["f1"] for h in history]
    best_idx = max(range(len(f1s)), key=lambda i: f1s[i])

    wandb.summary["best_f1"]    = f1s[best_idx]
    wandb.summary["best_epoch"] = best_idx + 1
    wandb.summary["final_f1"]   = f1s[-1]
    wandb.summary.update(stability_metrics(history, key="f1"))

    # Per-class summaries land automatically when the run is 3-class.
    if "per_class" in history[-1]:
        for cname in history[-1]["per_class"]:
            class_f1s = [h["per_class"][cname]["f1"] for h in history]
            wandb.summary[f"best_f1/{cname}"]  = max(class_f1s)
            wandb.summary[f"final_f1/{cname}"] = class_f1s[-1]

    if verdict:
        wandb.summary["verdict"] = verdict

    # Log the best checkpoint as an artifact so it can be pulled later
    # by alias (``model-...:best`` or ``model-...:phase0h``).
    if best_ckpt is not None:
        ckpt_path = Path(best_ckpt)
        if ckpt_path.exists():
            run = wandb.run
            art = wandb.Artifact(
                f"model-{run.name}",
                type="model",
                metadata={
                    "phase":      run.config["phase"],
                    "best_f1":    float(f1s[best_idx]),
                    "best_epoch": int(best_idx + 1),
                },
            )
            art.add_file(str(ckpt_path))
            run.log_artifact(
                art,
                aliases=["best", f"phase{run.config['phase']}"],
            )

    # Soft alert if F1 collapsed at the end of training — useful for
    # sweeps where you might not be watching every run.
    if f1s[best_idx] > 0.1 and f1s[-1] < 0.5 * f1s[best_idx]:
        try:
            wandb.alert(
                title="F1 collapsed at end of training",
                text=(f"Best {f1s[best_idx]:.3f} at epoch {best_idx + 1}, "
                      f"final {f1s[-1]:.3f}"),
                level=wandb.AlertLevel.WARN,
            )
        except Exception:
            pass

    wandb.finish()
