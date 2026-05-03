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
import random
import time
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA) so a phase script
    produces the same F1 trajectory across runs.

    Call this once at the very top of ``main()``, before any model,
    dataset, or DataLoader is constructed — DataLoader workers inherit
    the RNG state at construction time, so order matters.

    Parameters
    ----------
    seed : int
        Master seed. The same value drives Python ``random``, NumPy,
        and torch CPU + CUDA generators.
    deterministic : bool
        If True, also forces cuDNN into deterministic mode. This makes
        runs bit-identical at the cost of ~10-30% throughput. Leave
        False for development; turn on for the final report run.

    Returns
    -------
    int
        The seed that was set, so you can stuff it straight into the
        wandb config: ``seed = wbu.seed_everything(42)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seeded_dataloader_kwargs(seed: int) -> dict:
    """
    Returns kwargs to pass to ``DataLoader(...)`` so that shuffle order
    and worker RNG state are reproducible.

    Usage::

        train_loader = DataLoader(
            train_ds, batch_size=..., shuffle=True,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True,
            **wbu.seeded_dataloader_kwargs(seed),
        )
    """
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        # Each worker gets a different but deterministic seed.
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {"generator": g, "worker_init_fn": _worker_init}


# ---------------------------------------------------------------------------
# Project-level config — set these once.
# ---------------------------------------------------------------------------

WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  "bio-dcase")
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
    "0i": dict(
        parent="0h",
        hypothesis=("Focal loss (α=0.25, γ=2) on top of weighted BCE to "
                    "dampen overconfident FPs and stabilise the rare-class "
                    "swings introduced in 0h."),
        interventions=["focal_loss"],
    ),
    "0j": dict(
        parent="0g",  # default; --parent_override at the call site picks the
                      # actual source phase based on which checkpoint is tuned
        hypothesis=("Per-class threshold tuning on the official val split. "
                    "DCASE protocol: pick θ_c per class from the PR curve."),
        interventions=["tuned_thresholds"],
    ),
    "0k": dict(
        parent="0g",
        hypothesis=("Scale training data from 4 → 8 Antarctic sites with "
                    "the same plain-BCE 3-class recipe. Tests whether bmabz "
                    "(common class) gains from doubling annotation count."),
        interventions=["all_8_train_sites"],
    ),
    "0l": dict(
        parent="0g",
        hypothesis=("Scale LSTM 32→128 hidden, 1→2 layers (paper config) "
                    "with 0g's 4-site data. Tests whether model capacity is "
                    "the bottleneck closing the gap to F1=0.443."),
        interventions=["paper_lstm"],
    ),
    "0m": dict(
        parent="0k",
        hypothesis=("Full paper recipe: 8 sites + paper LSTM + 7-class "
                    "training collapsed to 3 at eval. Combines 0k's data "
                    "with 0l's capacity, then adds finer-grained training "
                    "supervision."),
        interventions=["paper_lstm", "seven_class_train"],
    ),
    "0n": dict(
        parent="0m",
        hypothesis=("Per-class threshold tuning applied to the 0m "
                    "7-class checkpoint. Same DCASE protocol as 0j but "
                    "with 7→3 collapse before the sweep."),
        interventions=["tuned_thresholds"],
    ),
    "0o": dict(
        parent="0m",
        hypothesis=("Add weighted BCE + focal loss (α=0.25, γ=2) to the "
                    "0m recipe. Tests whether at full data/model scale "
                    "focal does the rare-class job it failed at in 0i."),
        interventions=["weighted_bce", "focal_loss"],
    ),
    "0p": dict(
        parent="0m",
        hypothesis=("3-class direct training instead of 7-class collapse, "
                    "everything else fixed at 0m. Tests whether fine-"
                    "grained subclass supervision actually contributes."),
        # Negation-of-parent intervention: 0p removes seven_class_train
        # by reverting to direct 3-class output. Tag is descriptive
        # (``three_class_direct``) so the chain reads coherently.
        interventions=["three_class_direct"],
    ),
    "1a": dict(
        parent="baseline",
        hypothesis=("Time-mask augmentation on the F1=0.474 baseline. "
                    "Zero a random 0.2-1.5s window of the spectrogram "
                    "during training; loss mask zeroed in the same range "
                    "so erased frames don't contribute. Tests whether "
                    "SpecAugment-style time masking improves cross-site "
                    "generalisation."),
        interventions=["time_mask"],
    ),
    "1b": dict(
        parent="baseline",
        hypothesis=("Narrowband freq-mask augmentation on the baseline, "
                    "deliberately avoiding the protected 12-52 Hz call "
                    "band. Tests whether erasing random non-call "
                    "frequency bands improves robustness to unseen "
                    "ambient noise textures (notably casey2017's ice)."),
        interventions=["freq_mask_safe"],
    ),
    "1c": dict(
        parent="baseline",
        hypothesis=("Whole-segment volume scaling on the baseline. "
                    "Multiply each training sample's audio by a "
                    "log-uniform factor in [0.5, 2.0]. Tests whether "
                    "exposing the model to gain variation improves "
                    "invariance across hydrophones and recording "
                    "distances."),
        interventions=["volume_scaling"],
    ),
    "1e": dict(
        parent="baseline",
        hypothesis=("Cross-site noise mixing on the baseline. For each "
                    "positive sample, mix in a no-call clip from a "
                    "different training site at controlled SNR. Directly "
                    "addresses the cross-site failure mode diagnosed in "
                    "Phase 0a."),
        interventions=["cross_site_mix"],
    ),
    "2a": dict(
        parent="baseline",
        hypothesis=("Bigger BiLSTM on the baseline: hidden 128→256, "
                    "layers 2→3 (~2.7× the recurrent param count). "
                    "Tests whether the existing temporal model is "
                    "capacity-limited before swapping to attention."),
        interventions=["bigger_bilstm"],
    ),
    "2b": dict(
        parent="baseline",
        hypothesis=("Replace the BiLSTM with a 4-layer Transformer "
                    "encoder (d_model=128, nhead=4, sinusoidal PE) at "
                    "matched parameter count. Tests whether self-"
                    "attention beats recurrence on 30s/1500-frame "
                    "sequences."),
        interventions=["transformer_encoder"],
    ),
    "baseline": dict(
        parent=None,
        hypothesis=("Production training baseline. Paper recipe (8 sites, "
                    "paper LSTM, 7→3 collapse) plus the production-grade "
                    "stability extensions: early stopping, ReduceLROnPlateau, "
                    "periodic negative resampling, per-epoch threshold "
                    "tuning. Optional contrastive-pretrained encoder. "
                    "Starting point for the next round of experiments."),
        interventions=[
            "paper_recipe",
            "early_stopping",
            "lr_scheduler",
            "periodic_resample",
            "per_epoch_threshold_tuning",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _walk_chain(phase: str, parent: Optional[str]) -> list[str]:
    """
    Walk a parent chain explicitly, so an override can be used without
    mutating the registry.
    """
    out = list(PHASE_REGISTRY[phase]["interventions"])
    p = parent
    while p is not None:
        out = PHASE_REGISTRY[p]["interventions"] + out
        p = PHASE_REGISTRY[p]["parent"]
    return out


def cumulative_interventions(phase: str) -> list[str]:
    """Walk the parent chain and collect every intervention up to ``phase``."""
    return _walk_chain(phase, PHASE_REGISTRY[phase]["parent"])


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
               parent_override: Optional[str] = None,
               job_type: Optional[str] = None,
               mode: str = "online"):
    """
    Start a wandb run for one phase of the ablation ladder.

    Call this once at the top of ``main()``. Everything the prof needs
    to understand the run — phase id, parent, hypothesis, full
    intervention chain, git sha — lands on the run automatically.

    Parameters
    ----------
    phase : str
        Phase identifier (must exist in ``PHASE_REGISTRY``).
    config : dict
        Hyperparameters / data choices to log to wandb config.
    name_suffix : str
        Optional suffix appended to the auto-generated run name.
    extra_tags : list of str, optional
        Additional tags beyond the auto-generated phase + intervention
        ones. Useful for marking, e.g., ``"leaderboard_submission"``.
    parent_override : str, optional
        Override the registry parent for this run. Used by phase 0j
        (threshold tuning), which can attach to any of 0g/0h/0i
        depending on which checkpoint is being tuned. The cumulative
        intervention chain is rebuilt from the override.
    job_type : str, optional
        Wandb job type (``"train"``, ``"eval"``, ``"sweep"``, ...).
        Defaults to ``"phase<phase>"``.
    mode : str
        Wandb run mode. ``"disabled"`` for smoke tests.
    """
    if phase not in PHASE_REGISTRY:
        raise KeyError(
            f"Phase {phase!r} not in PHASE_REGISTRY. "
            f"Add it to wandb_utils.py before launching the run."
        )

    meta = dict(PHASE_REGISTRY[phase])
    if parent_override is not None:
        if parent_override not in PHASE_REGISTRY:
            raise KeyError(
                f"parent_override {parent_override!r} is not in PHASE_REGISTRY."
            )
        meta["parent"] = parent_override
    cumulative = _walk_chain(phase, parent=meta["parent"])

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
        job_type=job_type or f"phase{phase}",
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


def finalize_eval_phase(summary: dict,
                        verdict: str = "",
                        artifact_path: Optional[Path | str] = None,
                        artifact_type: str = "evaluation",
                        artifact_metadata: Optional[dict] = None) -> None:
    """
    Stamp evaluation-run results onto wandb and finish.

    Phase 0j (and any future eval-only phases) doesn't have an epoch
    loop, so the regular ``finalize_phase`` doesn't fit. Instead, the
    caller passes a flat ``summary`` dict (which becomes
    ``wandb.summary`` directly) and optionally an artifact to log.

    Parameters
    ----------
    summary : dict
        Key/value pairs to write to ``wandb.summary``. Nested dicts are
        flattened by joining keys with ``/`` so they render cleanly in
        the wandb UI.
    verdict : str
        Plain-English outcome stamped onto ``summary.verdict``.
    artifact_path : Path or str, optional
        File to upload (e.g. ``tuned_thresholds.pt``).
    artifact_type : str
        Wandb artifact type. ``"evaluation"`` is a sensible catch-all;
        threshold-tuning runs can pass ``"thresholds"`` for clarity.
    artifact_metadata : dict, optional
        Extra metadata to attach to the artifact (source checkpoint,
        baseline F1, tuned F1, etc).
    """
    if wandb.run is None:
        return

    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, prefix=f"{key}/"))
            else:
                out[key] = v
        return out

    for k, v in _flatten(summary).items():
        wandb.summary[k] = v
    if verdict:
        wandb.summary["verdict"] = verdict

    if artifact_path is not None:
        ap = Path(artifact_path)
        if ap.exists():
            run = wandb.run
            art = wandb.Artifact(
                f"{artifact_type}-{run.name}",
                type=artifact_type,
                metadata=artifact_metadata or {},
            )
            art.add_file(str(ap))
            run.log_artifact(
                art,
                aliases=["latest", f"phase{run.config['phase']}"],
            )

    wandb.finish()
