"""
Triton — Weights & Biases integration
=====================================

Run lifecycle helpers that wrap the ``wandb`` SDK with the conventions
this project uses: tagged runs, structured per-epoch logging, summary
stamping at the end, and best-checkpoint artifact upload.

Design notes
------------
The earlier ``WhaleVAD`` codebase carried a ``PHASE_REGISTRY`` parent
chain encoding every ablation experiment. That registry served its
purpose but is tied to historical experiments. Triton starts clean:
runs are identified by a short ``name`` plus a free-form list of
``interventions`` tags, and parent-chains can be reconstructed by the
caller if needed.

Public API
----------
- ``init_run``       — start a wandb run with tags and config
- ``log_epoch``      — single dict-based per-epoch log (works for any
                       class count; the metric dict shape is what
                       ``train.py`` already produces)
- ``finalize_run``   — stamp summary + upload best checkpoint as
                       artifact + close the run
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import wandb


# ----------------------------------------------------------------------
# Project-level config (overridable via env vars)
# ----------------------------------------------------------------------

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "bio-dcase")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "biodcase26-task2-triton")
WANDB_GROUP = os.environ.get("WANDB_GROUP", "triton")


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def init_run(
    name: str,
    config: dict,
    *,
    interventions: Optional[list[str]] = None,
    extra_tags: Optional[list[str]] = None,
    notes: str = "",
    job_type: Optional[str] = None,
    mode: str = "online",
) -> wandb.sdk.wandb_run.Run:
    """
    Start a wandb run with the project-level defaults.

    Parameters
    ----------
    name : str
        Short identifier for the experiment family (e.g. ``"triton"``,
        ``"triton_tide"``). The seed and a timestamp are appended
        automatically so the wandb run name is unique.
    config : dict
        Hyperparameters and data choices logged to wandb's ``config``.
    interventions : list of str, optional
        Free-form labels describing what's different about this run
        (e.g. ``["focal_loss", "weighted_bce", "frozen_encoder_5e"]``).
        Each becomes a wandb tag.
    extra_tags : list of str, optional
        Additional tags on top of the interventions list.
    notes : str
        Free-text description visible in the wandb run header.
    job_type : str, optional
        Wandb job type. Defaults to ``name``.
    mode : str
        Wandb mode. ``"disabled"`` for smoke tests.
    """
    seed = config.get("seed", "noseed")
    timestamp = time.strftime("%m%d-%H%M")
    run_name = f"{name}__seed{seed}__{timestamp}"

    full_config = {
        **config,
        "run_name": name,
        "interventions": list(interventions or []),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }

    tags = [name]
    if interventions:
        tags.extend(interventions)
    if extra_tags:
        tags.extend(extra_tags)

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type=job_type or name,
        name=run_name,
        tags=tags,
        notes=notes,
        config=full_config,
        mode=mode,
    )


def log_epoch(epoch: int, payload: dict) -> None:
    """
    Log an arbitrary payload at the given step.

    Triton's ``train.py`` already constructs the per-epoch dict it
    wants on wandb (per-class F1/P/R, learning rate, train/val loss,
    tuned thresholds, classifier bias). This function is a thin
    indirection that adds the ``epoch`` key if missing and delegates
    to ``wandb.log``.
    """
    if "epoch" not in payload:
        payload["epoch"] = epoch
    wandb.log(payload, step=epoch)


def finalize_run(
    best_f1: float,
    best_epoch: int,
    epochs_run: int,
    *,
    verdict: str = "",
    extra_summary: Optional[dict] = None,
    artifact_paths: Optional[list[Path | str]] = None,
    artifact_aliases: Optional[list[str]] = None,
    artifact_metadata: Optional[dict] = None,
) -> None:
    """
    Stamp summary metrics on the run, upload checkpoints as a model
    artifact, and close the run.

    Parameters
    ----------
    best_f1 : float
        Headline metric (validation F1 at the best checkpoint).
    best_epoch : int
        Epoch index of the best checkpoint.
    epochs_run : int
        Total number of epochs actually run (may be less than the
        configured cap because of early stopping).
    verdict : str
        Plain-English summary of how the run went. Visible in the
        wandb UI as ``summary.verdict``.
    extra_summary : dict, optional
        Additional summary fields. Nested dicts are flattened via
        ``/`` joins so they render cleanly.
    artifact_paths : list of Path or str, optional
        Files to upload (e.g. ``best_model.pt``, ``final_model.pt``).
    artifact_aliases : list of str, optional
        Aliases for the uploaded artifact. Defaults to ``["best"]``.
    artifact_metadata : dict, optional
        Extra metadata stored on the artifact.
    """
    if wandb.run is None:
        return

    wandb.summary["best_f1"] = float(best_f1)
    wandb.summary["best_epoch"] = int(best_epoch)
    wandb.summary["epochs_run"] = int(epochs_run)
    if verdict:
        wandb.summary["verdict"] = verdict

    if extra_summary:
        for k, v in _flatten(extra_summary).items():
            wandb.summary[k] = v

    if artifact_paths:
        run = wandb.run
        art = wandb.Artifact(
            f"model-{run.name}",
            type="model",
            metadata=artifact_metadata or {
                "best_f1": float(best_f1),
                "best_epoch": int(best_epoch),
                "epochs_run": int(epochs_run),
            },
        )
        for p in artifact_paths:
            pp = Path(p)
            if pp.exists():
                art.add_file(str(pp))
        run.log_artifact(art, aliases=artifact_aliases or ["best"])

    wandb.finish()


# ----------------------------------------------------------------------
# Internal: nested-dict flattener for the summary
# ----------------------------------------------------------------------

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}/"))
        else:
            out[key] = v
    return out
