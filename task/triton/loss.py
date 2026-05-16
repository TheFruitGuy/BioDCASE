"""
Triton — Training losses
========================

Two loss classes plus a shared utility:

- ``TritonLoss``      — weighted-BCE-with-logits + optional focal
                        modulation. Numerically stable; used with
                        ``Triton`` (baseline) where the model emits
                        ``{"logits": ..., "probs": sigmoid(logits)}``
                        and the gate is the identity.
- ``TritonTIDELoss``  — weighted-BCE-on-probabilities + optional focal
                        modulation + optional auxiliary BCE on the
                        TIDE gate against "any-class-active" targets.
                        Used with ``TritonTIDE`` where ``probs`` are
                        already gated and applying ``sigmoid`` again
                        would be wrong.
- ``compute_class_weights`` — file-level positive weights w_c = N / P_c
                        normalised so the minimum weight is 1.0.

Both losses accept the model-output dict whole (matching the unified
``train.py`` convention) and the padded-frame mask from the collator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ======================================================================
# Baseline loss (BCE-with-logits, numerically stable)
# ======================================================================

class TritonLoss(nn.Module):
    """
    Weighted binary cross-entropy with optional focal modulation, operating
    on raw logits.

    Used with the baseline ``Triton`` model. Numerically more stable than
    BCE-on-probabilities because ``binary_cross_entropy_with_logits`` uses
    the log-sum-exp trick internally.

    Parameters
    ----------
    pos_weight : torch.Tensor, optional
        Per-class positive weights, shape ``(num_classes,)``. Pass
        ``compute_class_weights()`` to weight by ``w_c = N / P_c``.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        if pos_weight is not None:
            # Buffer so .to(device) propagates correctly.
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.alpha = cfg.FOCAL_ALPHA
        self.gamma = cfg.FOCAL_GAMMA
        self.use_focal = cfg.USE_FOCAL_LOSS

    def forward(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        targets: torch.Tensor,                              # (B, T, C)
        padding_mask: torch.Tensor | None = None,           # (B, T) bool
    ) -> torch.Tensor:
        """
        Compute loss for a mini-batch.

        ``outputs`` may be either the model's dict (preferred — uses
        ``outputs["logits"]``) or a bare logits tensor for backward
        compatibility with older callers.
        """
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        pw = self.pos_weight.view(1, 1, -1) if self.pos_weight is not None else None

        # Element-wise BCE (no reduction) so we can apply focal modulation
        # and the padding mask afterwards.
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none",
        )

        if self.use_focal:
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.gamma
            loss = focal_mod * bce
        else:
            loss = bce

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            loss = loss * mask
            denom = mask.sum() * logits.size(-1) + 1e-8
            return loss.sum() / denom

        return loss.mean()


# ======================================================================
# TIDE loss (BCE-on-probabilities + optional aux on the gate)
# ======================================================================

class TritonTIDELoss(nn.Module):
    """
    TIDE-aware loss = weighted/focal BCE on **gated probabilities** +
    optional auxiliary BCE on the gate.

    Differences from ``TritonLoss``:
      - Operates on probabilities (not logits) because TIDE has already
        applied sigmoid and multiplied by the gate.
      - Probabilities are clamped to ``[ε, 1−ε]`` before the BCE to
        guard against numerical issues from the gate driving any
        per-frame probability to exactly 0 or 1.
      - Adds an optional auxiliary loss on the gate against the
        per-frame "any class active" target. When enabled, the gate
        receives a direct supervision signal, which empirically helps
        with stability during early training.

    Parameters
    ----------
    pos_weight : torch.Tensor, optional
        Per-class positive weights, shape ``(num_classes,)``.
    aux_loss : bool
        If True, add an auxiliary BCE on the gate against the per-frame
        "any class active" target.
    aux_weight : float
        Weight of the auxiliary loss in the total.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        aux_loss: bool = False,
        aux_weight: float = 0.1,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.aux_loss = aux_loss
        self.aux_weight = aux_weight
        self.alpha = cfg.FOCAL_ALPHA
        self.gamma = cfg.FOCAL_GAMMA
        self.use_focal = cfg.USE_FOCAL_LOSS
        self.eps = 1e-7

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = outputs["probs"].clamp(self.eps, 1 - self.eps)
        gate = outputs["gate"]

        # Element-wise BCE in probability space.
        bce = -(targets * torch.log(probs)
                + (1 - targets) * torch.log(1 - probs))

        # Focal modulation in probability space.
        if self.use_focal:
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.gamma
            bce = focal_mod * bce

        # Per-class positive weighting.
        if self.pos_weight is not None:
            pw = self.pos_weight.view(1, 1, -1)
            bce = bce * (targets * pw + (1 - targets))

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            bce = bce * mask
            denom = mask.sum() * bce.size(-1) + 1e-8
            main_loss = bce.sum() / denom
        else:
            main_loss = bce.mean()

        total = main_loss

        # Auxiliary: gate should track "any class active". Without this
        # the gate only sees gradient through the multiplicative path,
        # which can stall when the gate starts near 1.0 everywhere.
        if self.aux_loss:
            any_call = targets.max(dim=-1).values.float()        # (B, T)
            gate_c = gate.clamp(self.eps, 1 - self.eps)
            aux = -(any_call * torch.log(gate_c)
                    + (1 - any_call) * torch.log(1 - gate_c))
            if padding_mask is not None:
                aux = aux * padding_mask.float()
                aux_loss = aux.sum() / (padding_mask.sum() + 1e-8)
            else:
                aux_loss = aux.mean()
            total = total + self.aux_weight * aux_loss

        return total


# ======================================================================
# Class weights (shared by both losses)
# ======================================================================

def compute_class_weights() -> torch.Tensor:
    """
    Per-class positive weights using ``w_c = N / P_c``.

    ``N`` is the number of negative segments (interpreted as files
    containing no annotation of class ``c``) and ``P_c`` is the number
    of positive segments (files containing at least one annotation of
    class ``c``). The raw weights are normalised so the minimum is 1.0,
    preserving the relative weighting while ensuring no class is
    actively *downweighted* (which would slow down learning on the
    majority class with the most training signal).

    Returns
    -------
    torch.Tensor
        Shape ``(num_classes,)``, on CPU. Move to device before passing
        into a loss constructor.
    """
    # Local import to avoid a circular dependency: dataset.py imports
    # nothing from loss.py, so this is safe.
    from dataset import load_annotations

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    total_files = annotations.groupby(["dataset", "filename"]).ngroups

    class_names = cfg.class_names()
    weights = []
    for c_name in class_names:
        if cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]

        p_c = max(class_annots.groupby(["dataset", "filename"]).ngroups, 1)
        n_neg = max(total_files - p_c, 1)
        weights.append(n_neg / p_c)

    result = torch.tensor(weights, dtype=torch.float32)
    # Normalise so min = 1.0: keeps the ratios while avoiding any
    # class being weighted < 1 (which would actively downweight it).
    result = result / result.min()

    print(f"Class weights (w_c = N/P_c, normalised to min=1):")
    for name, w in zip(class_names, result.tolist()):
        print(f"  {name}: {w:.3f}")
    return result
