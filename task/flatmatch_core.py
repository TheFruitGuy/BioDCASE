"""
FlatMatch — Cross-Sharpness for Semi-Supervised Learning
========================================================

Implementation of the cross-sharpness regulariser from

    Huang, Shen, Yu, Han, Liu.
    "FlatMatch: Bridging Labeled Data and Unlabeled Data with
    Cross-Sharpness for Semi-Supervised Learning." NeurIPS 2023.

Mechanism
---------
::

    Labeled batch ──► Student(θ) ──► supervised loss ──► backward ─► ∇L_l in .grad
                                                                          │
                                                      ε* = ρ · g / ‖g‖₂  ◄┤  (mode "full": g = ∇L_l now;
                                                          │                  mode "e":   g = EMA buffer M_t)
                                                          ▼
    Unlabeled  ──► Student(θ)   ─── no_grad ───► f_u  ────────────────────► detach
                ╲                                                         ╲
                 ╲                                                         ▼
                  ──► Student(θ+ε*) ─── with grad ─► f̃_u ─► cross-sharpness loss
                                                              (MSE on sigmoid probs)

The supervised gradient (∇L_l) and the cross-sharpness gradient (∇R_xs)
both accumulate into ``.grad`` across two separate ``backward()`` calls,
so the optimiser step descends Eq. 5 of the paper without ever
materialising the combined loss as a single tensor.

Differences from the paper
--------------------------
* Multi-label, per-frame SED head → the paper's softmax-CE prediction
  distance becomes MSE between sigmoid probabilities, matching the
  existing ``consistency_loss`` in this codebase. The semantics
  ("penalise disagreement between θ and θ̃ on unlabeled data") are
  unchanged.
* The "full" variant in the paper does two labeled forward+backward
  passes per step (once for ε*, once at θ+ε*). Per Eq. 2 the
  second-order correction is negligible, so we evaluate L_l at θ and
  reuse the same labeled backward for both the ε* gradient source and
  the supervised contribution to the optimiser step. Net cost of
  ``mode="full"`` over ``mode="e"`` is therefore *zero* extra labeled
  work — the modes differ only in whether ε* comes from this step's
  fresh ∇L_l or from the EMA buffer of past ∇L_l's.

This module deliberately depends on nothing in the codebase so it can
be slotted into any training loop that has a labeled batch and an
unlabeled batch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
# Cross-sharpness loss
# ======================================================================

def cross_sharpness_loss(
    perturbed_logits: torch.Tensor,           # f̃_u = g_{θ+ε*}(x_u), with grad
    clean_logits:     torch.Tensor,           # f_u  = g_θ(x_u),    must be detached
    mask:             Optional[torch.Tensor] = None,  # (B, T)
) -> torch.Tensor:
    """
    MSE between perturbed and clean sigmoid-probability predictions on the
    unlabeled batch.

    Shape contract matches ``mean_teacher_core.consistency_loss``:
    ``perturbed_logits`` and ``clean_logits`` are both ``(B, T, C)``, mask
    is ``(B, T)`` if supplied. Gradients flow only through
    ``perturbed_logits``; the caller is responsible for ``.detach()``-ing
    ``clean_logits`` (we don't re-detach so accidental grad flow is loud).
    """
    p = torch.sigmoid(perturbed_logits)
    q = torch.sigmoid(clean_logits)
    sq = (p - q).pow(2)
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        denom = m.sum() * sq.size(-1) + 1e-8
        return (sq * m).sum() / denom
    return sq.mean()


# ======================================================================
# Perturbation
# ======================================================================

class FlatMatchPerturbation:
    """
    Manages the worst-case perturbation ε* and (for ``mode="e"``) the EMA
    buffer of past labeled gradients.

    Lifecycle of one training step::

        # supervised forward + backward populate .grad with ∇L_l
        loss_sup.backward()

        # mode "full" reads ∇L_l straight from .grad;
        # mode "e"    reads the EMA buffer M_t (state from previous step)
        applied = fmp.apply()

        # at this point: params = θ + ε*, .grad still holds ∇L_l
        # f̃_u = student(spec_u) tracks grad back through the perturbed
        # params to θ (autograd treats in-place .data edits as invisible).

        if mode == "e":
            fmp.update_buffer()           # snapshot ∇L_l → M_{t+1}
                                          # done BEFORE the xs backward so
                                          # the buffer stays clean

        loss_xs.backward()                # accumulates ∇R_xs into .grad

        fmp.restore()                     # subtracts ε* from .data
        optimizer.step()                  # descends ∇L_l + λ·∇R_xs

    Either ordering of ``apply``/``restore`` around the perturbed forward
    is supported — the perturbation is stored as an explicit ε* dict
    (one tensor per parameter), so restoration doesn't depend on .grad
    being unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        rho: float = 0.1,
        mode: str = "e",
        ema_alpha: float = 0.999,
    ):
        if mode not in ("e", "full"):
            raise ValueError(f"mode must be 'e' or 'full', got {mode!r}")
        self.model = _unwrap(model)
        self.rho = float(rho)
        self.mode = mode
        self.ema_alpha = float(ema_alpha)

        # EMA buffer of past ∇L_l's. ``None`` until the first step's
        # labeled backward populates it.
        self.buffer: Optional[dict[str, torch.Tensor]] = None

        # Per-parameter ε* tensors, populated by ``apply``, consumed by
        # ``restore``. ``None`` between steps.
        self._epsilons: Optional[dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # ε* source
    # ------------------------------------------------------------------

    def _grad_source(self) -> Optional[dict[str, torch.Tensor]]:
        """Pick the gradient dict used to construct ε* for this step."""
        if self.mode == "full":
            return {
                name: p.grad
                for name, p in self.model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
        # mode "e"
        return self.buffer

    # ------------------------------------------------------------------
    # Apply / restore
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply(self) -> bool:
        """
        Compute ε* = ρ · g / ‖g‖₂ and add it to model parameters in place.

        Returns
        -------
        bool
            ``True`` if ε* was applied. ``False`` if there's no gradient
            source available (e.g. ``mode="e"`` on the very first step
            before the buffer has been populated). Caller should skip the
            perturbed forward and the cross-sharpness loss in that case.
        """
        src = self._grad_source()
        if not src:
            return False

        # Restrict to params that (a) require grad and (b) have a matching
        # entry in src. Anything else is silently skipped.
        items: list[tuple[str, nn.Parameter, torch.Tensor]] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            g = src.get(name)
            if g is None:
                continue
            items.append((name, p, g))
        if not items:
            return False

        # Global L2 norm over all selected parameter gradients.
        flat_sq = sum((g.detach() ** 2).sum() for _, _, g in items)
        norm = flat_sq.sqrt() + 1e-12
        scale = self.rho / norm

        self._epsilons = {}
        for name, p, g in items:
            # ε* lives outside autograd. Use a fresh tensor so subsequent
            # in-place modifications to .grad (by the xs backward) don't
            # poison restoration.
            eps = (g.detach() * scale).to(dtype=p.data.dtype)
            self._epsilons[name] = eps
            p.data.add_(eps)
        return True

    @torch.no_grad()
    def restore(self) -> None:
        """Undo the last ``apply`` by subtracting the stored ε* in place."""
        if not self._epsilons:
            return
        params = dict(self.model.named_parameters())
        for name, eps in self._epsilons.items():
            p = params.get(name)
            if p is not None:
                p.data.sub_(eps)
        self._epsilons = None

    # ------------------------------------------------------------------
    # EMA buffer (mode "e" only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_buffer(self) -> None:
        """
        EMA-update the buffer M_{t+1} ← α·M_t + (1-α)·∇L_l using the
        gradient currently in ``.grad``.

        Call this AFTER the supervised backward but BEFORE the
        cross-sharpness backward, while ``.grad`` still equals ∇L_l. Safe
        to call in ``mode="full"`` too (it just keeps the buffer warm in
        case the mode is flipped at runtime), but normally a no-op for
        that mode.
        """
        new_grads: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                new_grads[name] = p.grad.detach().clone()
        if not new_grads:
            return

        if self.buffer is None:
            # Bootstrap: just adopt the first gradient as the buffer.
            self.buffer = new_grads
            return

        a = self.ema_alpha
        for name, g in new_grads.items():
            prev = self.buffer.get(name)
            if prev is None or prev.shape != g.shape:
                self.buffer[name] = g
            else:
                # M ← α·M + (1-α)·g
                prev.mul_(a).add_(g, alpha=(1.0 - a))


# ======================================================================
# Helpers
# ======================================================================

def _unwrap(model: nn.Module) -> nn.Module:
    """Return ``model.module`` if wrapped in ``DataParallel``/``DDP``,
    else ``model``. Mirrors the convention in ``mean_teacher_core``."""
    return getattr(model, "module", model)
