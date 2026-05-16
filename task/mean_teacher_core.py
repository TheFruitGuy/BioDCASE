"""
Mean Teacher Self-Training — Core Components
=============================================

Mean-teacher consistency-regularised semi-supervised learning, adapted
from Tarvainen & Valpola (NeurIPS 2017) to the per-frame multi-label
SED setting used by WhaleVAD. This is the standard DCASE Task 4 SOTA
recipe; the first application to whale call detection.

Mechanism
---------
::

    Labeled batch ──► Student ──────────────► Supervised loss
                                                     │
                                                     ▼
                                                Total loss
                                                     ▲
    Unlabeled  ┬── weak aug ──► Teacher (no_grad) ┐  │
    batch      │                                   ├──► Consistency loss
               └── strong aug ─► Student ──────────┘  │   (× λ(epoch))
                                                     │
    Student ──── EMA update (every step) ───► Teacher

Teacher weights are an EMA of the student. Teacher is never trained by
gradient descent. The consistency loss is MSE between sigmoid outputs.

Why mean teacher rather than one-shot pseudo-labeling
-----------------------------------------------------
- No confirmation bias: teacher continuously updates as student improves
- No threshold cliff: soft targets contribute graded signal everywhere,
  including in regions where the model is genuinely uncertain
- Iterative by construction: the set of confidently-labeled examples
  grows monotonically across training without manual re-mining rounds

Reference
---------
Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role
models: Weight-averaged consistency targets improve semi-supervised
deep learning results. NeurIPS 2017.

DCASE Task 4 lineage (mean teacher with strong/weak augmentation as
the dominant paradigm): Delphin-Poulat 2019, JiaKai 2019, Miyazaki
2020, Nam 2022, Kim/Park 2023.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
# EMA teacher
# ======================================================================

class EMATeacher:
    """
    Exponential moving-average copy of a student model.

    The teacher's weights are updated as

        θ' ← α · θ' + (1 - α) · θ

    after every student optimiser step. The teacher is never trained
    directly — gradients do not flow through it. It is always in eval
    mode so dropout and BatchNorm use running statistics.

    BatchNorm running stats are EMA-tracked the same way as parameters,
    which matches the Tarvainen & Valpola reference implementation.
    Integer buffers (e.g. ``num_batches_tracked``) are copied verbatim.

    Parameters
    ----------
    student : nn.Module
        Source model. The teacher is a ``copy.deepcopy`` of this at
        construction time, so any subsequent re-init of the student
        won't propagate. ``DataParallel`` is unwrapped automatically.
    alpha : float, default 0.999
        Default EMA decay. The per-step ``update`` call can override
        this for schedules (see ``cosine_alpha``).
    """

    def __init__(self, student: nn.Module, alpha: float = 0.999):
        self.alpha = alpha
        src = _unwrap(student)
        self.teacher = copy.deepcopy(src)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module, alpha: Optional[float] = None) -> None:
        """In-place EMA update of teacher params and BN buffers."""
        a = self.alpha if alpha is None else alpha
        src = _unwrap(student)

        for p_t, p_s in zip(self.teacher.parameters(), src.parameters()):
            p_t.data.mul_(a).add_(p_s.data, alpha=1.0 - a)

        for b_t, b_s in zip(self.teacher.buffers(), src.buffers()):
            if b_t.dtype.is_floating_point:
                b_t.data.mul_(a).add_(b_s.data, alpha=1.0 - a)
            else:
                # num_batches_tracked and other integer counters
                b_t.data.copy_(b_s.data)

    def state_dict(self) -> dict:
        return self.teacher.state_dict()

    def to(self, device) -> "EMATeacher":
        self.teacher.to(device)
        return self


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip ``DataParallel`` wrapping if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ======================================================================
# Consistency loss
# ======================================================================

def consistency_loss(
    student_logits: torch.Tensor,           # (B, T, C)
    teacher_logits: torch.Tensor,           # (B, T, C) — should be detached
    mask: Optional[torch.Tensor] = None,    # (B, T) bool/float
) -> torch.Tensor:
    """
    MSE between student and teacher sigmoid-probability outputs.

    Operates in probability space rather than logits because
        (a) Tarvainen & Valpola's original recipe uses probability MSE,
        (b) it bounds the loss naturally to [0, 1] per element, and
        (c) sigmoid saturation provides automatic gradient damping for
            already-confident predictions (which is the desired effect
            on a per-frame multi-label SED head).

    The caller is responsible for detaching ``teacher_logits``. We do
    not redundantly detach here so accidental gradient flow into the
    teacher is loud rather than silent.
    """
    s = torch.sigmoid(student_logits)
    t = torch.sigmoid(teacher_logits)
    sq = (s - t).pow(2)

    if mask is not None:
        m = mask.unsqueeze(-1).float()
        denom = m.sum() * sq.size(-1) + 1e-8
        return (sq * m).sum() / denom
    return sq.mean()


# ======================================================================
# Schedules
# ======================================================================

def sigmoid_ramp(epoch: int, ramp_epochs: int) -> float:
    """
    Tarvainen & Valpola's exponential ramp-up: 0 → 1 over
    ``ramp_epochs`` using ``exp(-5 · (1 - p)²)`` where ``p`` is the
    fractional progress through the ramp clipped to [0, 1].

    The shape is "slow start, fast middle, gentle end" — keeps the
    consistency loss out of the picture while supervised loss is still
    rapidly shaping early features, then phases it in.
    """
    if ramp_epochs <= 0 or epoch >= ramp_epochs:
        return 1.0
    p = max(0.0, min(1.0, epoch / float(ramp_epochs)))
    return math.exp(-5.0 * (1.0 - p) ** 2)


def cosine_alpha(
    epoch: int,
    alpha_start: float = 0.99,
    alpha_end: float = 0.999,
    warmup_epochs: int = 10,
) -> float:
    """
    Cosine ramp of the EMA decay from ``alpha_start`` → ``alpha_end``
    over ``warmup_epochs`` epochs, then ``alpha_end`` thereafter.

    Lower α early lets the teacher track the rapidly-changing student;
    higher α later produces a smoother, more reliable target signal
    once the student has stabilised. This is a common refinement to
    the constant-α schedule in the original mean teacher paper.
    """
    if epoch >= warmup_epochs:
        return alpha_end
    t = epoch / max(warmup_epochs, 1)
    cos = 0.5 * (1.0 - math.cos(math.pi * t))
    return alpha_start + (alpha_end - alpha_start) * cos


# ======================================================================
# Frame alignment helpers
# ======================================================================

def align_lengths_pair(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trim student and teacher logits to a common time length.

    Strong and weak augmentations operate on identical-length audio
    and produce identical-length spectrograms, but lazy CNN-projection
    quirks (e.g. one-frame discrepancies from edge handling) can
    surface a ±1 frame mismatch. Trim to the minimum to be safe.
    """
    T_s, T_t = s_logits.size(1), t_logits.size(1)
    if T_s == T_t:
        return s_logits, t_logits
    T = min(T_s, T_t)
    return s_logits[:, :T, :], t_logits[:, :T, :]


def align_supervised_lengths(
    logits: torch.Tensor,   # (B, T_m, C)
    targets: torch.Tensor,  # (B, T_t, C)
    mask: torch.Tensor,     # (B, T_t)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match the alignment logic from ``train.py``: trim or zero-pad
    targets/mask to the model's output frame count.

    Returns the aligned ``targets`` and ``mask`` (logits are returned
    by the model and unchanged here).
    """
    T_m, T_t = logits.size(1), targets.size(1)
    if T_m == T_t:
        return targets, mask
    if T_m < T_t:
        return targets[:, :T_m, :], mask[:, :T_m]
    # T_m > T_t: pad targets with zeros, mark padded frames invalid
    pad_t = torch.zeros(
        targets.size(0), T_m - T_t, targets.size(2),
        device=targets.device, dtype=targets.dtype,
    )
    pad_m = torch.zeros(
        mask.size(0), T_m - T_t,
        device=mask.device, dtype=mask.dtype,
    )
    return torch.cat([targets, pad_t], dim=1), torch.cat([mask, pad_m], dim=1)


# ======================================================================
# View builders for the unlabeled stream
# ======================================================================
# These are thin wrappers around ssl_augmentations.make_view that pin
# the parameter combinations used by the mean teacher. Keeping them
# named here documents the design choice at the call site:
#
#   - WEAK view  → teacher input. Minimal perturbation. Spec extraction
#                  with light volume jitter only, so the teacher's soft
#                  target is well-calibrated.
#   - STRONG view → student input. The full SSL augmentation pipeline.
#                  The augmentation gap between weak and strong is the
#                  source of the consistency learning signal.

def make_weak_view(audio, sites, spec_extractor):
    """Light perturbation only — teacher-side."""
    # Lazy import keeps this module decoupled from ssl_augmentations
    # for unit tests that don't need them.
    from ssl_augmentations import make_view
    return make_view(
        audio, sites, spec_extractor,
        use_volume=True,
        use_time_mask=False,
        use_noise=False,
        use_freq_mask=False,
        use_cross_site=False,
        volume_p=0.3,
    )


def make_strong_view(audio, sites, spec_extractor, no_call_pool=None):
    """Full SSL aug pipeline — student-side."""
    from ssl_augmentations import make_view
    return make_view(
        audio, sites, spec_extractor,
        use_volume=True,
        use_time_mask=True,
        use_noise=True,
        use_freq_mask=True,
        use_cross_site=(no_call_pool is not None),
        no_call_pool=no_call_pool,
        volume_p=1.0,
        time_mask_p=1.0,
        noise_p=1.0,
        freq_mask_p=1.0,
        cross_site_p=0.5,
    )

    def freeze_bn_running_stats(model: nn.Module) -> None:
        """
        Set all BatchNorm modules to eval mode: forward uses frozen running
        statistics and no statistics are updated. Affine parameters and the
        rest of the network still train normally.

        Required when the unlabeled stream is from a different audio domain
        than the labeled stream — otherwise BN running stats drift toward
        the average of the two distributions and feature alignment with the
        classifier breaks at val time.
        """
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()