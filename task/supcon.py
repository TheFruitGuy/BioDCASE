"""
Supervised Contrastive Loss (Khosla et al. 2020)
================================================

Generalization of InfoNCE / NT-Xent (SimCLR) to the multi-positive
labelled setting: instead of exactly one positive per anchor (the
augmented view), every other in-batch sample with the same group label
is a positive.

Why for the verifier
--------------------
Each candidate already has a (class_idx, label) pair. Treating
``cell = class_idx * 2 + label`` as the group id gives 6 groups:

    0 = bmabz-FP   1 = bmabz-TP
    2 = d-FP       3 = d-TP
    4 = bp-FP      5 = bp-TP

With the balanced sampler producing ~5 samples per cell per batch of 32,
every anchor sees ~5 positives and ~26 negatives. The contrastive
gradient (a) pulls D-TPs toward each other so the model learns "what
real D-calls look like" rather than memorizing 99 specific examples,
(b) pushes D-TPs away from D-FPs which is exactly the verifier's job at
the embedding level, and (c) provides O(B²) pairwise constraints per
batch on top of the B per-sample BCE losses — sample-efficient
regularization for the small-data regime that broke v1.

Mathematical form
-----------------
For anchor ``i`` with positive set ``P(i)`` (same group, not self):

    L_i = - (1 / |P(i)|) · Σ_{p ∈ P(i)}  log(
        exp(z_i · z_p / τ)
        / Σ_{a ≠ i} exp(z_i · z_a / τ)
    )

We average ``L_i`` over anchors that have at least one positive (skip
singletons). ``z`` is L2-normalized so the dot product is cosine
similarity. Temperature ``τ`` controls how hard the contrast is — 0.1
is the standard default.

References
----------
Khosla, P. et al. (2020). Supervised Contrastive Learning. NeurIPS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(
    features: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss with multiple positives per anchor.

    Parameters
    ----------
    features : torch.Tensor, shape (B, D)
        Will be L2-normalized internally; pass raw or normalized either
        way works.
    group_ids : torch.Tensor, shape (B,) long
        Samples sharing a group_id are mutual positives. For the
        verifier use ``class_idx * 2 + label.long()``.
    temperature : float, default 0.1

    Returns
    -------
    torch.Tensor, scalar
        The loss. Returns 0 if the batch contains fewer than 2 samples
        or no anchor has any positive.
    """
    B = features.shape[0]
    if B < 2:
        return features.new_zeros(())

    # L2-normalize so the dot product is the cosine similarity.
    z = F.normalize(features, dim=-1)

    # Pairwise scaled similarities. Subtracting the row-wise max
    # before exp would be the numerically-stable way; logsumexp
    # below is equivalent and more readable.
    sim = (z @ z.T) / temperature                    # (B, B)

    # Exclude the self pair from the denominator and the positive set.
    # Use a large finite negative value instead of -inf so that the
    # later (log_prob * pos_mask) multiplication doesn't hit
    # 0 * (-inf) = NaN on the masked-out diagonal.
    diag_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)

    # log p(a | i) = sim_{i,a} - logsumexp_a sim_{i,a}
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)


    # Positive set mask: same group, not self.
    same_group = group_ids[:, None] == group_ids[None, :]
    pos_mask = same_group & ~diag_mask                # (B, B)

    # Mean log-prob over each anchor's positives.
    n_pos = pos_mask.sum(dim=1).float()               # (B,)
    has_pos = n_pos > 0
    # Avoid division by zero on rows without positives — those rows
    # are skipped via has_pos masking below.
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    avg_log_prob = pos_log_prob / n_pos.clamp(min=1.0)

    # Anchors without any positive contribute nothing; otherwise the
    # group-id distribution is too unbalanced and we'd be biasing toward
    # them. (Should be rare with the balanced sampler.)
    if has_pos.any():
        return -avg_log_prob[has_pos].mean()
    return features.new_zeros(())


# ======================================================================
# Standalone sanity check
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # 6 cells, ~5 per cell — what the balanced sampler produces.
    n_per_cell = 5
    n_cells = 6
    B = n_per_cell * n_cells
    group_ids = torch.arange(n_cells).repeat_interleave(n_per_cell)

    # Random features → loss should be near log(B-1) ≈ log(29) ≈ 3.37
    feats_random = torch.randn(B, 128)
    loss_random = supcon_loss(feats_random, group_ids)
    print(f"Random features:   loss = {loss_random.item():.4f}  "
          f"(expected ~{torch.tensor(B - 1).log().item():.4f})")

    # Features perfectly aligned with group → loss should be near 0.
    feats_aligned = torch.zeros(B, 128)
    for i in range(B):
        feats_aligned[i, group_ids[i]] = 1.0
    feats_aligned += 0.01 * torch.randn(B, 128)
    loss_aligned = supcon_loss(feats_aligned, group_ids)
    print(f"Aligned features:  loss = {loss_aligned.item():.4f}  "
          f"(expected near 0)")

    assert loss_aligned < loss_random, \
        "aligned features should give lower loss"
    print("OK — aligned beats random as expected")
