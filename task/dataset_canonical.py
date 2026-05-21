"""
Canonical-Recipe Dataset Extensions
===================================

Companion to ``dataset.py`` for ``train_canonical.py``. Adds the two pieces
that the email from Geldenhuys (2026-05-21) flagged as worth scrutinising:

  1. ``compute_pos_weights_framewise(segments)`` — recomputes per-class BCE
     positive weights at the **frame** level rather than the **file** level.
     The original ``compute_class_weights`` in ``model.py`` counts files,
     which is a coarse proxy that does not match the actual per-frame BCE
     loss the model is trained against. Christiaan's hint was: confirm that
     "padded frames are properly masked out of both the loss computation
     and the per-class positive/negative counts used for BCE weighting."
     This function is the second half of that audit.

     NOTE: The training segments themselves contain no padding — padding
     is introduced only at collation time. So counting frames over the
     ``Segment`` objects directly is already padding-free; we just need
     to count them.

  2. ``assert_mask_is_correct(audio_pad, target_pad, mask_pad, metas)`` — a
     drop-in defensive assertion you can call inside the training loop to
     verify that the boolean ``mask_pad`` actually marks the padded frames
     and only the padded frames. Cheap, runs in microseconds, and catches
     the class of silent bug where the mask drifts out of sync with the
     real lengths of the underlying audio.

Both helpers are designed to be imported alongside the standard
``dataset.py`` module — they do not replace it. ``train_canonical.py``
uses the regular segment builders from ``dataset.py`` and only swaps in
the new pos_weight computation here.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

import config as cfg
from dataset import Segment


# ======================================================================
# Frame-level positive weight computation
# ======================================================================

def compute_pos_weights_framewise(
    segments: Sequence[Segment],
    n_classes: int | None = None,
    frame_stride_s: float | None = None,
    sample_rate: int | None = None,
    normalize_min_to_one: bool = True,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute per-class ``pos_weight`` from segment annotations at frame level.

    For ``BCEWithLogitsLoss`` (which is what ``WhaleVADLoss`` wraps), the
    canonical interpretation of ``pos_weight[c]`` is::

        pos_weight[c] = (# negative frames for class c) / (# positive frames for class c)

    where positive / negative are determined by the binary target value at
    each (frame, class) cell. This function computes that ratio directly
    from the ``Segment`` objects, *before* any collation-time padding is
    introduced, so the result is unaffected by per-batch padding choices.

    Parameters
    ----------
    segments : sequence of Segment
        The training segments (positives + currently-resampled negatives).
        Negative segments contribute only to the negative count.
    n_classes : int, optional
        Number of output classes. Defaults to ``cfg.n_classes()``.
    frame_stride_s : float, optional
        Temporal frame stride in seconds. Defaults to ``cfg.FRAME_STRIDE_S``.
    sample_rate : int, optional
        Audio sample rate in Hz. Defaults to ``cfg.SAMPLE_RATE``.
    normalize_min_to_one : bool
        If True (default), divide the resulting weights by the minimum so
        the smallest weight equals 1.0. This matches the convention used
        by ``compute_class_weights`` in ``model.py`` and prevents the
        common class from being actively down-weighted (which would slow
        learning on the class with the most training signal).
    verbose : bool
        If True, print the per-class frame counts and resulting weights.

    Returns
    -------
    torch.Tensor
        Per-class ``pos_weight``, shape ``(n_classes,)``. Suitable for use
        as ``WhaleVADLoss(pos_weight=...)`` or directly as the ``pos_weight``
        argument of ``F.binary_cross_entropy_with_logits``.
    """
    if n_classes is None:
        n_classes = cfg.n_classes()
    if frame_stride_s is None:
        frame_stride_s = cfg.FRAME_STRIDE_S
    if sample_rate is None:
        sample_rate = cfg.SAMPLE_RATE
    class_idx = cfg.class_to_idx()
    stride_samp = int(frame_stride_s * sample_rate)

    pos_counts = torch.zeros(n_classes, dtype=torch.float64)
    total_frames = 0  # over all segments combined

    for seg in segments:
        n_samples = seg.end_sample - seg.start_sample
        n_frames = n_samples // stride_samp
        if n_frames <= 0:
            continue
        total_frames += n_frames
        seg_start_s = seg.start_sample / sample_rate
        # Mirror the painting logic from WhaleDataset.__getitem__ but
        # count frames directly rather than allocating a target tensor.
        # Multiple annotations may overlap, so we OR them per-class via a
        # boolean buffer to avoid double-counting overlapping calls of
        # the same class within a segment.
        per_class_frame_mask = torch.zeros(n_classes, n_frames, dtype=torch.bool)
        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label not in class_idx:
                continue
            c = class_idx[label]
            local_start_s = max(0.0, a["start_s"] - seg_start_s)
            local_end_s = min(n_samples / sample_rate, a["end_s"] - seg_start_s)
            f0 = int(local_start_s / frame_stride_s)
            f1 = int(local_end_s / frame_stride_s)
            if f1 > f0:
                per_class_frame_mask[c, f0:f1] = True
        pos_counts += per_class_frame_mask.sum(dim=1).to(torch.float64)

    # Per-class negative frame count = (total frames) - (positive frames).
    # Per-class weight = N_c / P_c with a tiny epsilon for the degenerate
    # all-positive case (will not happen in practice, but guards against
    # divide-by-zero if a class is missing entirely).
    neg_counts = float(total_frames) - pos_counts
    eps = 1.0
    weights = (neg_counts.clamp(min=eps)) / (pos_counts.clamp(min=eps))
    weights = weights.to(torch.float32)

    if normalize_min_to_one:
        weights = weights / weights.min()

    if verbose:
        print("Per-class frame-level pos_weight (canonical recipe):")
        names = cfg.class_names()
        for i, name in enumerate(names):
            pc = int(pos_counts[i].item())
            nc = int(neg_counts[i].item())
            ratio = nc / max(pc, 1)
            print(f"  {name:6s}  pos={pc:>10,d}  neg={nc:>10,d}  "
                  f"raw={ratio:>8.2f}  w={float(weights[i]):.3f}")
        print(f"  total frames considered: {total_frames:,}")

    return weights


# ======================================================================
# Padding-mask audit
# ======================================================================

def assert_mask_is_correct(
    audio_pad: torch.Tensor,
    target_pad: torch.Tensor,
    mask_pad: torch.Tensor,
    metas: list[dict],
    frame_stride_s: float | None = None,
    sample_rate: int | None = None,
) -> None:
    """
    Verify that the padding mask actually matches the segment-level lengths.

    Intended to be called inside the training loop on the first few batches
    of the first epoch as a one-time sanity check. If the assertion passes
    on a couple of batches there is no reason to suspect the rest will
    differ — collation logic is deterministic. The cost is negligible
    (a few CPU ops per batch), and the failure mode it catches is silent.

    The check is:
        For every sample i in the batch, the mask should be True for
        exactly the first ``ceil((end_sample - start_sample) / hop)`` frames
        and False thereafter — i.e. ``mask_pad[i].sum() == n_real_frames_i``,
        where ``n_real_frames_i`` is derived from the meta dict.

    Parameters
    ----------
    audio_pad : torch.Tensor, shape (B, max_samples)
    target_pad : torch.Tensor, shape (B, max_frames, n_classes)
    mask_pad : torch.Tensor, shape (B, max_frames), bool
    metas : list of dict
        Per-sample metadata with at least ``start_sample`` and ``end_sample``.
    frame_stride_s, sample_rate : optional overrides

    Raises
    ------
    AssertionError
        If the mask does not match the implied segment lengths.
    """
    if frame_stride_s is None:
        frame_stride_s = cfg.FRAME_STRIDE_S
    if sample_rate is None:
        sample_rate = cfg.SAMPLE_RATE
    hop = int(frame_stride_s * sample_rate)

    B, max_frames = mask_pad.shape
    assert target_pad.shape[:2] == (B, max_frames), (
        f"target_pad shape {target_pad.shape} disagrees with mask_pad {mask_pad.shape}"
    )

    for i, meta in enumerate(metas):
        n_real_samples = meta["end_sample"] - meta["start_sample"]
        n_real_frames = n_real_samples // hop
        mask_true_count = int(mask_pad[i].sum().item())
        # Allow off-by-at-most-one because integer rounding of (samples / hop)
        # can disagree by 1 frame depending on which end the rounding falls.
        assert abs(mask_true_count - n_real_frames) <= 1, (
            f"sample {i} ({meta.get('dataset')}/{meta.get('filename')}): "
            f"mask has {mask_true_count} True frames but segment implies "
            f"{n_real_frames} frames (samples={n_real_samples}, hop={hop})"
        )
        # Padded positions must have all-zero targets, otherwise the loss
        # will leak gradient through positions that don't correspond to
        # real audio. The collator zero-initialises target_pad, so this
        # should always pass; the assertion is here to catch future drift.
        if mask_true_count < max_frames:
            pad_targets = target_pad[i, mask_true_count:, :]
            assert pad_targets.abs().sum().item() == 0, (
                f"sample {i}: padded target rows are nonzero — "
                f"target_pad[{i}, {mask_true_count}:, :].sum() != 0"
            )


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Quick functional check: build the training set as train.py would,
    # then compute frame-level pos_weights. Compares against the file-level
    # weights from model.compute_class_weights so the divergence is visible.
    print("=" * 70)
    print("Canonical-recipe pos_weight self-test")
    print("=" * 70)

    from dataset import (
        get_file_manifest, load_annotations,
        build_positive_segments, build_negative_segments,
    )
    from model import compute_class_weights as file_level_weights

    print("\nLoading training manifest and annotations...")
    manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    annotations = load_annotations(cfg.TRAIN_DATASETS, manifest=manifest)
    pos = build_positive_segments(annotations, manifest)
    neg = build_negative_segments(
        annotations, manifest,
        n_segments=int(len(pos) * cfg.NEG_RATIO),
    )
    print(f"  positives: {len(pos)},  negatives: {len(neg)}")

    print("\n--- Frame-level (canonical) ---")
    w_frame = compute_pos_weights_framewise(pos + neg)

    print("\n--- File-level (current model.compute_class_weights) ---")
    w_file = file_level_weights()

    print("\n--- Comparison ---")
    for i, name in enumerate(cfg.class_names()):
        wf, wl = float(w_frame[i]), float(w_file[i])
        print(f"  {name:6s}  frame={wf:.3f}  file={wl:.3f}  ratio={wf/wl:.3f}")
