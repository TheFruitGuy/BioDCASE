"""
Segment length core
===================

Fixed-length segment extension utilities, factored out of the original
``train_phase0e`` experiment script so they can be reused without pulling
in the whole phase-0 training chain (``train_phase0e`` →
``train_phase0`` → ``train_phase0c``).

The key fix this implements: training segments must match the validation
tile length. The BiLSTM was originally trained on ~10s sequences and
validated on 30s tiles, producing miscalibrated confidence at eval.
Forcing every training segment to a fixed 30s removed the oscillation.

These helpers are imported by the HNM and mean-teacher training scripts
as well as the 30s dataset builder.
"""

from __future__ import annotations

import random as random_module
from dataclasses import replace

import config as cfg
from dataset import Segment


#: Target segment length in seconds, matching validation tiles.
PHASE0E_SEGMENT_S = 30.0


def extend_segment_to_fixed_length(
    seg: Segment,
    target_seconds: float,
    file_duration_s: float,
    sample_rate: int = cfg.SAMPLE_RATE,
    rng: random_module.Random = None,
) -> Segment:
    """
    Return a copy of ``seg`` whose ``[start_sample, end_sample)`` window
    is exactly ``target_seconds`` long.

    The original segment is grown to the target length by adding context
    on both sides. The position of the original content within the new
    window is randomised so the model doesn't learn a fixed
    "calls-always-near-frame-N" prior. If the original segment is already
    at or above the target length, it's returned unchanged.

    The segment's ``annotations`` field is NOT modified — annotation
    times are file-relative, so they remain valid regardless of how the
    segment window moves. The frame-level target tensor is rebuilt by
    ``WhaleDataset.__getitem__`` based on which annotations intersect
    the new ``[start_sample, end_sample)`` range.

    Parameters
    ----------
    seg : Segment
        The original variable-length training segment.
    target_seconds : float
        Desired final length, e.g. 30.0 for matching validation tiles.
    file_duration_s : float
        Total duration of the underlying audio file, used to clamp the
        extended window to file boundaries.
    sample_rate : int
    rng : random.Random, optional
        For reproducible randomised positioning. ``None`` uses the
        module-level ``random``.

    Returns
    -------
    Segment
        A new Segment dataclass instance; original is not mutated.
    """
    if rng is None:
        rng = random_module

    target_samples = int(target_seconds * sample_rate)
    file_samples = int(file_duration_s * sample_rate)
    cur_length = seg.end_sample - seg.start_sample

    # Already at or above target; nothing to do.
    if cur_length >= target_samples:
        return seg

    extra = target_samples - cur_length

    # File is shorter than target — center on midpoint, clamp ends.
    if file_samples <= target_samples:
        return replace(seg, start_sample=0, end_sample=file_samples)

    # How much can we add on each side without leaving the file?
    pre_room = seg.start_sample
    post_room = file_samples - seg.end_sample

    # Randomise the split between pre and post when both sides have room.
    # Bound by what each side actually has available, then by what we need.
    pre_extra = min(pre_room, rng.randint(0, extra))
    post_extra = min(post_room, extra - pre_extra)

    # If one side runs out of room, push the leftover to the other side.
    deficit = extra - pre_extra - post_extra
    if deficit > 0:
        if pre_room - pre_extra >= deficit:
            pre_extra += deficit
        else:
            post_extra += deficit

    new_start = max(0, seg.start_sample - pre_extra)
    new_end = min(file_samples, new_start + target_samples)
    # Final guarantee on length.
    new_start = max(0, new_end - target_samples)

    return replace(seg, start_sample=new_start, end_sample=new_end)


def extend_all_segments(segments, manifest, target_seconds: float):
    """
    Apply ``extend_segment_to_fixed_length`` to every segment in a list.

    Looks up each segment's ``duration_s`` from the file manifest so the
    extension can clamp to file boundaries.
    """
    rng = random_module.Random(0xC0FFEE)
    duration_lookup = {
        (r["dataset"], r["filename"]): r["duration_s"]
        for _, r in manifest.iterrows()
    }
    extended = []
    for seg in segments:
        dur = duration_lookup.get((seg.dataset, seg.filename))
        if dur is None:
            continue
        extended.append(
            extend_segment_to_fixed_length(seg, target_seconds, dur, rng=rng)
        )
    return extended
