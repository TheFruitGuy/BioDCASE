"""
Precompute a Cross-Site No-Call Audio Pool for Phase 1e
========================================================

Phase 1e (cross-site noise mixing) needs to draw, on every training
batch, a clip of pure ocean noise from a site *other* than the one a
positive segment came from. Doing this lazily during training is slow
(every sample requires a disk seek) and non-deterministic (results
depend on filesystem caching state).

This script does it once, upfront. We sample a fixed number of no-call
30-second clips from each training site, decode them to int16 waveforms
on disk, and save them as a single ``.pt`` file. The Phase 1e training
script then loads this file into a ``NoCallPool`` (defined in
``phase1_baseline.py``) which serves clips from in-RAM tensors.

Storage cost
------------
Per clip: 30 s × 250 Hz × 2 bytes (int16) = 15 kB
Per site: 50 clips × 15 kB = 750 kB
8 sites: ~6 MB total

Easily fits in RAM, instantly accessible.

Why ``int16``
-------------
The original audio files are typically int16 PCM. Storing the pool as
int16 keeps the file small without precision loss (the spectrogram
extractor doesn't care — it normalises internally), and eliminates
the need to keep the values in [-1, 1] floats.

Identifying no-call segments
----------------------------
We reuse the existing ``build_negative_segments`` logic from
``dataset.py``, which already handles the "find a 30s window with no
annotations overlapping" problem correctly. We just need ``N_PER_SITE``
draws per site rather than the usual one-per-positive-segment.

Determinism
-----------
A fixed seed (1337, matching Phase 0i+ runs) is set before sampling so
re-running this script produces the same pool. This matters because
the entire Phase 1e ablation should be reproducible from the same
no-call pool — otherwise Phase 1e would have an extra source of
between-run variance the other phases don't have.

Usage
-----
::

    python precompute_no_call_pool.py
    # or:
    python precompute_no_call_pool.py --output /path/to/pool.pt --n_per_site 50

Output is ``no_call_pool.pt`` in the project root (or ``--output`` if
specified). Phase 1e loads this from a hardcoded path that defaults to
``no_call_pool.pt`` next to ``train.py``.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import config as cfg
from dataset import (
    load_annotations, get_file_manifest, build_negative_segments,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=str, default="no_call_pool.pt",
        help="Path to write the pool .pt file (default: no_call_pool.pt)",
    )
    p.add_argument(
        "--n_per_site", type=int, default=50,
        help="Number of no-call clips to sample per training site (default: 50)",
    )
    p.add_argument(
        "--seed", type=int, default=1337,
        help="Seed for reproducible sampling (default: 1337)",
    )
    return p.parse_args()


def load_segment_audio(path: str, start_sample: int, n_samples: int) -> np.ndarray:
    """
    Read a fixed-length audio chunk starting at start_sample.

    Returns a 1-D ``np.ndarray`` of dtype ``int16`` and shape
    ``(n_samples,)``. If the file ends before n_samples are read
    (shouldn't happen given how negative segments are sampled, but
    defensive), the result is zero-padded.
    """
    audio, sr = sf.read(
        path, start=start_sample, frames=n_samples,
        dtype="int16", always_2d=False,
    )
    if sr != cfg.SAMPLE_RATE:
        # Negative segments come from the same files as positives, so
        # this shouldn't fire — but if it does, something's wrong with
        # the manifest.
        raise RuntimeError(
            f"Expected sample rate {cfg.SAMPLE_RATE}, got {sr} for {path}"
        )
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.shape[0] < n_samples:
        # Zero-pad if the file was shorter than expected.
        audio = np.concatenate([
            audio,
            np.zeros(n_samples - audio.shape[0], dtype=np.int16),
        ])
    return audio


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    print(f"Output file: {output_path.resolve()}")
    print(f"Per-site clips: {args.n_per_site}")
    print(f"Seed: {args.seed}")
    print(f"Training sites: {list(cfg.TRAIN_DATASETS)}")

    # Each clip is exactly 30 seconds of audio at the project sample rate.
    # We use the same length everywhere in the codebase (validation tile
    # length, Phase 0e training segment length), so positive segments
    # mixed with these clips will share a sample count exactly.
    clip_samples = int(30.0 * cfg.SAMPLE_RATE)

    # Load annotations and manifest once for the whole training set.
    print(f"\nLoading manifest and annotations...")
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    train_annotations = load_annotations(
        cfg.TRAIN_DATASETS, manifest=train_manifest,
    )
    print(f"  {len(train_manifest)} files, "
          f"{len(train_annotations)} annotations")

    # We sample one site at a time so build_negative_segments only sees
    # within-site files (and so the "exclude this site" logic during
    # training cleanly maps to a per-site bucket).
    pool: dict[str, list[np.ndarray]] = {}

    for site in cfg.TRAIN_DATASETS:
        site_manifest = train_manifest[train_manifest["dataset"] == site]
        site_annots = train_annotations[train_annotations["dataset"] == site]
        if len(site_manifest) == 0:
            print(f"\n  [{site}] no files in manifest, skipping")
            continue

        print(f"\n  [{site}] {len(site_manifest)} files, "
              f"{len(site_annots)} annotations")
        t0 = time.time()

        # ``build_negative_segments`` already handles the "30-second
        # window with no overlapping annotation" problem. Asking for
        # ``n_per_site`` segments returns that many.
        neg_segs = build_negative_segments(
            site_annots, site_manifest, n_segments=args.n_per_site,
        )
        if len(neg_segs) == 0:
            print(f"    no negative segments produced for {site}, skipping")
            continue

        clips: list[np.ndarray] = []
        for seg in neg_segs:
            try:
                clip = load_segment_audio(
                    seg.path, seg.start_sample, clip_samples,
                )
                clips.append(clip)
            except Exception as exc:
                print(f"    skipping {seg.path}@{seg.start_sample}: {exc}")
                continue
        if not clips:
            print(f"    no clips loaded for {site}")
            continue
        pool[site] = clips
        print(f"    loaded {len(clips)} clips ({time.time() - t0:.1f}s)")

    if not pool:
        raise RuntimeError(
            "Pool is empty across all sites — check that audio files "
            "are accessible and that build_negative_segments isn't "
            "returning empty lists."
        )

    # Stack per-site clips into a single tensor for fast indexing during
    # training. Layout: ``pool[site] -> tensor of shape (n_clips, n_samples)``.
    serialised = {}
    total_clips = 0
    for site, clips in pool.items():
        arr = np.stack(clips, axis=0)  # (n_clips, n_samples), int16
        serialised[site] = torch.from_numpy(arr)
        total_clips += arr.shape[0]
    print(f"\nTotal: {total_clips} clips across {len(serialised)} sites")
    print(f"Memory footprint: "
          f"{sum(t.numel() * 2 for t in serialised.values()) / 1024:.0f} kB")

    # Save with metadata so the loader can sanity-check on load.
    torch.save({
        "version": 1,
        "sample_rate": cfg.SAMPLE_RATE,
        "clip_samples": clip_samples,
        "n_per_site_requested": args.n_per_site,
        "seed": args.seed,
        "sites": list(serialised.keys()),
        "clips": serialised,
    }, output_path)
    print(f"\nWrote {output_path} "
          f"({output_path.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
