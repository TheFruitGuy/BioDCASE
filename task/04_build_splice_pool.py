"""
Stage 4: Build the Splice Host Pool
====================================

Combines ``aadc_clips.pt`` from stage 2 with the optional
``aadc_scores.pt`` from stage 3 into a final ``splice_host_pool.pt``
ready to drop into the augmentation. The output schema is
deliberately self-contained so the augmentation needs only this one
file at training time.

Two pool modes
--------------

* **Strict** (``--scores`` provided, default threshold):
  retain only clips with ``max_p_overall < confidence_threshold``.
  These are the clips the current model already classifies as no-call
  with high confidence, so they're the cleanest possible canvas for a
  planted call. A small random fraction
  (``random_window_fraction``) of unfiltered clips is also kept as a
  guardrail against selection bias.

* **Unfiltered** (``--scores`` omitted): every clip from stage 2 is
  kept. Fine for a first iteration; you can always re-build the pool
  later once a scoring checkpoint exists.

Output schema
-------------
``splice_host_pool.pt`` is a torch-pickled dict::

    {
        "hosts": [
            {
                "audio":         torch.float32, (cfg.SAMPLE_RATE * duration_s,)
                "site":          str
                "max_p":         float | None  (None if unscored)
                "is_filtered":   bool   (True = passed confidence filter)
            },
            ...
        ],
        "by_site": {site: [idx, ...]},
        "config": {
            "sample_rate":            int,
            "duration_s":             float,
            "confidence_threshold":   float | None,
            "random_window_fraction": float,
            "n_hosts_total":          int,
            "n_filtered":             int,
            "n_random":               int,
            "donor_sites":            list[str],
        },
    }

Usage
-----
::

    # With confidence filtering (recommended once a checkpoint exists):
    python 04_build_splice_pool.py \\
        --aadc_clips aadc_clips.pt \\
        --scores aadc_scores.pt \\
        --threshold 0.05 \\
        --random_fraction 0.2 \\
        --out splice_host_pool.pt

    # First-iteration unfiltered:
    python 04_build_splice_pool.py \\
        --aadc_clips aadc_clips.pt \\
        --out splice_host_pool.pt
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402


def build_pool(
    aadc_clips_path: Path,
    scores_path: Path | None,
    threshold: float | None,
    random_fraction: float,
    seed: int,
    out_path: Path,
) -> None:
    """Filter and re-pack the AADC clips into the final host pool."""
    print(f"Loading AADC clips from {aadc_clips_path}...")
    pool = torch.load(aadc_clips_path, map_location="cpu")
    clips = pool["clips"]
    donor_sites = pool["config"]["sites"]
    duration_s = pool["config"]["duration_s"]
    print(f"  {len(clips)} clips, sites = {donor_sites}\n")

    scores_by_idx: dict[int, float] = {}
    if scores_path is not None:
        print(f"Loading scores from {scores_path}...")
        scores = torch.load(scores_path, map_location="cpu")
        for s in scores["clip_scores"]:
            if s is None:
                continue
            scores_by_idx[s["clip_idx"]] = s["max_p_overall"]
        print(f"  {len(scores_by_idx)} scored clips\n")

    rng = random.Random(seed)

    use_filter = (scores_by_idx and threshold is not None)
    n_filtered = 0
    n_random = 0
    hosts: list[dict] = []
    by_site: dict[str, list[int]] = defaultdict(list)

    for clip_idx, clip in enumerate(clips):
        max_p = scores_by_idx.get(clip_idx)
        is_filtered = False

        if use_filter:
            below_thr = (max_p is not None) and (max_p < threshold)
            if below_thr:
                is_filtered = True
            elif rng.random() < random_fraction:
                # Guardrail: include a small fraction of unfiltered
                # clips so the model isn't only trained on canvases
                # it already classifies confidently.
                is_filtered = False
            else:
                continue
        # else: keep all clips

        entry = {
            "audio":       clip["audio"],
            "site":        clip["site"],
            "max_p":       max_p,
            "is_filtered": is_filtered,
        }
        idx = len(hosts)
        hosts.append(entry)
        by_site[entry["site"]].append(idx)
        if is_filtered:
            n_filtered += 1
        else:
            n_random += 1

    print(f"Built pool of {len(hosts)} hosts")
    if use_filter:
        print(f"  passed confidence filter (< {threshold}): {n_filtered}")
        print(f"  random-window guardrail:                  {n_random}")
    print(f"\nPer site:")
    for site in donor_sites:
        ids = by_site.get(site, [])
        print(f"  {site:<22} {len(ids):5d} hosts  "
              f"({len(ids) * duration_s / 3600:.1f} h)")

    out = {
        "hosts":   hosts,
        "by_site": dict(by_site),
        "config": {
            "sample_rate":            cfg.SAMPLE_RATE,
            "duration_s":             duration_s,
            "confidence_threshold":   threshold if use_filter else None,
            "random_window_fraction": random_fraction if use_filter else None,
            "n_hosts_total":          len(hosts),
            "n_filtered":             n_filtered,
            "n_random":               n_random,
            "donor_sites":            donor_sites,
            "source_clips":           str(aadc_clips_path),
            "source_scores":          str(scores_path) if scores_path else None,
        },
    }
    torch.save(out, out_path)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aadc_clips", type=Path, default=Path("aadc_clips.pt"))
    p.add_argument("--scores", type=Path, default=None,
                   help="Optional scores from stage 3. Enables filtering.")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Max-p cutoff for filtered hosts (used only with --scores).")
    p.add_argument("--random_fraction", type=float, default=0.2,
                   help="Fraction of non-filtered clips kept as guardrail.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("splice_host_pool.pt"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pool(
        aadc_clips_path=args.aadc_clips,
        scores_path=args.scores,
        threshold=args.threshold,
        random_fraction=args.random_fraction,
        seed=args.seed,
        out_path=args.out,
    )
