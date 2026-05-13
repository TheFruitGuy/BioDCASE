"""
Stage 2: Extract AADC Host Clips
=================================

Walks the safe AADC site directories (sites that are neither in
``cfg.TRAIN_DATASETS`` nor ``cfg.EVAL_DATASETS``) and extracts 30 s
no-call clips to use as background hosts for call splicing. The
output is a ``aadc_clips.pt`` file consumed by stage 3 (optional
scoring) and stage 4 (pool finalization).

Test-site quarantine
--------------------
``cfg.EVAL_DATASETS = ["kerguelen2020", "ddu2021"]`` are the BioDCASE
2025 evaluation sites. This script hard-asserts that no entry in the
SAFE list overlaps with them. If you add new donor sites, update SAFE
below and the assertion will protect you.

Per-clip random offset
----------------------
For each AADC WAV file, we extract ``clips_per_file`` non-overlapping
30 s windows at random offsets. We deliberately do NOT use every
window of every file — that would over-represent long files. Random
sampling gives more donor-site diversity per CPU minute.

Output schema
-------------
``aadc_clips.pt`` is a torch-pickled dict::

    {
        "clips": [
            {
                "audio":       torch.float32, shape (cfg.SAMPLE_RATE * 30,)
                "site":        str
                "source_file": str   (basename only, for compactness)
                "offset_s":    float (clip start within source file)
                "sample_rate": int
                "duration_s":  float (always 30.0)
            },
            ...
        ],
        "by_site":  {site: [idx, ...]},
        "config":   {sample_rate, duration_s, clips_per_file, sites},
    }

Usage
-----
::

    python 02_extract_aadc_hosts.py \\
        --aadc_root /path/to/aadc_archive \\
        --out aadc_clips.pt \\
        --clips_per_file 10
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402


# ----------------------------------------------------------------------
# Safe donor sites — sites that are neither training nor evaluation.
# ----------------------------------------------------------------------

# Current safe sites confirmed downloaded and clear of test-set overlap.
# Scott2019 is included with a caveat: it has documented quality issues
# (annotator notes flag noisy recordings); set --exclude_scott to drop it.
# Names match the CamelCase convention used by download_aadc.py and the
# AADC archive itself. The case-insensitive lookup in _find_wav_files
# accepts lowercase variants too.
SAFE_DEFAULT = ["Casey2018", "Scott2019", "Prydz2013"]

# Future high-value additions once downloaded:
#   "DDU2018", "DDU2019"                 — same hydrophone as ddu2021
#   "Kerguelen2016", "Kerguelen2018",
#   "Kerguelen2019", "Kerguelen2021",
#   "Kerguelen2023", "Kerguelen2024"     — same deployment as kerguelen2020
SAFE_FUTURE = [
    "DDU2018", "DDU2019",
    "Kerguelen2016", "Kerguelen2018", "Kerguelen2019",
    "Kerguelen2021", "Kerguelen2023", "Kerguelen2024",
]


def assert_safe(donor_sites: list[str]) -> None:
    """
    Hard assertion that no donor site is on the BioDCASE evaluation list.

    Case-insensitive: the AADC archive uses CamelCase folder names
    (``DDU2018``, ``Casey2018``) while ``cfg.EVAL_DATASETS`` uses
    lowercase (``ddu2021``, ``kerguelen2020``). A naïve set intersection
    would let ``DDU2021`` slip through. We lower-case everything before
    comparing.
    """
    donor_lc = {s.lower() for s in donor_sites}
    forbidden_lc = {s.lower() for s in cfg.EVAL_DATASETS}
    train_lc = {s.lower() for s in cfg.TRAIN_DATASETS}

    forbidden_hits = donor_lc & forbidden_lc
    if forbidden_hits:
        raise RuntimeError(
            f"\n\n*** TEST-SITE LEAK ATTEMPTED ***\n"
            f"Refusing to extract from: {sorted(forbidden_hits)}\n"
            f"These are the BioDCASE evaluation sites and MUST remain quarantined.\n"
            f"Pipeline aborted.\n"
        )
    train_hits = donor_lc & train_lc
    if train_hits:
        raise RuntimeError(
            f"\n\n*** TRAINING-SITE CONTAMINATION ***\n"
            f"Donor list includes training sites: {sorted(train_hits)}\n"
            f"Donor backgrounds must be UNSEEN. Remove these and re-run.\n"
        )


# ----------------------------------------------------------------------
# Resampling helper (duplicated from 01 to keep scripts independent)
# ----------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resample of a 1-D float32 waveform."""
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(src_sr, dst_sr)
    from scipy.signal import resample_poly
    return resample_poly(audio, up=dst_sr // g, down=src_sr // g).astype(np.float32)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def _find_wav_files(aadc_root: Path, site: str) -> list[Path]:
    """
    Locate all WAV files for one donor site.

    Case-insensitive directory lookup: matches ``DDU2018``,
    ``ddu2018``, ``Ddu2018`` etc. to whichever folder name actually
    exists. The AADC archive uses CamelCase, our config uses lowercase,
    so we accept either.

    Tolerates either ``<root>/<site>/**/*.wav`` (preferred AADC layout)
    or a flat ``<root>/<site>/*.wav`` layout.
    """
    # Try exact match first.
    site_dir = aadc_root / site
    if not site_dir.exists():
        # Fall back to case-insensitive search of immediate subdirs.
        site_lc = site.lower()
        candidates = [
            d for d in aadc_root.iterdir()
            if d.is_dir() and d.name.lower() == site_lc
        ]
        if not candidates:
            print(f"  WARNING: no folder matching '{site}' in {aadc_root}, skipping")
            return []
        site_dir = candidates[0]
        if site_dir.name != site:
            print(f"  (case-insensitive match: '{site}' → '{site_dir.name}')")
    wavs = sorted(site_dir.rglob("*.wav"))
    if not wavs:
        wavs = sorted(site_dir.rglob("*.WAV"))
    return wavs


def _extract_random_clips(
    wav_path: Path,
    n_clips: int,
    duration_s: float,
    target_sr: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, float]]:
    """
    Extract up to ``n_clips`` random 30 s windows from a single WAV.

    If the file is shorter than ``duration_s`` it is skipped. If it's
    only marginally longer we may extract fewer than ``n_clips`` to
    avoid overlap.

    Returns
    -------
    list of (audio_array, offset_s)
        ``audio_array`` is already resampled to ``target_sr``.
    """
    try:
        info = sf.info(str(wav_path))
    except RuntimeError:
        return []
    src_sr = info.samplerate
    file_dur_s = info.frames / src_sr

    if file_dur_s < duration_s + 1.0:
        return []

    # Random non-overlapping offsets. Allocate via stride to avoid
    # overlap; ``shuffle`` then ``take(n_clips)`` ensures randomness.
    stride_s = duration_s
    n_possible = int((file_dur_s - 1.0) // stride_s)
    if n_possible <= 0:
        return []
    offsets_s = [i * stride_s + rng.uniform(0.0, max(0.0, stride_s - 0.5))
                 for i in range(n_possible)]
    # Trim any that would run off the end after the random jitter.
    offsets_s = [o for o in offsets_s if o + duration_s <= file_dur_s]
    rng.shuffle(offsets_s)
    offsets_s = offsets_s[:n_clips]

    clips: list[tuple[np.ndarray, float]] = []
    for offset_s in offsets_s:
        start_frame = int(offset_s * src_sr)
        end_frame = start_frame + int(duration_s * src_sr)
        try:
            audio, _ = sf.read(
                str(wav_path), start=start_frame, stop=end_frame, dtype="float32",
            )
        except RuntimeError:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample(audio, src_sr, target_sr)
        # Pad or trim to exactly the expected length, in case resampling
        # produced an off-by-one length difference.
        expected = int(duration_s * target_sr)
        if len(audio) > expected:
            audio = audio[:expected]
        elif len(audio) < expected:
            audio = np.pad(audio, (0, expected - len(audio)))
        clips.append((audio.astype(np.float32), float(offset_s)))
    return clips


def extract_aadc_hosts(
    aadc_root: Path,
    donor_sites: list[str],
    clips_per_file: int,
    duration_s: float,
    seed: int,
    out_path: Path,
) -> None:
    """Drive the per-site, per-file extraction and save the pool."""
    assert_safe(donor_sites)
    rng = random.Random(seed)

    print(f"Donor sites: {donor_sites}")
    print(f"AADC root:   {aadc_root}")
    print(f"Clips/file:  {clips_per_file}")
    print(f"Duration:    {duration_s} s @ {cfg.SAMPLE_RATE} Hz")
    print(f"Quarantined: {cfg.EVAL_DATASETS}\n")

    clips_all: list[dict] = []
    by_site: dict[str, list[int]] = defaultdict(list)

    for site in donor_sites:
        wavs = _find_wav_files(aadc_root, site)
        print(f"[{site}] {len(wavs)} WAV files found")
        if not wavs:
            continue
        for wav_path in tqdm(wavs, desc=f"  {site}", leave=False):
            extracted = _extract_random_clips(
                wav_path=wav_path,
                n_clips=clips_per_file,
                duration_s=duration_s,
                target_sr=cfg.SAMPLE_RATE,
                rng=rng,
            )
            for audio, offset_s in extracted:
                entry = {
                    "audio":       torch.from_numpy(audio),
                    "site":        site,
                    "source_file": wav_path.name,
                    "offset_s":    offset_s,
                    "sample_rate": cfg.SAMPLE_RATE,
                    "duration_s":  duration_s,
                }
                idx = len(clips_all)
                clips_all.append(entry)
                by_site[site].append(idx)

    print()
    print(f"Extracted {len(clips_all)} host clips")
    total_hours = len(clips_all) * duration_s / 3600.0
    print(f"  {total_hours:.1f} hours of donor background audio")
    for site in donor_sites:
        n = len(by_site.get(site, []))
        print(f"  {site:<22} {n:5d} clips  ({n * duration_s / 3600:.1f} h)")

    pool = {
        "clips":   clips_all,
        "by_site": dict(by_site),
        "config": {
            "sample_rate":    cfg.SAMPLE_RATE,
            "duration_s":     duration_s,
            "clips_per_file": clips_per_file,
            "sites":          donor_sites,
            "seed":           seed,
        },
    }
    torch.save(pool, out_path)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--aadc_root", type=Path, required=True,
        help="Directory containing per-site subdirectories (e.g. /data/aadc/).",
    )
    p.add_argument(
        "--out", type=Path, default=Path("aadc_clips.pt"),
        help="Destination .pt file.",
    )
    p.add_argument(
        "--sites", type=str, nargs="*", default=None,
        help="Donor sites (default: SAFE_DEFAULT). Test sites are blocked.",
    )
    p.add_argument(
        "--clips_per_file", type=int, default=10,
        help="Max random 30s windows extracted per donor WAV.",
    )
    p.add_argument(
        "--duration_s", type=float, default=30.0,
        help="Clip duration in seconds (should match training segment length).",
    )
    p.add_argument(
        "--exclude_scott", action="store_true",
        help="Drop scott2019 from the donor list (has documented quality issues).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for random clip-offset selection.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sites = args.sites if args.sites else list(SAFE_DEFAULT)
    if args.exclude_scott:
        sites = [s for s in sites if s.lower() != "scott2019"]
        print("Note: scott2019 dropped via --exclude_scott")
    extract_aadc_hosts(
        aadc_root=args.aadc_root,
        donor_sites=sites,
        clips_per_file=args.clips_per_file,
        duration_s=args.duration_s,
        seed=args.seed,
        out_path=args.out,
    )
