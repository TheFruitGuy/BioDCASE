"""
SSL Pretraining Dataset
=======================

Builds an audio-only manifest from labeled BioDCASE training audio + unlabeled
AADC pretraining sites, and provides a PyTorch Dataset that yields random 30s
clips together with their source-site name.

The clip-pair construction (clip_a, clip_b = aug(x), aug(x)) lives in the
training loop, not here — the Dataset returns plain clips and the source site
so 3α and 3β share exactly the same Dataset.

Why a separate manifest builder
-------------------------------
``dataset.get_file_manifest`` only walks ``cfg.DATA_ROOT/{train|validation}/
audio/{ds}/`` and parses ATBFL-style filenames (``2014-06-29T23-00-00_000.wav``).
AADC files live under a different root, use a different filename convention
(``14_2018-02-02_00-00-00.wav``), and have no annotation CSVs.

For SSL we don't need timestamps — only paths and durations to draw random
windows. The builder here is intentionally simple: walk a directory list,
read WAV headers, return a DataFrame.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

import config as cfg


# ======================================================================
# Manifest construction
# ======================================================================

def build_pretrain_manifest(
    train_datasets: Optional[list[str]] = None,
    aadc_sites: Optional[list[str]] = None,
    aadc_root: Optional[Path] = None,
    min_duration_s: float = 60.0,
    expected_sample_rate: int = 250,
) -> pd.DataFrame:
    """
    Walk labeled BioDCASE training audio + AADC pretraining audio.

    Parameters
    ----------
    train_datasets : list of str, optional
        BioDCASE training site names (e.g. ``["ballenyislands2015", ...]``).
        Audio is read from ``cfg.DATA_ROOT/train/audio/{ds}/*.wav``.
    aadc_sites : list of str, optional
        AADC site names (e.g. ``["DDU2018", ...]``). Audio is read from
        ``aadc_root/{site}/*.wav``.
    aadc_root : Path, optional
        Root directory under which AADC site subfolders live. Required if
        ``aadc_sites`` is non-empty.
    min_duration_s : float, default 60.0
        Skip files shorter than this. Avoids edge cases where a 30s clip
        cannot be drawn from a file.
    expected_sample_rate : int, default 250
        Verify all files have this SR. Raises if not — catches forgotten
        decimation steps before training starts.

    Returns
    -------
    pd.DataFrame with columns: ``site``, ``path``, ``duration_s``,
    ``n_samples``, ``sample_rate``.
    """
    rows: list[dict] = []

    # Labeled training-site audio (from cfg.DATA_ROOT)
    for ds in train_datasets or []:
        audio_dir = cfg.DATA_ROOT / "train" / "audio" / ds
        if not audio_dir.exists():
            print(f"[manifest] WARNING: missing training audio dir {audio_dir}")
            continue
        n_added = 0
        for wav in sorted(audio_dir.glob("*.wav")):
            info = sf.info(str(wav))
            if info.duration < min_duration_s:
                continue
            rows.append({
                "site": ds, "path": str(wav),
                "duration_s": float(info.duration),
                "n_samples": int(info.frames),
                "sample_rate": int(info.samplerate),
            })
            n_added += 1
        print(f"[manifest] {ds}: {n_added} files")

    # Unlabeled AADC pretraining audio (from aadc_root)
    if aadc_sites:
        if aadc_root is None:
            raise ValueError("aadc_root must be provided when aadc_sites is non-empty")
        for site in aadc_sites:
            audio_dir = Path(aadc_root) / site
            if not audio_dir.exists():
                print(f"[manifest] WARNING: missing AADC audio dir {audio_dir}")
                continue
            n_added = 0
            for wav in sorted(audio_dir.glob("*.wav")):
                info = sf.info(str(wav))
                if info.duration < min_duration_s:
                    continue
                rows.append({
                    "site": site, "path": str(wav),
                    "duration_s": float(info.duration),
                    "n_samples": int(info.frames),
                    "sample_rate": int(info.samplerate),
                })
                n_added += 1
            print(f"[manifest] {site}: {n_added} files")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "Pretrain manifest is empty. Check --aadc-root and --aadc-sites, "
            "and that cfg.DATA_ROOT contains train/audio/ for the requested sites."
        )

    # Sample-rate sanity check — protects against forgetting to decimate.
    sr_unique = df["sample_rate"].unique()
    bad = [int(sr) for sr in sr_unique if sr != expected_sample_rate]
    if bad:
        raise RuntimeError(
            f"Manifest contains files at unexpected SR(s) {bad}. "
            f"Expected {expected_sample_rate} Hz everywhere. "
            f"Check that AADC audio was decimated."
        )

    total_h = df["duration_s"].sum() / 3600.0
    print(f"[manifest] total: {len(df)} files, "
          f"{total_h:.1f} hours, {df['site'].nunique()} sites")
    return df


# ======================================================================
# Clip-pair dataset
# ======================================================================

class SSLClipDataset(Dataset):
    """
    Random 30s clips at 250 Hz from the pretrain manifest.

    Length is virtual: ``epoch_clips`` samples are drawn per epoch by
    randomly picking a file (weighted by duration so longer recordings
    get proportionally more draws) and a uniform-random start offset.

    Each ``__getitem__`` returns a dict ``{"audio": (n_samples,) float32,
    "site": str}``. The training loop is responsible for producing two
    augmented views per clip.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        clip_seconds: float = 30.0,
        sample_rate: int = 250,
        epoch_clips: int = 50_000,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(clip_seconds * sample_rate)
        self.epoch_clips = int(epoch_clips)

        durations = self.manifest["duration_s"].to_numpy()
        if (durations <= 0).any():
            raise ValueError("Manifest contains files with non-positive duration")
        self.file_weights = durations / durations.sum()

    def __len__(self) -> int:
        return self.epoch_clips

    def __getitem__(self, idx: int):
        # Per-call random sampling. Each worker has its own RNG seeded
        # by torch's worker_init; combining with idx makes draws unique
        # across (worker, idx) without becoming a global bottleneck.
        winfo = torch.utils.data.get_worker_info()
        seed = (winfo.seed ^ idx) & 0xFFFFFFFF if winfo is not None else idx
        rng = np.random.default_rng(seed)

        file_idx = int(rng.choice(len(self.manifest), p=self.file_weights))
        row = self.manifest.iloc[file_idx]
        n_avail = int(row["n_samples"])

        if n_avail < self.clip_samples:
            offset = 0
            need_pad = self.clip_samples - n_avail
            n_read = n_avail
        else:
            offset = int(rng.integers(0, n_avail - self.clip_samples + 1))
            need_pad = 0
            n_read = self.clip_samples

        audio, _sr = sf.read(
            row["path"], start=offset, frames=n_read,
            dtype="float32", always_2d=False,
        )
        if audio.ndim > 1:                         # collapse stereo if any
            audio = audio[:, 0]
        if need_pad > 0:
            audio = np.pad(audio, (0, need_pad))
        if len(audio) != self.clip_samples:        # defensive trim/pad
            audio = audio[:self.clip_samples]
            if len(audio) < self.clip_samples:
                audio = np.pad(audio, (0, self.clip_samples - len(audio)))

        return {
            "audio": torch.from_numpy(audio.astype(np.float32, copy=False)),
            "site": str(row["site"]),
        }


def collate_ssl(batch: list[dict]) -> dict:
    """Stack audio into ``(B, n_samples)``, keep site names as a list."""
    audio = torch.stack([b["audio"] for b in batch], dim=0)
    sites = [b["site"] for b in batch]
    return {"audio": audio, "sites": sites}
