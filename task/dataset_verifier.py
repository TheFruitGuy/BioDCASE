"""
Verifier Dataset — Audio Crops Around Candidate Events
======================================================

PyTorch ``Dataset`` that serves cropped audio windows centred on stage-1
candidate events, paired with the candidate's predicted class identity and
the ground-truth TP/FP label produced by ``extract_candidates.py``.

v2 (2026-05-08): added optional augmentation (random time shift, volume
scaling, time masking) and made the crop window configurable. The v1
defaults (30 s, no augmentation) remain available via constructor args
for direct comparison.

Why the v2 changes
------------------
v1's first run peaked at epoch 1 (macro F1 = 0.7393) and degraded
afterwards while training loss kept dropping (0.59 → 0.16 over 9 epochs).
Classic overfitting on a 14k-sample dataset with a 200k-parameter model
seeing the same audio crops every epoch. The fix has three legs:
  1. Halve the crop length (30 s → 15 s) — less audio per sample, less
     to memorise. 15 s still covers the longest BMABZ Z-calls (~20 s)
     when the call is centred (the ~2.5 s clipped from each end is on
     the call's quiet tails, not its informative middle).
  2. Augment during training so the same TP is seen with different
     position, volume, and dropouts each time.
  3. Stronger regularization in the model (handled in train_verifier.py).

Augmentation in __getitem__, not as a wrapper
---------------------------------------------
We gate every augmentation step on ``self.train`` so the same Dataset
class can serve both phases — just construct a separate val instance with
``train=False``. This is simpler than chaining transforms and avoids
accidentally augmenting validation crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import config as cfg


# ======================================================================
# Lightweight per-row record
# ======================================================================

@dataclass(slots=True)
class CandidateRecord:
    """
    One stage-1 candidate event, ready to be turned into an audio crop.

    Attributes
    ----------
    cand_id : int
        Globally unique row id from the candidates parquet.
    dataset, filename : str
        Origin of the candidate.
    path : str
        Absolute path to the WAV file.
    file_dur_s : float
        Total duration of the source file in seconds.
    start_s, end_s : float
        Candidate event span (file-relative).
    class_idx : int
        Index of the predicted coarse class (0 = bmabz, 1 = d, 2 = bp).
    label : int
        1 for TP (matched a GT event), 0 for FP.
    stage1_score : float
        Mean per-frame probability over the candidate span.
    """
    cand_id: int
    dataset: str
    filename: str
    path: str
    file_dur_s: float
    start_s: float
    end_s: float
    class_idx: int
    label: int
    stage1_score: float


# ======================================================================
# Loading the parquet output of extract_candidates.py
# ======================================================================

def load_candidates(parquet_path: str | Path) -> list[CandidateRecord]:
    """Read a candidates parquet into a list of ``CandidateRecord``."""
    df = pd.read_parquet(parquet_path)
    records = []
    for _, r in df.iterrows():
        records.append(CandidateRecord(
            cand_id=int(r["cand_id"]),
            dataset=str(r["dataset"]),
            filename=str(r["filename"]),
            path=str(r["path"]),
            file_dur_s=float(r["file_dur_s"]),
            start_s=float(r["start_s"]),
            end_s=float(r["end_s"]),
            class_idx=int(r["class_idx"]),
            label=int(r["label"]),
            stage1_score=float(r["stage1_score"]),
        ))
    return records


# ======================================================================
# Dataset
# ======================================================================

class VerifierDataset(Dataset):
    """
    Map-style dataset of (audio_crop, class_idx, label, meta) 4-tuples.

    Parameters
    ----------
    records : list of CandidateRecord
    crop_s : float, default 15.0
        Crop window length in seconds. v1 used 30; v2 default 15 cuts
        memorisable surface area in half.
    train : bool, default False
        Enables augmentation. Validation should always be ``False`` for
        deterministic eval.
    time_shift_max_s : float, default 2.0
        Max absolute shift of the crop centre, sampled uniformly per call.
        Stops the verifier from memorising the always-centred event
        position.
    volume_scale_range : tuple, default (0.7, 1.3)
        Multiplicative gain range — approximates per-recording level
        variation.
    time_mask_max_s : float, default 1.0
        Max width of the random zero-mask applied to the waveform.
    time_mask_prob : float, default 0.5
        Probability of applying time mask in any given training call.
    """

    def __init__(
        self,
        records: list[CandidateRecord],
        crop_s: float = 15.0,
        train: bool = False,
        time_shift_max_s: float = 2.0,
        volume_scale_range: tuple[float, float] = (0.7, 1.3),
        time_mask_max_s: float = 1.0,
        time_mask_prob: float = 0.5,
    ):
        self.records = records
        self.crop_s = crop_s
        self.crop_samples = int(round(crop_s * cfg.SAMPLE_RATE))
        self.train = train
        self.time_shift_max_s = time_shift_max_s
        self.volume_scale_range = volume_scale_range
        self.time_mask_max_s = time_mask_max_s
        self.time_mask_prob = time_mask_prob

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # Crop geometry
    # ------------------------------------------------------------------

    def _compute_crop_window(
        self, rec: CandidateRecord, shift_s: float = 0.0
    ) -> tuple[int, int]:
        """
        Compute ``[start_sample, end_sample)`` for the audio crop.

        The candidate is centred (plus optional ``shift_s``) when
        possible; if centring would push either edge outside
        ``[0, file_dur_s]``, the window is shifted inward.
        """
        sr = cfg.SAMPLE_RATE
        file_samples = int(round(rec.file_dur_s * sr))
        center_s = (rec.start_s + rec.end_s) / 2 + shift_s
        center_sample = int(round(center_s * sr))

        half = self.crop_samples // 2
        start = center_sample - half
        end = start + self.crop_samples

        if start < 0:
            shift = -start
            start += shift
            end += shift
        if end > file_samples:
            shift = end - file_samples
            start -= shift
            end -= shift
        start = max(0, start)
        end = min(file_samples, end)
        return start, end

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Per-worker numpy RNG (seeded via ``worker_init_fn`` in
        # train_verifier.py).
        if self.train and self.time_shift_max_s > 0:
            shift_s = float(np.random.uniform(
                -self.time_shift_max_s, self.time_shift_max_s,
            ))
        else:
            shift_s = 0.0

        start, end = self._compute_crop_window(rec, shift_s=shift_s)
        audio_np, sr = sf.read(
            rec.path, start=start, stop=end, dtype="float32",
        )
        assert sr == cfg.SAMPLE_RATE, (
            f"Expected {cfg.SAMPLE_RATE} Hz, got {sr} for {rec.path}"
        )

        # Pad / truncate to exact crop length.
        n_loaded = audio_np.shape[0]
        if n_loaded < self.crop_samples:
            pad = np.zeros(self.crop_samples - n_loaded, dtype=np.float32)
            audio_np = np.concatenate([audio_np, pad], axis=0)
        elif n_loaded > self.crop_samples:
            audio_np = audio_np[: self.crop_samples]

        # Augmentations — training only.
        if self.train:
            scale = float(np.random.uniform(*self.volume_scale_range))
            audio_np = audio_np * scale

            if np.random.random() < self.time_mask_prob:
                mask_s = float(np.random.uniform(0.1, self.time_mask_max_s))
                mask_len = int(round(mask_s * cfg.SAMPLE_RATE))
                mask_len = min(mask_len, self.crop_samples - 1)
                if mask_len > 0:
                    mask_start = int(np.random.randint(
                        0, self.crop_samples - mask_len + 1,
                    ))
                    audio_np[mask_start:mask_start + mask_len] = 0.0

        audio = torch.from_numpy(audio_np)
        class_idx = torch.tensor(rec.class_idx, dtype=torch.long)
        label = torch.tensor(rec.label, dtype=torch.float32)
        meta = {
            "cand_id": rec.cand_id,
            "stage1_score": rec.stage1_score,
        }
        return audio, class_idx, label, meta

    # ------------------------------------------------------------------
    # Balanced sampler
    # ------------------------------------------------------------------

    def make_balanced_sampler(
        self,
        num_samples: int | None = None,
        replacement: bool = True,
    ) -> WeightedRandomSampler:
        """
        ``WeightedRandomSampler`` that equalises (class × label) cells in
        expectation.
        """
        if num_samples is None:
            num_samples = len(self.records)

        counts: dict[tuple[int, int], int] = {}
        for rec in self.records:
            counts[(rec.class_idx, rec.label)] = (
                counts.get((rec.class_idx, rec.label), 0) + 1
            )
        n_cells = len(counts)

        weights = np.zeros(len(self.records), dtype=np.float64)
        for i, rec in enumerate(self.records):
            cell_size = counts[(rec.class_idx, rec.label)]
            weights[i] = 1.0 / (n_cells * cell_size)

        return WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=num_samples,
            replacement=replacement,
        )


# ======================================================================
# Collation
# ======================================================================

def verifier_collate_fn(batch):
    """Stack same-shape tensors; pass meta through as list of dicts."""
    audios, class_idxs, labels, metas = zip(*batch)
    audio = torch.stack(audios, dim=0)
    class_idx = torch.stack(class_idxs, dim=0)
    label = torch.stack(labels, dim=0)
    return audio, class_idx, label, list(metas)


# ======================================================================
# Sanity helper (CLI)
# ======================================================================

def _summarise(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df)} candidates from {parquet_path}")
    print(f"  splits: {df['source_split'].value_counts().to_dict()}")
    for cls_name in cfg.CALL_TYPES_3:
        sub = df[df["predicted_class"] == cls_name]
        n_tp = int((sub["label"] == 1).sum())
        n_fp = int((sub["label"] == 0).sum())
        prec = n_tp / max(n_tp + n_fp, 1)
        print(f"  {cls_name}: TP={n_tp}  FP={n_fp}  precision={prec:.3f}  "
              f"mean stage1_score(TP)={sub.loc[sub['label']==1, 'stage1_score'].mean():.3f}  "
              f"mean stage1_score(FP)={sub.loc[sub['label']==0, 'stage1_score'].mean():.3f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python dataset_verifier.py CANDIDATES.parquet")
        sys.exit(1)
    _summarise(sys.argv[1])
