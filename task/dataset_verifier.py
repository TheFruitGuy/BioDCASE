"""
Verifier Dataset — Audio Crops Around Candidate Events
======================================================

PyTorch ``Dataset`` that serves cropped audio windows centred on stage-1
candidate events, paired with the candidate's predicted class identity and
the ground-truth TP/FP label produced by ``extract_candidates.py``.

Design choices
--------------
**30-second crop window (configurable).** The crop matches ``train.py``'s
30-second eval segments, which means:
    - ``SpectrogramExtractor`` works on the crop without any reshaping logic,
    - even the longest whale call (BMABZ Z-calls run ~20 s) fits inside one
      window with margin on either side,
    - batch tensors have a uniform shape, so collation is trivial.

**Edge handling: shift, don't pad.** If centring the crop would place its
left edge before sample 0 or its right edge past end-of-file, we shift the
window so it stays inside the file. This means the event is no longer at
the centre near file boundaries, but every crop has the full window length
with real audio. Pad-with-zeros would have been an alternative but training
on synthetic silence at boundaries is its own bias.

**Class identity passed alongside audio.** The verifier model is multi-head:
one shared backbone, three sigmoid output heads (one per coarse class), and
the dataset returns the candidate's ``class_idx`` so the training loop can
read the correct head. This shares features across classes while keeping
per-class specialisation in the final layer.

**Balanced sampling.** ``VerifierDataset.make_balanced_sampler`` produces a
``WeightedRandomSampler`` whose weights equalise (class × TP/FP) cells in
expectation. This matters because:
    - BMABZ has many candidates, D has few — uniform sampling would make
      the verifier good at BMABZ and indifferent to D, exactly the
      opposite of what we want;
    - within each class, FPs typically outnumber TPs at the low operating
      threshold, so without rebalancing the model would learn "predict
      everything is FP" and reach high training accuracy that doesn't
      transfer to event-level F1.

What this dataset does NOT do
-----------------------------
- No spectrogram conversion. Audio is returned raw; the training script
  pipes it through ``SpectrogramExtractor`` so the verifier and stage-1
  share the same feature extractor at inference time.
- No on-the-fly augmentation. Augmentations belong in the training loop
  (or a wrapper Dataset) so they can be toggled without touching the
  data-loading code path.
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
        Absolute path to the WAV file (cached at extraction time so the
        DataLoader workers don't have to redo the manifest walk).
    file_dur_s : float
        Total duration of the source file in seconds. Needed for the
        edge-shift logic below.
    start_s, end_s : float
        Candidate event span (file-relative).
    class_idx : int
        Index of the predicted coarse class (0 = bmabz, 1 = d, 2 = bp).
    label : int
        1 for TP (matched a GT event), 0 for FP.
    stage1_score : float
        Mean per-frame probability over the candidate span. Useful as an
        auxiliary input feature, or for diagnostic plotting.
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
    """
    Read a candidates parquet into a list of ``CandidateRecord``.

    Parameters
    ----------
    parquet_path : str or Path

    Returns
    -------
    list of CandidateRecord
    """
    df = pd.read_parquet(parquet_path)
    records = []
    for _, r in df.iterrows():
        # Defensive cast: parquet round-trip can leave numeric columns as
        # numpy dtypes which trip dataclass slots later on.
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
    Map-style dataset of (audio_crop, class_idx, label) triples.

    Parameters
    ----------
    records : list of CandidateRecord
    crop_s : float, default = 30.0
        Crop window length in seconds. 30 s matches the stage-1 eval
        window so SpectrogramExtractor works without any reshaping.
    rng_seed : int, optional
        Currently unused (no random augmentation), reserved so a future
        augmenting wrapper can stay reproducible.
    """

    def __init__(
        self,
        records: list[CandidateRecord],
        crop_s: float = 30.0,
        rng_seed: int | None = None,
    ):
        self.records = records
        self.crop_s = crop_s
        self.crop_samples = int(round(crop_s * cfg.SAMPLE_RATE))
        self.rng_seed = rng_seed

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # Crop geometry
    # ------------------------------------------------------------------

    def _compute_crop_window(self, rec: CandidateRecord) -> tuple[int, int]:
        """
        Compute ``[start_sample, end_sample)`` for the audio crop.

        The candidate is centred when possible; if centring would push
        either edge outside ``[0, file_dur_s]``, the window is shifted
        inward so it always has the configured length and contains only
        real audio. For files shorter than ``crop_s`` (very rare), the
        whole file is returned and the caller's spectrogram path will see
        a shorter sequence — handled by zero-padding in ``__getitem__``.

        Parameters
        ----------
        rec : CandidateRecord

        Returns
        -------
        start_sample, end_sample : int
        """
        sr = cfg.SAMPLE_RATE
        file_samples = int(round(rec.file_dur_s * sr))
        center_sample = int(round(((rec.start_s + rec.end_s) / 2) * sr))

        half = self.crop_samples // 2
        start = center_sample - half
        end = start + self.crop_samples

        # Shift inward if outside file bounds.
        if start < 0:
            shift = -start
            start += shift
            end += shift
        if end > file_samples:
            shift = end - file_samples
            start -= shift
            end -= shift
        # If the file is shorter than crop_s, clamp to file bounds and let
        # __getitem__ pad. Both must be non-negative.
        start = max(0, start)
        end = min(file_samples, end)

        return start, end

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        """
        Load the audio crop for one candidate and return a 4-tuple.

        Returns
        -------
        audio : torch.Tensor, shape (crop_samples,)
            Float32 waveform. Zero-padded on the right if the source file
            is shorter than ``crop_s``.
        class_idx : torch.Tensor, scalar long
            0 = bmabz, 1 = d, 2 = bp.
        label : torch.Tensor, scalar float
            1.0 for TP, 0.0 for FP. Float so it slots straight into BCE.
        meta : dict
            ``{"cand_id": int, "stage1_score": float}`` for downstream
            score combination during inference.
        """
        rec = self.records[idx]
        start, end = self._compute_crop_window(rec)

        audio_np, sr = sf.read(
            rec.path, start=start, stop=end, dtype="float32",
        )
        assert sr == cfg.SAMPLE_RATE, (
            f"Expected {cfg.SAMPLE_RATE} Hz, got {sr} for {rec.path}"
        )

        # Zero-pad on the right if file was shorter than crop_s.
        n_loaded = audio_np.shape[0]
        if n_loaded < self.crop_samples:
            pad = np.zeros(self.crop_samples - n_loaded, dtype=np.float32)
            audio_np = np.concatenate([audio_np, pad], axis=0)
        elif n_loaded > self.crop_samples:
            # Defensive: should not happen given our shift logic, but
            # truncate just in case.
            audio_np = audio_np[: self.crop_samples]

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
        Build a ``WeightedRandomSampler`` that equalises (class × label).

        Each of the up to 6 cells (3 classes × {TP, FP}) is sampled with
        equal expected frequency: weights for each record are
        ``1 / (n_cells × n_records_in_its_cell)``. Cells that happen to be
        empty (e.g. no D-class TPs at all) contribute nothing — their
        absence is correctly handled by uniform-over-non-empty-cells.

        Parameters
        ----------
        num_samples : int, optional
            Total samples per epoch. Defaults to ``len(self)``.
        replacement : bool, default True
            Whether to sample with replacement. With six cells of varying
            sizes and replacement = True, an "epoch" loosely covers each
            cell ~``num_samples / 6`` times — the right behaviour for
            balancing.

        Returns
        -------
        WeightedRandomSampler
        """
        if num_samples is None:
            num_samples = len(self.records)

        # Count records per (class_idx, label) cell.
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
    """
    Stack same-shape tensors and pass meta through as a list of dicts.

    All audio crops are the same length (``crop_samples``) by
    construction, so this is just a stack — much simpler than
    ``dataset.collate_fn``.

    Parameters
    ----------
    batch : list of 4-tuples
        Each element is ``(audio, class_idx, label, meta)``.

    Returns
    -------
    audio : torch.Tensor, shape (B, crop_samples)
    class_idx : torch.Tensor, shape (B,) long
    label : torch.Tensor, shape (B,) float
    metas : list of dict
    """
    audios, class_idxs, labels, metas = zip(*batch)
    audio = torch.stack(audios, dim=0)
    class_idx = torch.stack(class_idxs, dim=0)
    label = torch.stack(labels, dim=0)
    return audio, class_idx, label, list(metas)


# ======================================================================
# Sanity helper (CLI)
# ======================================================================

def _summarise(parquet_path: str):
    """
    Print a small summary of the candidates parquet without loading audio.
    Intended for ``python -m dataset_verifier path.parquet`` ad-hoc checks.
    """
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
