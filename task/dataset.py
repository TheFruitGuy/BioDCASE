"""
Dataset and data loading for BioDCASE 2026 Task 2.

Handles:
  - Loading ATBFL audio files (250 Hz WAV)
  - Parsing annotation CSVs
  - Segment extraction with random collars (training) / fixed windows (eval)
  - Per-frame binary multi-label target construction at 20 ms resolution
  - Stochastic negative mini-batch undersampling
  - Collation with zero-padding + padding masks
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# The 7 original call types and the 3 collapsed evaluation classes
CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
CALL_TYPES_3 = ["bmabz", "d", "bp"]

# Mapping from 7-class to 3-class
COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d",
    "bpd": "bp", "bp20": "bp", "bp20plus": "bp",
}

CLASS_TO_IDX_7 = {c: i for i, c in enumerate(CALL_TYPES_7)}
CLASS_TO_IDX_3 = {c: i for i, c in enumerate(CALL_TYPES_3)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_root: str = "./data"               # root containing site-year folders
    sample_rate: int = 250
    frame_stride_s: float = 0.02            # 20 ms classification resolution
    use_3class: bool = True                 # collapse to 3-class problem
    collar_min_s: float = 1.0               # min collar around annotation
    collar_max_s: float = 5.0               # max collar around annotation
    eval_segment_s: float = 30.0            # fixed segment length at eval time
    eval_overlap_s: float = 2.0             # overlap between eval segments
    neg_ratio: float = 1.0                  # ratio of neg-to-pos segments per epoch
    min_call_duration_s: float = 0.5
    max_call_duration_s: float = 30.0

    train_datasets: list[str] = field(default_factory=lambda: [
        "ballenyisland2015", "casey2014", "elephantisland2013",
        "elephantisland2014", "greenwich2015", "kerguelen2005",
        "maudrise2014", "rosssea2014",
    ])
    val_datasets: list[str] = field(default_factory=lambda: [
        "casey2017", "kerguelen2014", "kerguelen2015",
    ])


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def load_annotations(data_root: str, dataset_names: list[str]) -> pd.DataFrame:
    """
    Load and merge annotation CSVs for the given site-year datasets.
    Expects one CSV per dataset at: {data_root}/{dataset}/annotation.csv
    """
    frames = []
    for ds in dataset_names:
        csv_path = Path(data_root) / ds / "annotation.csv"
        if not csv_path.exists():
            # Try alternate naming conventions
            for alt in ["annotations.csv", f"{ds}.csv"]:
                alt_path = Path(data_root) / ds / alt
                if alt_path.exists():
                    csv_path = alt_path
                    break
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["dataset"] = ds
            frames.append(df)
        else:
            print(f"Warning: no annotation file found for {ds}")
    if not frames:
        raise FileNotFoundError(f"No annotation files found in {data_root}")
    annotations = pd.concat(frames, ignore_index=True)

    # Parse datetimes
    for col in ["start_datetime", "end_datetime"]:
        annotations[col] = pd.to_datetime(annotations[col], utc=True)

    # Add collapsed class
    annotations["label_3class"] = annotations["annotation"].map(COLLAPSE_MAP)

    return annotations


def get_file_manifest(data_root: str, dataset_names: list[str]) -> pd.DataFrame:
    """Build a manifest of all WAV files with their durations."""
    records = []
    for ds in dataset_names:
        ds_dir = Path(data_root) / ds
        wav_files = sorted(ds_dir.glob("*.wav"))
        for wf in wav_files:
            info = sf.info(str(wf))
            records.append({
                "dataset": ds,
                "filename": wf.name,
                "filepath": str(wf),
                "duration_s": info.duration,
                "sample_rate": info.samplerate,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A training/eval segment with metadata."""
    filepath: str
    dataset: str
    filename: str
    start_sample: int
    end_sample: int
    annotations: list  # list of (start_sample_rel, end_sample_rel, class_idx)
    is_positive: bool


def build_training_segments(
    annotations: pd.DataFrame,
    file_manifest: pd.DataFrame,
    config: DataConfig,
) -> tuple[list[Segment], list[Segment]]:
    """
    Build positive segments (centered on annotations with random collars)
    and negative segments (from unannotated regions).
    """
    sr = config.sample_rate
    n_classes = 3 if config.use_3class else 7
    class_map = CLASS_TO_IDX_3 if config.use_3class else CLASS_TO_IDX_7
    label_col = "label_3class" if config.use_3class else "annotation"

    positive_segments = []
    negative_segments = []

    for _, file_row in file_manifest.iterrows():
        ds = file_row["dataset"]
        fname = file_row["filename"]
        fpath = file_row["filepath"]
        file_dur_samples = int(file_row["duration_s"] * sr)

        # Get annotations for this file
        file_annots = annotations[
            (annotations["dataset"] == ds) & (annotations["filename"] == fname)
        ]

        if len(file_annots) == 0:
            # Entire file is negative — extract fixed-length negative segments
            seg_len = int(config.eval_segment_s * sr)
            for start in range(0, file_dur_samples - seg_len, seg_len):
                negative_segments.append(Segment(
                    filepath=fpath, dataset=ds, filename=fname,
                    start_sample=start, end_sample=start + seg_len,
                    annotations=[], is_positive=False,
                ))
            continue

        # Parse file start time from filename (format: YYYY-MM-DDTHH-MM-SS_000.wav)
        try:
            base = fname.replace(".wav", "").split("_")[0]
            file_start_dt = datetime.strptime(base, "%Y-%m-%dT%H-%M-%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        # Build positive segments per annotation
        for _, ann in file_annots.iterrows():
            ann_start = (ann["start_datetime"] - file_start_dt).total_seconds()
            ann_end = (ann["end_datetime"] - file_start_dt).total_seconds()
            call_dur = ann_end - ann_start

            if call_dur < config.min_call_duration_s or call_dur > config.max_call_duration_s:
                continue

            # Random collar
            collar_before = random.uniform(config.collar_min_s, config.collar_max_s)
            collar_after = random.uniform(config.collar_min_s, config.collar_max_s)

            seg_start_s = max(0, ann_start - collar_before)
            seg_end_s = min(file_row["duration_s"], ann_end + collar_after)

            seg_start_samp = int(seg_start_s * sr)
            seg_end_samp = int(seg_end_s * sr)

            # Find ALL annotations overlapping with this segment
            seg_annots = []
            for _, a2 in file_annots.iterrows():
                a2_start = (a2["start_datetime"] - file_start_dt).total_seconds()
                a2_end = (a2["end_datetime"] - file_start_dt).total_seconds()
                if a2_end > seg_start_s and a2_start < seg_end_s:
                    rel_start = max(0, int((a2_start - seg_start_s) * sr))
                    rel_end = min(seg_end_samp - seg_start_samp, int((a2_end - seg_start_s) * sr))
                    cls_idx = class_map.get(a2[label_col], -1)
                    if cls_idx >= 0:
                        seg_annots.append((rel_start, rel_end, cls_idx))

            positive_segments.append(Segment(
                filepath=fpath, dataset=ds, filename=fname,
                start_sample=seg_start_samp, end_sample=seg_end_samp,
                annotations=seg_annots, is_positive=True,
            ))

        # Also extract negative regions from this file
        annotated_intervals = []
        for _, ann in file_annots.iterrows():
            s = (ann["start_datetime"] - file_start_dt).total_seconds()
            e = (ann["end_datetime"] - file_start_dt).total_seconds()
            annotated_intervals.append((s, e))
        annotated_intervals.sort()

        # Merge overlapping intervals
        merged = [annotated_intervals[0]]
        for s, e in annotated_intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Extract gaps as negative segments
        seg_len_s = config.eval_segment_s
        prev_end = 0.0
        for s, e in merged:
            gap = s - prev_end
            if gap >= seg_len_s:
                start_s = prev_end
                while start_s + seg_len_s <= s:
                    negative_segments.append(Segment(
                        filepath=fpath, dataset=ds, filename=fname,
                        start_sample=int(start_s * sr),
                        end_sample=int((start_s + seg_len_s) * sr),
                        annotations=[], is_positive=False,
                    ))
                    start_s += seg_len_s
            prev_end = e

    return positive_segments, negative_segments


def build_eval_segments(
    file_manifest: pd.DataFrame, config: DataConfig
) -> list[Segment]:
    """Build fixed-length overlapping segments for evaluation."""
    sr = config.sample_rate
    seg_len = int(config.eval_segment_s * sr)
    hop = int((config.eval_segment_s - config.eval_overlap_s) * sr)
    segments = []

    for _, row in file_manifest.iterrows():
        file_dur_samples = int(row["duration_s"] * sr)
        start = 0
        while start + seg_len <= file_dur_samples:
            segments.append(Segment(
                filepath=row["filepath"], dataset=row["dataset"],
                filename=row["filename"],
                start_sample=start, end_sample=start + seg_len,
                annotations=[], is_positive=False,
            ))
            start += hop
        # Handle last partial segment
        if start < file_dur_samples:
            segments.append(Segment(
                filepath=row["filepath"], dataset=row["dataset"],
                filename=row["filename"],
                start_sample=file_dur_samples - seg_len,
                end_sample=file_dur_samples,
                annotations=[], is_positive=False,
            ))
    return segments


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class WhaleCallDataset(Dataset):
    """
    Returns (audio_tensor, target_tensor, padding_mask, metadata_dict).

    target_tensor: (T_frames, n_classes) binary labels at 20 ms resolution
    """
    def __init__(
        self,
        segments: list[Segment],
        config: DataConfig,
        is_train: bool = True,
    ):
        self.segments = segments
        self.config = config
        self.is_train = is_train
        self.n_classes = 3 if config.use_3class else 7
        self.frame_stride_samples = int(config.frame_stride_s * config.sample_rate)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]

        # Load audio
        audio, sr = sf.read(
            seg.filepath,
            start=seg.start_sample,
            stop=seg.end_sample,
            dtype="float32",
        )
        assert sr == self.config.sample_rate, f"Expected {self.config.sample_rate} Hz, got {sr}"

        # Mean subtraction (per-segment, as in Whale-VAD)
        audio = audio - audio.mean()

        n_samples = len(audio)
        n_frames = n_samples // self.frame_stride_samples

        # Build target: (n_frames, n_classes)
        target = np.zeros((n_frames, self.n_classes), dtype=np.float32)
        for rel_start, rel_end, cls_idx in seg.annotations:
            frame_start = rel_start // self.frame_stride_samples
            frame_end = rel_end // self.frame_stride_samples
            frame_start = max(0, min(frame_start, n_frames - 1))
            frame_end = max(0, min(frame_end, n_frames))
            target[frame_start:frame_end, cls_idx] = 1.0

        metadata = {
            "dataset": seg.dataset,
            "filename": seg.filename,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
        }

        return (
            torch.from_numpy(audio),
            torch.from_numpy(target),
            metadata,
        )


# ---------------------------------------------------------------------------
# Collation & negative undersampling
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Zero-pad variable-length segments to batch max and create padding masks.
    """
    audios, targets, metas = zip(*batch)

    # Pad audio
    max_audio_len = max(a.size(0) for a in audios)
    padded_audio = torch.zeros(len(audios), max_audio_len)
    for i, a in enumerate(audios):
        padded_audio[i, : a.size(0)] = a

    # Pad targets — need to know frame counts
    max_frames = max(t.size(0) for t in targets)
    n_classes = targets[0].size(1)
    padded_targets = torch.zeros(len(targets), max_frames, n_classes)
    padding_mask = torch.zeros(len(targets), max_frames, dtype=torch.bool)
    for i, t in enumerate(targets):
        padded_targets[i, : t.size(0)] = t
        padding_mask[i, : t.size(0)] = True

    return padded_audio, padded_targets, padding_mask, list(metas)


class NegativeUndersamplingDataset(Dataset):
    """
    Wraps positive + negative segments with epoch-level resampling of negatives.
    Call resample_negatives() at the start of each epoch.
    """
    def __init__(
        self,
        positive_segments: list[Segment],
        negative_segments: list[Segment],
        config: DataConfig,
    ):
        self.positive_segments = positive_segments
        self.all_negatives = negative_segments
        self.config = config
        self.inner_dataset: WhaleCallDataset | None = None
        self.resample_negatives()

    def resample_negatives(self):
        """Sample a new subset of negatives for this epoch."""
        n_neg = int(len(self.positive_segments) * self.config.neg_ratio)
        n_neg = min(n_neg, len(self.all_negatives))
        sampled_neg = random.sample(self.all_negatives, n_neg)
        all_segments = self.positive_segments + sampled_neg
        random.shuffle(all_segments)
        self.inner_dataset = WhaleCallDataset(
            all_segments, self.config, is_train=True
        )

    def __len__(self) -> int:
        return len(self.inner_dataset)

    def __getitem__(self, idx: int):
        return self.inner_dataset[idx]


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: DataConfig, batch_size: int = 16, num_workers: int = 4,
) -> tuple[NegativeUndersamplingDataset, DataLoader, DataLoader]:
    """
    Build training dataset (with undersampling) and validation dataloader.
    Returns (train_dataset, train_loader, val_loader).
    """
    # Load annotations
    all_datasets = config.train_datasets + config.val_datasets
    annotations = load_annotations(config.data_root, all_datasets)
    train_annots = annotations[annotations["dataset"].isin(config.train_datasets)]

    # File manifests
    train_manifest = get_file_manifest(config.data_root, config.train_datasets)
    val_manifest = get_file_manifest(config.data_root, config.val_datasets)

    # Training segments
    pos_segs, neg_segs = build_training_segments(train_annots, train_manifest, config)
    print(f"Training: {len(pos_segs)} positive, {len(neg_segs)} negative segments")

    train_ds = NegativeUndersamplingDataset(pos_segs, neg_segs, config)

    # Validation segments (fixed windows, no undersampling)
    val_segs = build_eval_segments(val_manifest, config)
    print(f"Validation: {len(val_segs)} segments")
    val_ds = WhaleCallDataset(val_segs, config, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_ds, train_loader, val_loader
