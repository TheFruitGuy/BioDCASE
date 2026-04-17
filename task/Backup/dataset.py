"""
Dataset and data loading for BioDCASE 2026 Task 2.

CRITICAL FIX from previous version:
  build_eval_segments() creates segments with NO annotations → val F1 always 0.
  Added build_val_segments_with_annotations() for proper validation.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def load_annotations(dataset_names: list[str]) -> pd.DataFrame:
    """
    Load and merge annotation CSVs for the given site-year datasets.
    Searches in both train/annotations/ and validation/annotations/.
    """
    frames = []
    search_dirs = [
        cfg.DATA_ROOT / "train" / "annotations",
        cfg.DATA_ROOT / "validation" / "annotations",
    ]

    for ds in dataset_names:
        found = False
        for s_dir in search_dirs:
            csv_path = s_dir / f"{ds}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if "dataset" not in df.columns:
                    df["dataset"] = ds
                frames.append(df)
                found = True
                break
        if not found:
            print(f"Warning: no annotation file found for {ds}")

    if not frames:
        raise FileNotFoundError(
            f"No annotation files found in train/ or validation/ under {cfg.DATA_ROOT}"
        )

    annotations = pd.concat(frames, ignore_index=True)
    for col in ["start_datetime", "end_datetime"]:
        annotations[col] = pd.to_datetime(annotations[col], format="mixed", utc=True)
    annotations["label_3class"] = annotations["annotation"].map(cfg.COLLAPSE_MAP)
    return annotations


def get_file_manifest(dataset_names: list[str]) -> pd.DataFrame:
    """Build a manifest of all WAV files with their durations."""
    records = []
    search_dirs = [
        cfg.DATA_ROOT / "train" / "audio",
        cfg.DATA_ROOT / "validation" / "audio",
    ]

    for ds in dataset_names:
        found = False
        for s_dir in search_dirs:
            ds_dir = s_dir / ds
            if ds_dir.exists():
                found = True
                for wf in sorted(ds_dir.glob("*.wav")):
                    info = sf.info(str(wf))
                    records.append({
                        "dataset": ds,
                        "filename": wf.name,
                        "filepath": str(wf),
                        "duration_s": info.duration,
                        "sample_rate": info.samplerate,
                    })
                break
        if not found:
            print(f"Warning: Audio directory not found for {ds}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    filepath: str
    dataset: str
    filename: str
    start_sample: int
    end_sample: int
    annotations: list       # list of (start_sample_rel, end_sample_rel, class_idx)
    is_positive: bool


def _parse_file_start_dt(filename: str) -> datetime | None:
    try:
        base = filename.replace(".wav", "").split("_")[0]
        return datetime.strptime(base, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Training segments (collared around annotations + negatives)
# ---------------------------------------------------------------------------

def build_training_segments(
    annotations: pd.DataFrame,
    file_manifest: pd.DataFrame,
) -> tuple[list[Segment], list[Segment]]:
    """Build positive segments (with random collars) and negative segments."""
    sr = cfg.SAMPLE_RATE
    class_map = cfg.class_to_idx()
    label_col = "label_3class" if cfg.USE_3CLASS else "annotation"

    positive_segments = []
    negative_segments = []

    for _, file_row in file_manifest.iterrows():
        ds = file_row["dataset"]
        fname = file_row["filename"]
        fpath = file_row["filepath"]
        file_dur_samples = int(file_row["duration_s"] * sr)

        file_annots = annotations[
            (annotations["dataset"] == ds) & (annotations["filename"] == fname)
        ]

        # --- fully negative file ---
        if len(file_annots) == 0:
            seg_len = int(cfg.EVAL_SEGMENT_S * sr)
            for start in range(0, file_dur_samples - seg_len, seg_len):
                negative_segments.append(Segment(
                    filepath=fpath, dataset=ds, filename=fname,
                    start_sample=start, end_sample=start + seg_len,
                    annotations=[], is_positive=False,
                ))
            continue

        file_start_dt = _parse_file_start_dt(fname)
        if file_start_dt is None:
            continue

        # --- positive segments per annotation ---
        for _, ann in file_annots.iterrows():
            ann_start = (ann["start_datetime"] - file_start_dt).total_seconds()
            ann_end = (ann["end_datetime"] - file_start_dt).total_seconds()
            call_dur = ann_end - ann_start
            if call_dur < cfg.MIN_CALL_DURATION_S or call_dur > cfg.MAX_CALL_DURATION_S:
                continue

            collar_before = random.uniform(cfg.COLLAR_MIN_S, cfg.COLLAR_MAX_S)
            collar_after = random.uniform(cfg.COLLAR_MIN_S, cfg.COLLAR_MAX_S)
            seg_start_s = max(0, ann_start - collar_before)
            seg_end_s = min(file_row["duration_s"], ann_end + collar_after)
            seg_start_samp = int(seg_start_s * sr)
            seg_end_samp = int(seg_end_s * sr)

            # collect ALL annotations overlapping this segment
            seg_annots = []
            for _, a2 in file_annots.iterrows():
                a2_start = (a2["start_datetime"] - file_start_dt).total_seconds()
                a2_end = (a2["end_datetime"] - file_start_dt).total_seconds()
                if a2_end > seg_start_s and a2_start < seg_end_s:
                    rel_start = max(0, int((a2_start - seg_start_s) * sr))
                    rel_end = min(seg_end_samp - seg_start_samp,
                                  int((a2_end - seg_start_s) * sr))
                    cls_idx = class_map.get(a2[label_col], -1)
                    if cls_idx >= 0:
                        seg_annots.append((rel_start, rel_end, cls_idx))

            positive_segments.append(Segment(
                filepath=fpath, dataset=ds, filename=fname,
                start_sample=seg_start_samp, end_sample=seg_end_samp,
                annotations=seg_annots, is_positive=True,
            ))

        # --- negative segments from gaps ---
        intervals = sorted(
            ((ann["start_datetime"] - file_start_dt).total_seconds(),
             (ann["end_datetime"] - file_start_dt).total_seconds())
            for _, ann in file_annots.iterrows()
        )
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        seg_len_s = cfg.EVAL_SEGMENT_S
        prev_end = 0.0
        for s, e in merged:
            if s - prev_end >= seg_len_s:
                pos = prev_end
                while pos + seg_len_s <= s:
                    negative_segments.append(Segment(
                        filepath=fpath, dataset=ds, filename=fname,
                        start_sample=int(pos * sr),
                        end_sample=int((pos + seg_len_s) * sr),
                        annotations=[], is_positive=False,
                    ))
                    pos += seg_len_s
            prev_end = e

    return positive_segments, negative_segments


# ---------------------------------------------------------------------------
# Validation segments — fixed windows WITH annotations
# ---------------------------------------------------------------------------

def build_val_segments_with_annotations(
    annotations: pd.DataFrame,
    file_manifest: pd.DataFrame,
) -> list[Segment]:
    """
    Build fixed-length overlapping segments for validation,
    WITH annotations loaded so we can compute proper F1.

    CRITICAL FIX: the old build_eval_segments() had annotations=[]
    which meant all val targets were zero → F1 always 0.
    """
    sr = cfg.SAMPLE_RATE
    seg_len = int(cfg.EVAL_SEGMENT_S * sr)
    hop = int((cfg.EVAL_SEGMENT_S - cfg.EVAL_OVERLAP_S) * sr)
    class_map = cfg.class_to_idx()
    label_col = "label_3class" if cfg.USE_3CLASS else "annotation"
    segments = []

    for _, row in file_manifest.iterrows():
        ds = row["dataset"]
        fname = row["filename"]
        fpath = row["filepath"]
        file_dur_samples = int(row["duration_s"] * sr)

        file_start_dt = _parse_file_start_dt(fname)
        if file_start_dt is None:
            continue

        # Get all annotations for this file
        file_annots = annotations[
            (annotations["dataset"] == ds) & (annotations["filename"] == fname)
        ]

        # Pre-compute annotation times relative to file start (in seconds)
        ann_times = []
        for _, ann in file_annots.iterrows():
            a_start = (ann["start_datetime"] - file_start_dt).total_seconds()
            a_end = (ann["end_datetime"] - file_start_dt).total_seconds()
            cls_idx = class_map.get(ann[label_col], -1)
            if cls_idx >= 0:
                ann_times.append((a_start, a_end, cls_idx))

        # Build fixed segments
        start = 0
        while start + seg_len <= file_dur_samples:
            seg_start_s = start / sr
            seg_end_s = (start + seg_len) / sr

            # Find overlapping annotations
            seg_annots = []
            for a_start, a_end, cls_idx in ann_times:
                if a_end > seg_start_s and a_start < seg_end_s:
                    rel_start = max(0, int((a_start - seg_start_s) * sr))
                    rel_end = min(seg_len, int((a_end - seg_start_s) * sr))
                    seg_annots.append((rel_start, rel_end, cls_idx))

            segments.append(Segment(
                filepath=fpath, dataset=ds, filename=fname,
                start_sample=start, end_sample=start + seg_len,
                annotations=seg_annots,
                is_positive=len(seg_annots) > 0,
            ))
            start += hop

        # Last partial
        if start < file_dur_samples and file_dur_samples >= seg_len:
            last_start = file_dur_samples - seg_len
            seg_start_s = last_start / sr
            seg_end_s = file_dur_samples / sr

            seg_annots = []
            for a_start, a_end, cls_idx in ann_times:
                if a_end > seg_start_s and a_start < seg_end_s:
                    rel_start = max(0, int((a_start - seg_start_s) * sr))
                    rel_end = min(seg_len, int((a_end - seg_start_s) * sr))
                    seg_annots.append((rel_start, rel_end, cls_idx))

            segments.append(Segment(
                filepath=fpath, dataset=ds, filename=fname,
                start_sample=last_start, end_sample=file_dur_samples,
                annotations=seg_annots,
                is_positive=len(seg_annots) > 0,
            ))

    return segments


def build_eval_segments(file_manifest: pd.DataFrame) -> list[Segment]:
    """Build fixed-length overlapping segments for INFERENCE (no annotations)."""
    sr = cfg.SAMPLE_RATE
    seg_len = int(cfg.EVAL_SEGMENT_S * sr)
    hop = int((cfg.EVAL_SEGMENT_S - cfg.EVAL_OVERLAP_S) * sr)
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
        if start < file_dur_samples and file_dur_samples >= seg_len:
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
    """Returns (audio, target, metadata).  target: (T_frames, C) binary."""

    def __init__(self, segments: list[Segment], is_train: bool = True):
        self.segments = segments
        self.is_train = is_train
        self.n_classes = cfg.n_classes()
        self.frame_stride_samples = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        audio, sr = sf.read(seg.filepath, start=seg.start_sample,
                            stop=seg.end_sample, dtype="float32")
        assert sr == cfg.SAMPLE_RATE
        audio = audio - audio.mean()

        n_frames = len(audio) // self.frame_stride_samples
        target = np.zeros((n_frames, self.n_classes), dtype=np.float32)
        for rel_start, rel_end, cls_idx in seg.annotations:
            fs = max(0, min(rel_start // self.frame_stride_samples, n_frames - 1))
            fe = max(0, min(rel_end // self.frame_stride_samples, n_frames))
            target[fs:fe, cls_idx] = 1.0

        meta = {"dataset": seg.dataset, "filename": seg.filename,
                "start_sample": seg.start_sample, "end_sample": seg.end_sample}
        return torch.from_numpy(audio), torch.from_numpy(target), meta


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    audios, targets, metas = zip(*batch)

    max_audio = max(a.size(0) for a in audios)
    padded_audio = torch.zeros(len(audios), max_audio)
    for i, a in enumerate(audios):
        padded_audio[i, :a.size(0)] = a

    max_frames = max(t.size(0) for t in targets)
    nc = targets[0].size(1)
    padded_targets = torch.zeros(len(targets), max_frames, nc)
    padding_mask = torch.zeros(len(targets), max_frames, dtype=torch.bool)
    for i, t in enumerate(targets):
        padded_targets[i, :t.size(0)] = t
        padding_mask[i, :t.size(0)] = True

    return padded_audio, padded_targets, padding_mask, list(metas)


# ---------------------------------------------------------------------------
# Negative undersampling wrapper
# ---------------------------------------------------------------------------

class NegativeUndersamplingDataset(Dataset):
    def __init__(self, positive: list[Segment], negative: list[Segment]):
        self.positive = positive
        self.all_negatives = negative
        self.inner: WhaleCallDataset | None = None
        self.resample_negatives()

    def resample_negatives(self):
        n_neg = int(len(self.positive) * cfg.NEG_RATIO)
        n_neg = min(n_neg, len(self.all_negatives))
        sampled = random.sample(self.all_negatives, n_neg)
        combined = self.positive + sampled
        random.shuffle(combined)
        self.inner = WhaleCallDataset(combined, is_train=True)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]


# ---------------------------------------------------------------------------
# Build loaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> tuple[NegativeUndersamplingDataset, DataLoader, DataLoader]:
    """Returns (train_dataset, train_loader, val_loader)."""
    all_datasets = cfg.TRAIN_DATASETS + cfg.VAL_DATASETS
    annotations = load_annotations(all_datasets)
    train_annots = annotations[annotations["dataset"].isin(cfg.TRAIN_DATASETS)]
    val_annots = annotations[annotations["dataset"].isin(cfg.VAL_DATASETS)]

    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)

    # Training: collared segments + negatives
    pos, neg = build_training_segments(train_annots, train_manifest)
    print(f"Training: {len(pos)} positive, {len(neg)} negative segments")
    train_ds = NegativeUndersamplingDataset(pos, neg)

    # Validation: fixed windows WITH annotations (critical fix!)
    val_segs = build_val_segments_with_annotations(val_annots, val_manifest)
    n_pos_val = sum(1 for s in val_segs if s.is_positive)
    print(f"Validation: {len(val_segs)} segments ({n_pos_val} with calls, "
          f"{len(val_segs) - n_pos_val} empty)")
    val_ds = WhaleCallDataset(val_segs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=True)
    return train_ds, train_loader, val_loader
