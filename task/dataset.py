"""
Dataset loading for the Acoustic Trends Blue Fin Library (ATBFL).

Directory layout (BioDCASE 2026):
    DATA_ROOT/
      train/
        annotations/{dataset}.csv
        audio/{dataset}/*.wav
      validation/
        annotations/{dataset}.csv
        audio/{dataset}/*.wav

Handles:
  - Reading per-dataset boundary annotation CSVs
  - Matching annotations to audio files via datetime overlap
    (CSVs may not have a 'filename' column)
  - Training segments: each positive annotation + random collar (Section 5.1)
  - Negative segments: random "no-call" windows, sampled fresh each epoch (5.5)
  - Validation segments: 30s windows with 2s overlap (Section 5.1)
  - Frame-level target construction at 20ms resolution
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


# ----------------------------------------------------------------------
# Path resolution — maps dataset name → split directory
# ----------------------------------------------------------------------

def _split_for_dataset(ds: str) -> str:
    """Return 'train' or 'validation' for a given dataset name."""
    if ds in cfg.TRAIN_DATASETS:
        return "train"
    if ds in cfg.VAL_DATASETS:
        return "validation"
    for split in ("train", "validation"):
        if (cfg.DATA_ROOT / split / "audio" / ds).exists():
            return split
    raise FileNotFoundError(f"Cannot find split for dataset '{ds}'")


# ----------------------------------------------------------------------
# Filename → datetime parsing
# ----------------------------------------------------------------------

def _parse_file_start_dt(filename: str):
    stem = Path(filename).stem
    stem = stem.split("_")[0]
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# File manifest
# ----------------------------------------------------------------------

def get_file_manifest(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        try:
            split = _split_for_dataset(ds)
        except FileNotFoundError:
            print(f"Warning: cannot locate {ds}")
            continue
        audio_dir = cfg.DATA_ROOT / split / "audio" / ds
        if not audio_dir.exists():
            print(f"Warning: audio directory missing for {ds}: {audio_dir}")
            continue
        for wav in sorted(audio_dir.glob("*.wav")):
            info = sf.info(str(wav))
            start_dt = _parse_file_start_dt(wav.name)
            end_dt = start_dt + timedelta(seconds=info.duration) if start_dt else None
            rows.append({
                "dataset": ds,
                "filename": wav.name,
                "path": str(wav),
                "duration_s": info.duration,
                "start_dt": start_dt,
                "end_dt": end_dt,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Annotation loading
# ----------------------------------------------------------------------

def load_annotations(datasets: list[str]) -> pd.DataFrame:
    """
    Load annotations. CSVs may not have a 'filename' column — in that case
    we match each annotation to a file by finding the audio file whose
    datetime range contains the annotation's start_datetime.
    """
    all_rows = []
    manifest = get_file_manifest(datasets)

    for ds in datasets:
        try:
            split = _split_for_dataset(ds)
        except FileNotFoundError:
            continue
        ann_path = cfg.DATA_ROOT / split / "annotations" / f"{ds}.csv"
        if not ann_path.exists():
            print(f"Warning: no annotations for {ds}: {ann_path}")
            continue

        df = pd.read_csv(ann_path)
        df["dataset"] = ds

        df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)
        df["end_datetime"]   = pd.to_datetime(df["end_datetime"], utc=True)

        # Match annotations to files via datetime overlap if 'filename' missing
        if "filename" not in df.columns:
            ds_files = manifest[manifest["dataset"] == ds].sort_values("start_dt").reset_index(drop=True)
            filenames = []
            for _, row in df.iterrows():
                ann_start = row["start_datetime"]
                match = ds_files[
                    (ds_files["start_dt"] <= ann_start) &
                    (ds_files["end_dt"]   >  ann_start)
                ]
                filenames.append(match.iloc[0]["filename"] if len(match) else None)
            df["filename"] = filenames
            n_before = len(df)
            df = df[df["filename"].notna()].reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(f"  {ds}: dropped {n_dropped}/{n_before} annotations with no matching file")

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    ann = pd.concat(all_rows, ignore_index=True)
    ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
    return ann


# ----------------------------------------------------------------------
# Segment definition
# ----------------------------------------------------------------------

@dataclass
class Segment:
    dataset: str
    filename: str
    path: str
    start_sample: int
    end_sample: int
    file_start_dt: datetime
    annotations: list[dict]
    is_positive: bool


# ----------------------------------------------------------------------
# Build training segments (Section 5.1)
# ----------------------------------------------------------------------

def build_positive_segments(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    collar_min_s: float = cfg.COLLAR_MIN_S,
    collar_max_s: float = cfg.COLLAR_MAX_S,
) -> list[Segment]:
    segments = []
    if manifest.empty or annotations.empty:
        return segments

    manifest_idx = manifest.set_index(["dataset", "filename"])

    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        if key not in manifest_idx.index:
            continue
        file_row = manifest_idx.loc[key]
        file_start_dt = file_row["start_dt"]
        if file_start_dt is None:
            continue

        call_start_s = (row["start_datetime"] - file_start_dt).total_seconds()
        call_end_s   = (row["end_datetime"]   - file_start_dt).total_seconds()

        if call_end_s <= call_start_s or call_end_s <= 0:
            continue
        if call_end_s - call_start_s > cfg.MAX_CALL_DURATION_S:
            continue
        if call_end_s - call_start_s < cfg.MIN_CALL_DURATION_S:
            continue

        pre  = random.uniform(collar_min_s, collar_max_s)
        post = random.uniform(collar_min_s, collar_max_s)
        seg_start_s = max(0.0, call_start_s - pre)
        seg_end_s   = min(file_row["duration_s"], call_end_s + post)

        file_anns = annotations[
            (annotations["dataset"] == row["dataset"]) &
            (annotations["filename"] == row["filename"])
        ]
        inter_anns = []
        for _, a in file_anns.iterrows():
            a_start_s = (a["start_datetime"] - file_start_dt).total_seconds()
            a_end_s   = (a["end_datetime"]   - file_start_dt).total_seconds()
            if a_end_s > seg_start_s and a_start_s < seg_end_s:
                inter_anns.append({
                    "start_s": a_start_s,
                    "end_s":   a_end_s,
                    "label":   a["annotation"],
                    "label_3class": a["label_3class"],
                })

        segments.append(Segment(
            dataset=row["dataset"],
            filename=row["filename"],
            path=file_row["path"],
            start_sample=int(seg_start_s * cfg.SAMPLE_RATE),
            end_sample=int(seg_end_s * cfg.SAMPLE_RATE),
            file_start_dt=file_start_dt,
            annotations=inter_anns,
            is_positive=True,
        ))

    return segments


def build_negative_segments(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    n_segments: int,
    min_dur_s: float = 5.0,
    max_dur_s: float = 30.0,
) -> list[Segment]:
    segments = []
    if manifest.empty:
        return segments

    call_intervals: dict[tuple, list[tuple[float, float]]] = {}
    if not annotations.empty:
        for _, a in annotations.iterrows():
            key = (a["dataset"], a["filename"])
            file_rows = manifest[(manifest["dataset"] == a["dataset"]) &
                                 (manifest["filename"] == a["filename"])]
            if file_rows.empty:
                continue
            fsd = file_rows.iloc[0]["start_dt"]
            if fsd is None:
                continue
            s = (a["start_datetime"] - fsd).total_seconds()
            e = (a["end_datetime"]   - fsd).total_seconds()
            call_intervals.setdefault(key, []).append((s, e))

    files = manifest.to_dict("records")
    tries, max_tries = 0, n_segments * 20

    while len(segments) < n_segments and tries < max_tries:
        tries += 1
        file_row = random.choice(files)
        key = (file_row["dataset"], file_row["filename"])
        dur = file_row["duration_s"]
        seg_len = random.uniform(min_dur_s, max_dur_s)
        if dur <= seg_len + 1.0:
            continue
        seg_start_s = random.uniform(0, dur - seg_len)
        seg_end_s   = seg_start_s + seg_len

        intervals = call_intervals.get(key, [])
        overlap = any(seg_end_s > cs and seg_start_s < ce for cs, ce in intervals)
        if overlap:
            continue

        segments.append(Segment(
            dataset=file_row["dataset"],
            filename=file_row["filename"],
            path=file_row["path"],
            start_sample=int(seg_start_s * cfg.SAMPLE_RATE),
            end_sample=int(seg_end_s * cfg.SAMPLE_RATE),
            file_start_dt=file_row["start_dt"],
            annotations=[],
            is_positive=False,
        ))

    return segments


# ----------------------------------------------------------------------
# Build validation segments
# ----------------------------------------------------------------------

def build_val_segments(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    segment_s: float = cfg.EVAL_SEGMENT_S,
    overlap_s: float = cfg.EVAL_OVERLAP_S,
) -> list[Segment]:
    segments = []
    if manifest.empty:
        return segments

    step_s = segment_s - overlap_s

    ann_by_file: dict[tuple, list[dict]] = {}
    if not annotations.empty:
        for _, a in annotations.iterrows():
            key = (a["dataset"], a["filename"])
            file_rows = manifest[(manifest["dataset"] == a["dataset"]) &
                                 (manifest["filename"] == a["filename"])]
            if file_rows.empty:
                continue
            fsd = file_rows.iloc[0]["start_dt"]
            if fsd is None:
                continue
            ann_by_file.setdefault(key, []).append({
                "start_s": (a["start_datetime"] - fsd).total_seconds(),
                "end_s":   (a["end_datetime"]   - fsd).total_seconds(),
                "label":   a["annotation"],
                "label_3class": a["label_3class"],
            })

    for _, f in manifest.iterrows():
        key = (f["dataset"], f["filename"])
        dur = f["duration_s"]
        fsd = f["start_dt"]
        file_anns = ann_by_file.get(key, [])

        t = 0.0
        while t + segment_s <= dur + 1e-6:
            inter = [a for a in file_anns
                     if a["end_s"] > t and a["start_s"] < t + segment_s]
            segments.append(Segment(
                dataset=f["dataset"],
                filename=f["filename"],
                path=f["path"],
                start_sample=int(t * cfg.SAMPLE_RATE),
                end_sample=int((t + segment_s) * cfg.SAMPLE_RATE),
                file_start_dt=fsd,
                annotations=inter,
                is_positive=len(inter) > 0,
            ))
            t += step_s

    return segments


# ----------------------------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------------------------

class WhaleDataset(Dataset):
    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self.stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)
        self.class_idx = cfg.class_to_idx()
        self.n_classes = cfg.n_classes()

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        n_samples = seg.end_sample - seg.start_sample

        audio, sr = sf.read(
            seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32"
        )
        assert sr == cfg.SAMPLE_RATE, f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}"
        audio = torch.from_numpy(audio)

        n_frames = n_samples // self.stride_samp
        targets = torch.zeros(n_frames, self.n_classes)

        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE
        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label not in self.class_idx:
                continue
            c = self.class_idx[label]
            local_start_s = max(0.0, a["start_s"] - seg_start_s)
            local_end_s   = min(n_samples / cfg.SAMPLE_RATE, a["end_s"] - seg_start_s)
            f0 = int(local_start_s / cfg.FRAME_STRIDE_S)
            f1 = int(local_end_s / cfg.FRAME_STRIDE_S)
            targets[f0:f1, c] = 1.0

        mask = torch.ones(n_frames, dtype=torch.bool)

        meta = {
            "dataset": seg.dataset,
            "filename": seg.filename,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
        }
        return audio, targets, mask, meta


def collate_fn(batch):
    audios, targets, masks, metas = zip(*batch)
    max_samp   = max(a.size(0) for a in audios)
    max_frames = max(t.size(0) for t in targets)
    n_classes  = targets[0].size(1)
    B = len(audios)

    audio_pad  = torch.zeros(B, max_samp)
    target_pad = torch.zeros(B, max_frames, n_classes)
    mask_pad   = torch.zeros(B, max_frames, dtype=torch.bool)

    for i in range(B):
        audio_pad[i, :audios[i].size(0)]    = audios[i]
        target_pad[i, :targets[i].size(0)]  = targets[i]
        mask_pad[i, :masks[i].size(0)]      = masks[i]

    return audio_pad, target_pad, mask_pad, list(metas)


# ----------------------------------------------------------------------
# High-level
# ----------------------------------------------------------------------

class TrainingDatasetWithResample(WhaleDataset):
    def __init__(self, positive_segments, manifest, annotations):
        self.positive_segments = positive_segments
        self.manifest = manifest
        self.annotations = annotations
        self.negative_segments = []
        self.resample_negatives()
        super().__init__(self.positive_segments + self.negative_segments)

    def resample_negatives(self):
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.segments = self.positive_segments + self.negative_segments


def build_dataloaders():
    print(f"Loading train datasets: {cfg.TRAIN_DATASETS}")
    train_annotations = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest    = get_file_manifest(cfg.TRAIN_DATASETS)
    print(f"  Found {len(train_manifest)} audio files, {len(train_annotations)} annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    train_ds = TrainingDatasetWithResample(pos_segs, train_manifest, train_annotations)
    print(f"Training: {len(pos_segs)} positive + "
          f"{len(train_ds.negative_segments)} negative segments")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"Loading val datasets: {cfg.VAL_DATASETS}")
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest    = get_file_manifest(cfg.VAL_DATASETS)
    print(f"  Found {len(val_manifest)} audio files, {len(val_annotations)} annotations")

    val_segs = build_val_segments(val_manifest, val_annotations)
    val_ds = WhaleDataset(val_segs)
    print(f"Validation: {len(val_segs)} segments "
          f"({sum(s.is_positive for s in val_segs)} with calls)")

    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    return train_ds, train_loader, val_loader