"""
Triton — Dataset loading and segmentation
=========================================

Handles all data-side responsibilities:

    1. Locating audio files and annotations on disk (BioDCASE 2026 layout)
    2. Parsing per-dataset CSV annotation files, including datetime-to-file
       matching for CSVs that lack an explicit ``filename`` column
    3. Constructing training segments: each positive annotation + random
       collar, plus randomly sampled negative segments
    4. Constructing fixed-length, overlapping validation segments for
       deterministic evaluation
    5. Frame-level target tensor construction at 20 ms resolution
    6. PyTorch Dataset and DataLoader wrappers with batched collation

Directory layout expected
-------------------------
::

    DATA_ROOT/
      train/
        annotations/{dataset_name}.csv
        audio/{dataset_name}/*.wav
      validation/
        annotations/{dataset_name}.csv
        audio/{dataset_name}/*.wav

Annotation CSV format
---------------------
Each CSV must contain at minimum ``start_datetime``, ``end_datetime``,
and ``annotation`` columns. An explicit ``filename`` column is optional
— it's inferred from ``start_datetime`` against per-file ranges if absent.
Filenames are expected to encode their start time, e.g.
``2014-06-29T23-00-00_000.wav``.

Caching
-------
Both ``get_file_manifest`` and ``load_annotations`` cache results to disk
under ``./.cache/`` keyed on the sorted dataset list. After modifying
audio files or CSVs, call ``clear_cache()`` (or ``rm -rf .cache/``).
"""

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


# ======================================================================
# Disk cache
# ======================================================================
# Parquet-based; survives pandas upgrades (unlike pickle, which can break
# across versions when DataFrame internals change).

_CACHE_DIR = Path("./.cache")
_CACHE_EXT = ".parquet"


def _cache_path(name: str, datasets: list[str]) -> Path:
    """Cache file path for a (name, datasets) pair."""
    _CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.md5(",".join(sorted(datasets)).encode()).hexdigest()[:8]
    return _CACHE_DIR / f"{name}_{key}{_CACHE_EXT}"


def _cache_load(path: Path) -> pd.DataFrame | None:
    """Load a cached DataFrame, return None on miss / corruption."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  Cache miss (corrupt): {path.name} ({e}); rebuilding")
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _cache_save(path: Path, df: pd.DataFrame) -> None:
    """Atomically write a DataFrame to parquet (Ctrl-C-safe)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def clear_cache() -> None:
    """Wipe cached manifests and annotations. Call after data changes."""
    if _CACHE_DIR.exists():
        for pattern in ("*.parquet", "*.pkl"):
            for f in _CACHE_DIR.glob(pattern):
                f.unlink()
        print(f"Cleared cache directory: {_CACHE_DIR}")


# ======================================================================
# Path resolution
# ======================================================================

def _split_for_dataset(ds: str) -> str:
    """Return "train" or "validation" for a dataset name."""
    if ds in cfg.TRAIN_DATASETS:
        return "train"
    if ds in cfg.VAL_DATASETS:
        return "validation"
    for split in ("train", "validation"):
        if (cfg.DATA_ROOT / split / "audio" / ds).exists():
            return split
    raise FileNotFoundError(f"Cannot find split for dataset '{ds}'")


def _parse_file_start_dt(filename: str):
    """
    Parse ATBFL ``YYYY-MM-DDTHH-MM-SS[_msec].wav`` → UTC datetime, or
    None on parse failure.
    """
    stem = Path(filename).stem.split("_")[0]
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ======================================================================
# File manifest
# ======================================================================

def _build_file_manifest_uncached(datasets: list[str]) -> pd.DataFrame:
    """Slow path for ``get_file_manifest``: scans disk and reads headers."""
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


def get_file_manifest(datasets: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame of all audio files in the requested datasets.

    Cached. Columns: dataset, filename, path, duration_s, start_dt, end_dt.
    """
    cp = _cache_path("manifest", datasets)
    cached = _cache_load(cp)
    if cached is not None:
        return cached

    print(f"  Building file manifest for {len(datasets)} dataset(s)...")
    df = _build_file_manifest_uncached(datasets)
    _cache_save(cp, df)
    print(f"  Manifest cached to {cp}")
    return df


# ======================================================================
# Annotation loading
# ======================================================================

def _infer_filenames_vectorized(
    df: pd.DataFrame, ds_files: pd.DataFrame
) -> pd.Series:
    """
    Vectorized filename inference via ``pd.merge_asof``.

    For each annotation, finds the audio file whose start time is the
    largest value not exceeding the annotation's start time, then
    verifies the annotation also begins before the file ends.
    """
    if df.empty or ds_files.empty:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    df_sorted = df.sort_values("start_datetime").copy()
    files_sorted = (
        ds_files[["start_dt", "end_dt", "filename"]]
        .sort_values("start_dt")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        df_sorted, files_sorted,
        left_on="start_datetime", right_on="start_dt",
        direction="backward",
    )

    # Reject annotations that fall in a gap between recordings.
    out_of_range = merged["start_datetime"] >= merged["end_dt"]
    merged.loc[out_of_range, "filename"] = pd.NA

    merged.index = df_sorted.index
    return merged.sort_index()["filename"]


def _load_annotations_uncached(
    datasets: list[str], manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Slow path for ``load_annotations``."""
    all_rows = []
    if manifest is None:
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
        df["end_datetime"] = pd.to_datetime(df["end_datetime"], utc=True)

        if "filename" not in df.columns:
            ds_files = manifest[manifest["dataset"] == ds]
            df["filename"] = _infer_filenames_vectorized(df, ds_files)
            n_before = len(df)
            df = df[df["filename"].notna()].reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(f"  {ds}: dropped {n_dropped}/{n_before} annotations "
                      f"with no matching file")

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    ann = pd.concat(all_rows, ignore_index=True)
    ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
    return ann


def load_annotations(
    datasets: list[str], manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate annotations for the requested datasets.

    Cached. Columns: dataset, filename, start_datetime, end_datetime,
    annotation, label_3class.
    """
    cp = _cache_path("annotations", datasets)
    cached = _cache_load(cp)
    if cached is not None:
        return cached

    print(f"  Loading annotations for {len(datasets)} dataset(s)...")
    df = _load_annotations_uncached(datasets, manifest=manifest)
    _cache_save(cp, df)
    print(f"  Annotations cached to {cp}")
    return df


# ======================================================================
# Segment dataclass
# ======================================================================

@dataclass
class Segment:
    """Lightweight description of an audio segment to be loaded on demand."""

    dataset: str
    filename: str
    path: str
    start_sample: int
    end_sample: int
    file_start_dt: datetime
    annotations: list[dict]
    is_positive: bool


# ======================================================================
# Per-file annotation index helper
# ======================================================================

def _build_annotations_by_file(
    annotations: pd.DataFrame, manifest: pd.DataFrame,
) -> dict:
    """
    Group annotations by ``(dataset, filename)`` for O(1) lookup with
    file-relative ``start_s, end_s`` instead of absolute datetimes.
    """
    if annotations.empty or manifest.empty:
        return {}

    file_starts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }

    out: dict = {}
    for _, a in annotations.iterrows():
        key = (a["dataset"], a["filename"])
        fsd = file_starts.get(key)
        # pd.isna catches both None and pandas NaT (which the parquet
        # cache materialises None datetimes as).
        if fsd is None or pd.isna(fsd):
            continue
        out.setdefault(key, []).append({
            "start_s": (a["start_datetime"] - fsd).total_seconds(),
            "end_s": (a["end_datetime"] - fsd).total_seconds(),
            "label": a["annotation"],
            "label_3class": a["label_3class"],
        })
    return out


# ======================================================================
# Training segment construction
# ======================================================================

def build_positive_segments(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    collar_min_s: float = cfg.COLLAR_MIN_S,
    collar_max_s: float = cfg.COLLAR_MAX_S,
) -> list[Segment]:
    """
    One training segment per positive annotation. Each segment spans the
    call plus random silent collars on either side; all annotations that
    intersect the resulting segment are attached.
    """
    segments: list[Segment] = []
    if manifest.empty or annotations.empty:
        return segments

    manifest_idx = manifest.set_index(["dataset", "filename"])
    ann_by_file = _build_annotations_by_file(annotations, manifest)

    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        if key not in manifest_idx.index:
            continue
        file_row = manifest_idx.loc[key]
        file_start_dt = file_row["start_dt"]
        if file_start_dt is None or pd.isna(file_start_dt):
            continue

        call_start_s = (row["start_datetime"] - file_start_dt).total_seconds()
        call_end_s = (row["end_datetime"] - file_start_dt).total_seconds()

        # Sanity filters.
        if call_end_s <= call_start_s or call_end_s <= 0:
            continue
        if call_end_s - call_start_s > cfg.MAX_CALL_DURATION_S:
            continue
        if call_end_s - call_start_s < cfg.MIN_CALL_DURATION_S:
            continue

        pre = random.uniform(collar_min_s, collar_max_s)
        post = random.uniform(collar_min_s, collar_max_s)
        seg_start_s = max(0.0, call_start_s - pre)
        seg_end_s = min(file_row["duration_s"], call_end_s + post)

        file_anns = ann_by_file.get(key, [])
        inter_anns = [
            a for a in file_anns
            if a["end_s"] > seg_start_s and a["start_s"] < seg_end_s
        ]

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
    """
    Sample ``n_segments`` random call-free windows. Rejection sampler
    bails out after 20× ``n_segments`` retries; returns fewer if the
    sampler can't find enough non-overlapping windows.
    """
    segments: list[Segment] = []
    if manifest.empty:
        return segments

    ann_by_file = _build_annotations_by_file(annotations, manifest)
    call_intervals: dict = {
        key: [(a["start_s"], a["end_s"]) for a in anns]
        for key, anns in ann_by_file.items()
    }

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
        seg_end_s = seg_start_s + seg_len

        intervals = call_intervals.get(key, [])
        if any(seg_end_s > cs and seg_start_s < ce for cs, ce in intervals):
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


# ======================================================================
# Validation segment construction
# ======================================================================

def build_val_segments(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    segment_s: float = cfg.EVAL_SEGMENT_S,
    overlap_s: float = cfg.EVAL_OVERLAP_S,
) -> list[Segment]:
    """
    Tile each file with fixed-length overlapping windows. The overlap
    margin allows overlapping predictions to be averaged during the
    stitching step, smoothing segment boundaries.
    """
    segments: list[Segment] = []
    if manifest.empty:
        return segments

    step_s = segment_s - overlap_s
    ann_by_file = _build_annotations_by_file(annotations, manifest)

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


# ======================================================================
# PyTorch Dataset and collation
# ======================================================================

class TritonDataset(Dataset):
    """
    Map-style dataset yielding ``(audio, targets, mask, meta)`` per segment.

    Audio is read lazily from disk on every ``__getitem__`` call. For
    the BioDCASE corpus the full data fits in OS page cache, so this is
    fast in practice.
    """

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
            seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32",
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
            local_end_s = min(n_samples / cfg.SAMPLE_RATE, a["end_s"] - seg_start_s)
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
    """
    Pad variable-length segments to the batch maximum and assemble a
    boolean validity mask for the padded frames.
    """
    audios, targets, masks, metas = zip(*batch)
    max_samp = max(a.size(0) for a in audios)
    max_frames = max(t.size(0) for t in targets)
    n_classes = targets[0].size(1)
    B = len(audios)

    audio_pad = torch.zeros(B, max_samp)
    target_pad = torch.zeros(B, max_frames, n_classes)
    mask_pad = torch.zeros(B, max_frames, dtype=torch.bool)

    for i in range(B):
        audio_pad[i, :audios[i].size(0)] = audios[i]
        target_pad[i, :targets[i].size(0)] = targets[i]
        mask_pad[i, :masks[i].size(0)] = masks[i]

    return audio_pad, target_pad, mask_pad, list(metas)


# ======================================================================
# High-level helpers
# ======================================================================

class TrainingDatasetWithResample(TritonDataset):
    """
    Training dataset with epoch-level negative resampling.

    Positive segments are fixed at construction time; negative segments
    are redrawn on each ``resample_negatives()`` call, implementing the
    stochastic undersampling protocol.
    """

    def __init__(self, positive_segments, manifest, annotations):
        self.positive_segments = positive_segments
        self.manifest = manifest
        self.annotations = annotations
        self.negative_segments = []
        self.resample_negatives()
        super().__init__(self.positive_segments + self.negative_segments)

    def resample_negatives(self):
        """Draw a new set of negative segments for the next epoch."""
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
            # 30-60s overrides build_negative_segments' default of 5-30s.
            # The canonical seed-42 baseline (runs/whalevad_20260502_175547,
            # F1=0.474) was trained with this override. Dropping it lets
            # the function defaults take over and shortens the training
            # segment distribution to 5-30s — which causes the BiLSTM to
            # specialise to short contexts and degrade catastrophically on
            # validation when the eval window exceeds 30s. The collar +
            # positive durations span 1-30s; pairing those with 30-60s
            # negatives gives the model the full 1-60s context range that
            # the validation segments (and downstream inference) live in.
            min_dur_s=30.0, max_dur_s=60.0,
        )
        self.segments = self.positive_segments + self.negative_segments


def build_dataloaders():
    """
    Construct train and validation DataLoaders.

    Returns ``(train_ds, train_loader, val_loader)``. The training
    dataset retains its ``resample_negatives()`` method so the caller
    can refresh negatives between epochs.
    """
    print(f"Loading train datasets: {cfg.TRAIN_DATASETS}")
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    train_annotations = load_annotations(cfg.TRAIN_DATASETS, manifest=train_manifest)
    print(f"  Found {len(train_manifest)} audio files, "
          f"{len(train_annotations)} annotations")

    pos_segs = build_positive_segments(train_annotations, train_manifest)
    train_ds = TrainingDatasetWithResample(pos_segs, train_manifest, train_annotations)
    print(f"Training: {len(pos_segs)} positive + "
          f"{len(train_ds.negative_segments)} negative segments")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"Loading val datasets: {cfg.VAL_DATASETS}")
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_annotations = load_annotations(cfg.VAL_DATASETS, manifest=val_manifest)
    print(f"  Found {len(val_manifest)} audio files, "
          f"{len(val_annotations)} annotations")

    val_segs = build_val_segments(val_manifest, val_annotations)
    val_ds = TritonDataset(val_segs)
    print(f"Validation: {len(val_segs)} segments "
          f"({sum(s.is_positive for s in val_segs)} with calls)")

    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    return train_ds, train_loader, val_loader
