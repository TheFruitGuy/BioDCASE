"""
Final Pipeline - Dataset Loading and Segmentation
=================================================

All data responsibilities for the clean Whale-VAD training pipeline:

    1. Locating audio files and annotations on disk (BioDCASE layout).
    2. Parsing per-dataset annotation CSVs, inferring the owning file from
       ``start_datetime`` when no ``filename`` column is present.
    3. Building positive training segments (call + random collar), negative
       segments (random call-free windows), and fixed-length overlapping
       validation segments.
    4. Extending every training segment to a fixed length so the model sees
       the same context window at train and eval time.
    5. Frame-level (20 ms) target tensor construction.
    6. A PyTorch ``Dataset`` and a padding collator.

Directory layout
----------------
::

    DATA_ROOT/
      train/      annotations/{dataset}.csv   audio/{dataset}/*.wav
      validation/ annotations/{dataset}.csv   audio/{dataset}/*.wav

Caching
-------
``get_file_manifest`` and ``load_annotations`` cache their results as parquet
under ``./.cache/``, keyed on the sorted dataset list. Delete ``.cache/`` (or
call :func:`clear_cache`) after changing audio files or annotation CSVs.
"""

import hashlib
import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

import config_final as cfg


# ======================================================================
# Disk cache (parquet, keyed on the sorted dataset list)
# ======================================================================

_CACHE_DIR = Path("./.cache")
_CACHE_EXT = ".parquet"


def _cache_path(name: str, datasets: list[str]) -> Path:
    """Return the cache file path for a (name, datasets) pair."""
    _CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.md5(",".join(sorted(datasets)).encode()).hexdigest()[:8]
    return _CACHE_DIR / f"{name}_{key}{_CACHE_EXT}"


def _cache_load(path: Path) -> pd.DataFrame | None:
    """Load a cached DataFrame, or return None on miss / corruption."""
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
    """Atomically write a DataFrame to parquet (write-tmp-then-rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def clear_cache() -> None:
    """Remove all cached manifests and annotations."""
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob(f"*{_CACHE_EXT}"):
            f.unlink()
        print(f"Cleared cache directory: {_CACHE_DIR}")


# ======================================================================
# Path resolution
# ======================================================================

def _split_for_dataset(ds: str) -> str:
    """
    Return the split directory ("train" or "validation") for a dataset name.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be located in either split directory.
    """
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
    Parse the UTC start datetime encoded in an ATBFL audio filename, e.g.
    ``2014-06-29T23-00-00_000.wav`` -> 2014-06-29 23:00:00 UTC.

    Returns ``None`` if the filename does not match the expected pattern.
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
    """Scan the filesystem and read each WAV header for its duration."""
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
    Return a DataFrame of all audio files in the requested datasets, cached
    to disk.

    Columns: ``dataset``, ``filename``, ``path``, ``duration_s``,
    ``start_dt``, ``end_dt``.
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
    Infer the owning file for each annotation via ``pd.merge_asof``.

    For each annotation, find the file whose start time is the largest not
    exceeding the annotation start, then verify the annotation begins before
    the file ends. Files in a dataset are non-overlapping in time, so this is
    unique. Rows with no match get ``pd.NA``.
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

    out_of_range = merged["start_datetime"] >= merged["end_dt"]
    merged.loc[out_of_range, "filename"] = pd.NA

    merged.index = df_sorted.index
    return merged.sort_index()["filename"]


def _load_annotations_uncached(
    datasets: list[str], manifest: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Read annotation CSVs and infer filenames where absent."""
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
    datasets: list[str], manifest: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Load and concatenate annotations for the requested datasets, cached to
    disk.

    Columns: ``dataset``, ``filename``, ``start_datetime``, ``end_datetime``,
    ``annotation`` (fine 7-class label), ``label_3class`` (coarse label).
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
    """
    Lightweight description of an audio segment, loaded on demand.

    Attributes
    ----------
    dataset, filename, path : str
        Origin of the segment's audio.
    start_sample, end_sample : int
        Sample offsets into the file.
    file_start_dt : datetime
        UTC start time of the containing file.
    annotations : list of dict
        Intersecting annotations with file-relative ``start_s`` / ``end_s``
        plus ``label`` and ``label_3class``.
    is_positive : bool
        Whether the segment contains at least one annotated call.
    """

    dataset: str
    filename: str
    path: str
    start_sample: int
    end_sample: int
    file_start_dt: datetime
    annotations: list[dict]
    is_positive: bool


def _build_annotations_by_file(
    annotations: pd.DataFrame, manifest: pd.DataFrame
) -> dict:
    """
    Group annotations by ``(dataset, filename)`` for O(1) lookup, with
    file-relative second offsets.
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
        # pd.isna handles NaT from the parquet cache (None datetimes).
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
    rng: random.Random | None = None,
) -> list[Segment]:
    """
    Build one training segment per valid positive annotation.

    Each segment spans the call plus a random ``uniform(collar_min_s,
    collar_max_s)`` collar on each side. All annotations intersecting the
    resulting window are attached so multi-call segments are labelled
    correctly. Annotations with invalid or out-of-range durations are skipped.

    Parameters
    ----------
    rng : random.Random, optional
        Source of randomness for the collars. Pass a dedicated instance to
        decouple sampling from the global ``random`` stream (and thus from
        anything else that may consume it). Defaults to the global module.
    """
    if rng is None:
        rng = random

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

        if call_end_s <= call_start_s or call_end_s <= 0:
            continue
        if call_end_s - call_start_s > cfg.MAX_CALL_DURATION_S:
            continue
        if call_end_s - call_start_s < cfg.MIN_CALL_DURATION_S:
            continue

        pre = rng.uniform(collar_min_s, collar_max_s)
        post = rng.uniform(collar_min_s, collar_max_s)
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
    rng: random.Random | None = None,
) -> list[Segment]:
    """
    Sample up to ``n_segments`` random call-free windows from the manifest.

    Implements stochastic negative undersampling: candidate windows that
    overlap any annotated call are rejected. Returns fewer than requested only
    if the rejection sampler hits its retry cap (20x ``n_segments``).

    Parameters
    ----------
    rng : random.Random, optional
        Source of randomness for window selection. Pass a dedicated instance
        to decouple sampling from the global ``random`` stream. Defaults to
        the global module. Reusing the same instance across epochs makes each
        epoch draw a fresh, reproducible negative subset.
    """
    if rng is None:
        rng = random

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
        file_row = rng.choice(files)
        key = (file_row["dataset"], file_row["filename"])
        dur = file_row["duration_s"]
        seg_len = rng.uniform(min_dur_s, max_dur_s)

        if dur <= seg_len + 1.0:
            continue

        seg_start_s = rng.uniform(0, dur - seg_len)
        seg_end_s = seg_start_s + seg_len

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


# ======================================================================
# Fixed-length segment extension
# ======================================================================

def extend_segment_to_fixed_length(
    seg: Segment,
    target_seconds: float,
    file_duration_s: float,
    sample_rate: int = cfg.SAMPLE_RATE,
    rng: random.Random = None,
) -> Segment:
    """
    Return a copy of ``seg`` whose window is exactly ``target_seconds`` long.

    Context is added on both sides; the original content's position within the
    new window is randomised so the model does not learn a fixed
    "calls-always-here" prior. Segments already at or above the target length
    are returned unchanged. Annotation times are file-relative, so they remain
    valid as the window grows; the frame-level target tensor is rebuilt from
    the new window in :meth:`WhaleDataset.__getitem__`.
    """
    if rng is None:
        rng = random

    target_samples = int(target_seconds * sample_rate)
    file_samples = int(file_duration_s * sample_rate)
    cur_length = seg.end_sample - seg.start_sample

    if cur_length >= target_samples:
        return seg

    extra = target_samples - cur_length

    if file_samples <= target_samples:
        return replace(seg, start_sample=0, end_sample=file_samples)

    pre_room = seg.start_sample
    post_room = file_samples - seg.end_sample

    pre_extra = min(pre_room, rng.randint(0, extra))
    post_extra = min(post_room, extra - pre_extra)

    deficit = extra - pre_extra - post_extra
    if deficit > 0:
        if pre_room - pre_extra >= deficit:
            pre_extra += deficit
        else:
            post_extra += deficit

    new_start = max(0, seg.start_sample - pre_extra)
    new_end = min(file_samples, new_start + target_samples)
    new_start = max(0, new_end - target_samples)

    return replace(seg, start_sample=new_start, end_sample=new_end)


def extend_all_segments(segments, manifest, target_seconds: float):
    """
    Apply :func:`extend_segment_to_fixed_length` to every segment, looking up
    each file's duration from the manifest. Uses a fixed-seed RNG so the
    randomised positioning is reproducible.
    """
    rng = random.Random(0xC0FFEE)
    duration_lookup = {
        (r["dataset"], r["filename"]): r["duration_s"]
        for _, r in manifest.iterrows()
    }
    extended = []
    for seg in segments:
        dur = duration_lookup.get((seg.dataset, seg.filename))
        if dur is None:
            continue
        extended.append(
            extend_segment_to_fixed_length(seg, target_seconds, dur, rng=rng)
        )
    return extended


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
    Tile each file with fixed-length overlapping windows for evaluation.

    Consecutive windows overlap by ``overlap_s`` seconds so overlapping
    predictions can be averaged during post-processing.
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

class WhaleDataset(Dataset):
    """
    Map-style dataset yielding ``(audio, targets, mask, meta)`` per segment.

    Audio is read lazily from disk on each ``__getitem__`` (a 30 s 250 Hz
    segment is ~30 KB), keeping memory low. The target width follows
    ``cfg.n_classes()`` at construction time.
    """

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self.stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)
        self.class_idx = cfg.class_to_idx()
        self.n_classes = cfg.n_classes()

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        audio : torch.Tensor, shape (n_samples,)
        targets : torch.Tensor, shape (n_frames, n_classes)
            ``targets[t, c] = 1`` iff frame ``t`` overlaps a class-``c`` call.
        mask : torch.Tensor, shape (n_frames,), dtype bool
            All True before collation; the collator marks padded frames.
        meta : dict
            ``dataset``, ``filename``, ``start_sample``, ``end_sample``.
        """
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
    Pad variable-length segments to the batch maximum.

    Returns
    -------
    audio_pad : torch.Tensor, shape (B, max_samples)
    target_pad : torch.Tensor, shape (B, max_frames, n_classes)
    mask_pad : torch.Tensor, shape (B, max_frames), dtype bool
        True for valid (non-padded) frames.
    metas : list of dict
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
