"""
Dataset Loading and Segmentation
================================

Handles all data-related responsibilities for the Whale-VAD pipeline:

    1. Locating audio files and annotations on disk (BioDCASE 2026 layout)
    2. Parsing per-dataset CSV annotation files, including datetime-to-file
       matching for CSVs that lack an explicit ``filename`` column
    3. Constructing training segments: each positive annotation + random
       collar (Section 5.1), plus randomly sampled negative segments
       (Section 5.5)
    4. Constructing fixed-length, overlapping validation segments for
       deterministic evaluation (Section 5.1)
    5. Frame-level target tensor construction at 20 ms resolution
    6. PyTorch Dataset and DataLoader wrappers with batched collation

Directory layout expected by this module
----------------------------------------
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
Each CSV must contain at minimum:

    - ``start_datetime``  (ISO 8601 timestamp)
    - ``end_datetime``    (ISO 8601 timestamp)
    - ``annotation``      (call-type label, e.g. "bma", "bp20")
    - optionally ``filename`` (audio file containing the call)

If ``filename`` is missing, it is inferred by matching ``start_datetime``
against the datetime range covered by each WAV file in the same dataset.
Filenames are expected to encode their start time, e.g.
``2014-06-29T23-00-00_000.wav``.
"""

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
# Path resolution
# ======================================================================

def _split_for_dataset(ds: str) -> str:
    """
    Return the split directory ("train" or "validation") for a dataset name.

    Parameters
    ----------
    ds : str
        Dataset identifier (e.g. ``"casey2017"``).

    Returns
    -------
    str
        Either ``"train"`` or ``"validation"``.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be located in either split directory.
    """
    # First check the explicit config lists, which are authoritative for
    # known splits and avoid a filesystem hit.
    if ds in cfg.TRAIN_DATASETS:
        return "train"
    if ds in cfg.VAL_DATASETS:
        return "validation"
    # Fall back to checking the filesystem in case a dataset is referenced
    # that is not listed in the config (e.g. custom ablation setups).
    for split in ("train", "validation"):
        if (cfg.DATA_ROOT / split / "audio" / ds).exists():
            return split
    raise FileNotFoundError(f"Cannot find split for dataset '{ds}'")


def _parse_file_start_dt(filename: str):
    """
    Parse the start datetime encoded in an ATBFL audio filename.

    Example filename: ``2014-06-29T23-00-00_000.wav`` → 2014-06-29 23:00:00 UTC.

    The optional trailing ``_000`` represents milliseconds (always zero in
    the ATBFL corpus) and is discarded.

    Parameters
    ----------
    filename : str
        Audio filename (with or without path prefix and extension).

    Returns
    -------
    datetime or None
        UTC-aware datetime, or ``None`` if the filename does not match the
        expected pattern.
    """
    stem = Path(filename).stem
    stem = stem.split("_")[0]  # drop any trailing "_NNN" millisecond field
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ======================================================================
# File manifest
# ======================================================================

def get_file_manifest(datasets: list[str]) -> pd.DataFrame:
    """
    Scan the filesystem and return a DataFrame of all audio files in
    the requested datasets.

    Parameters
    ----------
    datasets : list of str
        Dataset identifiers to include in the manifest.

    Returns
    -------
    pd.DataFrame
        One row per audio file, with columns:

        - ``dataset``     (str) dataset identifier
        - ``filename``    (str) basename of the WAV file
        - ``path``        (str) absolute filesystem path
        - ``duration_s``  (float) duration in seconds
        - ``start_dt``    (datetime) UTC start time parsed from filename
        - ``end_dt``      (datetime) ``start_dt + duration_s``

    Notes
    -----
    Uses ``soundfile.info`` to read duration from each file's header,
    which is fast (no full decode needed).
    """
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


# ======================================================================
# Annotation loading
# ======================================================================

def load_annotations(datasets: list[str]) -> pd.DataFrame:
    """
    Load and concatenate boundary annotations for the requested datasets.

    Handles the common case where the CSV does not contain a ``filename``
    column by inferring it from ``start_datetime``: the annotation is
    attributed to the audio file whose recording interval contains the
    annotation's start time.

    Parameters
    ----------
    datasets : list of str
        Dataset identifiers to load.

    Returns
    -------
    pd.DataFrame
        Concatenated annotations with columns:

        - ``dataset``         (str)
        - ``filename``        (str) inferred if not present in CSV
        - ``start_datetime``  (datetime, UTC)
        - ``end_datetime``    (datetime, UTC)
        - ``annotation``      (str) fine-grained 7-class label
        - ``label_3class``    (str) coarse 3-class label from COLLAPSE_MAP

        Annotations with no matching audio file (possible when some files
        are missing from disk) are silently dropped, with a count printed
        for transparency.
    """
    all_rows = []
    manifest = get_file_manifest(datasets)  # needed for filename inference

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

        # Parse datetime columns as UTC-aware pandas Timestamps.
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)
        df["end_datetime"] = pd.to_datetime(df["end_datetime"], utc=True)

        # If the CSV does not identify which file each annotation belongs to,
        # infer it by finding the audio file whose time range contains the
        # annotation's start time. Because files in a dataset don't overlap
        # in time, there is at most one match.
        if "filename" not in df.columns:
            ds_files = (manifest[manifest["dataset"] == ds]
                        .sort_values("start_dt")
                        .reset_index(drop=True))
            filenames = []
            for _, row in df.iterrows():
                ann_start = row["start_datetime"]
                match = ds_files[
                    (ds_files["start_dt"] <= ann_start) &
                    (ds_files["end_dt"] > ann_start)
                ]
                filenames.append(match.iloc[0]["filename"] if len(match) else None)
            df["filename"] = filenames

            # Drop orphan annotations (file not on disk, or time gap).
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
    # Add the collapsed 3-class label as a convenience column. Labels not
    # in COLLAPSE_MAP pass through unchanged (shouldn't happen in practice).
    ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
    return ann


# ======================================================================
# Segment dataclass
# ======================================================================

@dataclass
class Segment:
    """
    Lightweight description of an audio segment to be loaded on demand.

    Attributes
    ----------
    dataset, filename : str
        Origin of the segment's audio.
    path : str
        Absolute path to the WAV file.
    start_sample, end_sample : int
        Sample offsets into the file. ``end_sample - start_sample`` gives
        the segment length in samples.
    file_start_dt : datetime
        UTC start time of the containing file. Used to translate segment
        offsets into absolute datetimes for CSV export.
    annotations : list of dict
        Per-annotation dicts with ``start_s``, ``end_s``, ``label``, and
        ``label_3class`` fields. Times are relative to the containing file,
        not to the segment itself.
    is_positive : bool
        ``True`` if the segment contains at least one annotated call.
    """

    dataset: str
    filename: str
    path: str
    start_sample: int
    end_sample: int
    file_start_dt: datetime
    annotations: list[dict]
    is_positive: bool


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
    Build one training segment per positive annotation.

    For each annotated call, construct a segment that spans the call plus
    random silent "collars" of ``uniform(collar_min_s, collar_max_s)``
    duration on either side. This ensures each training example contains
    a positive call while varying the surrounding context to prevent
    trivial segment-length heuristics.

    All annotations that intersect the resulting segment (not just the
    focal annotation) are included in the segment's ``annotations`` list,
    so frame-level targets correctly label multi-call segments.

    Parameters
    ----------
    annotations : pd.DataFrame
        Output of ``load_annotations``.
    manifest : pd.DataFrame
        Output of ``get_file_manifest`` (must include the files referenced
        in ``annotations``).
    collar_min_s, collar_max_s : float
        Range of random collar durations in seconds.

    Returns
    -------
    list of Segment
        One segment per valid positive annotation. Annotations with invalid
        durations (negative, zero, or exceeding ``MAX_CALL_DURATION_S``)
        are silently skipped.
    """
    segments = []
    if manifest.empty or annotations.empty:
        return segments

    # Build an indexed lookup once; per-row .loc is then O(1).
    manifest_idx = manifest.set_index(["dataset", "filename"])

    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        if key not in manifest_idx.index:
            continue
        file_row = manifest_idx.loc[key]
        file_start_dt = file_row["start_dt"]
        if file_start_dt is None:
            continue

        # Convert absolute datetime annotations to file-relative seconds.
        call_start_s = (row["start_datetime"] - file_start_dt).total_seconds()
        call_end_s = (row["end_datetime"] - file_start_dt).total_seconds()

        # Sanity filters. Negative or zero durations indicate corrupted CSVs;
        # extremely long annotations are typically errors (e.g. missing end
        # timestamp inherited from a previous row).
        if call_end_s <= call_start_s or call_end_s <= 0:
            continue
        if call_end_s - call_start_s > cfg.MAX_CALL_DURATION_S:
            continue
        if call_end_s - call_start_s < cfg.MIN_CALL_DURATION_S:
            continue

        # Random collars, clamped to the file's boundaries.
        pre = random.uniform(collar_min_s, collar_max_s)
        post = random.uniform(collar_min_s, collar_max_s)
        seg_start_s = max(0.0, call_start_s - pre)
        seg_end_s = min(file_row["duration_s"], call_end_s + post)

        # Gather all annotations that intersect this segment. This matters
        # when calls overlap or are closely spaced, so the frame-level
        # targets cover every annotated event inside the segment.
        file_anns = annotations[
            (annotations["dataset"] == row["dataset"]) &
            (annotations["filename"] == row["filename"])
        ]
        inter_anns = []
        for _, a in file_anns.iterrows():
            a_start_s = (a["start_datetime"] - file_start_dt).total_seconds()
            a_end_s = (a["end_datetime"] - file_start_dt).total_seconds()
            if a_end_s > seg_start_s and a_start_s < seg_end_s:
                inter_anns.append({
                    "start_s": a_start_s,
                    "end_s": a_end_s,
                    "label": a["annotation"],
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
    """
    Sample random call-free segments from the manifest.

    Implements the stochastic negative undersampling scheme (Section 5.5):
    each epoch, sample a fresh set of ``n_segments`` random windows that
    do not overlap any annotated call. Over many epochs the model sees a
    diverse sample of the negative distribution without being drowned in
    easy negatives.

    Parameters
    ----------
    annotations : pd.DataFrame
        Used to exclude windows that intersect any annotated call.
    manifest : pd.DataFrame
        Candidate files to sample from.
    n_segments : int
        Target number of negative segments to return.
    min_dur_s, max_dur_s : float
        Range of random segment durations (matches the expected positive
        segment duration distribution).

    Returns
    -------
    list of Segment
        Up to ``n_segments`` negative segments. Fewer may be returned if
        the rejection sampler hits its retry cap (20× ``n_segments``).
    """
    segments = []
    if manifest.empty:
        return segments

    # Precompute, for each file, the list of call intervals as
    # file-relative (start_s, end_s) tuples. Used to reject candidate
    # windows that intersect any call.
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
            e = (a["end_datetime"] - fsd).total_seconds()
            call_intervals.setdefault(key, []).append((s, e))

    files = manifest.to_dict("records")
    tries, max_tries = 0, n_segments * 20

    while len(segments) < n_segments and tries < max_tries:
        tries += 1
        file_row = random.choice(files)
        key = (file_row["dataset"], file_row["filename"])
        dur = file_row["duration_s"]
        seg_len = random.uniform(min_dur_s, max_dur_s)

        # Skip files that are too short to fit the candidate window (with
        # a 1-second margin to avoid edge artefacts).
        if dur <= seg_len + 1.0:
            continue

        seg_start_s = random.uniform(0, dur - seg_len)
        seg_end_s = seg_start_s + seg_len

        # Reject if this window overlaps any annotated call.
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

    Used for both validation (during training) and challenge inference.
    Each file is covered by consecutive ``segment_s``-long windows with
    ``overlap_s`` seconds of overlap between neighbours. The ``overlap_s``
    margin allows overlapping predictions to be averaged during the
    stitching step in post-processing, smoothing segment boundaries.

    Parameters
    ----------
    manifest : pd.DataFrame
        Files to tile.
    annotations : pd.DataFrame
        Annotations to attach to each window (empty DataFrame for inference
        on unlabeled data).
    segment_s : float
        Window duration in seconds.
    overlap_s : float
        Overlap between consecutive windows in seconds.

    Returns
    -------
    list of Segment
        One segment per window.
    """
    segments = []
    if manifest.empty:
        return segments

    step_s = segment_s - overlap_s

    # Precompute per-file list of annotations with file-relative times.
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
                "end_s": (a["end_datetime"] - fsd).total_seconds(),
                "label": a["annotation"],
                "label_3class": a["label_3class"],
            })

    # Tile each file with overlapping windows.
    for _, f in manifest.iterrows():
        key = (f["dataset"], f["filename"])
        dur = f["duration_s"]
        fsd = f["start_dt"]
        file_anns = ann_by_file.get(key, [])

        t = 0.0
        while t + segment_s <= dur + 1e-6:
            # All annotations that intersect this window
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

    Audio is read lazily from disk on every call to ``__getitem__``, which
    keeps memory usage low (a single 30-second 250 Hz segment is ~30 KB)
    at the cost of some disk I/O. For the BioDCASE dataset this trade-off
    is favourable because the full corpus easily fits in OS page cache.

    Parameters
    ----------
    segments : list of Segment
        Pre-built segments to serve.
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
        Load one segment from disk and build its frame-level targets.

        Returns
        -------
        audio : torch.Tensor, shape (n_samples,)
            Raw waveform for the segment.
        targets : torch.Tensor, shape (n_frames, n_classes)
            Binary per-frame targets. ``targets[t, c] = 1.0`` iff frame
            ``t`` overlaps an annotation of class ``c``.
        mask : torch.Tensor, shape (n_frames,), dtype=bool
            All ``True`` before collation; used by the collator to mark
            padded frames. Returned here for API uniformity.
        meta : dict
            Metadata for stitching and CSV export: ``dataset``, ``filename``,
            ``start_sample``, ``end_sample``.
        """
        seg = self.segments[idx]
        n_samples = seg.end_sample - seg.start_sample

        # Partial read: soundfile seeks directly to the requested region
        # rather than loading the whole file.
        audio, sr = sf.read(
            seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32"
        )
        assert sr == cfg.SAMPLE_RATE, f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}"
        audio = torch.from_numpy(audio)

        # Build frame-level targets by "painting" each annotation onto a
        # zero-initialized target matrix. Classes not in class_idx (e.g.
        # rare 7-class labels when 3-class mode is active) are skipped.
        n_frames = n_samples // self.stride_samp
        targets = torch.zeros(n_frames, self.n_classes)
        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE

        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label not in self.class_idx:
                continue
            c = self.class_idx[label]
            # Clamp annotation bounds to this segment.
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
    Custom collator that pads variable-length segments to the batch maximum.

    Training segments have random collar widths and therefore variable
    durations; this collator pads all tensors to the longest in the batch
    and flags padded frames as invalid via the boolean mask.

    Parameters
    ----------
    batch : list of 4-tuples
        Each element is ``(audio, targets, mask, meta)`` from
        ``WhaleDataset.__getitem__``.

    Returns
    -------
    audio_pad : torch.Tensor, shape (B, max_samples)
    target_pad : torch.Tensor, shape (B, max_frames, n_classes)
    mask_pad : torch.Tensor, shape (B, max_frames), dtype=bool
        ``True`` for valid (non-padded) frames.
    metas : list of dict
        Per-segment metadata (unmodified).
    """
    audios, targets, masks, metas = zip(*batch)
    max_samp = max(a.size(0) for a in audios)
    max_frames = max(t.size(0) for t in targets)
    n_classes = targets[0].size(1)
    B = len(audios)

    # Allocate padded tensors filled with zeros; mask starts all False
    # (invalid) and is set True only where real data is copied in.
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

class TrainingDatasetWithResample(WhaleDataset):
    """
    Training dataset whose negative segments can be resampled between epochs.

    Wraps the standard ``WhaleDataset`` with a ``resample_negatives`` method
    that draws a fresh random set of negative segments, implementing the
    stochastic undersampling protocol of Section 5.5. The positive segments
    are fixed once at init time (they are deterministic given the
    annotations).

    Parameters
    ----------
    positive_segments : list of Segment
        Output of ``build_positive_segments``.
    manifest : pd.DataFrame
        Used by the negative sampler.
    annotations : pd.DataFrame
        Used by the negative sampler to avoid call-overlap.
    """

    def __init__(self, positive_segments, manifest, annotations):
        self.positive_segments = positive_segments
        self.manifest = manifest
        self.annotations = annotations
        self.negative_segments = []
        # Initial negative sample so the dataset is valid immediately
        # after construction.
        self.resample_negatives()
        super().__init__(self.positive_segments + self.negative_segments)

    def resample_negatives(self):
        """
        Draw a new set of negative segments.

        Called once per epoch (or less frequently; see ``RESAMPLE_EVERY``
        in ``train.py``). Updates ``self.segments`` so that subsequent
        ``__getitem__`` calls see the new negatives.
        """
        n_neg = int(len(self.positive_segments) * cfg.NEG_RATIO)
        self.negative_segments = build_negative_segments(
            self.annotations, self.manifest, n_segments=n_neg,
        )
        self.segments = self.positive_segments + self.negative_segments


def build_dataloaders():
    """
    Construct train and validation DataLoaders.

    Returns
    -------
    train_ds : TrainingDatasetWithResample
        Training dataset (caller can invoke ``resample_negatives`` between
        epochs).
    train_loader : DataLoader
        Yields shuffled ``(audio, targets, mask, metas)`` batches.
    val_loader : DataLoader
        Yields unshuffled validation batches.
    """
    print(f"Loading train datasets: {cfg.TRAIN_DATASETS}")
    train_annotations = load_annotations(cfg.TRAIN_DATASETS)
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
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
    val_annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    print(f"  Found {len(val_manifest)} audio files, "
          f"{len(val_annotations)} annotations")

    val_segs = build_val_segments(val_manifest, val_annotations)
    val_ds = WhaleDataset(val_segs)
    print(f"Validation: {len(val_segs)} segments "
          f"({sum(s.is_positive for s in val_segs)} with calls)")

    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    return train_ds, train_loader, val_loader
