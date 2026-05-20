"""
Dedicated D-call detector
=========================

This script trains a standalone detector for the merged D-call class.
Synthetic call generation is intentionally kept out of this file.

By default, the script trains one merged D-call output.

During training, validation tunes the D threshold in the same spirit as the
main training scripts, so checkpoint selection reflects the post-processing
operating point. After training, the best checkpoint is reloaded once and the
D threshold is tuned again before saving the final model.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, sosfiltfilt
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import (
    Segment,
    WhaleDataset,
    _build_annotations_by_file,
    build_negative_segments,
    build_positive_segments,
    build_val_segments,
    collate_fn,
    get_file_manifest,
    load_annotations,
)
from model import WhaleVAD, WhaleVADLoss
from model_bpn_v2 import BPNConfigV2, WhaleVADBPNLossV2, WhaleVADBPNV2
from postprocess import (
    Detection,
    compute_metrics,
    merge_and_filter,
    smooth_probabilities,
    stitch_segments,
)
from spectrogram import SpectrogramExtractor

D_SUBTYPES = ["bmd", "bpd"]
D_COARSE_LABEL = "d"
DEFAULT_SEGMENT_S = 60.0
EXPERIMENT_NAME = "phase_d_detector"
SYNTHETIC_CALLS_ENABLED = False

RESAMPLE_EVERY = 5
EARLY_STOP_PATIENCE = 20
LR_PATIENCE = 7
LR_FACTOR = 0.5
MIN_LR = 1e-7

VALID_TAP_NAMES = ("feat_ext", "bottleneck", "depthwise")


def parse_args() -> argparse.Namespace:
    """Parse D-detector training options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general training settings
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--segment-s", type=float, default=DEFAULT_SEGMENT_S, help="Window length in seconds for train and validation segments.")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S, help="Overlap between validation windows in seconds.")
    p.add_argument("--highpass-hz", type=float, default=0.0, help=("Optional audio high-pass cutoff in Hz. Use 0 to keep the raw waveform."))
    p.add_argument("--val-every", type=int, default=1, help="Run full validation every N epochs.")
    p.add_argument("--train-threshold", type=float, default=0.3, help=("Initial threshold before the first tuning pass. Used when --threshold-tune-every is set to 0."))
    p.add_argument("--threshold-tune-every", type=int, default=1, help=("Tune D thresholds every N validation runs during training. Set 0 to disable training-time tuning."))
    p.add_argument("--target-mode", choices=["merged", "subtype"], default="merged", help=("merged trains one binary d output; subtype trains separate bmd/bpd outputs and collapses them to d."))
    p.add_argument("--arch", choices=["whalevad", "bpn_v2"], default="bpn_v2", help="Model architecture for the dedicated D detector.")
    p.add_argument("--run-name", type=str, default=None, help="Optional stable run directory name under runs/.")

    # negative mining options
    p.add_argument("--neg-ratio", type=float, default=1.0, help="Call-free background negatives per D-positive segment.")
    p.add_argument("--other-call-neg-ratio", type=float, default=1.0, help="Non-D whale-call hard negatives per D-positive segment.")
    p.add_argument("--no-pos-weights", dest="no_pos_weights", action="store_true", help="Disable positive-class weights in the loss.")

    # wandb logging
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")

    bpn_grp = p.add_argument_group("BPN-v2 options")
    bpn_grp.add_argument("--no-bpn", dest="bpn_enabled", action="store_false", default=True)
    bpn_grp.add_argument("--bpn-taps", type=str, default="feat_ext,bottleneck,depthwise")
    bpn_grp.add_argument("--bpn-rois", type=int, default=32)
    bpn_grp.add_argument("--bpn-proj-dim", type=int, default=128)
    bpn_grp.add_argument("--bpn-out-dim", type=int, default=64)
    bpn_grp.add_argument("--bpn-pool", choices=["softmax", "sigmoid", "mean"], default="softmax")
    bpn_grp.add_argument("--bpn-init", choices=["near_one", "random"], default="near_one")
    bpn_grp.add_argument("--bpn-temporal", choices=["bilstm", "lr", "none"], default="bilstm")
    bpn_grp.add_argument("--bpn-lstm-hidden", type=int, default=64)
    bpn_grp.add_argument("--bpn-lstm-layers", type=int, default=1)
    bpn_grp.add_argument("--bpn-spatial-dropout", type=float, default=0.2)
    bpn_grp.add_argument("--bpn-dropout", type=float, default=0.2)
    bpn_grp.add_argument("--bpn-dilations", type=str, default="2,4,8")

    bpn_grp.add_argument("--bpn-aux-loss", action="store_true")
    bpn_grp.add_argument("--bpn-aux-weight", type=float, default=0.1)
    bpn_grp.add_argument("--bpn-aux-pos-weight", type=float, default=19.0)
    bpn_grp.add_argument("--bpn-aux-warmup", type=int, default=5)
    bpn_grp.add_argument("--bpn-aux-ramp", type=int, default=5)
    bpn_grp.add_argument("--bpn-aux-sigma", type=float, default=2.0)
    bpn_grp.add_argument("--bpn-aux-recall-weight", type=float, default=1.0)

    return p.parse_args()


def target_labels(target_mode: str) -> list[str]:
    """Return model output labels for the selected D-detector target."""
    return [D_COARSE_LABEL] if target_mode == "merged" else list(D_SUBTYPES)


def parse_taps(taps_arg: str) -> tuple[str, ...]:
    """Convert the comma-separated BPN tap list into validated names."""
    names = tuple(t.strip() for t in taps_arg.split(",") if t.strip())
    unknown = set(names) - set(VALID_TAP_NAMES)
    if not names or unknown:
        raise ValueError(f"Invalid --bpn-taps={taps_arg!r}; valid names are {list(VALID_TAP_NAMES)}")
    return names


def build_bpn_cfg(args: argparse.Namespace) -> BPNConfigV2:
    """Translate CLI options into the model's BPN configuration object."""
    dilations = tuple(int(d) for d in args.bpn_dilations.split(",") if d.strip())
    return BPNConfigV2(
        enabled=args.bpn_enabled,
        taps=parse_taps(args.bpn_taps),
        proj_dim=args.bpn_proj_dim,
        out_dim=args.bpn_out_dim,
        n_rois=args.bpn_rois,
        pool_mode=args.bpn_pool,
        init_mode=args.bpn_init,
        temporal=args.bpn_temporal,
        lstm_hidden=args.bpn_lstm_hidden,
        lstm_layers=args.bpn_lstm_layers,
        spatial_dropout_p=args.bpn_spatial_dropout,
        bpn_dropout_p=args.bpn_dropout,
        aux_loss=args.bpn_aux_loss,
        aux_weight=args.bpn_aux_weight,
        aux_pos_weight=args.bpn_aux_pos_weight,
        aux_warmup_epochs=args.bpn_aux_warmup,
        aux_ramp_epochs=args.bpn_aux_ramp,
        aux_target_sigma=args.bpn_aux_sigma,
        aux_recall_weight=args.bpn_aux_recall_weight,
        dilations=dilations,
    )


class WhaleVADDictWrapper(nn.Module):
    """
    Wrap the original WhaleVAD so it matches the BPN output interface.

    The D-detector training loop expects a dict with probabilities for
    validation and logits for loss computation. Keeping this adapter local
    avoids special cases throughout the rest of the script.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = WhaleVAD(num_classes=num_classes)

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.model(spec)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "bpn_extras": {},
        }


class WhaleVADDictLoss(nn.Module):
    """Loss adapter for the dict output produced by WhaleVADDictWrapper."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = WhaleVADLoss(pos_weight=pos_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss(outputs["logits"], targets, padding_mask)


def build_model_and_loss(args: argparse.Namespace, n_outputs: int, pos_weight: torch.Tensor | None, device: torch.device) -> tuple[nn.Module, nn.Module, dict]:
    """
    Build the selected D-detector architecture and matching loss.

    'whalevad' is the original WhaleVAD architecture, while 'bpn_v2' is the
    stronger BPN-gated variant used in the current D-detector experiments.
    """
    if args.arch == "whalevad":
        model = WhaleVADDictWrapper(num_classes=n_outputs).to(device)
        criterion = WhaleVADDictLoss(pos_weight=pos_weight).to(device)
        return model, criterion, {
            "arch": "whalevad",
            "model_name": "WhaleVAD",
        }

    if args.arch == "bpn_v2":
        bpn_cfg = build_bpn_cfg(args)
        model = WhaleVADBPNV2(num_classes=n_outputs, bpn_cfg=bpn_cfg).to(device)
        criterion = WhaleVADBPNLossV2(pos_weight=pos_weight, bpn_cfg=bpn_cfg).to(device)
        return model, criterion, {
            "arch": "bpn_v2",
            "bpn_cfg_v2": bpn_cfg.to_dict(),
            "bpn_version": "v2",
        }

    raise ValueError(f"Unsupported architecture: {args.arch}")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extend_segment_to_fixed_length(seg: Segment, target_seconds: float, file_duration_s: float, rng: random.Random) -> Segment:
    """
    Grow a segment to a fixed duration while staying inside the audio file.

    Validation uses fixed windows. Training on shorter call-centered snippets
    caused a train/validation length mismatch in earlier experiments, so the
    D detector trains on the same window length that it validates on.
    """
    target_samples = int(target_seconds * cfg.SAMPLE_RATE)
    file_samples = int(file_duration_s * cfg.SAMPLE_RATE)
    cur_samples = seg.end_sample - seg.start_sample

    if cur_samples >= target_samples:
        return seg
    if file_samples <= target_samples:
        return replace(seg, start_sample=0, end_sample=file_samples)

    extra = target_samples - cur_samples
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


def extend_all_segments(segments: list[Segment], manifest: pd.DataFrame, target_seconds: float) -> list[Segment]:
    """Extend all segments to the validation window length."""
    rng = random.Random(0xDCA11)
    durations = {
        (row["dataset"], row["filename"]): float(row["duration_s"]) for _, row in manifest.iterrows()
    }
    out = []
    for seg in segments:
        duration = durations.get((seg.dataset, seg.filename))
        if duration is None:
            continue
        out.append(extend_segment_to_fixed_length(seg, target_seconds, duration, rng))
    return out


def align_lengths(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim or pad targets so they match the model's frame count."""
    t_model, t_target = probs.size(1), targets.size(1)
    if t_model < t_target:
        targets = targets[:, :t_model, :]
        mask = mask[:, :t_model]
    elif t_model > t_target:
        pad_targets = torch.zeros(
            targets.size(0), t_model - t_target, targets.size(2),
            device=targets.device,
        )
        pad_mask = torch.zeros(
            mask.size(0), t_model - t_target,
            dtype=torch.bool, device=mask.device,
        )
        targets = torch.cat([targets, pad_targets], dim=1)
        mask = torch.cat([mask, pad_mask], dim=1)
    return targets, mask


def align_outputs(outputs: dict[str, torch.Tensor], targets: torch.Tensor, mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Apply target/mask alignment consistently for BPN output dicts."""
    targets, mask = align_lengths(outputs["probs"], targets, mask)
    return outputs, targets, mask


def build_highpass_sos(highpass_hz: float) -> np.ndarray | None:
    """Create a stable high-pass filter, or return None when disabled."""
    if highpass_hz <= 0:
        return None
    return butter(
        N=4,
        Wn=float(highpass_hz),
        btype="highpass",
        fs=cfg.SAMPLE_RATE,
        output="sos",
    ).astype(np.float32)


def apply_highpass(audio: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    """Apply zero-phase high-pass filtering before spectrogram extraction."""
    if sos is None:
        return audio

    # sosfiltfilt avoids an artificial timing/phase shift. 
    # Very short clips can be shorter than scipy's padding requirement, 
    # so fall back to sosfilt instead of failing a training worker.
    try:
        filtered = sosfiltfilt(sos, audio, axis=0)
    except ValueError:
        filtered = sosfilt(sos, audio, axis=0)
    return np.ascontiguousarray(filtered, dtype=np.float32)


class DDetectorDataset(WhaleDataset):
    """
    Dataset that paints only D-call frame targets.

    Non-D calls inside a segment remain zero target frames. This is deliberate:
    for a D detector, other whale calls are hard negative context.
    """

    def __init__(self, segments: list[Segment], target_mode: str = "merged", highpass_hz: float = 0.0):
        super().__init__(segments)
        self.target_mode = target_mode
        self.output_labels = target_labels(target_mode)
        self.n_classes = len(self.output_labels)
        self.subtype_to_idx = {name: i for i, name in enumerate(D_SUBTYPES)}
        self.highpass_hz = float(highpass_hz)
        self.highpass_sos = build_highpass_sos(self.highpass_hz)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        n_samples = seg.end_sample - seg.start_sample

        audio, sr = sf.read(seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32")
        assert sr == cfg.SAMPLE_RATE, f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}"
        audio = apply_highpass(audio, self.highpass_sos)
        audio = torch.from_numpy(audio)

        n_frames = n_samples // self.stride_samp
        targets = torch.zeros(n_frames, self.n_classes)
        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE

        for ann in seg.annotations:
            label = ann["label"]
            if label not in D_SUBTYPES:
                continue
            local_start_s = max(0.0, ann["start_s"] - seg_start_s)
            local_end_s = min(n_samples / cfg.SAMPLE_RATE, ann["end_s"] - seg_start_s)
            f0 = int(local_start_s / cfg.FRAME_STRIDE_S)
            f1 = int(local_end_s / cfg.FRAME_STRIDE_S)
            if self.target_mode == "merged":
                targets[f0:f1, 0] = 1.0
                continue
            targets[f0:f1, self.subtype_to_idx[label]] = 1.0

        mask = torch.ones(n_frames, dtype=torch.bool)
        meta = {
            "dataset": seg.dataset,
            "filename": seg.filename,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
        }
        return audio, targets, mask, meta


class DDetectorTrainingDataset(DDetectorDataset):
    """
    Training set with D positives, non-D hard negatives, and background.

    Non-D hard negatives and background negatives are resampled every few
    epochs so the model sees a wider range of confusing whale calls and
    call-free audio.
    """

    def __init__(
        self,
        positive_segments: list[Segment],
        manifest: pd.DataFrame,
        all_annotations: pd.DataFrame,
        hard_neg_ratio: float,
        neg_ratio: float,
        segment_s: float,
        target_mode: str,
        highpass_hz: float,
    ):
        self.positive_segments = positive_segments
        self.manifest = manifest
        self.all_annotations = all_annotations
        self.hard_neg_ratio = hard_neg_ratio
        self.neg_ratio = neg_ratio
        self.segment_s = segment_s
        self.target_mode = target_mode
        self.highpass_hz = highpass_hz
        self.hard_negative_segments: list[Segment] = []
        self.background_negative_segments: list[Segment] = []
        super().__init__(self._all_segments(), target_mode=target_mode, highpass_hz=highpass_hz)

    def _all_segments(self) -> list[Segment]:
        return (self.positive_segments + self.hard_negative_segments + self.background_negative_segments)

    def resample_negatives(self) -> None:
        """Draw fresh fixed-length non-D hard negatives and background windows."""
        n_hard = int(len(self.positive_segments) * self.hard_neg_ratio)
        self.hard_negative_segments = sample_hard_negative_segments(
            self.all_annotations,
            self.manifest,
            n_hard,
            self.segment_s,
        )

        n_neg = int(len(self.positive_segments) * self.neg_ratio)
        self.background_negative_segments = build_negative_segments(
            self.all_annotations,
            self.manifest,
            n_segments=n_neg,
            min_dur_s=self.segment_s,
            max_dur_s=self.segment_s,
        )
        self.segments = self._all_segments()


def d_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return only bmd/bpd annotation rows."""
    return annotations[annotations["annotation"].isin(D_SUBTYPES)].reset_index(drop=True)


def non_d_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return all non-D annotations used to build hard negatives."""
    return annotations[~annotations["annotation"].isin(D_SUBTYPES)].reset_index(drop=True)


def build_d_intervals_by_file(annotations: pd.DataFrame, manifest: pd.DataFrame) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Build file-relative D-call intervals for overlap filtering."""
    manifest_idx = manifest.set_index(["dataset", "filename"])
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for _, row in d_annotations(annotations).iterrows():
        key = (row["dataset"], row["filename"])
        if key not in manifest_idx.index:
            continue
        file_start = manifest_idx.loc[key, "start_dt"]
        if file_start is None or pd.isna(file_start):
            continue
        start_s = (row["start_datetime"] - file_start).total_seconds()
        end_s = (row["end_datetime"] - file_start).total_seconds()
        intervals.setdefault(key, []).append((start_s, end_s))
    return intervals


def overlaps_d_call(segment: Segment, d_intervals_by_file: dict[tuple[str, str], list[tuple[float, float]]]) -> bool:
    """Check whether a candidate negative segment overlaps any D-call."""
    key = (segment.dataset, segment.filename)
    start_s = segment.start_sample / cfg.SAMPLE_RATE
    end_s = segment.end_sample / cfg.SAMPLE_RATE
    return any(
        end_s > d_start and start_s < d_end
        for d_start, d_end in d_intervals_by_file.get(key, [])
    )


def refresh_segment_annotations(segments: list[Segment], annotations: pd.DataFrame, manifest: pd.DataFrame) -> list[Segment]:
    """
    Refresh annotations after extending a segment to the training window length.

    If the added context contains another bmd/bpd call, it must become a
    positive target. Otherwise the model would see a real D-call as background.
    """
    ann_by_file = _build_annotations_by_file(annotations, manifest)
    refreshed = []
    for seg in segments:
        start_s = seg.start_sample / cfg.SAMPLE_RATE
        end_s = seg.end_sample / cfg.SAMPLE_RATE
        file_anns = ann_by_file.get((seg.dataset, seg.filename), [])
        inter = [
            ann for ann in file_anns
            if ann["end_s"] > start_s and ann["start_s"] < end_s
        ]
        refreshed.append(replace(seg, annotations=inter, is_positive=bool(inter)))
    return refreshed


def sample_hard_negative_segments(annotations: pd.DataFrame, manifest: pd.DataFrame, n_segments: int, segment_s: float) -> list[Segment]:
    """
    Build non-D whale-call negatives that do not overlap D-calls.

    These are harder than random background because they contain whale-call
    energy, but their D target should be zero.
    """
    if n_segments <= 0:
        return []

    candidates = build_positive_segments(non_d_annotations(annotations), manifest)
    candidates = extend_all_segments(candidates, manifest, segment_s)
    d_intervals = build_d_intervals_by_file(annotations, manifest)
    clean = [seg for seg in candidates if not overlaps_d_call(seg, d_intervals)]
    random.shuffle(clean)
    return clean[:min(n_segments, len(clean))]


def build_dataloaders(args: argparse.Namespace):
    """Load manifests/annotations and build train/validation dataloaders."""
    print(f"Loading train datasets: {cfg.TRAIN_DATASETS}")
    train_manifest = get_file_manifest(cfg.TRAIN_DATASETS)
    train_ann = load_annotations(cfg.TRAIN_DATASETS, manifest=train_manifest)
    train_d_ann = d_annotations(train_ann)

    print(f"  train files: {len(train_manifest)}")
    print(f"  train annotations: {len(train_ann)}")
    for label in D_SUBTYPES:
        print(f"  train {label}: {(train_d_ann['annotation'] == label).sum()}")

    labels = target_labels(args.target_mode)
    d_pos = build_positive_segments(train_d_ann, train_manifest)
    d_pos = extend_all_segments(d_pos, train_manifest, args.segment_s)
    d_pos = refresh_segment_annotations(d_pos, train_d_ann, train_manifest)
    d_pos = [seg for seg in d_pos if seg.annotations]

    train_ds = DDetectorTrainingDataset(
        positive_segments=d_pos,
        manifest=train_manifest,
        all_annotations=train_ann,
        hard_neg_ratio=args.other_call_neg_ratio,
        neg_ratio=args.neg_ratio,
        segment_s=args.segment_s,
        target_mode=args.target_mode,
        highpass_hz=args.highpass_hz,
    )

    # Draw the first negative set before we print counts or build the loader.
    # Later epochs refresh this pool on the configured resampling schedule.
    print("  Sampling initial non-D hard negatives and background negatives")
    train_ds.resample_negatives()

    print("Training segments:")
    print(f"  target mode: {args.target_mode} ({labels})")
    print(f"  segment length: {args.segment_s:.1f}s")
    print(f"  high-pass: {args.highpass_hz:.1f} Hz" if args.highpass_hz > 0 else "  high-pass: disabled")
    print(f"  D positives: {len(d_pos)}")
    print(f"  non-D hard negatives: {len(train_ds.hard_negative_segments)}")
    print(f"  background negatives: {len(train_ds.background_negative_segments)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"\nLoading validation datasets: {cfg.VAL_DATASETS}")
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_ann = load_annotations(cfg.VAL_DATASETS, manifest=val_manifest)
    val_d_ann = d_annotations(val_ann)
    val_segments = build_val_segments(val_manifest, val_ann, segment_s=args.segment_s, overlap_s=args.overlap_s)
    val_ds = DDetectorDataset(val_segments, target_mode=args.target_mode, highpass_hz=args.highpass_hz)

    print(f"  validation files: {len(val_manifest)}")
    print(f"  validation annotations: {len(val_ann)}")
    for label in D_SUBTYPES:
        print(f"  validation {label}: {(val_d_ann['annotation'] == label).sum()}")
    print(f"  validation windows: {len(val_segments)}")
    print(f"  validation overlap: {args.overlap_s:.1f}s")
    print(f"  validation high-pass: {args.highpass_hz:.1f} Hz" if args.highpass_hz > 0 else "  validation high-pass: disabled")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    file_start_dts = {(row.dataset, row.filename): row.start_dt for _, row in val_manifest.iterrows()}
    return train_ds, train_loader, train_ann, val_loader, val_ann, file_start_dts


def compute_subtype_pos_weights(train_annotations: pd.DataFrame) -> torch.Tensor:
    """
    Upweight rarer subtype frames.

    The ratio is computed at file level to avoid a few dense files dominating
    the estimate. The common subtype gets weight 1.0.
    """
    total_files = max(train_annotations.groupby(["dataset", "filename"]).ngroups, 1)
    weights = []
    for label in D_SUBTYPES:
        rows = train_annotations[train_annotations["annotation"] == label]
        pos_files = max(rows.groupby(["dataset", "filename"]).ngroups, 1)
        weights.append(max(total_files - pos_files, 1) / pos_files)

    out = torch.tensor(weights, dtype=torch.float32)
    out = out / out.min()
    print("Subtype positive weights:")
    for label, weight in zip(D_SUBTYPES, out.tolist()):
        print(f"  {label}: {weight:.3f}")
    return out


def compute_target_pos_weights(train_annotations: pd.DataFrame, target_mode: str) -> torch.Tensor:
    """
    Compute positive weights for either merged-D or subtype outputs.

    For the merged detector, bmd and bpd are treated as one positive class.
    This matches the final evaluation label and avoids separate calibration
    for two very similar D-call subtypes.
    """
    if target_mode == "subtype":
        return compute_subtype_pos_weights(train_annotations)

    total_files = max(train_annotations.groupby(["dataset", "filename"]).ngroups, 1)
    d_rows = d_annotations(train_annotations)
    pos_files = max(d_rows.groupby(["dataset", "filename"]).ngroups, 1)
    weight = max(total_files - pos_files, 1) / pos_files
    out = torch.tensor([weight], dtype=torch.float32)
    print("Merged-D positive weight:")
    print(f"  {D_COARSE_LABEL}: {out.item():.3f}")
    return out


def d_detections_from_file_probs(file_probs: dict[tuple[str, str], np.ndarray], thresholds: np.ndarray, target_mode: str) -> list[Detection]:
    """Threshold model outputs and return merged coarse-d detections."""
    output_labels = target_labels(target_mode)
    raw: list[Detection] = []
    for (dataset, filename), probs in file_probs.items():
        for c, label in enumerate(output_labels):
            active = probs[:, c] > thresholds[c]
            diffs = np.diff(active.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for start, end in zip(starts, ends):
                raw.append(Detection(
                    dataset=dataset,
                    filename=filename,
                    label=label,
                    start_s=start * cfg.FRAME_STRIDE_S,
                    end_s=end * cfg.FRAME_STRIDE_S,
                    confidence=float(probs[start:end, c].mean()),
                ))
    return merge_and_filter(raw)


def smooth_file_probs(file_probs: dict[tuple[str, str], np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    """Smooth stitched probability streams once before threshold scans."""
    return {
        key: smooth_probabilities(probs)
        for key, probs in file_probs.items()
    }


def d_detections_from_probs(all_probs: dict[tuple[str, str, int], np.ndarray], thresholds: np.ndarray, target_mode: str) -> list[Detection]:
    """Stitch validation windows before thresholding D streams."""
    file_probs = smooth_file_probs(stitch_segments(all_probs))
    return d_detections_from_file_probs(file_probs, thresholds, target_mode)


def build_d_ground_truth(val_annotations: pd.DataFrame, file_start_dts: dict[tuple[str, str], object]) -> list[Detection]:
    """Build coarse-d ground truth from bmd/bpd validation annotations."""
    gt: list[Detection] = []
    for _, row in d_annotations(val_annotations).iterrows():
        key = (row["dataset"], row["filename"])
        file_start = file_start_dts.get(key)
        if file_start is None or pd.isna(file_start):
            continue
        gt.append(Detection(
            dataset=row["dataset"],
            filename=row["filename"],
            label=D_COARSE_LABEL,
            start_s=(row["start_datetime"] - file_start).total_seconds(),
            end_s=(row["end_datetime"] - file_start).total_seconds(),
        ))
    return gt


def tune_d_thresholds(all_probs: dict[tuple[str, str, int], np.ndarray], gt_events: list[Detection], initial_thresholds: np.ndarray, target_mode: str) -> tuple[np.ndarray, dict]:
    """Tune D output thresholds against collapsed-d F1."""
    thresholds = np.asarray(initial_thresholds, dtype=np.float64).copy()
    output_labels = target_labels(target_mode)
    candidates = np.concatenate([
        np.arange(0.02, 0.40, 0.02),
        np.arange(0.40, 0.90, 0.05),
        np.arange(0.90, 0.99, 0.02),
    ])
    candidates = np.unique(np.round(candidates, 4))

    # Stitch and smooth once per validation pass. Doing this inside every
    # threshold trial makes validation look frozen for a long time.
    print("    stitching validation windows for threshold tuning")
    tune_start = time.time()
    file_probs = smooth_file_probs(stitch_segments(all_probs))
    total_trials = len(candidates) * len(output_labels)
    print(f"    threshold tuning over {total_trials} trials")
    best_metrics = {}
    with tqdm(
        total=total_trials,
        desc="    threshold tuning",
        unit="trial",
        leave=True,
        dynamic_ncols=True,
    ) as pbar:
        for c, label in enumerate(output_labels):
            best_f1 = -1.0
            best_t = thresholds[c]
            best_fp = float("inf")
            best_precision = -1.0
            for trial_t in candidates:
                trial = thresholds.copy()
                trial[c] = float(trial_t)
                preds = d_detections_from_file_probs(file_probs, trial, target_mode)
                metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
                d_metrics = metrics.get(D_COARSE_LABEL, {})
                f1 = d_metrics.get("f1", 0.0)
                precision = d_metrics.get("precision", 0.0)
                fp = d_metrics.get("fp", 0)

                # Low F1 regions can contain many nearly equivalent thresholds.
                # In ties, prefer the stricter threshold with fewer false positives
                # so the tuned curve does not jump to a very permissive operating
                # point for no real F1 gain.
                better = f1 > best_f1 + 1e-12
                tied = abs(f1 - best_f1) <= 1e-12
                if tied:
                    better = (
                        fp < best_fp
                        or (fp == best_fp and precision > best_precision + 1e-12)
                        or (fp == best_fp and abs(precision - best_precision) <= 1e-12 and trial_t > best_t)
                    )

                if better:
                    best_f1 = f1
                    best_t = float(trial_t)
                    best_metrics = metrics
                    best_fp = fp
                    best_precision = precision

                pbar.set_postfix(
                    label=label,
                    threshold=f"{trial_t:.2f}",
                    best_f1=f"{best_f1:.3f}",
                    refresh=False,
                )
                pbar.update(1)

            thresholds[c] = best_t
            pbar.write(f"    {label:<3} threshold={best_t:.2f}  collapsed-d F1={best_f1:.3f}")

    preds = d_detections_from_file_probs(file_probs, thresholds, target_mode)
    final_metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
    print(f"    threshold tuning done in {time.time() - tune_start:.1f}s")
    return thresholds, final_metrics or best_metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    spec_extractor: SpectrogramExtractor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: np.ndarray,
    val_annotations: pd.DataFrame,
    file_start_dts: dict[tuple[str, str], object],
    target_mode: str,
    tune_thresholds: bool = True,
    track_bpn_mask: bool = False,
) -> dict:
    """Run validation and score the collapsed coarse d class."""
    model.eval()
    val_start = time.time()
    total_loss = 0.0
    n_batches = 0
    all_probs: dict[tuple[str, str, int], np.ndarray] = {}
    bpn_mask_total = 0.0
    bpn_mask_n = 0

    print(f"    validation inference over {len(loader)} batches")
    for audio, targets, mask, metas in tqdm(loader, desc="Validating", leave=True):
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        outputs = model(spec_extractor(audio))
        outputs, targets, mask = align_outputs(outputs, targets, mask)

        loss = criterion(outputs, targets, mask)
        total_loss += loss.item()
        n_batches += 1

        if track_bpn_mask and "mask" in outputs:
            valid_bpn_mask = outputs["mask"][mask]
            if valid_bpn_mask.numel() > 0:
                bpn_mask_total += valid_bpn_mask.mean().item()
                bpn_mask_n += 1

        probs = outputs["probs"].detach().float().cpu().numpy()
        hop = spec_extractor.hop_length
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samples = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samples // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    print(f"    validation inference done in {time.time() - val_start:.1f}s ({len(all_probs)} windows)")

    metric_start = time.time()
    gt_events = build_d_ground_truth(val_annotations, file_start_dts)
    if tune_thresholds:
        used_thresholds, metrics = tune_d_thresholds(all_probs, gt_events, initial_thresholds=thresholds, target_mode=target_mode)
    else:
        print("    scoring with existing thresholds")
        used_thresholds = thresholds
        preds = d_detections_from_probs(all_probs, used_thresholds, target_mode)
        metrics = compute_metrics(preds, gt_events, iou_threshold=0.3)
    print(f"    event scoring done in {time.time() - metric_start:.1f}s")

    d_metrics = metrics.get(D_COARSE_LABEL, {})
    print("\n  Collapsed D validation:")
    threshold_text = ", ".join(
        f"{label}={threshold:.2f}"
        for label, threshold in zip(target_labels(target_mode), used_thresholds)
    )
    print(f"    thresholds: {threshold_text}")
    print(f"    TP={d_metrics.get('tp', 0):5d} "
          f"FP={d_metrics.get('fp', 0):5d} "
          f"FN={d_metrics.get('fn', 0):5d}  "
          f"P={d_metrics.get('precision', 0.0):.3f} "
          f"R={d_metrics.get('recall', 0.0):.3f} "
          f"F1={d_metrics.get('f1', 0.0):.3f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "f1": d_metrics.get("f1", 0.0),
        "precision": d_metrics.get("precision", 0.0),
        "recall": d_metrics.get("recall", 0.0),
        "tp": d_metrics.get("tp", 0),
        "fp": d_metrics.get("fp", 0),
        "fn": d_metrics.get("fn", 0),
        "thresholds": used_thresholds,
        "bpn_mask_mean": (
            bpn_mask_total / bpn_mask_n
            if track_bpn_mask and bpn_mask_n > 0
            else None
        ),
    }


def train_epoch(
    model: torch.nn.Module,
    spec_extractor: SpectrogramExtractor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for audio, targets, mask, _ in pbar:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(spec_extractor(audio))
        outputs, targets, mask = align_outputs(outputs, targets, mask)
        loss = criterion(outputs, targets, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            print("  WARNING: NaN/Inf loss, skipping batch")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    best_f1: float,
    thresholds: np.ndarray,
    model_cfg: dict,
    segment_s: float,
    overlap_s: float,
    highpass_hz: float,
    target_mode: str,
    extra: dict | None = None,
) -> None:
    """Save model weights plus metadata needed for later ensembling."""
    model_state = (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
    payload = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "best_f1": float(best_f1),
        "thresholds": torch.tensor(thresholds, dtype=torch.float32),
        "target_mode": target_mode,
        "target_labels": target_labels(target_mode),
        "collapsed_label": D_COARSE_LABEL,
        "num_classes": len(target_labels(target_mode)),
        "segment_s": float(segment_s),
        "overlap_s": float(overlap_s),
        "highpass_hz": float(highpass_hz),
        "experiment": EXPERIMENT_NAME,
        "synthetic_calls": SYNTHETIC_CALLS_ENABLED,
    }
    payload.update(model_cfg)
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def init_wandb(args: argparse.Namespace, model_cfg: dict, run_dir: Path):
    """Start a plain wandb run for the dedicated D-detector experiment."""
    if args.no_wandb or args.wandb_mode == "disabled":
        return None

    import wandb
    import wandb_utils as wbu

    return wandb.init(
        entity=wbu.WANDB_ENTITY,
        project=wbu.WANDB_PROJECT,
        group=EXPERIMENT_NAME,
        job_type="d_detector_training",
        name=(f"d_detector_{args.arch}_{args.target_mode}__seed{args.seed}__{time.strftime('%m%d-%H%M')}"),
        mode=args.wandb_mode,
        tags=[
            "d_detector",
            f"{args.arch}_arch",
            f"{args.target_mode}_target",
            "collapsed_d_eval",
        ],
        config={
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "segment_s": args.segment_s,
            "overlap_s": args.overlap_s,
            "highpass_hz": args.highpass_hz,
            "val_every": args.val_every,
            "initial_threshold": args.train_threshold,
            "threshold_tune_every": args.threshold_tune_every,
            "neg_ratio": args.neg_ratio,
            "other_call_neg_ratio": args.other_call_neg_ratio,
            "target_mode": args.target_mode,
            "target_labels": target_labels(args.target_mode),
            "collapsed_label": D_COARSE_LABEL,
            "synthetic_calls": SYNTHETIC_CALLS_ENABLED,
            "model": model_cfg,
            "run_dir": str(run_dir),
        },
    )


def main() -> None:
    args = parse_args()
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")
    if args.threshold_tune_every < 0:
        raise ValueError("--threshold-tune-every must be >= 0")
    if not 0.0 <= args.train_threshold <= 1.0:
        raise ValueError("--train-threshold must be between 0 and 1")
    if args.segment_s <= 0:
        raise ValueError("--segment-s must be > 0")
    if args.overlap_s < 0 or args.overlap_s >= args.segment_s:
        raise ValueError("--overlap-s must be >= 0 and smaller than --segment-s")
    nyquist_hz = cfg.SAMPLE_RATE / 2.0
    if args.highpass_hz < 0 or args.highpass_hz >= nyquist_hz:
        raise ValueError(f"--highpass-hz must be >= 0 and < Nyquist ({nyquist_hz:.1f} Hz)")

    set_seed(args.seed)
    labels = target_labels(args.target_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or (f"d_detector_{args.arch}_{args.target_mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = Path(cfg.OUTPUT_DIR) / EXPERIMENT_NAME / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Architecture: {args.arch}")
    print(f"Target mode: {args.target_mode}")
    print(f"Output labels: {labels} -> {D_COARSE_LABEL}")
    print(f"Synthetic calls: {'enabled' if SYNTHETIC_CALLS_ENABLED else 'disabled'}")
    print(f"Batch size: {args.batch_size}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Segment length: {args.segment_s:.1f}s")
    if args.highpass_hz > 0:
        print(f"Audio high-pass: {args.highpass_hz:.1f} Hz")
    else:
        print("Audio high-pass: disabled")
    print(f"Validation every: {args.val_every} epoch(s)")
    print(f"Initial validation threshold: {args.train_threshold:.2f}")
    if args.threshold_tune_every > 0:
        print(f"Training-time threshold tuning every: {args.threshold_tune_every} validation(s)")
    else:
        print("Training-time threshold tuning: disabled; tuning once after training")

    train_ds, train_loader, train_ann, val_loader, val_ann, file_start_dts = build_dataloaders(args)
    pos_weight = (
        None
        if args.no_pos_weights
        else compute_target_pos_weights(train_ann, args.target_mode).to(device)
    )

    spec_extractor = SpectrogramExtractor().to(device)
    model, criterion, model_cfg = build_model_and_loss(args, n_outputs=len(labels), pos_weight=pos_weight, device=device)
    track_bpn_mask = bool(model_cfg.get("bpn_cfg_v2", {}).get("enabled", False))
    print(f"Model config: {model_cfg}")

    # The projection layer is lazy, so run one dummy window before checkpointing or loading weights.
    with torch.no_grad():
        dummy = torch.randn(1, int(cfg.SAMPLE_RATE * args.segment_s), device=device)
        model(spec_extractor(dummy))

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(cfg.BETA1, cfg.BETA2))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    wandb_run = init_wandb(args, model_cfg, run_dir)

    # initial values
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    thresholds = np.full(len(labels), args.train_threshold, dtype=np.float64)
    n_validations = 0

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}  LR={lr:.2e}")
        print(f"{'=' * 60}")

        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
            
        aux_enabled = (model_cfg.get("bpn_cfg_v2", {}).get("aux_loss", False))
        if aux_enabled and hasattr(criterion, "_aux_weight_now"):
            print(f"  Aux loss effective weight: {criterion._aux_weight_now():.4f}")

        if epoch > 1 and (epoch - 1) % RESAMPLE_EVERY == 0:
            print("  Resampling non-D hard negatives and background negatives")
            train_ds.resample_negatives()
            print(f"    non-D hard negatives: {len(train_ds.hard_negative_segments)}")
            print(f"    background negatives: {len(train_ds.background_negative_segments)}")
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        # run training iteration
        train_loss = train_epoch(model, spec_extractor, train_loader, criterion, optimizer, device, epoch)
        print(f"\n  Train loss: {train_loss:.4f}")

        
        should_validate = epoch % args.val_every == 0 or epoch == args.epochs
        val = None
        if should_validate:
            n_validations += 1
            tune_thresholds = (
                args.threshold_tune_every > 0
                and (
                    n_validations == 1
                    or n_validations % args.threshold_tune_every == 0
                    or epoch == args.epochs
                )
            )
            print("  Running validation")
            if tune_thresholds:
                print("  Training-time threshold tuning enabled for this validation")
            else:
                print(f"  Using fixed training threshold(s): {thresholds.tolist()}")
            val = validate(
                model, spec_extractor, val_loader, criterion, device,
                thresholds, val_ann, file_start_dts, args.target_mode,
                tune_thresholds=tune_thresholds,
                track_bpn_mask=track_bpn_mask,
            )
            thresholds = val["thresholds"]
            scheduler.step(val["f1"])

            print(f"  Val loss: {val['loss']:.4f}")
            print(f"  D F1: {val['f1']:.3f}  Best D F1: {best_f1:.3f}")
            if val["bpn_mask_mean"] is not None:
                print(f"  BPN mask mean: {val['bpn_mask_mean']:.3f}")
        else:
            print(f"  Skipping validation this epoch (--val-every {args.val_every})")

        if wandb_run is not None:
            import wandb
            print("  Logging metrics to wandb")
            log_data = {
                "epoch": epoch,
                "lr": lr,
                "train/loss": train_loss,
                "train/aux_weight_eff": (criterion._aux_weight_now() if aux_enabled and hasattr(criterion, "_aux_weight_now") else 0.0)
            }
            if val is not None:
                log_data.update({
                    "val/loss": val["loss"],
                    "val/f1/d": val["f1"],
                    "val/precision/d": val["precision"],
                    "val/recall/d": val["recall"],
                    "val/tp/d": val["tp"],
                    "val/fp/d": val["fp"],
                    "val/fn/d": val["fn"],
                })
                if val["bpn_mask_mean"] is not None:
                    log_data["val/bpn_mask_mean"] = val["bpn_mask_mean"]
                for label, threshold in zip(labels, thresholds):
                    log_data[f"val/threshold/{label}"] = float(threshold)
            wandb.log(log_data, step=epoch)
            print("  Wandb log done")

        print("  Saving latest checkpoint")
        save_checkpoint(
            run_dir / "latest_model.pt",
            model, epoch, best_f1, thresholds, model_cfg,
            args.segment_s, args.overlap_s, args.highpass_hz, args.target_mode,
        )
        print("  Latest checkpoint saved")

        if val is not None:
            if val["f1"] > best_f1 or best_epoch == 0:
                best_f1 = val["f1"]
                best_epoch = epoch
                no_improve = 0
                print("  Saving best checkpoint")
                save_checkpoint(
                    run_dir / "best_model.pt",
                    model, epoch, best_f1, thresholds, model_cfg,
                    args.segment_s, args.overlap_s, args.highpass_hz, args.target_mode,
                    extra={"val": val},
                )
                print(f"  *** New best D F1: {best_f1:.3f}")
            else:
                no_improve += 1
                print(f"  No improvement for {no_improve}/{EARLY_STOP_PATIENCE} validations")

            if no_improve >= EARLY_STOP_PATIENCE:
                print("\n  Early stopping")
                break

        print("  Epoch complete")

    # final threshold tuning after training
    final_thresholds = thresholds
    final_tuned_val = None
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        model_to_save.load_state_dict(best_ckpt["model_state_dict"])

        print("\nPost-training threshold tuning on best checkpoint")
        start_thresholds = best_ckpt["thresholds"].cpu().numpy()
        final_tuned_val = validate(
            model_to_save,
            spec_extractor,
            val_loader,
            criterion,
            device,
            start_thresholds,
            val_ann,
            file_start_dts,
            args.target_mode,
            tune_thresholds=True,
            track_bpn_mask=track_bpn_mask,
        )
        final_thresholds = final_tuned_val["thresholds"]

        tuned_path = run_dir / "tuned_thresholds.pt"
        torch.save({
            "thresholds": final_thresholds.tolist(),
            "initial_threshold": float(args.train_threshold),
            "best_training_f1": float(best_ckpt.get("best_f1", best_f1)),
            "tuned_f1": float(final_tuned_val["f1"]),
            "tuned_precision": float(final_tuned_val["precision"]),
            "tuned_recall": float(final_tuned_val["recall"]),
            "tuned_tp": int(final_tuned_val["tp"]),
            "tuned_fp": int(final_tuned_val["fp"]),
            "tuned_fn": int(final_tuned_val["fn"]),
            "highpass_hz": float(args.highpass_hz),
            "target_mode": args.target_mode,
            "target_labels": labels,
            "model": model_cfg,
        }, tuned_path)
        print(f"Saved tuned thresholds to {tuned_path}")

        save_checkpoint(
            run_dir / "final_model.pt",
            model_to_save,
            int(best_ckpt.get("epoch", best_epoch)),
            float(final_tuned_val["f1"]),
            final_thresholds,
            model_cfg,
            args.segment_s,
            args.overlap_s,
            args.highpass_hz,
            args.target_mode,
            extra={
                "selected_from": "best_model.pt",
                "best_training_f1": float(best_ckpt.get("best_f1", best_f1)),
                "post_tuned_val": final_tuned_val,
            },
        )

    # print final results
    print(f"\nDone. Best training D F1: {best_f1:.3f} at epoch {best_epoch}")
    if final_tuned_val is not None:
        print(f"Post-tuned D F1: {final_tuned_val['f1']:.3f}")
    final_threshold_text = ", ".join(
        f"{label}={threshold:.2f}"
        for label, threshold in zip(labels, final_thresholds)
    )
    print(f"Final thresholds: {final_threshold_text}")
    print(f"Run dir: {run_dir}")

    if wandb_run is not None:
        import wandb
        wandb.summary["best_f1/d"] = float(best_f1)
        wandb.summary["best_epoch"] = int(best_epoch)
        if final_tuned_val is not None:
            wandb.summary["post_tuned_f1/d"] = float(final_tuned_val["f1"])
            wandb.summary["post_tuned_precision/d"] = float(final_tuned_val["precision"])
            wandb.summary["post_tuned_recall/d"] = float(final_tuned_val["recall"])
            wandb.summary["post_tuned_fp/d"] = int(final_tuned_val["fp"])
        for label, threshold in zip(labels, final_thresholds):
            wandb.summary[f"final_threshold/{label}"] = float(threshold)
        if best_path.exists():
            artifact = wandb.Artifact(
                f"model-{wandb_run.name}",
                type="model",
                metadata={
                    "experiment": EXPERIMENT_NAME,
                    "best_training_f1_d": float(best_f1),
                    "post_tuned_f1_d": (
                        float(final_tuned_val["f1"])
                        if final_tuned_val is not None else None
                    ),
                    "target_mode": args.target_mode,
                    "target_labels": labels,
                    "collapsed_label": D_COARSE_LABEL,
                    "synthetic_calls": SYNTHETIC_CALLS_ENABLED,
                    "highpass_hz": float(args.highpass_hz),
                    "model": model_cfg,
                },
            )
            artifact.add_file(str(best_path))
            final_path = run_dir / "final_model.pt"
            if final_path.exists():
                artifact.add_file(str(final_path))
            tuned_path = run_dir / "tuned_thresholds.pt"
            if tuned_path.exists():
                artifact.add_file(str(tuned_path))
            wandb_run.log_artifact(artifact, aliases=["best", "d_detector"])
        wandb.finish()


if __name__ == "__main__":
    main()
