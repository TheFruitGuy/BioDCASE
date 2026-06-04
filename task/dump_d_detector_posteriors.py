"""
dump_d_detector_posteriors.py
=============================

Generate a ``best_posteriors.npz`` for a dedicated single-class D detector
(``train_d_detector.py`` / ``WhaleVADBPNV2``) in the exact format that
``postproc_search.py`` loads, so the detector can be added as a member of
the per-class-weighted ensemble.

train_d_detector.py only saves checkpoints + tuned thresholds; it never
writes the stitched posteriors npz that postproc_search expects. This
script fills that gap.

Format produced (matches train_hnm_final.save_posteriors /
rescore_base_epochs exactly)
----------------------------------------------------------------------
- ``pp.stitch_segments`` turns per-tile probs into per-file streams.
- npz keys are ``"{dataset}|{filename}"``.
- each value is a stitched per-file float16 array of shape ``(T_file, 3)``.
- channel order is ``cfg.CALL_TYPES_3`` = [bmabz, d, bp].

For the D detector, only the D channel (index 1) carries the detector's
probability. BMABZ and BP are written as zeros: in your postproc_search
command the detector gets weight 0 for those classes anyway, so the zeros
never contribute, and a per-class weighted average over them stays 0.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python dump_d_detector_posteriors.py \
        --checkpoint runs/phase_d_detector/d_detector_bpn_v2_merged_20260523_080641/best_model.pt

By default the npz is written next to the checkpoint, named
``best_posteriors.npz`` (the name postproc_search looks for first). Then:

    HNM=(runs/hnm_3class_s{42,2024,7777,9999,1337}_ens_nopgi/best_model.pt)
    MT=(runs/mtfb_3class_s{42,2024,7777,9999,1337}_ens_asymmetric_mse/best_model.pt)
    DDET=runs/phase_d_detector/d_detector_bpn_v2_merged_20260523_080641/best_model.pt
    python postproc_search.py \
        --checkpoints "${HNM[@]}" "${MT[@]}" "$DDET" \
        --weights-bmabz 1 1 1 1 1 0 0 0 0 0 0 \
        --weights-bp    1 1 1 1 1 0 0 0 0 0 0 \
        --weights-d     0 0 0 0 0 1 1 1 1 1 1

Notes
-----
- Run at the SAME eval tiling the other members used so the stitched
  per-file frame counts line up. The default (cfg.EVAL_SEGMENT_S /
  cfg.EVAL_OVERLAP_S = 30s / 2s) is the standard val tiling used by the
  training validation path that wrote the HNM/MT npz files. stitch is
  full-file-length regardless, so small tile-length differences usually
  do not matter, but matching avoids any boundary off-by-one.
- The npz is keyed by the run-DIRECTORY (postproc_search reads
  ``<run_dir>/best_posteriors.npz``), so pass a checkpoint and the file
  lands in its parent dir automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import postprocess as pp
from dataset import (
    WhaleDataset, build_val_segments, collate_fn,
    get_file_manifest, load_annotations,
)
from spectrogram import SpectrogramExtractor
from model_bpn_v2 import BPNConfigV2, WhaleVADBPNV2

D_IDX = cfg.CALL_TYPES_3.index("d")  # 1


# ----------------------------------------------------------------------
# Detector construction
# ----------------------------------------------------------------------

def build_d_detector(ckpt: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    """Rebuild the dedicated D detector from its checkpoint (bpn_v2 or whalevad)."""
    arch = ckpt.get("arch")
    n_out = int(ckpt.get("num_classes", len(ckpt.get("target_labels", ["d"]))))

    if arch == "bpn_v2" or "bpn_cfg_v2" in ckpt:
        if "bpn_cfg_v2" not in ckpt:
            raise KeyError("bpn_v2 checkpoint without 'bpn_cfg_v2'; cannot rebuild BPN config.")
        bpn_cfg = BPNConfigV2(**dict(ckpt["bpn_cfg_v2"]))
        return WhaleVADBPNV2(num_classes=n_out, bpn_cfg=bpn_cfg).to(device), "bpn_v2"

    if arch == "whalevad":
        from model import WhaleVAD
        return WhaleVAD(num_classes=n_out).to(device), "whalevad"

    raise ValueError(f"Unrecognised D-detector arch={arch!r}; expected train_d_detector.py output.")


# ----------------------------------------------------------------------
# High-pass (only if the detector trained with one)
# ----------------------------------------------------------------------

def _highpass_sos(highpass_hz: float) -> np.ndarray | None:
    if highpass_hz <= 0:
        return None
    return butter(N=4, Wn=float(highpass_hz), btype="highpass",
                  fs=cfg.SAMPLE_RATE, output="sos").astype(np.float32)


def _apply_highpass_batch(audio_np: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    if sos is None:
        return audio_np
    try:
        out = sosfiltfilt(sos, audio_np, axis=1)
    except ValueError:
        out = sosfilt(sos, audio_np, axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


# ----------------------------------------------------------------------
# Inference -> per-tile (T, 3) with D in column 1, zeros elsewhere
# ----------------------------------------------------------------------

@torch.no_grad()
def infer_d_as_3class(
    model: torch.nn.Module,
    kind: str,
    spec_extractor: SpectrogramExtractor,
    loader: DataLoader,
    device: torch.device,
    highpass_hz: float,
    target_mode: str,
) -> dict[tuple, np.ndarray]:
    """Per-tile probs keyed (dataset, filename, start_sample), shape (T, 3)."""
    model.eval()
    sos = _highpass_sos(highpass_hz)
    hop = spec_extractor.hop_length
    out: dict[tuple, np.ndarray] = {}

    for audio, _, _, metas in tqdm(loader, desc="  d-detector inference", leave=False):
        if sos is not None:
            audio = torch.from_numpy(_apply_highpass_batch(audio.cpu().numpy(), sos))
        audio = audio.to(device, non_blocking=True)
        result = model(spec_extractor(audio))
        probs = result["probs"] if kind == "bpn_v2" else torch.sigmoid(result)
        probs_np = probs.detach().float().cpu().numpy()

        for j, meta in enumerate(metas):
            n_samp = meta["end_sample"] - meta["start_sample"]
            nf = min(n_samp // hop, probs_np[j].shape[0])
            arr = probs_np[j, :nf, :]
            if target_mode == "subtype" and arr.shape[1] > 1:
                d_col = arr.max(axis=1)            # collapse bmd/bpd -> d
            else:
                d_col = arr[:, 0]                  # merged: column 0 is d
            three = np.zeros((nf, 3), dtype=np.float32)
            three[:, D_IDX] = d_col
            out[(meta["dataset"], meta["filename"], meta["start_sample"])] = three
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True,
                   help="D-detector checkpoint (e.g. .../best_model.pt or final_model.pt).")
    p.add_argument("--out-name", default="best_posteriors.npz",
                   help="npz filename written into the checkpoint's run dir. "
                        "Use 'paper_best_posteriors.npz' if that is what your "
                        "postproc_search --posterior-name expects.")
    p.add_argument("--out-path", default=None,
                   help="Explicit output path. Overrides --out-name / run-dir placement.")
    p.add_argument("--segment-s", type=float, default=cfg.EVAL_SEGMENT_S,
                   help="Eval tile length (s). Default matches the standard val tiling.")
    p.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S,
                   help="Eval tile overlap (s).")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out_path) if args.out_path else ckpt_path.parent / args.out_name

    # Val loader at the standard tiling (same as the HNM/MT members used).
    print(f"\nLoading val data  (tiles {args.segment_s:.0f}s / {args.overlap_s:.0f}s overlap)")
    val_anns = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments(
        val_manifest, val_anns,
        segment_s=args.segment_s, overlap_s=args.overlap_s,
    )
    loader = DataLoader(
        WhaleDataset(val_segs), batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    n_files = val_manifest[["dataset", "filename"]].drop_duplicates().shape[0]
    print(f"  {len(val_segs)} tiles over {n_files} files")

    # Build + load detector.
    print(f"\nLoading detector: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, kind = build_d_detector(ckpt, device)
    highpass_hz = float(ckpt.get("highpass_hz", 0.0))
    target_mode = ckpt.get("target_mode", "merged")
    print(f"  kind={kind}  target_mode={target_mode}  highpass_hz={highpass_hz:.1f}  "
          f"trained_segment_s={ckpt.get('segment_s')}")
    if highpass_hz > 0:
        print(f"  -> re-applying {highpass_hz:.1f} Hz high-pass to match training")

    spec_extractor = SpectrogramExtractor().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)
        _ = model(spec_extractor(dummy))  # materialise lazy feat_proj
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (showing 3): {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (showing 3): {unexpected[:3]}")

    # Inference -> per-tile (T,3) -> stitch -> save (exact train_hnm_final format).
    print("\nRunning inference...")
    probs3 = infer_d_as_3class(
        model, kind, spec_extractor, loader, device,
        highpass_hz=highpass_hz, target_mode=target_mode,
    )

    cfg.USE_3CLASS = True  # ensure stitch operates in 3-class space
    stitched = pp.stitch_segments(probs3)
    flat = {f"{ds}|{fn}": arr.astype(np.float16) for (ds, fn), arr in stitched.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **flat)

    # Sanity report.
    sample_key = next(iter(flat))
    sample = flat[sample_key]
    print(f"\nWrote {out_path}")
    print(f"  {len(flat)} files  (members expect 587)")
    print(f"  per-file array shape e.g. {sample_key}: {sample.shape}  dtype={sample.dtype}")
    d_max = max(float(a[:, D_IDX].max()) for a in flat.values())
    bmabz_max = max(float(a[:, 0].max()) for a in flat.values())
    bp_max = max(float(a[:, cfg.CALL_TYPES_3.index('bp')].max()) for a in flat.values())
    print(f"  channel maxima  bmabz={bmabz_max:.3f} (expect 0)  "
          f"d={d_max:.3f}  bp={bp_max:.3f} (expect 0)")
    if bmabz_max > 0 or bp_max > 0:
        print("  WARNING: bmabz/bp channels are non-zero; expected all zeros for a D detector.")
    if len(flat) != 587:
        print(f"  NOTE: file count {len(flat)} != 587 — confirm it matches the other members "
              f"or postproc_search's combine may drop/mismatch keys.")

    print("\nDone. Re-run your postproc_search command; the detector now loads like any member.")


if __name__ == "__main__":
    main()
