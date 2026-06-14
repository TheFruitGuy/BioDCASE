"""
dump_eval_posteriors.py — per-model eval posteriors for the submission
======================================================================

Runs the forward pass on a set of audio files and writes, into each checkpoint's
run dir, the file-level stitched 3-class posteriors that ``make_submission.py``
(and ``loso_eval.py``) consume: a float16 npz keyed ``"<dataset>|<filename.wav>"``
-> (T, 3).

It is your training-time ``validate()`` forward with the gt-dependent scoring
removed. It reuses your own ``build_model`` and ``collapse_to_3class`` (from
``rescore_base_epochs``) and ``stitch_segments`` (from ``postprocess_final``), so
the inference is identical to how ``best_posteriors*.npz`` was produced — only the
data split changes. ``n_classes`` is read from each checkpoint, exactly as the
predict path does, so 7-class and 3-class checkpoints are both handled.

VERIFY BEFORE TRUSTING IT ON EVAL (free, exact)
-----------------------------------------------
Point ``--eval-audio-root`` at your VALIDATION audio and dump to a scratch name,
then compare to the ``best_posteriors_60s.npz`` you already have::

    python dump_eval_posteriors.py \
        --checkpoints runs/hnm_3class_s42_ens_nopgi/best_model.pt \
        --eval-audio-root "$DATA_ROOT"/validation/audio \
        --segment-s 60 --out-name best_posteriors_60s_CHECK.npz

    python - <<'PY'
    import numpy as np
    a = np.load("runs/hnm_3class_s42_ens_nopgi/best_posteriors_60s.npz")
    b = np.load("runs/hnm_3class_s42_ens_nopgi/best_posteriors_60s_CHECK.npz")
    assert set(a.files) == set(b.files), "key mismatch"
    print("max abs diff:", max(np.abs(a[k].astype("f4") - b[k].astype("f4")).max() for k in a.files))
    PY

A max-abs-diff at the float16 noise floor (~1e-3) means the forward is faithful;
then run on the eval audio for real.

Usage (eval)
------------
::

    python dump_eval_posteriors.py \
        --checkpoints \
            runs/hnm_3class_s{42,2024,7777,9999,1337}_ens_nopgi/best_model.pt \
            runs/mtfb_3class_s{42,2024,7777,9999,1337}_solo_asymmetric_mse/best_model.pt \
        --eval-audio-root /path/to/zenodo_20485251/audio \
        --segment-s 60 --out-name eval_posteriors_60s.npz

``--eval-audio-root`` should be a directory whose immediate subdirectories are the
eval site-years (each holding ``*.wav``); if it directly contains wavs, the root's
own name is used as the single dataset. Filenames must follow the ATBFL pattern
``YYYY-MM-DDTHH-MM-SS_mmm.wav`` so the absolute timestamps resolve downstream.

GPU note (sxrk10): GPUs 2 and 7 are Blackwell (sm_120) and crash the cu118 build
at the LSTM — restrict with e.g. ``CUDA_VISIBLE_DEVICES=0`` or run on sxrk09.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_final as cfg
import postprocess_final as pp
from dataset_final import (
    WhaleDataset, collate_fn, build_val_segments, _parse_file_start_dt,
)

# server-side forward primitives — the exact ones training/validation use
from rescore_base_epochs import build_model, collapse_to_3class


def build_eval_manifest(audio_root: Path) -> pd.DataFrame:
    """One row per wav under ``audio_root``. Immediate subdirs are datasets;
    if there are none, ``audio_root`` itself is the single dataset."""
    audio_root = Path(audio_root)
    if not audio_root.exists():
        sys.exit(f"--eval-audio-root does not exist: {audio_root}")
    subdirs = [d for d in sorted(audio_root.iterdir()) if d.is_dir()]
    sites = subdirs if subdirs else [audio_root]
    rows = []
    for site in sites:
        for wav in sorted(site.glob("*.wav")):
            info = sf.info(str(wav))
            sdt = _parse_file_start_dt(wav.name)
            if sdt is None:
                print(f"  WARN: {wav.name} does not match the ATBFL datetime "
                      "pattern — its events will have no absolute timestamp.")
            rows.append({
                "dataset": site.name,
                "filename": wav.name,
                "path": str(wav),
                "duration_s": info.duration,
                "start_dt": sdt,
                "end_dt": sdt + timedelta(seconds=info.duration) if sdt else None,
            })
    if not rows:
        sys.exit(f"no .wav files found under {audio_root}")
    df = pd.DataFrame(rows)
    print(f"  manifest: {len(df)} files across {df.dataset.nunique()} site(s): "
          f"{', '.join(sorted(df.dataset.unique()))}")
    return df


@torch.no_grad()
def forward_file_probs(model, spec, loader, device):
    """probs[(ds, fn, start_sample)] = (n_frames, n_classes). Copy of the
    forward half of validate(): model(spec(audio)) -> sigmoid -> per-segment."""
    model.eval()
    hop = spec.hop_length
    probs = {}
    for audio, _targets, _mask, metas in tqdm(loader, desc="fwd", leave=False):
        audio = audio.to(device, non_blocking=True)
        logits = model(spec(audio))
        p = torch.sigmoid(logits).cpu().numpy()
        for j, m in enumerate(metas):
            nf = min((m["end_sample"] - m["start_sample"]) // hop, p[j].shape[0])
            probs[(m["dataset"], m["filename"], m["start_sample"])] = p[j, :nf, :]
    return probs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--eval-audio-root", required=True,
                    help="dir of per-site wav subdirs (zenodo eval audio, or "
                         "$DATA_ROOT/validation/audio for the verification check)")
    ap.add_argument("--segment-s", type=float, default=60.0)
    ap.add_argument("--overlap-s", type=float, default=cfg.EVAL_OVERLAP_S)
    ap.add_argument("--out-name", default="eval_posteriors_60s.npz",
                    help="npz filename written into each checkpoint's run dir")
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  window {args.segment_s}s  overlap {args.overlap_s}s")

    manifest = build_eval_manifest(Path(args.eval_audio_root))

    # eval has no labels; build_val_segments only uses annotations for the
    # is_positive flag, which the forward ignores. An empty frame -> no anns.
    empty_anns = pd.DataFrame(columns=["dataset", "filename", "annotation",
                                       "start_s", "end_s"])
    segments = build_val_segments(manifest, empty_anns,
                                  segment_s=args.segment_s, overlap_s=args.overlap_s)
    if not segments:
        sys.exit("no segments — are the files shorter than --segment-s?")
    print(f"  tiled into {len(segments)} windows")
    loader = DataLoader(WhaleDataset(segments), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=True)

    for ck in args.checkpoints:
        ck = Path(ck)
        run_dir = ck.parent
        if not ck.exists():
            print(f"  SKIP missing checkpoint: {ck}")
            continue
        print(f"\n[{run_dir.name}]")
        ckpt = torch.load(ck, map_location=device, weights_only=False)
        n_classes = int(ckpt["model_state_dict"]["classifier.weight"].shape[0])
        cfg.USE_3CLASS = (n_classes == 3)          # mirror validate() before forward
        model, spec = build_model(n_classes, device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        probs = forward_file_probs(model, spec, loader, device)
        probs3 = collapse_to_3class(probs, n_classes)   # 7->3 pool; ends USE_3CLASS=True
        stitched = pp.stitch_segments(probs3)
        flat = {f"{ds}|{fn}": arr.astype(np.float16)
                for (ds, fn), arr in stitched.items()}
        out = run_dir / args.out_name
        np.savez_compressed(out, **flat)
        print(f"  wrote {out}  ({len(flat)} files, {n_classes}-class ckpt)")

        del model, spec
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nDone. Verify against best_posteriors_60s.npz on the val audio "
          "(see header) before the real eval run, then feed --out-name to "
          "make_submission.py.")


if __name__ == "__main__":
    main()
