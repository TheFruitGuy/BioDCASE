"""
Stage 3: Score AADC Host Clips (Optional but Recommended)
==========================================================

Runs a trained WhaleVAD checkpoint over every clip in ``aadc_clips.pt``
and saves the maximum per-frame, per-class probability for each clip.
Stage 4 uses these scores to filter the host pool down to
high-confidence no-call regions, which serves two purposes:

  1. Filters out the rare AADC clips that contain unannotated whale
     activity (the AADC archive has no exhaustive labels — most clips
     are no-call but not all are).
  2. Provides a cleaner training signal: when we plant a call into a
     host the model already classifies as no-call, the only positive
     in the segment is the one we planted, so the supervision is
     unambiguous.

This script is **optional**. Without it, stage 4 will use random
windows from ``aadc_clips.pt``. The base-rate noise level from
unfiltered AADC clips is probably tolerable, but filtering is a
cheap safety net.

The scoring model
-----------------
Pass any WhaleVAD checkpoint via ``--checkpoint``. Two natural choices:

  - The best multi-seed baseline (F1 ≈ 0.47). This is probably the
    most stable choice — well-calibrated, low FP rate.
  - The official Geldenhuys checkpoint (loaded via the remapping in
    ``load_official_checkpoint.py``). Slightly lower F1 but trained
    on the same data, so its confidence calibration should be similar.

Either works. Stick with one across the pipeline so the threshold
in stage 4 means the same thing each run.

Output schema
-------------
``aadc_scores.pt`` is a torch-pickled dict::

    {
        "clip_scores": [
            {
                "clip_idx":      int       (index into aadc_clips["clips"])
                "max_p_overall": float     (max over all frames AND classes)
                "max_p_per_class": dict    ({"bmabz": float, "d": ..., "bp": ...})
                "mean_p_overall": float    (sanity-check, usually << max)
                "n_high_p_frames": int     (frames where max(p) > 0.5)
            },
            ...
        ],
        "config": {
            "checkpoint":    str,
            "n_clips":       int,
            "n_classes":     int,
        },
    }

Usage
-----
::

    python 03_score_aadc_hosts.py \\
        --aadc_clips aadc_clips.pt \\
        --checkpoint runs/best_baseline.pt \\
        --out aadc_scores.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from model import WhaleVAD  # noqa: E402
from spectrogram import SpectrogramExtractor  # noqa: E402


# ----------------------------------------------------------------------
# In-memory dataset wrapping the AADC clip list
# ----------------------------------------------------------------------

class AADCClipDataset(torch.utils.data.Dataset):
    """
    Trivial dataset over the pre-extracted host clips.

    Returns ``(audio, clip_idx)`` so we can write scores back to the
    original ordering after batched inference.
    """

    def __init__(self, clips: list[dict]):
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.clips[idx]["audio"], idx


def _collate(batch):
    audio = torch.stack([b[0] for b in batch], dim=0)
    idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return audio, idxs


# ----------------------------------------------------------------------
# Scoring loop
# ----------------------------------------------------------------------

def score_clips(
    aadc_clips_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    batch_size: int,
    device: str,
) -> None:
    """Run inference over the pool, save per-clip score summaries."""
    print(f"Loading AADC clip pool from {aadc_clips_path}...")
    pool = torch.load(aadc_clips_path, map_location="cpu")
    clips = pool["clips"]
    print(f"  {len(clips)} clips to score\n")

    print(f"Loading checkpoint from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
        n_classes = state.get("num_classes", 3)
    elif "state_dict" in state:
        sd = state["state_dict"]
        n_classes = 3
    else:
        sd = state
        n_classes = 3
    print(f"  Building WhaleVAD with num_classes={n_classes}")

    model = WhaleVAD(num_classes=n_classes).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING: missing keys at load: {len(missing)}")
    if unexpected:
        print(f"  WARNING: unexpected keys at load: {len(unexpected)}")
    model.eval()

    spec_ext = SpectrogramExtractor().to(device)

    # 3-class collapse name list (matches phase_splice training).
    class_names = ["bmabz", "d", "bp"][:n_classes]

    dataset = AADCClipDataset(clips)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=_collate, pin_memory=(device != "cpu"),
    )

    clip_scores: list[dict] = [None] * len(clips)  # filled by index

    with torch.no_grad():
        for audio, idxs in tqdm(loader, desc="Scoring"):
            audio = audio.to(device, non_blocking=True)
            spec = spec_ext(audio)
            logits = model(spec)            # (B, T, C)
            probs = torch.sigmoid(logits)   # (B, T, C)

            # Per-clip summaries.
            max_pc = probs.amax(dim=1)              # (B, C) — over time
            max_p_all, _ = max_pc.max(dim=1)        # (B,)   — over classes
            mean_p_all = probs.mean(dim=(1, 2))     # (B,)
            n_high = (probs.amax(dim=2) > 0.5).sum(dim=1)  # (B,) frame count

            for i, clip_idx in enumerate(idxs.tolist()):
                per_class = {
                    class_names[c]: float(max_pc[i, c].item())
                    for c in range(probs.size(2))
                }
                clip_scores[clip_idx] = {
                    "clip_idx":        clip_idx,
                    "max_p_overall":   float(max_p_all[i].item()),
                    "max_p_per_class": per_class,
                    "mean_p_overall":  float(mean_p_all[i].item()),
                    "n_high_p_frames": int(n_high[i].item()),
                }

    # Diagnostics.
    p_values = [s["max_p_overall"] for s in clip_scores]
    p_tensor = torch.tensor(p_values)
    print()
    print("Score distribution (max p over each 30s clip, over all classes):")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        print(f"  p{int(q*100):>2}: {p_tensor.quantile(q).item():.3f}")
    for thr in [0.05, 0.1, 0.2, 0.5, 0.7]:
        n_below = (p_tensor < thr).sum().item()
        frac = n_below / len(p_tensor)
        print(f"  fraction of clips with max_p < {thr}:  "
              f"{frac:.1%} ({n_below}/{len(p_tensor)})")

    out = {
        "clip_scores": clip_scores,
        "config": {
            "checkpoint": str(checkpoint_path),
            "n_clips":    len(clips),
            "n_classes":  n_classes,
        },
    }
    torch.save(out, out_path)
    print(f"\nWrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--aadc_clips", type=Path, default=Path("aadc_clips.pt"),
        help="Pool from stage 2.",
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Trained WhaleVAD checkpoint to use for scoring.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("aadc_scores.pt"),
        help="Destination .pt file.",
    )
    p.add_argument(
        "--batch_size", type=int, default=cfg.BATCH_SIZE,
        help="Inference batch size.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_clips(
        aadc_clips_path=args.aadc_clips,
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        batch_size=args.batch_size,
        device=args.device,
    )
