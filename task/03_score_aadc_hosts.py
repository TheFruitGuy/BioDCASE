"""
Stage 3: Score AADC Host Clips with an Ensemble (Optional but Recommended)
==========================================================================

Runs a trained WhaleVAD (or WhaleVAD-BPN) **ensemble** over every clip
in ``aadc_clips.pt`` and saves the per-clip maximum probability across
all frames and all classes. Stage 4 uses these scores to filter the
host pool down to high-confidence no-call regions.

Multi-checkpoint mode
---------------------
Pass multiple ``--checkpoints`` and the script averages their per-frame
sigmoid outputs (optionally weighted) before computing the per-clip
max. This gives you ensemble-quality filtering — strictly better than
any single checkpoint at distinguishing call from no-call, which is
exactly what we need for cleaning the splice host pool.

The script auto-detects each checkpoint's architecture (WhaleVAD vs
WhaleVADBPN) via the same ``build_model_for_ckpt`` helper used by
``ensemble_predict.py``. So you can mix baseline and BPN checkpoints
in a single ensemble — no per-checkpoint adapter needed.

Output schema
-------------
``aadc_scores.pt`` is a torch-pickled dict::

    {
        "clip_scores": [
            {
                "clip_idx":      int       (index into aadc_clips["clips"])
                "max_p_overall": float     (max over all frames AND classes
                                            of the averaged ensemble probs)
                "max_p_per_class": dict    ({"bmabz": float, "d": ..., "bp": ...})
                "mean_p_overall": float
                "n_high_p_frames": int     (frames where max(p) > 0.5)
            },
            ...
        ],
        "config": {
            "checkpoints":  list[str],
            "weights":      list[float],
            "n_clips":      int,
            "n_classes":    int,
        },
    }

Usage
-----
Single checkpoint::

    python 03_score_aadc_hosts.py \\
        --aadc_clips aadc_clips.pt \\
        --checkpoints runs/best_baseline_seed1337/best_model.pt \\
        --out aadc_scores.pt

Your F1=0.518 four-model ensemble (same checkpoints + weights as
hybrid_ensemble_predict.py)::

    python 03_score_aadc_hosts.py \\
        --aadc_clips aadc_clips.pt \\
        --checkpoints \\
            runs/whalevad_<ts1>/best_model.pt \\
            runs/phase5_<ts1>/best_model.pt \\
            runs/whalevad_<ts2>/best_model.pt \\
            runs/phase5_<ts2>/best_model.pt \\
        --weights 1 1 1 2 \\
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
from spectrogram import SpectrogramExtractor  # noqa: E402

# Reuse the project's checkpoint-type detector + builder so we don't
# duplicate the BPN config reconstruction logic.
from ensemble_predict import build_model_for_ckpt, detect_model_type  # noqa: E402


# ----------------------------------------------------------------------
# In-memory dataset wrapping the AADC clip list
# ----------------------------------------------------------------------

class AADCClipDataset(torch.utils.data.Dataset):
    """Trivial dataset: yields ``(audio, clip_idx)`` for index-aware writeback."""

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
# Checkpoint loading
# ----------------------------------------------------------------------

def _load_checkpoint_to_model(ckpt_path: Path, device: str):
    """
    Load one checkpoint via the project's auto-detecting builder.

    Mirrors what ensemble_predict.py does: ``torch.load`` → inspect
    saved keys → ``build_model_for_ckpt`` constructs the right
    architecture (WhaleVAD or WhaleVADBPN) and runs a dummy forward
    pass to instantiate lazy modules → ``load_state_dict(strict=False)``
    to tolerate minor key drift.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model, model_type = build_model_for_ckpt(ckpt, torch.device(device))

    sd = (ckpt.get("model_state_dict")
          or ckpt.get("state_dict")
          or {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)})

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt_path.name} ({model_type}): "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, model_type


# ----------------------------------------------------------------------
# Scoring loop
# ----------------------------------------------------------------------

def score_clips(
    aadc_clips_path: Path,
    checkpoint_paths: list[Path],
    weights: list[float] | None,
    out_path: Path,
    batch_size: int,
    device: str,
) -> None:
    """Run inference with all checkpoints, average sigmoid outputs, save scores."""
    print(f"Loading AADC clip pool from {aadc_clips_path}...")
    pool = torch.load(aadc_clips_path, map_location="cpu")
    clips = pool["clips"]
    print(f"  {len(clips)} clips to score\n")

    # Normalize weights so they sum to 1.0 → averaged prob stays in [0,1].
    n_ckpts = len(checkpoint_paths)
    if weights is None:
        weights = [1.0 / n_ckpts] * n_ckpts
    else:
        if len(weights) != n_ckpts:
            raise SystemExit(
                f"--weights has {len(weights)} entries but "
                f"--checkpoints has {n_ckpts}"
            )
        total = float(sum(weights))
        weights = [w / total for w in weights]
    print(f"Ensemble of {n_ckpts} checkpoint(s), normalized weights: "
          f"{[f'{w:.3f}' for w in weights]}\n")

    spec_extractor = SpectrogramExtractor().to(device)

    print("Loading checkpoints...")
    models: list[tuple[torch.nn.Module, str]] = []
    n_classes = None
    for ckpt_path in checkpoint_paths:
        model, model_type = _load_checkpoint_to_model(ckpt_path, device)
        models.append((model, model_type))
        # Probe class count via a tiny forward pass.
        with torch.no_grad():
            dummy_audio = torch.zeros(1, cfg.SAMPLE_RATE * 30, device=device)
            dummy_spec = spec_extractor(dummy_audio)
            dummy_out = model(dummy_spec)
            if isinstance(dummy_out, dict):
                dummy_out = dummy_out["logits"]
            nc = int(dummy_out.size(-1))
        if n_classes is None:
            n_classes = nc
        elif n_classes != nc:
            raise SystemExit(
                f"Checkpoints have mismatched class counts ({n_classes} vs "
                f"{nc} for {ckpt_path}). Mixing 3-class and 4-class unsupported."
            )
    print(f"  All {n_ckpts} checkpoint(s) loaded with num_classes={n_classes}\n")

    class_names = ["bmabz", "d", "bp"][:n_classes]

    dataset = AADCClipDataset(clips)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=_collate, pin_memory=(device != "cpu"),
    )

    clip_scores: list[dict | None] = [None] * len(clips)

    with torch.no_grad():
        for audio, idxs in tqdm(loader, desc="Scoring"):
            audio = audio.to(device, non_blocking=True)
            spec = spec_extractor(audio)

            # Weighted ensemble of sigmoid outputs.
            ens_probs = None
            for (model, _mtype), w in zip(models, weights):
                out = model(spec)
                if isinstance(out, dict):
                    out = out["logits"]
                probs = torch.sigmoid(out)            # (B, T, C)
                ens_probs = (w * probs) if ens_probs is None else (ens_probs + w * probs)

            # Per-clip summaries on the averaged probs.
            max_pc = ens_probs.amax(dim=1)              # (B, C)
            max_p_all, _ = max_pc.max(dim=1)            # (B,)
            mean_p_all = ens_probs.mean(dim=(1, 2))     # (B,)
            n_high = (ens_probs.amax(dim=2) > 0.5).sum(dim=1)

            for i, clip_idx in enumerate(idxs.tolist()):
                per_class = {
                    class_names[c]: float(max_pc[i, c].item())
                    for c in range(ens_probs.size(2))
                }
                clip_scores[clip_idx] = {
                    "clip_idx":        clip_idx,
                    "max_p_overall":   float(max_p_all[i].item()),
                    "max_p_per_class": per_class,
                    "mean_p_overall":  float(mean_p_all[i].item()),
                    "n_high_p_frames": int(n_high[i].item()),
                }

    # Diagnostics.
    p_values = [s["max_p_overall"] for s in clip_scores if s is not None]
    p_tensor = torch.tensor(p_values)
    print()
    print("Score distribution (ensemble max p over each 30s clip):")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        print(f"  p{int(q*100):>2}: {p_tensor.quantile(q).item():.3f}")
    print()
    for thr in [0.05, 0.1, 0.2, 0.5, 0.7]:
        n_below = (p_tensor < thr).sum().item()
        frac = n_below / len(p_tensor)
        print(f"  fraction of clips with max_p < {thr}:  "
              f"{frac:.1%} ({n_below}/{len(p_tensor)})")

    out = {
        "clip_scores": clip_scores,
        "config": {
            "checkpoints": [str(p) for p in checkpoint_paths],
            "weights":     weights,
            "n_clips":     len(clips),
            "n_classes":   n_classes,
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
        "--checkpoints", type=Path, nargs="+", required=True,
        help="One or more trained WhaleVAD / WhaleVADBPN checkpoints. "
             "Multiple → ensemble (averaged sigmoid outputs).",
    )
    p.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="Optional per-checkpoint weights (same length as --checkpoints). "
             "Will be normalized to sum to 1.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("aadc_scores.pt"),
        help="Destination .pt file.",
    )
    p.add_argument(
        "--batch_size", type=int, default=cfg.BATCH_SIZE,
        help="Inference batch size. Lower if you hit OOM with many checkpoints.",
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
        checkpoint_paths=list(args.checkpoints),
        weights=args.weights,
        out_path=args.out,
        batch_size=args.batch_size,
        device=args.device,
    )
