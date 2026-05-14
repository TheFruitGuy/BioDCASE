"""
LEAF vs STFT frontend sanity check.

Run this BEFORE launching a full training run. Verifies that:

1. ``LeafFrontend`` output has the right shape for ``WhaleVAD(feat_channels=1)``.
2. At random init, LEAF features look like a sensible time-frequency
   representation (not all zeros, not pure noise) on real whale audio.
3. Gradients flow into the Gabor parameters.
4. (Optional) After a short trial-train, the Gabor filters actually move
   from their initialization — guards against the documented "LEAF didn't
   learn" failure mode from Anderson, Kinnunen & Harte (ICASSP 2023).

Usage
-----
::

    # Just plot and check shapes
    python check_leaf.py

    # Plot, plus brief grad-descent to verify filters move
    python check_leaf.py --train_steps 200

    # Compare against mel init instead of linear
    python check_leaf.py --init mel
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from dataset import build_dataloaders
from spectrogram import SpectrogramExtractor
from leaf_frontend import LeafFrontend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_segments", type=int, default=4)
    ap.add_argument("--n_filters", type=int, default=128)
    ap.add_argument("--init", choices=["linear", "mel"], default="linear")
    ap.add_argument("--use_pcen", action="store_true")
    ap.add_argument("--out", type=str, default="leaf_sanity.png")
    ap.add_argument(
        "--train_steps", type=int, default=0,
        help="If >0, run a brief auxiliary training loop and re-plot."
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Real batch from the training pipeline — same dataset code as your
    # actual trainer uses.
    _, train_loader, _ = build_dataloaders()
    batch = next(iter(train_loader))
    audio = batch[0][:args.n_segments].to(device)
    print(f"Audio batch: {audio.shape}")

    # Build both frontends
    stft = SpectrogramExtractor().to(device)
    leaf = LeafFrontend(
        n_filters=args.n_filters,
        init=args.init,
        use_pcen=args.use_pcen,
    ).to(device)

    with torch.no_grad():
        stft_feat = stft(audio)
        leaf_feat = leaf(audio)

    print(f"\nShape check:")
    print(f"  STFT: {tuple(stft_feat.shape)} (expected (B, 3, 129, ~1449))")
    print(f"  LEAF: {tuple(leaf_feat.shape)} "
          f"(expected (B, 1, {args.n_filters}, ~similar T))")

    print(f"\nLEAF feature stats:")
    print(f"  min={leaf_feat.min():.4f}  max={leaf_feat.max():.4f}  "
          f"mean={leaf_feat.mean():.4f}  std={leaf_feat.std():.4f}")
    if leaf_feat.abs().max() < 1e-6:
        print("  WARNING: LEAF features are ~zero. Something is wrong "
              "with init or sample-rate config.")

    # Gradient-flow check
    loss = leaf_feat.pow(2).mean()
    loss.backward()
    grad_ok = 0
    grad_total = 0
    for name, p in leaf.named_parameters():
        if not p.requires_grad:
            continue
        grad_total += 1
        if p.grad is not None and p.grad.abs().max() > 0:
            grad_ok += 1
    print(f"\nGradient flow: {grad_ok}/{grad_total} LEAF params received "
          f"non-zero gradients")
    leaf.zero_grad()

    # Optional: brief trial training to verify filters actually move.
    if args.train_steps > 0:
        print(f"\nTrial-training LEAF for {args.train_steps} steps "
              "(predict input RMS energy from mean LEAF activation)...")
        # Find the Gabor parameter tensor to snapshot before/after.
        gabor_param = None
        for name, p in leaf.named_parameters():
            if "kernel" in name.lower() and "complex_conv" in name.lower():
                gabor_param = p
                break
        if gabor_param is None:
            print("  Could not locate Gabor param to snapshot; skipping.")
        else:
            before = gabor_param.detach().clone()
            optim = torch.optim.AdamW(leaf.parameters(), lr=1e-3)
            for step in range(args.train_steps):
                feat = leaf(audio)
                pred = feat.mean(dim=(1, 2, 3))
                target = audio.pow(2).mean(dim=-1).sqrt()
                target = (target - target.mean()) / (target.std() + 1e-8)
                pred = (pred - pred.mean()) / (pred.std() + 1e-8)
                loss = torch.nn.functional.mse_loss(pred, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
            after = gabor_param.detach().clone()
            d_mu = (after[:, 0] - before[:, 0]).abs()
            d_sigma = (after[:, 1] - before[:, 1]).abs()
            sr = leaf.sample_rate
            print(f"  Center freq drift: mean={d_mu.mean()*sr:.3f} Hz, "
                  f"max={d_mu.max()*sr:.3f} Hz")
            print(f"  Bandwidth drift:   mean={d_sigma.mean()*sr:.3f} Hz, "
                  f"max={d_sigma.max()*sr:.3f} Hz")
            print(f"  Centers before (first 5, Hz): "
                  f"{(before[:5, 0]*sr).cpu().numpy().round(2)}")
            print(f"  Centers after  (first 5, Hz): "
                  f"{(after[:5, 0]*sr).cpu().numpy().round(2)}")
            with torch.no_grad():
                leaf_feat = leaf(audio)

    # Side-by-side plot: STFT mag (top) vs LEAF (bottom)
    fig, axes = plt.subplots(2, args.n_segments,
                             figsize=(4 * args.n_segments, 6))
    if args.n_segments == 1:
        axes = axes.reshape(2, 1)
    for i in range(args.n_segments):
        stft_mag = stft_feat[i, 0].cpu().numpy()
        leaf_mag = leaf_feat[i, 0].cpu().numpy()
        axes[0, i].imshow(np.log1p(stft_mag), aspect="auto", origin="lower")
        axes[0, i].set_title(f"STFT magnitude (seg {i})")
        axes[0, i].set_ylabel("freq bin (0..125 Hz)")
        axes[1, i].imshow(np.log1p(leaf_mag), aspect="auto", origin="lower")
        axes[1, i].set_title(f"LEAF ({args.init} init) — seg {i}")
        axes[1, i].set_ylabel("filter idx (low→high freq)")
        axes[1, i].set_xlabel("frame (20 ms)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main()
