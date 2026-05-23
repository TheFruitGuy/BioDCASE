"""
LEAF vs STFT frontend sanity check.

Run BEFORE launching training. Verifies:
1. ``LeafFrontend`` output shape is right for ``WhaleVAD(feat_channels=1)``.
2. LEAF features actually vary with input (no degenerate constant output).
3. Gradients flow into the Gabor parameters.
4. (Optional) Filters actually move from init during a short trial-train.

Usage
-----
::

    python check_leaf.py                     # linear init (default)
    python check_leaf.py --init mel          # compare with mel init
    python check_leaf.py --train_steps 200   # confirm filters move
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

    # Load a real batch.
    _, train_loader, _ = build_dataloaders()
    batch = next(iter(train_loader))
    audio = batch[0][:args.n_segments].to(device)
    print(f"Audio batch shape: {audio.shape}")
    print(f"Audio stats: min={audio.min():.4f} max={audio.max():.4f} "
          f"mean={audio.mean():.4f} std={audio.std():.4f}")

    # Build both frontends.
    stft = SpectrogramExtractor().to(device)
    leaf = LeafFrontend(
        n_filters=args.n_filters,
        init=args.init,
        use_pcen=args.use_pcen,
    ).to(device)

    # ------------------------------------------------------------------
    # Diagnostic: dump the LEAF module structure and key submodules.
    # If the output is degenerate (mean≈1, std≈0), this often reveals
    # which submodule is misbehaving.
    # ------------------------------------------------------------------
    print(f"\nLEAF module structure:")
    for name, mod in leaf.leaf.named_children():
        print(f"  leaf.{name}: {type(mod).__name__}")
    print(f"\nLEAF compression attribute: "
          f"{getattr(leaf.leaf, 'compression', 'NOT FOUND')}")

    # ------------------------------------------------------------------
    # Forward pass — keep gradients enabled so we can run the gradient
    # check below using the same tensor.
    # ------------------------------------------------------------------
    with torch.no_grad():
        stft_feat = stft(audio)
    leaf_feat = leaf(audio)   # NOTE: outside no_grad so .backward() works

    print(f"\nShape check:")
    print(f"  STFT: {tuple(stft_feat.shape)} (expected (B, 3, 129, ~T))")
    print(f"  LEAF: {tuple(leaf_feat.shape)} "
          f"(expected (B, 1, {args.n_filters}, ~T))")

    print(f"\nLEAF feature stats:")
    print(f"  min={leaf_feat.min().item():.6f}  "
          f"max={leaf_feat.max().item():.6f}  "
          f"mean={leaf_feat.mean().item():.6f}  "
          f"std={leaf_feat.std().item():.6f}")
    # Per-filter variance — if all near zero, filters are essentially
    # producing the same constant for every input.
    per_filter_std = leaf_feat[0, 0].std(dim=-1).cpu()  # std over time per filter
    print(f"  per-filter std across time: "
          f"min={per_filter_std.min().item():.6f} "
          f"max={per_filter_std.max().item():.6f} "
          f"mean={per_filter_std.mean().item():.6f}")
    if per_filter_std.max() < 1e-4:
        print("  ** WARNING: LEAF features barely vary with time. "
              "Filters likely degenerate. **")

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------
    loss = leaf_feat.pow(2).mean()
    loss.backward()
    grad_ok = grad_total = 0
    for name, p in leaf.named_parameters():
        if not p.requires_grad:
            continue
        grad_total += 1
        if p.grad is not None and p.grad.abs().max() > 0:
            grad_ok += 1
    print(f"\nGradient flow: {grad_ok}/{grad_total} LEAF params received "
          f"non-zero gradients")
    leaf.zero_grad()

    # ------------------------------------------------------------------
    # Optional: brief auxiliary training
    # ------------------------------------------------------------------
    if args.train_steps > 0:
        print(f"\nTrial-training LEAF for {args.train_steps} steps "
              "(MSE between mean LEAF activation and audio RMS)...")
        try:
            gabor_param = leaf._gabor_param()
        except Exception as e:
            print(f"  Could not locate Gabor param: {e}; skipping.")
            gabor_param = None

        if gabor_param is not None:
            before = gabor_param.detach().clone()
            optim = torch.optim.AdamW(leaf.parameters(), lr=1e-3)
            for step in range(args.train_steps):
                feat = leaf(audio)
                pred = feat.mean(dim=(1, 2, 3))
                target = audio.pow(2).mean(dim=-1).sqrt()
                pred_n = (pred - pred.mean()) / (pred.std() + 1e-8)
                target_n = (target - target.mean()) / (target.std() + 1e-8)
                loss = torch.nn.functional.mse_loss(pred_n, target_n)
                optim.zero_grad()
                loss.backward()
                optim.step()
            after = gabor_param.detach().clone()
            sr = leaf.sample_rate
            # Convert mu drift to Hz assuming angular-radian convention.
            d_mu_rad = (after[:, 0] - before[:, 0]).abs()
            d_mu_hz = d_mu_rad / torch.pi * (sr / 2)
            d_sigma_rad = (after[:, 1] - before[:, 1]).abs()
            print(f"  mu (center freq) drift: "
                  f"mean={d_mu_hz.mean():.4f} Hz, "
                  f"max={d_mu_hz.max():.4f} Hz")
            print(f"  sigma drift (raw): "
                  f"mean={d_sigma_rad.mean():.6f}, "
                  f"max={d_sigma_rad.max():.6f}")
            # Re-forward with trained leaf for the plot
            with torch.no_grad():
                leaf_feat = leaf(audio)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, args.n_segments,
                             figsize=(4 * args.n_segments, 6))
    if args.n_segments == 1:
        axes = axes.reshape(2, 1)
    for i in range(args.n_segments):
        stft_mag = stft_feat[i, 0].detach().cpu().numpy()
        leaf_mag = leaf_feat[i, 0].detach().cpu().numpy()
        axes[0, i].imshow(np.log1p(stft_mag), aspect="auto", origin="lower")
        axes[0, i].set_title(f"STFT magnitude (seg {i})")
        axes[0, i].set_ylabel("freq bin (0..125 Hz)")
        axes[1, i].imshow(np.log1p(leaf_mag), aspect="auto", origin="lower")
        axes[1, i].set_title(f"LEAF ({args.init} init) — seg {i}")
        axes[1, i].set_ylabel("filter idx (low→high freq)")
        axes[1, i].set_xlabel("frame")
    plt.tight_layout()
    plt.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main()
