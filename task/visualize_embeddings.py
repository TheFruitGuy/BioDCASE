"""
Visualize learned embeddings with UMAP — for your project write-up (Chapter 3).

Extracts embeddings from the Conformer encoder on labeled validation data,
then plots UMAP showing how well call types cluster.

Run this twice:
  1. With random init weights → "before pretraining" 
  2. With pretrained weights → "after pretraining"

The comparison shows what the model learned from unlabeled data.

Usage:
    # Before pretraining (random weights)
    python visualize_embeddings.py --title "Random Init"

    # After contrastive pretraining
    python visualize_embeddings.py \
        --checkpoint ./runs/pretrain/contrastive_XXXX/best_pretrained.pt \
        --title "After Contrastive Pretraining"

    # After fine-tuning
    python visualize_embeddings.py \
        --checkpoint ./runs/finetune_XXXX/best_model.pt \
        --title "After Fine-tuning"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from model import WhaleConformer
from dataset import (
    load_annotations, get_file_manifest,
    build_val_segments_with_annotations, WhaleCallDataset, collate_fn,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Encoder checkpoint (None = random init)")
    p.add_argument("--title", type=str, default="Embeddings")
    p.add_argument("--output", type=str, default="embeddings.png")
    p.add_argument("--n_segments", type=int, default=2000,
                   help="Max segments to embed (for speed)")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(model, loader, device, n_max=2000):
    """
    Extract segment-level embeddings and their dominant class labels.
    Returns (embeddings, labels) numpy arrays.
    """
    model.eval()
    all_embs = []
    all_labels = []
    n = 0

    for audio, targets, mask, metas in tqdm(loader, desc="Extracting"):
        audio = audio.to(device)

        # Forward through encoder only (not classifier)
        x = model.frontend(audio)
        x = model.pos_enc(x)
        for block in model.conformer_blocks:
            x = block(x)
        # (B, T, d_model)

        # Global average pool → segment-level embedding
        emb = x.mean(dim=1)  # (B, d_model)
        all_embs.append(emb.cpu().numpy())

        # Determine dominant label for each segment
        # 0=bmabz, 1=d, 2=bp, 3=noise (no calls)
        for i in range(audio.size(0)):
            t = targets[i]  # (T, C)
            m = mask[i]     # (T,)
            t_valid = t[m.bool()]
            if t_valid.size(0) == 0 or t_valid.sum() == 0:
                all_labels.append(3)  # noise
            else:
                # Dominant class = most positive frames
                class_sums = t_valid.sum(dim=0)
                all_labels.append(class_sums.argmax().item())

        n += audio.size(0)
        if n >= n_max:
            break

    embeddings = np.concatenate(all_embs, axis=0)[:n_max]
    labels = np.array(all_labels)[:n_max]
    return embeddings, labels


def plot_umap(embeddings, labels, title, output_path):
    """Create UMAP plot colored by call type."""
    try:
        import umap
    except ImportError:
        print("Install umap-learn: pip install umap-learn")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"Fitting UMAP on {embeddings.shape[0]} embeddings...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    label_names = cfg.class_names() + ["noise"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9E9E9E"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i, (name, color) in enumerate(zip(label_names, colors)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=color, label=f"{name} (n={mask.sum()})",
                      s=8, alpha=0.6)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=0.0, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, sample_rate=cfg.SAMPLE_RATE,
    ).to(device)

    # Load weights
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "encoder_state_dict" in ckpt:
            model.load_state_dict(ckpt["encoder_state_dict"], strict=False)
            print(f"Loaded pretrained encoder from {args.checkpoint}")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded fine-tuned model from {args.checkpoint}")
    else:
        print("Using random initialization (no checkpoint)")

    # Build validation data with annotations
    annotations = load_annotations(cfg.VAL_DATASETS)
    val_manifest = get_file_manifest(cfg.VAL_DATASETS)
    val_segs = build_val_segments_with_annotations(
        annotations[annotations["dataset"].isin(cfg.VAL_DATASETS)],
        val_manifest,
    )

    # Subsample for speed — take a mix of positive and negative segments
    pos_segs = [s for s in val_segs if s.is_positive]
    neg_segs = [s for s in val_segs if not s.is_positive]
    n_pos = min(len(pos_segs), args.n_segments // 2)
    n_neg = min(len(neg_segs), args.n_segments // 2)

    import random
    random.seed(42)
    selected = random.sample(pos_segs, n_pos) + random.sample(neg_segs, n_neg)
    random.shuffle(selected)
    print(f"Selected {len(selected)} segments ({n_pos} positive, {n_neg} negative)")

    dataset = WhaleCallDataset(selected, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, collate_fn=collate_fn)

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, loader, device, args.n_segments)

    # Print class distribution
    label_names = cfg.class_names() + ["noise"]
    for i, name in enumerate(label_names):
        print(f"  {name}: {(labels == i).sum()}")

    # Plot
    plot_umap(embeddings, labels, args.title, args.output)


if __name__ == "__main__":
    main()
