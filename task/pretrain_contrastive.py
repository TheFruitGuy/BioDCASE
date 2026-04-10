"""
Contrastive self-supervised pretraining for Whale-Conformer.

SimCLR-style: two augmented views of the same audio → encoder → projection head
→ NT-Xent loss pulls positive pairs together, pushes negatives apart.

The encoder learns representations of Antarctic underwater soundscapes
(whale calls, ice noise, ship noise, ocean background) without any labels.

After pretraining:
  - Discard projection head
  - Attach classification head
  - Fine-tune on labeled ATBFL data

Usage:
    CUDA_VISIBLE_DEVICES=8 python pretrain_contrastive.py
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import soundfile as sf

import config as cfg
from model import WhaleConformer
from augmentations import AudioAugmentor


# ---------------------------------------------------------------------------
# Config — edit these for pretraining
# ---------------------------------------------------------------------------

PRETRAIN_DATA_DIR   = Path("./data_unlabeled")
PRETRAIN_OUTPUT_DIR = Path("./runs/pretrain")

PRETRAIN_EPOCHS     = 30
PRETRAIN_BATCH_SIZE = 72       # larger batches = more negatives = better contrastive learning
ACCUMULATION_STEPS  = 4         # effective batch = 16 × 4 = 64
PRETRAIN_LR         = 1e-3
PRETRAIN_WARMUP     = 3
SEGMENT_LENGTH_S    = 30.0     # segment length in seconds
TEMPERATURE         = 0.07     # NT-Xent temperature (lower = harder negatives)
PROJ_DIM            = 128      # projection head output dimension
PRETRAIN_WORKERS    = 16


# ---------------------------------------------------------------------------
# Unlabeled dataset — random segments from raw WAV files
# ---------------------------------------------------------------------------

class UnlabeledAudioDataset(Dataset):
    """
    Loads random segments from unlabeled WAV files.
    Each __getitem__ returns two augmented views of the same segment.
    """
    def __init__(
        self,
        data_dir: str | Path,
        segment_length_s: float = SEGMENT_LENGTH_S,
        sample_rate: int = cfg.SAMPLE_RATE,
        augmentor: AudioAugmentor | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.segment_samples = int(segment_length_s * sample_rate)
        self.sample_rate = sample_rate
        self.augmentor = augmentor or AudioAugmentor(sample_rate=sample_rate)

        # Build file list with durations
        self.files = []
        for wf in sorted(self.data_dir.rglob("*.wav")):
            try:
                info = sf.info(str(wf))
                if info.duration >= segment_length_s:
                    self.files.append({
                        "path": str(wf),
                        "duration_s": info.duration,
                        "n_samples": int(info.duration * sample_rate),
                    })
            except Exception as e:
                print(f"Warning: skipping {wf.name}: {e}")

        print(f"UnlabeledAudioDataset: {len(self.files)} files, "
              f"{sum(f['duration_s'] for f in self.files)/3600:.0f} hours")

    def __len__(self) -> int:
        # Each file can yield multiple segments; use file count × segments_per_file
        # But for simplicity, just return file count — random offset gives variety
        return len(self.files)

    def __getitem__(self, idx: int):
        f = self.files[idx]

        # Random start position
        max_start = f["n_samples"] - self.segment_samples
        start = random.randint(0, max(0, max_start))

        # Load audio
        audio, sr = sf.read(f["path"], start=start,
                            stop=start + self.segment_samples, dtype="float32")
        assert sr == self.sample_rate

        # Mean subtraction
        audio = audio - audio.mean()
        audio = torch.from_numpy(audio)

        # Two augmented views
        view1, view2 = self.augmentor(audio)

        return view1, view2


def collate_contrastive(batch):
    """Collate pairs of views into two batched tensors."""
    views1, views2 = zip(*batch)
    return torch.stack(views1), torch.stack(views2)


# ---------------------------------------------------------------------------
# Projection head — maps encoder output to contrastive embedding space
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    MLP projection head: encoder features → contrastive embedding.
    
    Following SimCLR, using a 2-layer MLP with ReLU.
    This head is discarded after pretraining.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) — pooled encoder output
        return self.net(x)


# ---------------------------------------------------------------------------
# NT-Xent loss (Normalized Temperature-scaled Cross-Entropy)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    SimCLR contrastive loss.
    
    For a batch of N pairs (2N total views):
    - Each view i has exactly 1 positive (its augmented partner)
    - And 2(N-1) negatives (all other views in the batch)
    
    Larger batch size = more negatives = better representations.
    """
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) L2-normalized embeddings from two views
        Returns:
            scalar loss
        """
        B = z1.size(0)
        device = z1.device

        # Concatenate: [z1_0, z1_1, ..., z1_B, z2_0, z2_1, ..., z2_B]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_indices = torch.arange(2 * B, device=device)
        pos_indices[:B] += B     # first half → second half
        pos_indices[B:] -= B     # second half → first half

        # NT-Xent: cross-entropy where the positive pair is the "correct class"
        loss = F.cross_entropy(sim, pos_indices)

        return loss


# ---------------------------------------------------------------------------
# Contrastive model wrapper
# ---------------------------------------------------------------------------

class ContrastiveModel(nn.Module):
    """
    Wraps the Conformer encoder + projection head for contrastive pretraining.
    
    encoder: Conformer (frontend + positional encoding + conformer blocks)
    projection_head: MLP that maps pooled features → contrastive space
    """
    def __init__(self, encoder: WhaleConformer, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.encoder = encoder
        # Average-pool the temporal dimension, then project
        self.projection_head = ProjectionHead(
            input_dim=cfg.D_MODEL,
            output_dim=proj_dim,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, T_samples) raw waveform
        Returns:
            z: (B, proj_dim) L2-normalized embedding
        """
        # Get per-frame features from encoder (skip the classification head)
        x = self.encoder.frontend(audio)       # (B, T, d_model)
        x = self.encoder.pos_enc(x)
        for block in self.encoder.conformer_blocks:
            x = block(x)
        # (B, T_frames, d_model)

        # Global average pooling over time
        h = x.mean(dim=1)  # (B, d_model)

        # Project to contrastive space
        z = self.projection_head(h)  # (B, proj_dim)

        # L2 normalize
        z = F.normalize(z, dim=1)

        return z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Pretrain {epoch}/{PRETRAIN_EPOCHS}", leave=False)
    optimizer.zero_grad()

    for step, (view1, view2) in enumerate(pbar):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2) / ACCUMULATION_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  *** NaN detected, skipping batch ***")
            continue

        scaler.scale(loss).backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        n += 1
        pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

    return total_loss / max(n, 1)


@torch.no_grad()
def compute_alignment_uniformity(model, loader, device, n_batches=20):
    """
    Diagnostic metrics for contrastive learning quality:
    
    Alignment: how close positive pairs are (lower = better)
    Uniformity: how uniformly distributed embeddings are on hypersphere (lower = better)
    
    Good pretraining: alignment decreases, uniformity stays stable or decreases.
    Collapse: both go to 0 (everything maps to same point).
    """
    model.eval()
    all_z1, all_z2 = [], []

    for i, (v1, v2) in enumerate(loader):
        if i >= n_batches:
            break
        z1 = model(v1.to(device))
        z2 = model(v2.to(device))
        all_z1.append(z1.cpu())
        all_z2.append(z2.cpu())

    z1 = torch.cat(all_z1)
    z2 = torch.cat(all_z2)

    # Alignment: mean squared distance between positive pairs
    alignment = (z1 - z2).pow(2).sum(dim=1).mean().item()

    # Uniformity: log of average pairwise Gaussian potential
    z = torch.cat([z1, z2])
    sq_pdist = torch.cdist(z, z).pow(2)
    # Exclude diagonal
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    uniformity = torch.log(torch.exp(-2 * sq_pdist[mask]).mean()).item()

    return alignment, uniformity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = PRETRAIN_OUTPUT_DIR / f"contrastive_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    augmentor = AudioAugmentor(sample_rate=cfg.SAMPLE_RATE)
    dataset = UnlabeledAudioDataset(
        PRETRAIN_DATA_DIR,
        segment_length_s=SEGMENT_LENGTH_S,
        augmentor=augmentor,
    )
    loader = DataLoader(
        dataset,
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=PRETRAIN_WORKERS,
        collate_fn=collate_contrastive,
        pin_memory=True,
        drop_last=True,  # important for contrastive: need consistent batch size
    )
    print(f"Batches per epoch: {len(loader)}")

    # --- model ---
    # Create the Conformer encoder (same architecture as supervised)
    encoder = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=cfg.DROPOUT, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, sample_rate=cfg.SAMPLE_RATE,
    )

    model = ContrastiveModel(encoder, proj_dim=PROJ_DIM).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Pretraining across {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # --- loss + optimizer ---
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.01)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=PRETRAIN_WARMUP)
    cosine = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS - PRETRAIN_WARMUP)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[PRETRAIN_WARMUP])

    scaler = GradScaler("cuda")

    # --- training loop ---
    best_loss = float("inf")

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        print(f"\n{'='*60}\nPretrain Epoch {epoch}/{PRETRAIN_EPOCHS}\n{'='*60}")

        loss = train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch)
        scheduler.step()

        # Diagnostics
        alignment, uniformity = compute_alignment_uniformity(model, loader, device)
        print(f"  Loss: {loss:.4f}  |  Alignment: {alignment:.4f}  |  "
              f"Uniformity: {uniformity:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best encoder weights (without projection head)
        if loss < best_loss:
            best_loss = loss

            full_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            encoder_state = {}
            for k, v in full_state.items():
                if k.startswith("encoder."):
                    encoder_state[k[len("encoder."):]] = v

            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder_state,
                "full_model_state_dict": full_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "alignment": alignment,
                "uniformity": uniformity,
            }, run_dir / "best_pretrained.pt")
            print(f"  *** New best loss: {best_loss:.4f} — saved ***")

            # Also save latest
        full_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        encoder_state = {}
        for k, v in full_state.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v

        torch.save({
            "epoch": epoch,
            "encoder_state_dict": encoder_state,
            "loss": loss,
        }, run_dir / "latest_pretrained.pt")

    print(f"\nPretraining done. Best loss: {best_loss:.4f}")
    print(f"Encoder weights saved to: {run_dir / 'best_pretrained.pt'}")
    print(f"\nTo fine-tune, load with:")
    print(f"  ckpt = torch.load('{run_dir}/best_pretrained.pt')")
    print(f"  model.load_state_dict(ckpt['encoder_state_dict'], strict=False)")


if __name__ == "__main__":
    main()
