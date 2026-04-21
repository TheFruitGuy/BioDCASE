"""
Self-Supervised Contrastive Pretraining
=======================================

Optional SimCLR-style pretraining on unlabeled Antarctic hydrophone audio.
Trains the Whale-VAD encoder (filterbank → feature extractor → bottleneck
→ depthwise aggregation → projection) on unlabeled WAV files using the
NT-Xent contrastive loss. The resulting encoder weights can be loaded by
``train.py`` via the ``--pretrained`` flag for supervised fine-tuning.

Extension rationale
-------------------
The BioDCASE training corpus has ~58k annotated events across 8 sites,
while the raw Antarctic hydrophone record contains thousands of hours
of unlabeled audio (e.g. Kerguelen2020: ~8500 files, ~8500 hours). SSL
pretraining lets us extract useful representations from this unlabeled
bulk, which can then accelerate and improve supervised fine-tuning. This
is not part of the paper's method and sits outside the strict
reproduction scope — it is tracked as a separate ablation.

The encoder architecture is identical to the supervised one (``WhaleVAD``
class with ``num_classes=cfg.n_classes()``) so checkpoints transfer
cleanly: fine-tuning just loads the encoder state dict into a fresh
``WhaleVAD`` and randomly initializes the classifier + LSTM.

Usage
-----
::

    python pretrain_contrastive.py

Then, for fine-tuning::

    python train.py --pretrained runs/pretrain/contrastive_XXXX/best_pretrained.pt \\
                    --freeze_epochs 5
"""

import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as cfg
from spectrogram import SpectrogramExtractor
from model import WhaleVAD


# ======================================================================
# Pretraining hyperparameters (separate from supervised config)
# ======================================================================

#: Number of pretraining epochs. SSL typically needs fewer epochs than
#: supervised training because contrastive learning is less sample-
#: efficient per update but operates on a much larger dataset.
PRETRAIN_EPOCHS = 30

#: Per-device batch size. Kept small to fit on consumer GPUs; effective
#: batch is this × ACCUMULATION_STEPS.
PRETRAIN_BATCH_SIZE = 16

#: Gradient accumulation steps. Contrastive losses benefit from large
#: effective batch sizes because every additional positive pair becomes
#: a negative for every other pair in the batch.
ACCUMULATION_STEPS = 4  # effective batch = 64

#: Peak learning rate (cosine schedule with warmup).
PRETRAIN_LR = 1e-3

#: Linear warmup epochs.
PRETRAIN_WARMUP = 3

#: Segment length (seconds) for each view. 30 s matches the supervised
#: validation window, so no resize mismatch at fine-tuning time.
SEGMENT_LENGTH_S = 30.0

#: SimCLR temperature. Lower → sharper distributions; higher → softer.
#: 0.07 is a common default.
TEMPERATURE = 0.07

#: Output dimension of the contrastive projection head.
PROJ_DIM = 128

#: Parallel data loading workers.
PRETRAIN_WORKERS = 16

#: Output directory for pretrained checkpoints.
PRETRAIN_OUTPUT_DIR = Path("./runs/pretrain")


# ======================================================================
# Waveform augmentations for contrastive views
# ======================================================================

class AudioAugmentor:
    """
    Generate two stochastically-augmented views of the same waveform.

    Implements three augmentations that preserve the semantic content
    of whale calls while varying the superficial characteristics:

        1. **Circular time shift**: shifts the waveform by a random
           amount. Teaches the encoder to be roughly shift-invariant.
        2. **Additive Gaussian noise**: at a random SNR in [5, 30] dB.
           Teaches noise robustness.
        3. **Random gain**: uniform in [-6, +6] dB. Teaches amplitude
           invariance.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate. Used to compute the time-shift bounds.
    """

    def __init__(self, sample_rate: int = 250):
        self.sr = sample_rate

    def __call__(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented views of the input."""
        return self._aug(audio.clone()), self._aug(audio.clone())

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one random augmentation pipeline."""
        # Time shift (circular): model should be invariant to small offsets.
        if random.random() < 0.5:
            shift = random.randint(-5 * self.sr, 5 * self.sr)
            x = torch.roll(x, shifts=shift, dims=0)
        # Additive noise at a random SNR.
        if random.random() < 0.5:
            snr_db = random.uniform(5, 30)
            sig_p = x.pow(2).mean()
            n_p = sig_p / (10 ** (snr_db / 10) + 1e-8)
            x = x + torch.randn_like(x) * n_p.sqrt()
        # Random gain in dB.
        if random.random() < 0.8:
            g = random.uniform(-6, 6)
            x = x * (10 ** (g / 20))
        return x


# ======================================================================
# Unlabeled dataset
# ======================================================================

class UnlabeledAudioDataset(Dataset):
    """
    Random 30-second windows drawn from all WAV files in a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing one or more ``.wav`` files.
    segment_length_s : float
        Window length in seconds. Files shorter than this are skipped.
    """

    def __init__(self, data_dir: Path, segment_length_s: float = SEGMENT_LENGTH_S):
        self.data_dir = Path(data_dir)
        self.seg_samples = int(segment_length_s * cfg.SAMPLE_RATE)
        self.augmentor = AudioAugmentor(sample_rate=cfg.SAMPLE_RATE)

        # Build a manifest of files long enough to contain at least one
        # full segment. Files with corrupted headers are silently skipped.
        self.files = []
        for wf in sorted(self.data_dir.glob("*.wav")):
            try:
                info = sf.info(str(wf))
                if info.duration >= segment_length_s:
                    self.files.append({
                        "path": str(wf),
                        "n_samples": int(info.duration * cfg.SAMPLE_RATE),
                    })
            except Exception:
                continue
        print(f"UnlabeledAudioDataset: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Return two augmented views of a random window from file ``idx``."""
        f = self.files[idx]
        # Random start offset within the file.
        start = random.randint(0, max(0, f["n_samples"] - self.seg_samples))
        audio, sr = sf.read(f["path"], start=start,
                            stop=start + self.seg_samples, dtype="float32")
        assert sr == cfg.SAMPLE_RATE
        # DC removal: centers the waveform around zero.
        audio = torch.from_numpy(audio - audio.mean())
        return self.augmentor(audio)


def collate(batch):
    """
    Simple collate for ``(view1, view2)`` tuples.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        Two tensors of shape ``(B, n_samples)``.
    """
    v1, v2 = zip(*batch)
    return torch.stack(v1), torch.stack(v2)


# ======================================================================
# Contrastive model
# ======================================================================

class ContrastiveEncoder(nn.Module):
    """
    Whale-VAD encoder + projection head for SimCLR pretraining.

    The backbone is the full supervised ``WhaleVAD`` model; we use every
    layer up to and including the 64-dim ``feat_proj`` projection (i.e.
    everything that feeds the BiLSTM in supervised training). A global
    temporal mean pool collapses the time axis, and a 2-layer MLP
    projects to the contrastive embedding space.

    After pretraining, only the backbone weights are saved; the
    projection head is discarded at fine-tuning time (standard SimCLR
    practice — the projection head is useful for the contrastive task
    but hurts downstream performance).
    """

    def __init__(self):
        super().__init__()
        self.spec_extractor = SpectrogramExtractor()
        # Use the full WhaleVAD class so the loaded weights match the
        # supervised model exactly at fine-tuning time.
        self.backbone = WhaleVAD(num_classes=cfg.n_classes())

        # Projection head: pooled 64-dim features → 128-dim contrastive
        # embedding. BatchNorm is critical for SimCLR stability.
        self.projection = nn.Sequential(
            nn.Linear(cfg.PROJECTION_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, PROJ_DIM),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute L2-normalized contrastive embeddings.

        Parameters
        ----------
        audio : torch.Tensor, shape (B, n_samples)

        Returns
        -------
        torch.Tensor, shape (B, PROJ_DIM)
            L2-normalized embedding vectors ready for NT-Xent.
        """
        spec = self.spec_extractor(audio)
        # Manually step through the backbone up to the projection layer;
        # we deliberately skip the BiLSTM and classifier at pretraining
        # time — they are randomly initialized during fine-tuning.
        x = self.backbone.filterbank(spec)
        x = self.backbone.feat_extractor(x)
        x = self.backbone.residual_stack(x)

        # Flatten (channels, freq) and project down.
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)
        self.backbone._init_projection(C * Fr, x.device)
        x = self.backbone.feat_proj(x)  # (B, T, 64)

        # Global temporal mean pool → (B, 64). Mean pooling is a mild
        # inductive bias that works well for low-frequency whale calls
        # whose relevant information is distributed throughout the
        # segment rather than localized in a specific frame.
        h = x.mean(dim=1)

        # Project to contrastive space and L2-normalize (required by
        # NT-Xent's cosine-similarity formulation).
        z = self.projection(h)
        return F.normalize(z, dim=1)


# ======================================================================
# NT-Xent contrastive loss
# ======================================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss (SimCLR).

    For each pair of views ``(z1_i, z2_i)`` the loss encourages the
    embedding of one view to be closer to the other view of the same
    sample than to any view of any other sample in the batch. With 2N
    embeddings per batch, this gives 2N(2N-1) candidate negatives per
    query.

    Parameters
    ----------
    temperature : float
        NT-Xent temperature parameter.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a pair of view batches.

        Parameters
        ----------
        z1, z2 : torch.Tensor, shape (B, D)
            L2-normalized embeddings of the two view sets.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        B = z1.size(0)
        device = z1.device
        # Concatenate both views into a 2B matrix and compute all-pairs
        # cosine similarity (pre-normalized → dot product).
        z = torch.cat([z1, z2], dim=0)                # (2B, D)
        sim = torch.mm(z, z.t()) / self.tau           # (2B, 2B)

        # Mask self-similarity (diagonal): setting to -inf excludes a
        # vector from being its own nearest neighbour.
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        # For each query row i in [0, 2B), the positive index is:
        #   - i + B   if i <  B   (view1 → view2)
        #   - i - B   if i >= B   (view2 → view1)
        pos = torch.arange(2 * B, device=device)
        pos[:B] += B
        pos[B:] -= B

        # Cross-entropy where the correct class is the positive pair.
        return F.cross_entropy(sim, pos)


# ======================================================================
# Diagnostics: alignment & uniformity (Wang & Isola, 2020)
# ======================================================================

@torch.no_grad()
def alignment_uniformity(model, loader, device, n_batches=20):
    """
    Compute alignment and uniformity metrics for the learned embeddings.

    Introduced by Wang & Isola (ICML 2020) as interpretable proxies for
    contrastive representation quality:

        - **Alignment**: mean squared distance between positive pairs.
          Lower is better (positives stay close together).
        - **Uniformity**: ``log E[exp(-2 || z_i - z_j ||²)]`` across all
          pairs. Lower is better (embeddings cover the hypersphere).

    Parameters
    ----------
    model : ContrastiveEncoder
    loader : DataLoader
    device : torch.device
    n_batches : int
        How many batches to average over.

    Returns
    -------
    (float, float)
        ``(alignment, uniformity)``.
    """
    model.eval()
    all1, all2 = [], []
    for i, (v1, v2) in enumerate(loader):
        if i >= n_batches:
            break
        all1.append(model(v1.to(device)).cpu())
        all2.append(model(v2.to(device)).cpu())
    z1 = torch.cat(all1)
    z2 = torch.cat(all2)

    align = (z1 - z2).pow(2).sum(dim=1).mean().item()

    # Uniformity: expected squared distance between all distinct pairs.
    z = torch.cat([z1, z2])
    pdist = torch.cdist(z, z).pow(2)
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    unif = torch.log(torch.exp(-2 * pdist[mask]).mean()).item()
    return align, unif


# ======================================================================
# Main
# ======================================================================

def main():
    """Pretrain the contrastive encoder on unlabeled audio."""
    # Seed everything for reproducibility.
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = PRETRAIN_OUTPUT_DIR / f"contrastive_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = UnlabeledAudioDataset(cfg.UNLABELED_DATA_DIR)
    loader = DataLoader(
        dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=PRETRAIN_WORKERS, collate_fn=collate,
        pin_memory=True, drop_last=True,
    )
    print(f"Batches/epoch: {len(loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ContrastiveEncoder().to(device)

    # Trigger lazy projection layer initialization.
    with torch.no_grad():
        dummy = torch.randn(2, cfg.SAMPLE_RATE * 30, device=device)
        model(dummy)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n:,}")

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = NTXentLoss()
    optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.01)

    # Linear warmup → cosine decay, a common SSL schedule.
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=PRETRAIN_WARMUP)
    cosine = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS - PRETRAIN_WARMUP)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[PRETRAIN_WARMUP])

    best_loss = float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        print(f"\n{'=' * 60}\nPretrain {epoch}/{PRETRAIN_EPOCHS}\n{'=' * 60}")

        model.train()
        total_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Pretrain {epoch}", leave=False)
        for step, (v1, v2) in enumerate(pbar):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            z1 = model(v1)
            z2 = model(v2)
            # Divide loss by accumulation steps so the final update has
            # the same magnitude as an equivalent single-step large-batch
            # update.
            loss = criterion(z1, z2) / ACCUMULATION_STEPS

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # Only step the optimizer every ACCUMULATION_STEPS micro-batches.
            if (step + 1) % ACCUMULATION_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

        scheduler.step()

        avg_loss = total_loss / max(n_steps, 1)
        align, unif = alignment_uniformity(model, loader, device)
        print(f"  Loss: {avg_loss:.4f} | Align: {align:.4f} | Unif: {unif:.4f}")
        print(f"  LR:   {optimizer.param_groups[0]['lr']:.6f}")

        # Save only when the loss improves. We save just the backbone
        # weights (not the projection head) so the checkpoint plugs
        # directly into supervised training.
        if avg_loss < best_loss:
            best_loss = avg_loss
            encoder_state = {
                k: v for k, v in model.backbone.state_dict().items()
            }
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder_state,
                "loss": avg_loss,
                "alignment": align,
                "uniformity": unif,
            }, run_dir / "best_pretrained.pt")
            print(f"  *** New best: {best_loss:.4f} saved")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"To use: python train.py --pretrained "
          f"{run_dir}/best_pretrained.pt --freeze_epochs 5")


if __name__ == "__main__":
    main()
