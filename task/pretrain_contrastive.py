"""
Optional SSL extension: SimCLR contrastive pretraining on unlabeled audio.

    python pretrain_contrastive.py

Trains the Whale-VAD encoder (filterbank + feature extractor + bottleneck +
aggregation) on unlabeled audio using NT-Xent. Saves weights that can be
loaded by train.py via --pretrained.

Everything about the feature extractor architecture is identical to the
supervised model, so weights transfer cleanly.
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


# ----------------------------------------------------------------------
# Pretraining config — separate from supervised config
# ----------------------------------------------------------------------

PRETRAIN_EPOCHS     = 30
PRETRAIN_BATCH_SIZE = 16
ACCUMULATION_STEPS  = 4                    # effective batch = 64
PRETRAIN_LR         = 1e-3
PRETRAIN_WARMUP     = 3
SEGMENT_LENGTH_S    = 30.0
TEMPERATURE         = 0.07
PROJ_DIM            = 128
PRETRAIN_WORKERS    = 16
PRETRAIN_OUTPUT_DIR = Path("./runs/pretrain")


# ----------------------------------------------------------------------
# Waveform augmentations for contrastive views
# ----------------------------------------------------------------------

class AudioAugmentor:
    def __init__(self, sample_rate: int = 250):
        self.sr = sample_rate

    def __call__(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._aug(audio.clone()), self._aug(audio.clone())

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        # Time shift (circular)
        if random.random() < 0.5:
            shift = random.randint(-5 * self.sr, 5 * self.sr)
            x = torch.roll(x, shifts=shift, dims=0)
        # Additive noise
        if random.random() < 0.5:
            snr_db = random.uniform(5, 30)
            sig_p = x.pow(2).mean()
            n_p = sig_p / (10 ** (snr_db / 10) + 1e-8)
            x = x + torch.randn_like(x) * n_p.sqrt()
        # Gain
        if random.random() < 0.8:
            g = random.uniform(-6, 6)
            x = x * (10 ** (g / 20))
        return x


# ----------------------------------------------------------------------
# Unlabeled dataset: random 30s segments from unlabeled WAV files
# ----------------------------------------------------------------------

class UnlabeledAudioDataset(Dataset):
    def __init__(self, data_dir: Path, segment_length_s: float = SEGMENT_LENGTH_S):
        self.data_dir = Path(data_dir)
        self.seg_samples = int(segment_length_s * cfg.SAMPLE_RATE)
        self.augmentor = AudioAugmentor(sample_rate=cfg.SAMPLE_RATE)

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
        f = self.files[idx]
        start = random.randint(0, max(0, f["n_samples"] - self.seg_samples))
        audio, sr = sf.read(f["path"], start=start,
                            stop=start + self.seg_samples, dtype="float32")
        assert sr == cfg.SAMPLE_RATE
        audio = torch.from_numpy(audio - audio.mean())
        return self.augmentor(audio)


def collate(batch):
    v1, v2 = zip(*batch)
    return torch.stack(v1), torch.stack(v2)


# ----------------------------------------------------------------------
# Contrastive model: supervised encoder + projection head
# ----------------------------------------------------------------------

class ContrastiveEncoder(nn.Module):
    """
    Wraps the Whale-VAD feature extractor (everything up to LSTM) plus
    a projection head for SimCLR. After pretraining we save just the
    encoder weights.
    """
    def __init__(self):
        super().__init__()
        self.spec_extractor = SpectrogramExtractor()
        # Full WhaleVAD model — we use its spec → residual stack path
        self.backbone = WhaleVAD(num_classes=cfg.n_classes())

        # Projection: pooled encoder features → contrastive space
        # We pool the per-frame features (before LSTM) → 64-dim vector,
        # then project to PROJ_DIM
        self.projection = nn.Sequential(
            nn.Linear(cfg.PROJECTION_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, PROJ_DIM),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self.spec_extractor(audio)
        # Go through encoder up to the projection layer (not the LSTM)
        x = self.backbone.filterbank(spec)
        x = self.backbone.feat_extractor(x)
        x = self.backbone.residual_stack(x)

        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)
        self.backbone._init_projection(C * F, x.device)
        x = self.backbone.feat_proj(x)                # (B, T, 64)

        # Global average pool time → (B, 64)
        h = x.mean(dim=1)
        # Project to contrastive space
        z = self.projection(h)
        return F.normalize(z, dim=1)


# ----------------------------------------------------------------------
# NT-Xent loss (SimCLR)
# ----------------------------------------------------------------------

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        device = z1.device
        z = torch.cat([z1, z2], dim=0)                # (2B, D)
        sim = torch.mm(z, z.t()) / self.tau           # (2B, 2B)
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        pos = torch.arange(2 * B, device=device)
        pos[:B] += B
        pos[B:] -= B
        return F.cross_entropy(sim, pos)


# ----------------------------------------------------------------------
# Diagnostics: alignment & uniformity
# ----------------------------------------------------------------------

@torch.no_grad()
def alignment_uniformity(model, loader, device, n_batches=20):
    model.eval()
    all1, all2 = [], []
    for i, (v1, v2) in enumerate(loader):
        if i >= n_batches:
            break
        all1.append(model(v1.to(device)).cpu())
        all2.append(model(v2.to(device)).cpu())
    z1 = torch.cat(all1); z2 = torch.cat(all2)
    align = (z1 - z2).pow(2).sum(dim=1).mean().item()
    z = torch.cat([z1, z2])
    pdist = torch.cdist(z, z).pow(2)
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    unif = torch.log(torch.exp(-2 * pdist[mask]).mean()).item()
    return align, unif


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

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

    dataset = UnlabeledAudioDataset(cfg.UNLABELED_DATA_DIR)
    loader = DataLoader(
        dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=PRETRAIN_WORKERS, collate_fn=collate,
        pin_memory=True, drop_last=True,
    )
    print(f"Batches/epoch: {len(loader)}")

    model = ContrastiveEncoder().to(device)

    # Warm up lazy projection
    with torch.no_grad():
        dummy = torch.randn(2, cfg.SAMPLE_RATE * 30, device=device)
        model(dummy)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n:,}")

    criterion = NTXentLoss()
    optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=PRETRAIN_WARMUP)
    cosine = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS - PRETRAIN_WARMUP)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[PRETRAIN_WARMUP])

    best_loss = float("inf")

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        print(f"\n{'='*60}\nPretrain {epoch}/{PRETRAIN_EPOCHS}\n{'='*60}")

        model.train()
        total_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Pretrain {epoch}", leave=False)
        for step, (v1, v2) in enumerate(pbar):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            z1 = model(v1); z2 = model(v2)
            loss = criterion(z1, z2) / ACCUMULATION_STEPS

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

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

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save just the backbone (WhaleVAD encoder) weights — these
            # load directly into train.py's model via load_state_dict
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
    print(f"To use: python train.py --pretrained {run_dir}/best_pretrained.pt --freeze_epochs 5")


if __name__ == "__main__":
    main()
