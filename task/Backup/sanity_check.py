"""
sanity_check.py — Overfit a tiny batch to verify the data pipeline works.

If even a simple model can't overfit 4 batches, the data/labels are broken.
If a simple model CAN overfit but the Conformer can't, it's an architecture issue.

    CUDA_VISIBLE_DEVICES=8 python sanity_check.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config as cfg
from dataset import build_dataloaders, collate_fn


class TinyModel(nn.Module):
    """Stupidly simple model: STFT → mean pool freq → Linear → per-frame output."""
    def __init__(self, n_classes=3, hidden=128):
        super().__init__()
        self.n_fft = cfg.N_FFT
        self.hop = cfg.HOP_LENGTH
        self.win_length = cfg.WIN_LENGTH
        n_freq = self.n_fft // 2 + 1  # 129

        self.net = nn.Sequential(
            nn.Linear(n_freq, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, audio):
        # audio: (B, T_samples)
        window = torch.hann_window(self.win_length, device=audio.device)
        if self.win_length < self.n_fft:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            window = F.pad(window, (pad_left, pad_right))

        spec = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=window,
            return_complex=True, center=True,
        )
        mag = spec.abs()  # (B, F, T)
        mag = mag.permute(0, 2, 1)  # (B, T, F)
        return self.net(mag)  # (B, T, n_classes)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build data
    train_ds, train_loader, val_loader = build_dataloaders(batch_size=8, num_workers=4)

    # Grab a few batches
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i >= 3:
            break

    print(f"\nGot {len(batches)} batches")

    # Check targets actually have positives
    for i, (audio, targets, mask, metas) in enumerate(batches):
        pos_frames = targets.sum().item()
        total_frames = mask.sum().item() * targets.size(-1)
        print(f"Batch {i}: audio={audio.shape} targets={targets.shape} "
              f"pos_frames={int(pos_frames)} / {int(total_frames)} "
              f"({100*pos_frames/max(total_frames,1):.1f}%)")

    # Try to overfit these batches
    print("\n" + "="*60)
    print("TEST 1: TinyModel — can we overfit 4 batches?")
    print("="*60)

    model = TinyModel(n_classes=cfg.n_classes()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([20.0, 50.0, 50.0], device=device)

    for epoch in range(50):
        total_loss = 0
        total_tp = [0, 0, 0]
        total_pos = [0, 0, 0]
        total_pred = [0, 0, 0]

        for audio, targets, mask, _ in batches:
            audio = audio.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            logits = model(audio)

            # Align lengths
            T_m, T_t = logits.size(1), targets.size(1)
            if T_m < T_t:
                targets = targets[:, :T_m, :]
                mask = mask[:, :T_m]
            elif T_m > T_t:
                targets = F.pad(targets, (0, 0, 0, T_m - T_t))
                mask = F.pad(mask, (0, T_m - T_t))

            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=pos_weight.view(1, 1, -1),
                reduction="none"
            )
            loss = (loss * mask.unsqueeze(-1).float()).sum() / (mask.sum() * 3 + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Stats
            probs = torch.sigmoid(logits)
            for c in range(3):
                m = mask.view(-1).bool()
                p = (probs[..., c].reshape(-1)[m] > 0.3).float()
                t = targets[..., c].reshape(-1)[m]
                total_tp[c] += (p * t).sum().item()
                total_pos[c] += t.sum().item()
                total_pred[c] += p.sum().item()

        if epoch % 5 == 0:
            avg_loss = total_loss / len(batches)
            f1s = []
            for c in range(3):
                prec = total_tp[c] / (total_pred[c] + 1e-8)
                rec = total_tp[c] / (total_pos[c] + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                f1s.append(f1)
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}  "
                  f"F1=[{f1s[0]:.3f}, {f1s[1]:.3f}, {f1s[2]:.3f}]  "
                  f"mean={np.mean(f1s):.3f}")

    print("\n" + "="*60)
    print("TEST 2: Conformer — same 4 batches, higher LR")
    print("="*60)

    from model import WhaleConformer

    model2 = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=0.0,  # no dropout for overfit test
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.WIN_LENGTH,
        sample_rate=cfg.SAMPLE_RATE,
    ).to(device)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    for epoch in range(50):
        total_loss = 0
        total_tp = [0, 0, 0]
        total_pos = [0, 0, 0]
        total_pred = [0, 0, 0]

        for audio, targets, mask, _ in batches:
            audio = audio.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            logits = model2(audio)

            T_m, T_t = logits.size(1), targets.size(1)
            if T_m < T_t:
                targets = targets[:, :T_m, :]
                mask = mask[:, :T_m]
            elif T_m > T_t:
                targets = F.pad(targets, (0, 0, 0, T_m - T_t))
                mask = F.pad(mask, (0, T_m - T_t))

            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=pos_weight.view(1, 1, -1),
                reduction="none"
            )
            loss = (loss * mask.unsqueeze(-1).float()).sum() / (mask.sum() * 3 + 1e-8)

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            for c in range(3):
                m = mask.view(-1).bool()
                p = (probs[..., c].reshape(-1)[m] > 0.3).float()
                t = targets[..., c].reshape(-1)[m]
                total_tp[c] += (p * t).sum().item()
                total_pos[c] += t.sum().item()
                total_pred[c] += p.sum().item()

        if epoch % 5 == 0:
            avg_loss = total_loss / len(batches)
            f1s = []
            for c in range(3):
                prec = total_tp[c] / (total_pred[c] + 1e-8)
                rec = total_tp[c] / (total_pos[c] + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                f1s.append(f1)
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}  "
                  f"F1=[{f1s[0]:.3f}, {f1s[1]:.3f}, {f1s[2]:.3f}]  "
                  f"mean={np.mean(f1s):.3f}")

    print("\nDone. If TinyModel overfits → data pipeline works.")
    print("If Conformer also overfits → increase LR for full training.")
    print("If Conformer fails → architecture issue (gradients, dimensions).")


if __name__ == "__main__":
    main()
