"""
Spectrogram feature extraction.

Reproduces their whalevad/spectrogram.py with:
  - 256-point STFT, Hann window
  - "trig" complex representation: [mag, cos θ, sin θ] → 3 channels
  - Per-frequency-bin mean subtraction over segment duration ("demean")

Paper Section 5.2 and 5.4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

import config as cfg


class SpectrogramExtractor(nn.Module):
    """
    Input:  audio (B, T_samples) — raw waveform
    Output: features (B, 3, F, T_frames) — phase-aware spectrogram

    Steps:
      1. STFT → complex spectrogram
      2. Extract magnitude r and phase θ
      3. Stack [r, cos θ, sin θ] as 3 channels
      4. Per-frequency-bin mean subtraction across time
    """

    def __init__(self):
        super().__init__()
        self.n_fft      = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.hop_length = cfg.HOP_LENGTH
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T_samples) or (T_samples,)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # STFT returns complex (B, F, T_frames)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,                      # paper: "without additional zero padding"
            return_complex=True,
        )

        # Magnitude and phase
        mag   = stft.abs()                     # (B, F, T)
        angle = stft.angle()                   # (B, F, T)

        # Per-frequency mean subtraction across time (Section 5.2)
        # "mean computed independently for each frequency bin over the duration"
        if cfg.NORM_FEATURES == "demean":
            mag = mag - mag.mean(dim=-1, keepdim=True)

        # Trigonometric complex representation (Section 5.4)
        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        # Stack into 3 channels: (B, 3, F, T)
        feat = torch.stack([mag, cos_ph, sin_ph], dim=1)
        return feat


if __name__ == "__main__":
    # Quick shape test
    extractor = SpectrogramExtractor()
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)   # 30s segment
    feat = extractor(audio)
    print(f"Input:  {audio.shape}")
    print(f"Output: {feat.shape}")
    # Expected: (2, 3, 129, ~1500) — 129 = n_fft//2+1, 1500 = 30s / 20ms
