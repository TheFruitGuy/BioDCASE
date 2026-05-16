"""
Triton — Spectrogram feature extraction
=======================================

Phase-aware, frequency-normalized STFT front end. Implemented as an
``nn.Module`` so it can sit on a CUDA device alongside the model and be
invoked inside the training/inference loops.

Output format
-------------
For waveform ``(batch, n_samples)``, returns ``(batch, 3, n_freq, n_frames)``
with the three channels:

    channel 0 — magnitude ``|STFT|`` (with optional per-freq mean
                subtraction in complex space, see ``cfg.NORM_FEATURES``)
    channel 1 — ``cos(phase)``
    channel 2 — ``sin(phase)``

The trigonometric phase encoding avoids the 2π discontinuity that breaks
convolutional layers when fed raw angle values, while still letting the
model exploit phase-sensitive features like onset timing.

Important reproduction detail
-----------------------------
``cfg.NORM_FEATURES == "demean"`` subtracts the time-mean of the
**complex** STFT before splitting into magnitude/phase. This is *not*
equivalent to subtracting the mean of the magnitudes. The official
WhaleVAD checkpoint was trained against complex-domain demeaning, so any
deviation here silently degrades inference even with correct model
weights. (This was the root cause of our long-running reproduction gap
during the WhaleVAD baseline phase.)
"""

import torch
import torch.nn as nn

import config as cfg


class SpectrogramExtractor(nn.Module):
    """
    Phase-aware spectrogram extractor.

    Stateless w.r.t. trainable parameters; only a Hann window is
    registered as a buffer (so it follows the module across devices and
    serialises with the state dict).
    """

    def __init__(self):
        super().__init__()
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.hop_length = cfg.HOP_LENGTH
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute (B, 3, F, T) trig-encoded spectrogram from (B, n_samples).

        Accepts unbatched ``(n_samples,)`` input as a convenience.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # ``center=False`` skips reflective padding; the dataset's collar
        # mechanism already ensures no interesting events sit at segment
        # boundaries.
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )  # (B, n_freq, n_frames), complex

        # Complex-domain demeaning. See module docstring for why this
        # is not equivalent to magnitude demeaning.
        if cfg.NORM_FEATURES == "demean":
            stft = stft - stft.mean(dim=-1, keepdim=True)

        mag = stft.abs()
        angle = stft.angle()
        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        # (B, n_freq, n_frames) × 3 → (B, 3, n_freq, n_frames)
        return torch.stack([mag, cos_ph, sin_ph], dim=1)


# ----------------------------------------------------------------------
# Module self-test. Run ``python spectrogram.py`` to verify shapes.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    extractor = SpectrogramExtractor()
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)  # 2× 30-second segments
    feat = extractor(audio)
    print(f"Input:  {audio.shape}")
    print(f"Output: {feat.shape}")
    # Expected: (2, 3, 129, ~1449)
    #   - 129 freq bins (n_fft // 2 + 1)
    #   - ~1449 time frames (30s / 20ms, less boundary frames from center=False)
