"""
Final Pipeline - Spectrogram Feature Extraction
===============================================

Phase-aware, frequency-normalised STFT front end. Implemented as an
``nn.Module`` so it can live on the same device as the model inside the
training and validation loops.

For an input waveform of shape ``(batch, n_samples)``, the extractor returns
a tensor ``(batch, 3, n_freq, n_frames)`` whose three channels are, in order,
the (optionally demeaned) magnitude, ``cos(phase)`` and ``sin(phase)``.
"""

import torch
import torch.nn as nn

import config_final as cfg


class SpectrogramExtractor(nn.Module):
    """
    Phase-aware spectrogram extractor with per-frequency mean subtraction.

    All STFT parameters are read from :mod:`config_final` at construction
    time. The module holds no trainable parameters; the Hann window is
    registered as a buffer so it moves with ``.to(device)``.
    """

    def __init__(self):
        super().__init__()
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.hop_length = cfg.HOP_LENGTH
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute phase-aware spectrogram features.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform of shape ``(batch, n_samples)`` or ``(n_samples,)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(batch, 3, n_freq, n_frames)``.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Complex STFT with center=False: no reflective padding, because the
        # collar mechanism guarantees no events at segment boundaries.
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )  # (B, n_freq, n_frames), complex

        # Per-frequency mean subtraction on the complex spectrum, matching the
        # official reference implementation: demean before magnitude/phase.
        if cfg.NORM_FEATURES == "demean":
            stft = stft - stft.mean(dim=-1, keepdim=True)

        mag = stft.abs()
        angle = stft.angle()

        # Trigonometric phase encoding avoids the 2*pi discontinuity that
        # confuses convolutions while preserving onset-timing information.
        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        return torch.stack([mag, cos_ph, sin_ph], dim=1)
