"""
Final Pipeline - Spectrogram Feature Extraction
===============================================

Phase-aware, frequency-normalised STFT front end. Implemented as an
``nn.Module`` so it can live on the same device as the model inside the
training and validation loops.

For an input waveform of shape ``(batch, n_samples)``, the extractor returns
a tensor ``(batch, C, n_freq, n_frames)``. The first three channels are, in
order, the (optionally demeaned) magnitude, ``cos(phase)`` and ``sin(phase)``.
With ``cfg.DUAL_RESOLUTION`` a 4th channel is appended: a short-window
magnitude (``cfg.SHORT_WIN_LENGTH``) on the same n_fft/hop grid, giving the
model a sharper time view of frequency-modulated calls. ``C`` is then 4.
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

        # Optional dual-resolution 4th channel: a short-window magnitude on the
        # SAME n_fft/hop grid (torch.stft zero-pads the short window up to
        # n_fft, so freq/time dims are identical and it stacks straight in).
        self.dual_resolution = cfg.DUAL_RESOLUTION
        if self.dual_resolution:
            self.short_win_length = cfg.SHORT_WIN_LENGTH
            self.register_buffer(
                "window_short", torch.hann_window(self.short_win_length))

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
            Feature tensor of shape ``(batch, C, n_freq, n_frames)``, with
            ``C`` = 3, or 4 when ``cfg.DUAL_RESOLUTION`` is set.
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

        feats = [mag, cos_ph, sin_ph]
        if self.dual_resolution:
            # Short-window magnitude on the identical n_fft/hop grid: sharper
            # time resolution for the D-call downsweep (channel 4).
            stft_s = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.short_win_length,
                window=self.window_short,
                center=False,
                return_complex=True,
            )
            if cfg.NORM_FEATURES == "demean":
                stft_s = stft_s - stft_s.mean(dim=-1, keepdim=True)
            feats.append(stft_s.abs())

        return torch.stack(feats, dim=1)
