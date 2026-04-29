"""
Spectrogram Feature Extraction Module
=====================================

Implements the phase-aware, frequency-normalized STFT front end described in
Section 5.2 and 5.4 of the Whale-VAD paper. The extractor is implemented as
an ``nn.Module`` so that it can be placed on a CUDA device alongside the main
model and invoked inside the training/inference loops.

Output format
-------------
For an input waveform of shape ``(batch, n_samples)``, the extractor returns
a 4-D tensor ``(batch, 3, n_freq, n_time_frames)`` where the three channels
contain, in order:

    channel 0:  magnitude ``|STFT|``, optionally per-frequency mean subtracted
    channel 1:  ``cos(phase)``
    channel 2:  ``sin(phase)``

This "trigonometric" encoding of phase information was shown in Table 3 of
the paper to yield a +5.5% absolute F1 improvement over magnitude-only input
by preserving the temporal fine structure of whale calls, which have
distinct phase signatures that help distinguish them from background noise.

Shape example
-------------
For a 30-second segment at 250 Hz (= 7500 samples), with ``n_fft=256``,
``hop_length=5``, and ``center=False``, the output shape is
``(B, 3, 129, 1449)``:
    - 129 = ``n_fft // 2 + 1`` frequency bins from 0 Hz to Nyquist (125 Hz)
    - 1449 = ``floor((7500 - 256) / 5) + 1``, the number of frames that fit
      without any zero-padding at segment boundaries
"""

import torch
import torch.nn as nn

import config as cfg


class SpectrogramExtractor(nn.Module):
    """
    Phase-aware spectrogram extractor with per-frequency mean subtraction.

    Parameters
    ----------
    None (all parameters are read from the ``config`` module at init time).

    Attributes
    ----------
    n_fft, win_length, hop_length : int
        STFT parameters, in samples.
    window : torch.Tensor
        Registered buffer holding a Hann window of length ``win_length``.
        As a buffer it is automatically moved with ``.to(device)`` and
        included in the module's state dict.

    Notes
    -----
    The module is stateless with respect to trainable parameters; all learning
    happens in the downstream ``WhaleVAD`` model. Using an ``nn.Module`` here
    rather than a plain function buys us automatic device placement of the
    Hann window and compatibility with ``nn.DataParallel``.
    """

    def __init__(self):
        super().__init__()
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.hop_length = cfg.HOP_LENGTH

        # Hann window is registered as a buffer (not a parameter) so it moves
        # with the module but receives no gradients and is not saved as a
        # tunable weight.
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute phase-aware spectrogram features.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform tensor of shape ``(batch, n_samples)`` or ``(n_samples,)``.
            Values should be in the usual ``[-1, 1]`` range; the model is
            robust to small amplitude variations thanks to the per-frequency
            normalization below.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(batch, 3, n_freq, n_frames)``. The
            three channels contain magnitude, ``cos(phase)``, and
            ``sin(phase)`` respectively. See module docstring for shape
            examples.
        """
        # Accept both unbatched and batched inputs for convenience.
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Complex STFT. ``center=False`` disables reflective padding; the
        # paper uses unpadded STFTs because the collar mechanism in the
        # dataset ensures no interesting events occur at segment boundaries.
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )  # shape: (B, n_freq, n_frames), complex

        # Per-frequency mean subtraction (Section 5.2). The official
        # Geldenhuys reference implementation
        # (whalevad/spectrogram.py: SpectrogramExtractor.forward) subtracts
        # the mean of the COMPLEX STFT values along the time axis, BEFORE
        # taking magnitude and phase:
        #
        #     feat = stft - stft.mean(time_axis)
        #     mag, angle = feat.abs(), feat.angle()
        #
        # This is mathematically distinct from subtracting the mean of the
        # magnitudes (our previous behaviour). Differences:
        #   1. Their magnitude is always >= 0, ours could go negative
        #   2. Their phase reflects the demeaned signal; ours kept the
        #      raw STFT phase
        #
        # The official trained checkpoint was fitted to inputs produced by
        # this complex-demean pipeline, so any deviation here yields
        # silently-degraded inference even with correct model weights —
        # which was the root cause of our reproduction gap.
        if cfg.NORM_FEATURES == "demean":
            stft = stft - stft.mean(dim=-1, keepdim=True)

        # Decompose the (possibly demeaned) complex spectrum into
        # magnitude and phase components.
        mag = stft.abs()
        angle = stft.angle()

        # Trigonometric phase encoding (Section 5.4). Using (cos, sin) rather
        # than raw phase angles avoids the 2π discontinuity that confuses
        # convolutional layers, while still allowing the model to learn
        # phase-sensitive features such as onset timing.
        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)

        # Stack the three channels along a new dimension. Output:
        #   (B, n_freq, n_frames) × 3   →   (B, 3, n_freq, n_frames)
        feat = torch.stack([mag, cos_ph, sin_ph], dim=1)
        return feat


# ----------------------------------------------------------------------
# Module self-test. Run ``python spectrogram.py`` to verify output shapes
# independently of the rest of the pipeline.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    extractor = SpectrogramExtractor()
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)  # 2× 30-second segments
    feat = extractor(audio)
    print(f"Input:  {audio.shape}")
    print(f"Output: {feat.shape}")
    # Expected: (2, 3, 129, ~1449)
    #   - 129 frequency bins (n_fft // 2 + 1)
    #   - ~1449 time frames (30s / 20ms, less boundary frames from center=False)
