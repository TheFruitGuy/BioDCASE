"""
Multi-Resolution Phase-Aware STFT Frontend
==========================================

Drop-in replacement for :class:`spectrogram_final.SpectrogramExtractor`.
Runs ``N`` parallel STFTs at a shared FFT size and hop length but with
different window lengths, then channel-concatenates their phase-aware
features. The downstream :class:`model_final.WhaleVAD` consumes the
concatenated tensor unchanged once built with the matching
``feat_channels``.

Motivation
----------
A single STFT window is a Pareto compromise on the time-frequency
resolution tradeoff. For BioDCASE Task 2 the optimal choice differs
sharply by call type:

* **Bm-Z A-parts** (~10 s tonals near 27 Hz): want narrow frequency
  bins; can accept temporal smearing -> long window.
* **Bp20 pulses** (~1 s bursts near 20 Hz): intermediate.
* **D-calls** (1-3 s downsweeps over 20-120 Hz): want fine time
  resolution to track instantaneous frequency; can accept coarser
  frequency bins -> short window.

The BioDCASE 2025 winning submission (Marolt & Bones, val macro F1 =
0.64) exploits exactly this with three parallel Swin transformers at
three STFT scales. The multi-resolution *input* is the load-bearing
idea; the downstream model is secondary. This module re-uses the
existing CNN-BiLSTM and feeds it concatenated features rather than
replacing the trunk.

Output shape
------------
``(B, 3 * len(win_lengths), n_fft // 2 + 1, T)``

Each branch contributes three channels in branch-grouped order
(magnitude, cos-phase, sin-phase). The grouping makes the first conv
layer's weights interpretable as per-branch importance and makes
single-branch ablations easy via channel slicing.

Frame alignment
---------------
All branches share ``hop_length`` so they emit the same number of
frames. To make the *frame positions* coincide across branches (which
they wouldn't with ``center=False`` since that places the first frame
at sample ``win_length // 2``), every branch uses ``center=True``.
This adds a small amount of reflective padding at segment boundaries;
the collar mechanism in :mod:`dataset_final` keeps real events well
inside the segment so the padded edges contain no labelled content.

Note that with ``center=True`` the output has ``L // hop + 1`` frames,
which is one more than the per-frame targets emitted by
:class:`dataset_final.WhaleDataset` (``L // stride_samp``). The
existing trimming step ``T = min(logits.size(1), targets.size(1))``
in :mod:`train_final` handles this without modification.
"""

import torch
import torch.nn as nn

import config_final as cfg


# Default branch window lengths in samples. At ``cfg.SAMPLE_RATE = 250``
# these correspond to 64 / 256 / 1024 ms windows. The longest matches the
# current single-resolution baseline (``cfg.WIN_LENGTH == 256``); the two
# shorter branches add the new time-frequency tradeoffs that target
# D-calls and Bp pulses respectively.
DEFAULT_WIN_LENGTHS: tuple[int, ...] = (16, 64, 256)


class MultiResSpectrogramExtractor(nn.Module):
    """
    Phase-aware multi-resolution STFT extractor.

    Parameters
    ----------
    win_lengths : sequence of int, optional
        Window lengths in samples, one per branch. Each value must be
        ``<= n_fft``. Shorter windows are zero-padded inside
        ``torch.stft`` to the shared FFT size, so all branches output
        the same number of frequency bins. Defaults to
        :data:`DEFAULT_WIN_LENGTHS`.
    n_fft : int, optional
        FFT size, shared across branches. Defaults to ``cfg.N_FFT``.
    hop_length : int, optional
        Hop length in samples, shared across branches. Defaults to
        ``cfg.HOP_LENGTH``.

    Attributes
    ----------
    hop_length : int
        Exposed at the same name as :class:`SpectrogramExtractor` so
        the validation path in :mod:`train_final` works unchanged.
    win_lengths : tuple of int
    n_branches : int
    n_out_channels : int
        Equal to ``3 * n_branches`` — the value to pass to
        ``WhaleVAD(feat_channels=...)`` when constructing the model.
    """

    def __init__(
        self,
        win_lengths=DEFAULT_WIN_LENGTHS,
        n_fft: int | None = None,
        hop_length: int | None = None,
    ):
        super().__init__()

        if n_fft is None:
            n_fft = cfg.N_FFT
        if hop_length is None:
            hop_length = cfg.HOP_LENGTH

        self.win_lengths: tuple[int, ...] = tuple(int(w) for w in win_lengths)
        self.n_fft: int = int(n_fft)
        self.hop_length: int = int(hop_length)
        self.n_branches: int = len(self.win_lengths)

        if self.n_branches == 0:
            raise ValueError("win_lengths must contain at least one entry.")

        for w in self.win_lengths:
            if w <= 0:
                raise ValueError(f"win_length must be positive, got {w}.")
            if w > self.n_fft:
                raise ValueError(
                    f"win_length {w} exceeds n_fft {self.n_fft}; "
                    f"increase n_fft or shorten the window."
                )

        # One Hann window per branch, registered as a buffer so .to(device)
        # moves the tensors with the module. A separate buffer per branch
        # (rather than a single shared one trimmed at forward time) keeps
        # the forward pass branch-free.
        for i, w in enumerate(self.win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(w))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_out_channels(self) -> int:
        """Number of output channels, equals ``3 * n_branches``."""
        return 3 * self.n_branches

    def describe(self) -> str:
        """Human-readable summary, useful for logging at startup."""
        ms = [w * 1000.0 / cfg.SAMPLE_RATE for w in self.win_lengths]
        parts = [f"{w}smp/{m:.0f}ms" for w, m in zip(self.win_lengths, ms)]
        return (
            f"MultiResSpectrogramExtractor("
            f"branches={self.n_branches}, "
            f"win=[{', '.join(parts)}], "
            f"n_fft={self.n_fft}, hop={self.hop_length}, "
            f"out_channels={self.n_out_channels})"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _stft_branch(self, audio: torch.Tensor, i: int) -> torch.Tensor:
        """
        Single-branch STFT returning ``(B, 3, n_freq, T)``.

        The three channels are (magnitude, cos(phase), sin(phase)) with
        per-frequency complex-domain demean applied first, matching
        :class:`SpectrogramExtractor`'s ``cfg.NORM_FEATURES == "demean"``
        behaviour on a per-branch basis.
        """
        win = getattr(self, f"window_{i}")
        win_length = self.win_lengths[i]

        # center=True for cross-branch frame alignment. See module docstring.
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window=win,
            center=True,
            return_complex=True,
        )

        if cfg.NORM_FEATURES == "demean":
            stft = stft - stft.mean(dim=-1, keepdim=True)

        mag = stft.abs()
        angle = stft.angle()
        cos_ph = torch.cos(angle)
        sin_ph = torch.sin(angle)
        return torch.stack([mag, cos_ph, sin_ph], dim=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        audio : torch.Tensor
            Waveform of shape ``(B, n_samples)`` or ``(n_samples,)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, 3 * n_branches, n_freq, T)``
            with channels ordered ``[mag_b0, cos_b0, sin_b0, mag_b1,
            cos_b1, sin_b1, ...]``.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        branches = [self._stft_branch(audio, i) for i in range(self.n_branches)]
        return torch.cat(branches, dim=1)
