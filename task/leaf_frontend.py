"""
LEAF Frontend for Whale-VAD
============================

A drop-in replacement for ``SpectrogramExtractor`` that uses the LEAF
learnable filterbank (Zeghidour et al., arXiv:2101.08596) instead of a
fixed phase-aware STFT. Wraps SpeechBrain's ``Leaf`` module (tested
with speechbrain==1.1.0) with two whale-specific adaptations:

1. **Sample-rate-aware configuration.** SpeechBrain's defaults target
   speech at 16 kHz / 60-7800 Hz; this wrapper defaults to 250 Hz SR,
   5-125 Hz passband, 1024 ms window, 20 ms stride to match the baseline
   STFT frame rate one-for-one.

2. **Linear Gabor initialization.** Published bioacoustic studies
   (Anderson, Kinnunen & Harte 2023; Schlüter & Gutenbrunner 2022)
   consistently find that LEAF filters barely move from their init, so
   init choice dominates the result. For narrow low-frequency bands the
   default mel spacing wastes filter capacity; linear spacing
   distributes filters evenly across the passband.

Output format
-------------
For input ``(B, n_samples)``, returns ``(B, 1, n_filters, n_time_frames)``.
This matches the rank of ``SpectrogramExtractor``'s output, so the rest
of the pipeline can consume it unchanged — construct
``WhaleVAD(feat_channels=1)`` instead of the default 3.

Dependencies
------------
::

    pip install speechbrain  # 1.0+, tested with 1.1.0
"""

import torch
import torch.nn as nn

import config as cfg


class LeafFrontend(nn.Module):
    """
    LEAF learnable frontend wrapped for the Whale-VAD pipeline.

    Parameters
    ----------
    n_filters : int, default=128
        Number of Gabor filters. 128 keeps downstream feature-map sizes
        comparable to the baseline 129-bin STFT.
    sample_rate : int, optional
        Sample rate in Hz. Defaults to ``cfg.SAMPLE_RATE``.
    window_len_ms : float, optional
        Gabor window length in ms; defaults to reproducing
        ``cfg.WIN_LENGTH`` samples.
    window_stride_ms : float, optional
        Output frame stride in ms; defaults to reproducing
        ``cfg.HOP_LENGTH`` samples.
    min_freq : float, default=5.0
        Lowest filter center in Hz.
    max_freq : float, optional
        Highest filter center in Hz; defaults to Nyquist.
    use_pcen : bool, default=False
        Apply learnable PCEN compression.
    init : {"linear", "mel"}, default="linear"
        Gabor filter initialization scheme.
    """

    def __init__(
        self,
        n_filters: int = 128,
        sample_rate: int | None = None,
        window_len_ms: float | None = None,
        window_stride_ms: float | None = None,
        min_freq: float = 5.0,
        max_freq: float | None = None,
        use_pcen: bool = False,
        init: str = "linear",
    ):
        super().__init__()

        sample_rate = sample_rate if sample_rate is not None else cfg.SAMPLE_RATE
        window_len_ms = (
            window_len_ms if window_len_ms is not None
            else cfg.WIN_LENGTH / sample_rate * 1000.0
        )
        window_stride_ms = (
            window_stride_ms if window_stride_ms is not None
            else cfg.HOP_LENGTH / sample_rate * 1000.0
        )
        max_freq = max_freq if max_freq is not None else sample_rate / 2.0

        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.window_len_ms = window_len_ms
        self.window_stride_ms = window_stride_ms
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.use_pcen = use_pcen
        self.init_mode = init

        try:
            from speechbrain.lobes.features import Leaf
        except ImportError as e:
            raise ImportError(
                "LeafFrontend requires speechbrain. Install with:\n"
                "    pip install speechbrain"
            ) from e

        n_fft = max(256, int(round(window_len_ms * sample_rate / 1000.0)))

        # NOTE: speechbrain >=1.0 requires `in_channels` (or `input_shape`).
        # For mono audio, in_channels=1.
        self.leaf = Leaf(
            in_channels=1,
            out_channels=n_filters,
            window_len=window_len_ms,
            window_stride=window_stride_ms,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            use_pcen=use_pcen,
            learnable_pcen=use_pcen,
            use_legacy_complex=False,
            skip_transpose=True,
            n_fft=n_fft,
        )

        if init == "linear":
            self._reinit_gabor_linear()
        elif init != "mel":
            raise ValueError(
                f"Unknown init mode '{init}', expected 'linear' or 'mel'."
            )

    # ------------------------------------------------------------------
    # Compatibility shim: expose `hop_length` so the existing validate()
    # in train.py (which calls ``spec_extractor.hop_length``) works
    # unchanged when LeafFrontend is dropped in for SpectrogramExtractor.
    # ------------------------------------------------------------------
    @property
    def hop_length(self) -> int:
        """Frame stride in samples — for validate() / postprocessing compat."""
        return int(round(self.window_stride_ms * self.sample_rate / 1000.0))

    # ------------------------------------------------------------------
    # Custom Gabor initialization
    # ------------------------------------------------------------------

    def _reinit_gabor_linear(self) -> None:
        """
        Replace SpeechBrain's mel-spaced Gabor centers/bandwidths with a
        linear spacing across ``[min_freq, max_freq]``.

        The Gabor parameter tensor has shape ``(n_filters, 2)``: column 0
        is the normalized center frequency (mu, in ``[0, 0.5]``, where
        0.5 = Nyquist) and column 1 is sigma in the same units.
        """
        gabor = self.leaf.complex_conv

        param = None
        # SpeechBrain has used several names across versions; try the
        # likely ones first, then scan all params for an (n_filters, 2) tensor.
        for name in ("kernel", "_kernel", "mu", "_mu"):
            if hasattr(gabor, name):
                attr = getattr(gabor, name)
                if isinstance(attr, torch.nn.Parameter):
                    param = attr
                    break
        if param is None:
            for n, p in gabor.named_parameters():
                if p.dim() == 2 and p.shape == (self.n_filters, 2):
                    param = p
                    break
        if param is None:
            named = list(gabor.named_parameters())
            raise RuntimeError(
                "Could not locate the Gabor (mu, sigma) parameter tensor on "
                "speechbrain Leaf.complex_conv. "
                f"Parameters present: {[n for n, _ in named]}. "
                "Update _reinit_gabor_linear() for this speechbrain version."
            )

        min_mu = self.min_freq / self.sample_rate
        max_mu = self.max_freq / self.sample_rate
        new_mu = torch.linspace(min_mu, max_mu, self.n_filters)
        spacing = (max_mu - min_mu) / max(self.n_filters - 1, 1)
        new_sigma = torch.full((self.n_filters,), spacing / 2.355)

        new_kernel = torch.stack([new_mu, new_sigma], dim=-1)
        with torch.no_grad():
            param.copy_(new_kernel.to(param.device).to(param.dtype))

        print(
            f"[LeafFrontend] Linear Gabor init: "
            f"{self.n_filters} filters across "
            f"[{self.min_freq:.1f}, {self.max_freq:.1f}] Hz, "
            f"spacing = {spacing * self.sample_rate:.2f} Hz"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        ``(B, n_samples)`` -> ``(B, 1, n_filters, n_time_frames)``.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # speechbrain Leaf accepts (B, T) for mono audio. If a future
        # version requires explicit (B, T, 1), the fallback below covers it.
        try:
            x = self.leaf(audio)
        except (RuntimeError, ValueError):
            x = self.leaf(audio.unsqueeze(-1))

        # Output is (B, n_filters, T_frames) with skip_transpose=True.
        # Add a singleton channel for downstream WhaleVAD(feat_channels=1).
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------

    def gabor_param_groups(
        self, base_lr: float, leaf_lr_scale: float = 0.1,
    ) -> list[dict]:
        """
        Optimizer param group for LEAF parameters with a smaller LR.
        10× smaller (0.1) is the standard starting point.
        """
        return [{
            "params": [p for p in self.parameters() if p.requires_grad],
            "lr": base_lr * leaf_lr_scale,
            "name": "leaf",
        }]


# ----------------------------------------------------------------------
# Self-test. Run: python leaf_frontend.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    extractor = LeafFrontend(n_filters=128, init="linear", use_pcen=False)
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    feat = extractor(audio)
    print(f"Input:  {audio.shape}")
    print(f"Output: {feat.shape}")
    print(f"hop_length: {extractor.hop_length} samples")
    n = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"LEAF trainable params: {n:,}")

    loss = feat.pow(2).mean()
    loss.backward()
    nonzero = sum(
        1 for p in extractor.parameters()
        if p.grad is not None and p.grad.abs().max() > 0
    )
    total = sum(1 for p in extractor.parameters() if p.requires_grad)
    print(f"Params with non-zero gradient: {nonzero}/{total}")
