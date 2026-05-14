"""
LEAF Frontend for Whale-VAD
============================

A drop-in replacement for ``SpectrogramExtractor`` that uses the LEAF
learnable filterbank (Zeghidour et al., arXiv:2101.08596) instead of a
fixed phase-aware STFT. Wraps SpeechBrain's ``Leaf`` module (tested with
speechbrain==1.1.0).

Whale-specific adaptations
--------------------------
1. **Sample-rate-aware configuration.** SpeechBrain's defaults target
   speech at 16 kHz / 60-7800 Hz; this wrapper defaults to 250 Hz SR,
   5-125 Hz passband, 1024 ms window, 20 ms stride to match the baseline
   STFT frame rate.

2. **Linear Gabor initialization.** Published bioacoustic studies
   (Anderson, Kinnunen & Harte 2023; Schlüter & Gutenbrunner 2022)
   find that LEAF filters barely move from their init, so init choice
   dominates. For narrow low-frequency bands the default mel spacing
   wastes filter capacity; linear spacing distributes filters evenly.

   SpeechBrain's ``GaborConv1d`` stores filter params in a single
   parameter tensor ``self.weights`` of shape ``(n_filters, 2)`` where
   column 0 is the **center frequency in radians** and column 1 is the
   **inverse bandwidth** (also in normalized angular-frequency units).
   See ``_reinit_gabor_linear`` for the unit conversion.

Output format
-------------
``(B, n_samples)`` -> ``(B, 1, n_filters, n_time_frames)``. Matches the
rank of ``SpectrogramExtractor``'s output (which was 3-channel); construct
``WhaleVAD(feat_channels=1)`` to consume.

Dependencies
------------
::

    pip install speechbrain   # 1.0+, tested with 1.1.0
"""

import math
import torch
import torch.nn as nn

import config as cfg


class LeafFrontend(nn.Module):
    """
    LEAF learnable frontend wrapped for the Whale-VAD pipeline.

    Parameters mirror ``speechbrain.lobes.features.Leaf`` with
    whale-appropriate defaults.
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
        verbose: bool = True,
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
        self.verbose = verbose

        try:
            from speechbrain.lobes.features import Leaf
        except ImportError as e:
            raise ImportError(
                "LeafFrontend requires speechbrain. Install with:\n"
                "    pip install speechbrain"
            ) from e

        n_fft = max(256, int(round(window_len_ms * sample_rate / 1000.0)))

        # speechbrain>=1.0 requires `in_channels` or `input_shape`.
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

        # Snapshot what SpeechBrain's mel init produced — useful when
        # comparing against our linear-init values for unit-convention bugs.
        if verbose:
            self._print_gabor_state("after SpeechBrain mel init")

        if init == "linear":
            self._reinit_gabor_linear()
            if verbose:
                self._print_gabor_state("after our linear init")
        elif init != "mel":
            raise ValueError(
                f"Unknown init mode '{init}', expected 'linear' or 'mel'."
            )

    # ------------------------------------------------------------------
    # Compat shim for validate() in train.py (uses spec_extractor.hop_length)
    # ------------------------------------------------------------------
    @property
    def hop_length(self) -> int:
        return int(round(self.window_stride_ms * self.sample_rate / 1000.0))

    # ------------------------------------------------------------------
    # Gabor parameter introspection + reinitialization
    # ------------------------------------------------------------------

    def _gabor_param(self) -> torch.nn.Parameter:
        """
        Locate the GaborConv1d's (mu, sigma) parameter tensor of shape
        (n_filters, 2). SpeechBrain has named this ``weights``,
        ``kernel``, and ``_kernel`` in different versions, so we try
        the known names first, then fall back to shape-based search.
        """
        gabor = self.leaf.complex_conv
        # Known attribute names across SpeechBrain versions.
        for name in ("weights", "kernel", "_kernel", "mu", "_mu"):
            attr = getattr(gabor, name, None)
            if isinstance(attr, torch.nn.Parameter):
                return attr
        # Fall back: scan named_parameters() for the right shape.
        candidates = [
            (n, p) for n, p in gabor.named_parameters()
            if p.dim() == 2 and p.shape == (self.n_filters, 2)
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            raise RuntimeError(
                "Multiple GaborConv1d parameters of shape "
                f"({self.n_filters}, 2) found: "
                f"{[n for n, _ in candidates]}. Cannot disambiguate."
            )
        named = list(gabor.named_parameters())
        raise RuntimeError(
            "Could not locate the Gabor (mu, sigma) parameter tensor on "
            f"speechbrain Leaf.complex_conv. Parameters present: "
            f"{[(n, tuple(p.shape)) for n, p in named]}"
        )

    def _print_gabor_state(self, tag: str) -> None:
        """Print a compact summary of the current Gabor params."""
        try:
            p = self._gabor_param().detach().cpu()
        except RuntimeError as e:
            print(f"[LeafFrontend] {tag}: could not read Gabor param: {e}")
            return
        mu = p[:, 0]
        sig = p[:, 1]
        # Print the param in *raw* tensor units (unit-agnostic), and also
        # converted assuming SpeechBrain's convention is angular radians
        # in [0, π] (the most common Gabor convention).
        sr = self.sample_rate
        mu_hz_radians = (mu / math.pi * (sr / 2)).cpu()
        mu_hz_normalized = (mu * sr).cpu()
        print(
            f"[LeafFrontend] {tag}: param shape={tuple(p.shape)}\n"
            f"  raw mu       : min={mu.min().item():.6f} "
            f"max={mu.max().item():.6f} "
            f"med={mu.median().item():.6f}\n"
            f"  if angular   : min={mu_hz_radians.min().item():.2f} Hz "
            f"max={mu_hz_radians.max().item():.2f} Hz "
            f"med={mu_hz_radians.median().item():.2f} Hz\n"
            f"  if normalized: min={mu_hz_normalized.min().item():.2f} Hz "
            f"max={mu_hz_normalized.max().item():.2f} Hz "
            f"med={mu_hz_normalized.median().item():.2f} Hz\n"
            f"  raw sigma    : min={sig.min().item():.6f} "
            f"max={sig.max().item():.6f} "
            f"med={sig.median().item():.6f}"
        )

    def _reinit_gabor_linear(self) -> None:
        """
        Replace SpeechBrain's mel-spaced Gabor centers/bandwidths with a
        linear spacing across ``[min_freq, max_freq]``.

        SpeechBrain's ``GaborConv1d`` uses **angular frequency in radians**
        for ``mu``: ``mu = 2π f / sample_rate``, so a center frequency f
        Hz maps to ``mu = π · (f / Nyquist)``, with the range being
        ``[0, π]`` rather than ``[0, 0.5]``. The bandwidth-related
        ``sigma`` is in the same units.

        If you observe degenerate output after reinitialization (e.g.
        near-constant features), inspect the snapshot printed before/
        after init: if SpeechBrain's mel init produced ``mu`` values in
        a different range, the unit convention has changed in your
        installed speechbrain and this function needs updating.
        """
        param = self._gabor_param()
        sr = self.sample_rate

        # Convert target Hz -> angular radians.
        # f_hz / nyquist ∈ [0, 1], times π gives [0, π]
        min_mu = math.pi * self.min_freq / (sr / 2.0)
        max_mu = math.pi * self.max_freq / (sr / 2.0)
        new_mu = torch.linspace(min_mu, max_mu, self.n_filters)

        # Bandwidth: aim for adjacent filters to overlap at FWHM.
        # FWHM(σ) ≈ 2.355 σ for a Gaussian, so σ = spacing / 2.355
        # gives roughly half-overlap.
        spacing = (max_mu - min_mu) / max(self.n_filters - 1, 1)
        new_sigma = torch.full((self.n_filters,), spacing / 2.355)

        new_kernel = torch.stack([new_mu, new_sigma], dim=-1)
        with torch.no_grad():
            param.copy_(new_kernel.to(param.device).to(param.dtype))

        spacing_hz = (self.max_freq - self.min_freq) / max(self.n_filters - 1, 1)
        print(
            f"[LeafFrontend] Linear Gabor init "
            f"(angular-radian units, [0, π] => [0, Nyquist={sr/2:.1f}] Hz):\n"
            f"  {self.n_filters} filters across "
            f"[{self.min_freq:.1f}, {self.max_freq:.1f}] Hz, "
            f"spacing {spacing_hz:.2f} Hz "
            f"(mu range [{min_mu:.4f}, {max_mu:.4f}] rad)"
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
        try:
            x = self.leaf(audio)
        except (RuntimeError, ValueError):
            x = self.leaf(audio.unsqueeze(-1))
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x

    # ------------------------------------------------------------------
    # Optimizer param group helper
    # ------------------------------------------------------------------

    def gabor_param_groups(
        self, base_lr: float, leaf_lr_scale: float = 0.1,
    ) -> list[dict]:
        return [{
            "params": [p for p in self.parameters() if p.requires_grad],
            "lr": base_lr * leaf_lr_scale,
            "name": "leaf",
        }]


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Self-test: LEAF with mel init (SpeechBrain default)")
    print("=" * 60)
    leaf_mel = LeafFrontend(n_filters=128, init="mel", use_pcen=False)
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30)
    feat_mel = leaf_mel(audio)
    print(f"  Output shape: {feat_mel.shape}")
    print(f"  Output stats: min={feat_mel.min():.6f} "
          f"max={feat_mel.max():.6f} "
          f"mean={feat_mel.mean():.6f} std={feat_mel.std():.6f}")

    print()
    print("=" * 60)
    print("Self-test: LEAF with linear init (whale-band)")
    print("=" * 60)
    leaf_lin = LeafFrontend(n_filters=128, init="linear", use_pcen=False)
    feat_lin = leaf_lin(audio)
    print(f"  Output shape: {feat_lin.shape}")
    print(f"  Output stats: min={feat_lin.min():.6f} "
          f"max={feat_lin.max():.6f} "
          f"mean={feat_lin.mean():.6f} std={feat_lin.std():.6f}")

    print()
    print(f"Gradient check (linear init):")
    feat_lin.pow(2).mean().backward()
    nonzero = sum(
        1 for p in leaf_lin.parameters()
        if p.grad is not None and p.grad.abs().max() > 0
    )
    total = sum(1 for p in leaf_lin.parameters() if p.requires_grad)
    print(f"  Params with non-zero gradient: {nonzero}/{total}")
