"""
LEAF Frontend for Whale-VAD
============================

A drop-in replacement for ``SpectrogramExtractor`` that uses the LEAF
learnable filterbank (Zeghidour et al., arXiv:2101.08596) instead of a
fixed phase-aware STFT. Wraps SpeechBrain's ``Leaf`` (tested with
speechbrain==1.1.0).

Whale-specific adaptations
--------------------------
1. **Sample-rate-aware configuration.** SpeechBrain's defaults target
   speech at 16 kHz / 60-7800 Hz; this wrapper defaults to 250 Hz SR,
   5-125 Hz passband, 1024 ms window, 20 ms stride.

2. **Linear Gabor initialization.** Published bioacoustic studies
   (Anderson, Kinnunen & Harte 2023; Schlüter & Gutenbrunner 2022)
   find that LEAF filters barely move from their init. For narrow
   low-frequency bands mel spacing wastes filter capacity; linear
   spacing distributes filters evenly across the whale band.

3. **Default log compression.** SpeechBrain's ``Leaf`` with
   ``use_pcen=False`` applies no compression, leaving the output with
   a ~1.0 bias floor and tiny variation on real whale audio (whose
   waveform amplitude is small to begin with). That degenerate input
   makes the downstream model collapse to predicting zeros (F1=0).
   This wrapper defaults to ``compression="log"``: subtracting the
   floor and applying ``log(clamp(x, 1e-6))`` expands the dynamic
   range into something BatchNorm can normalize. PCEN remains
   available via ``compression="pcen"``.

SpeechBrain GaborConv1d parameter conventions (verified empirically
against the mel init values for speechbrain==1.1.0)
-----------------------------------------------------------------
The Gabor parameter tensor on ``leaf.complex_conv`` has shape
``(n_filters, 2)``:

  - **Column 0 (mu)**: center frequency in angular radians,
    ``mu = π · f / Nyquist``. Range ``[0, π]`` maps to ``[0, Nyquist]`` Hz.
  - **Column 1 (sigma)**: Gabor envelope's temporal standard deviation
    in samples. Frequency FWHM = ``2.3548 / sigma``.

Output format
-------------
``(B, n_samples)`` -> ``(B, 1, n_filters, n_time_frames)``. Build
``WhaleVAD(feat_channels=1)`` to consume.
"""

import math
import torch
import torch.nn as nn

import config as cfg


class LeafFrontend(nn.Module):
    """
    LEAF learnable frontend for the Whale-VAD pipeline.

    Parameters
    ----------
    compression : {"log", "pcen", "none"}
        Output post-processing.
        - "log": apply ``log(clamp(x, 1e-6))`` after LEAF, expanding
          dynamic range. SAFE DEFAULT — comparable to log-mel.
        - "pcen": delegate to SpeechBrain's learnable PCEN (sets
          ``use_pcen=True`` and ``learnable_pcen=True`` on the
          underlying ``Leaf`` module).
        - "none": no compression. Empirically broken on whale data;
          provided only for diagnostic ablations.
    """

    def __init__(
        self,
        n_filters: int = 128,
        sample_rate: int | None = None,
        window_len_ms: float | None = None,
        window_stride_ms: float | None = None,
        min_freq: float = 5.0,
        max_freq: float | None = None,
        compression: str = "log",
        init: str = "linear",
        verbose: bool = True,
        # Backward-compat: --use_pcen still works (overrides compression).
        use_pcen: bool | None = None,
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

        if use_pcen is True:
            compression = "pcen"
        if compression not in ("log", "pcen", "none"):
            raise ValueError(
                f"compression must be 'log', 'pcen', or 'none'; got {compression!r}"
            )

        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.window_len_ms = window_len_ms
        self.window_stride_ms = window_stride_ms
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.compression = compression
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

        use_pcen_internal = (compression == "pcen")
        self.leaf = Leaf(
            in_channels=1,
            out_channels=n_filters,
            window_len=window_len_ms,
            window_stride=window_stride_ms,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            use_pcen=use_pcen_internal,
            learnable_pcen=use_pcen_internal,
            use_legacy_complex=False,
            skip_transpose=True,
            n_fft=n_fft,
        )

        if verbose:
            print(f"[LeafFrontend] compression={compression}, "
                  f"init={init}, n_filters={n_filters}")
            self._print_gabor_state("after SpeechBrain mel init")

        if init == "linear":
            self._reinit_gabor_linear()
            if verbose:
                self._print_gabor_state("after our linear init")
        elif init != "mel":
            raise ValueError(
                f"Unknown init mode '{init}', expected 'linear' or 'mel'."
            )

    @property
    def hop_length(self) -> int:
        return int(round(self.window_stride_ms * self.sample_rate / 1000.0))

    # ------------------------------------------------------------------
    # Gabor parameter introspection + reinitialization
    # ------------------------------------------------------------------

    def _gabor_param(self) -> torch.nn.Parameter:
        gabor = self.leaf.complex_conv
        for name in ("weights", "kernel", "_kernel", "mu", "_mu"):
            attr = getattr(gabor, name, None)
            if isinstance(attr, torch.nn.Parameter):
                if attr.dim() == 2 and attr.shape == (self.n_filters, 2):
                    return attr
        candidates = [
            (n, p) for n, p in gabor.named_parameters()
            if p.dim() == 2 and p.shape == (self.n_filters, 2)
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            raise RuntimeError(
                "Multiple parameters of shape "
                f"({self.n_filters}, 2) found: "
                f"{[n for n, _ in candidates]}."
            )
        raise RuntimeError(
            "Could not locate the Gabor (mu, sigma) parameter tensor. "
            f"GaborConv1d parameters: "
            f"{[(n, tuple(p.shape)) for n, p in gabor.named_parameters()]}"
        )

    def _print_gabor_state(self, tag: str) -> None:
        try:
            p = self._gabor_param().detach().cpu()
        except RuntimeError as e:
            print(f"[LeafFrontend] {tag}: could not read Gabor param: {e}")
            return
        mu = p[:, 0]
        sig = p[:, 1]
        sr = self.sample_rate
        mu_hz = mu / math.pi * (sr / 2.0)
        fwhm_hz = (2.3548 / sig.clamp(min=1e-8)) * (sr / 2.0) / math.pi
        print(
            f"[LeafFrontend] {tag}: param shape={tuple(p.shape)}\n"
            f"  mu (Hz)         : min={mu_hz.min().item():.2f} "
            f"max={mu_hz.max().item():.2f} "
            f"med={mu_hz.median().item():.2f}\n"
            f"  sigma (samples) : min={sig.min().item():.4f} "
            f"max={sig.max().item():.4f} "
            f"med={sig.median().item():.4f}\n"
            f"  FWHM (Hz)       : min={fwhm_hz.min().item():.2f} "
            f"max={fwhm_hz.max().item():.2f} "
            f"med={fwhm_hz.median().item():.2f}"
        )

    def _reinit_gabor_linear(self) -> None:
        """Replace mel-spaced Gabor params with linear-in-Hz spacing."""
        param = self._gabor_param()
        sr = self.sample_rate
        nyq = sr / 2.0

        min_mu = math.pi * self.min_freq / nyq
        max_mu = math.pi * self.max_freq / nyq
        new_mu = torch.linspace(min_mu, max_mu, self.n_filters)

        spacing_rad = (max_mu - min_mu) / max(self.n_filters - 1, 1)
        target_fwhm_rad = 2.0 * spacing_rad
        sigma_samples = 2.3548 / target_fwhm_rad

        new_sigma = torch.full((self.n_filters,), float(sigma_samples))
        new_kernel = torch.stack([new_mu, new_sigma], dim=-1)
        with torch.no_grad():
            param.copy_(new_kernel.to(param.device).to(param.dtype))

        spacing_hz = (self.max_freq - self.min_freq) / max(self.n_filters - 1, 1)
        fwhm_hz = target_fwhm_rad * (sr / 2.0) / math.pi
        print(
            f"[LeafFrontend] Linear Gabor init: "
            f"{self.n_filters} filters across "
            f"[{self.min_freq:.1f}, {self.max_freq:.1f}] Hz, "
            f"spacing {spacing_hz:.2f} Hz, FWHM {fwhm_hz:.2f} Hz, "
            f"sigma {sigma_samples:.2f} samples"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        try:
            x = self.leaf(audio)
        except (RuntimeError, ValueError):
            x = self.leaf(audio.unsqueeze(-1))

        if self.compression == "log":
            # LEAF outputs (B, n_filters, T) with bias floor ~1.
            # log(clamp(x, 1e-6)) gives 0 at the floor and expands the
            # information-carrying range above. Comparable to log-mel.
            x = torch.log(x.clamp(min=1e-6))
        # "pcen" already applied inside SpeechBrain Leaf when configured.
        # "none" leaves output untouched.

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
    audio = torch.randn(2, cfg.SAMPLE_RATE * 30) * 0.012  # whale-amplitude

    for comp in ("none", "log", "pcen"):
        print()
        print("=" * 60)
        print(f"Self-test: LEAF with compression={comp}")
        print("=" * 60)
        leaf = LeafFrontend(n_filters=128, init="linear", compression=comp)
        feat = leaf(audio)
        pfs = feat[0, 0].std(dim=-1)
        print(f"  Output shape: {feat.shape}")
        print(f"  Output stats: min={feat.min():.4f} max={feat.max():.4f} "
              f"mean={feat.mean():.4f} std={feat.std():.4f}")
        print(f"  Per-filter time std: "
              f"min={pfs.min():.4f} max={pfs.max():.4f} mean={pfs.mean():.4f}")
        feat.pow(2).mean().backward()
        nonzero = sum(
            1 for p in leaf.parameters()
            if p.grad is not None and p.grad.abs().max() > 0
        )
        total = sum(1 for p in leaf.parameters() if p.requires_grad)
        print(f"  Gradient flow: {nonzero}/{total}")
