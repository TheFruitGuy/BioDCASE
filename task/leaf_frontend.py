"""
LEAF Frontend for Whale-VAD
============================

A drop-in replacement for ``SpectrogramExtractor`` that uses the LEAF
learnable filterbank (Zeghidour et al., arXiv:2101.08596) instead of a
fixed phase-aware STFT. Wraps SpeechBrain's ``Leaf`` module with two
whale-specific adaptations:

1. **Sample-rate-aware configuration.** SpeechBrain's defaults are tuned
   for speech at 16 kHz with a 60-7800 Hz passband; this wrapper exposes
   the relevant arguments and defaults them to whale-appropriate values
   read from ``config.py`` (250 Hz SR, 5-125 Hz passband, 1024 ms window,
   20 ms stride to match the baseline STFT frame rate).

2. **Linear Gabor initialization.** Published bioacoustic studies
   (Anderson, Kinnunen & Harte 2023, ICASSP; Schlüter & Gutenbrunner 2022,
   EUSIPCO) consistently find that LEAF filters barely move from their
   initialization, so init choice dominates the result. For calls living
   in a narrow 5-150 Hz band the default mel spacing wastes filter
   capacity at low frequencies; linear spacing distributes filters evenly
   across the relevant band. This module applies a linear init by default.

Output format
-------------
For input audio of shape ``(B, n_samples)``, returns a tensor of shape
``(B, 1, n_filters, n_time_frames)``. This matches the rank of
``SpectrogramExtractor``'s output (which was ``(B, 3, 129, T)``) so the
rest of the pipeline can consume it unchanged — just construct
``WhaleVAD(feat_channels=1)`` instead of the default ``feat_channels=3``.

Why ``n_filters=128`` by default
--------------------------------
The downstream ``WhaleVAD.filterbank`` Conv2d uses a (7, 1) kernel with
stride (3, 1) along frequency, so a 128-filter LEAF output gives
``(128-7)/3 + 1 = 41`` bins entering ``feat_extractor``, almost exactly
the 41 bins produced by the original 129-bin STFT path. This keeps
downstream feature-map sizes comparable to baseline. Using fewer filters
(e.g. 64, which is what the bioacoustics literature recommends as the
"minimum sufficient" count) eventually breaks one of the MaxPool layers
in ``feat_extractor`` because the frequency dim collapses below the
kernel size; if you want to try fewer filters, also widen / shorten the
MaxPool kernels in ``model.py``.

Dependencies
------------
Requires ``speechbrain`` (tested with 1.x):
    pip install speechbrain
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
        Number of Gabor filters. See module docstring for why 128 is the
        default rather than the 32-64 typical for bioacoustics.
    sample_rate : int, optional
        Audio sample rate in Hz. Defaults to ``cfg.SAMPLE_RATE``.
    window_len_ms : float, optional
        Gabor convolution window length in ms. Defaults to a value that
        reproduces ``cfg.WIN_LENGTH`` samples at ``sample_rate``.
    window_stride_ms : float, optional
        Output frame stride in ms. Defaults to reproducing
        ``cfg.HOP_LENGTH`` samples.
    min_freq : float, default=5.0
        Lowest filter center in Hz. The lowest target calls (fin whale
        ~15 Hz, Antarctic blue Z-call ~17 Hz) sit comfortably above 5 Hz
        with room for Gabor bandwidth tails.
    max_freq : float, optional
        Highest filter center in Hz. Defaults to Nyquist.
    use_pcen : bool, default=False
        Apply learnable PCEN compression. Off by default so the first
        ablation isolates the learnable-filterbank effect from the
        learnable-compression effect. Re-enable as a separate experiment.
    init : {"linear", "mel"}, default="linear"
        Gabor filter initialization. "linear" spaces filter centers
        uniformly across ``[min_freq, max_freq]`` (recommended for narrow
        low-frequency passbands); "mel" uses SpeechBrain's default.
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

        # Defaults derived from config so the LEAF frame rate exactly
        # matches the STFT frame rate of the baseline.
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

        # Defer the SpeechBrain import so this module can at least be
        # imported (e.g. for type checking) without the dep installed.
        try:
            from speechbrain.lobes.features import Leaf
        except ImportError as e:
            raise ImportError(
                "LeafFrontend requires speechbrain. Install with:\n"
                "    pip install speechbrain"
            ) from e

        # SpeechBrain's Leaf takes n_fft only for internal init bookkeeping;
        # match the equivalent of cfg.WIN_LENGTH in samples.
        n_fft = max(256, int(round(window_len_ms * sample_rate / 1000.0)))

        self.leaf = Leaf(
            out_channels=n_filters,
            window_len=window_len_ms,
            window_stride=window_stride_ms,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            use_pcen=use_pcen,
            learnable_pcen=use_pcen,
            use_legacy_complex=False,
            skip_transpose=True,    # return (B, F, T), not (B, T, F)
            n_fft=n_fft,
        )

        # Override SpeechBrain's mel-based Gabor init if requested. Done
        # after construction so SpeechBrain runs its own init first; we
        # then overwrite the parameter tensor in place.
        if init == "linear":
            self._reinit_gabor_linear()
        elif init != "mel":
            raise ValueError(
                f"Unknown init mode '{init}', expected 'linear' or 'mel'."
            )

    # ------------------------------------------------------------------
    # Custom Gabor initialization
    # ------------------------------------------------------------------

    def _reinit_gabor_linear(self) -> None:
        """
        Replace mel-spaced Gabor centers/bandwidths with linear spacing.

        SpeechBrain's ``GaborConv1d`` stores its parameters as a single
        tensor of shape ``(n_filters, 2)``, where column 0 is the
        normalized center frequency (mu, in ``[0, 0.5]`` where 0.5 is
        Nyquist) and column 1 is the bandwidth-related sigma in the same
        normalized units.

        We use ``2.355`` as the FWHM-to-σ factor (a Gaussian's
        full-width-half-max is ``2 sqrt(2 ln 2) σ``), which makes the
        bandwidth equal to the spacing between adjacent centers — neighboring
        filters cover roughly half each other's response.
        """
        gabor = self.leaf.complex_conv

        # Be defensive about SpeechBrain's evolving naming.
        param = None
        for name in ("kernel", "_kernel"):
            if hasattr(gabor, name):
                attr = getattr(gabor, name)
                if isinstance(attr, torch.nn.Parameter):
                    param = attr
                    break
        if param is None:
            named = list(gabor.named_parameters())
            raise RuntimeError(
                "Could not locate the Gabor parameter tensor on "
                "speechbrain.lobes.features.Leaf.complex_conv. "
                f"Available parameters: {[n for n, _ in named]}. "
                "You may need to update _reinit_gabor_linear() for your "
                "speechbrain version."
            )

        # Compute linear-spaced centers and matched bandwidths, in the
        # normalized-frequency units the parameter expects.
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
        Run LEAF and shape its output to match SpectrogramExtractor.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform of shape ``(B, n_samples)`` or ``(n_samples,)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, 1, n_filters, n_time_frames)``.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # (B, n_filters, T_frames) — skip_transpose=True keeps it channels-first.
        x = self.leaf(audio)
        # Add a singleton channel dim so downstream code that expects 4-D
        # input (B, C, F, T) works unchanged with WhaleVAD(feat_channels=1).
        x = x.unsqueeze(1)
        return x

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------

    def gabor_param_groups(
        self, base_lr: float, leaf_lr_scale: float = 0.1,
    ) -> list[dict]:
        """
        Build optimizer parameter groups with a smaller LR for LEAF.

        LEAF parameters are notoriously sensitive: the original paper and
        follow-up bioacoustic studies use 5-10x smaller learning rates
        for the frontend than for the classifier. The default scale of
        0.1 (10x smaller) is a safe starting point — increase later if
        you observe LEAF underfitting (filters still in their init
        positions after training).

        Example
        -------
        ::

            extractor = LeafFrontend(...).to(device)
            model = WhaleVAD(num_classes=cfg.n_classes(),
                             feat_channels=1).to(device)
            groups = (
                extractor.gabor_param_groups(cfg.LR, leaf_lr_scale=0.1)
                + [{"params": model.parameters(), "lr": cfg.LR}]
            )
            optimizer = AdamW(groups, weight_decay=cfg.WEIGHT_DECAY,
                              betas=(cfg.BETA1, cfg.BETA2))
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
    n = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"LEAF trainable params: {n:,}")

    # Verify gradients flow to LEAF
    loss = feat.pow(2).mean()
    loss.backward()
    nonzero = sum(
        1 for p in extractor.parameters()
        if p.grad is not None and p.grad.abs().max() > 0
    )
    total = sum(1 for p in extractor.parameters() if p.requires_grad)
    print(f"Params with non-zero gradient: {nonzero}/{total}")
