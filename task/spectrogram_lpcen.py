"""
Phase 13o -- raw-magnitude extra channel (feeds the LEARNABLE PCEN in the model)
================================================================================

Companion to ``model_lpcen``. Channels 0..base_ch-1 are byte-identical to
``SpectrogramExtractor`` (== 13d/13n); the LAST channel is the RAW STFT
magnitude (non-demeaned). The model's ``LearnablePCEN`` turns that raw channel
into a trainable PCEN view before the FDY stem sees it.

Why the raw channel lives in the extractor but the PCEN lives in the model:
``conformer_core``'s optimiser only steps ``model.parameters()``, so a trainable
PCEN cannot sit in a (param-less) extractor. The extractor therefore stops at
the raw magnitude; the trainable AGC is a model submodule. This module is
param-less, exactly like ``PCENChannelExtractor`` -- only the 4th channel
differs (raw magnitude here vs fixed-PCEN there).
"""

import torch
import torch.nn as nn

import config_final as cfg
from spectrogram_final import SpectrogramExtractor


class RawMagChannelExtractor(nn.Module):
    """SpectrogramExtractor channels + one RAW-magnitude channel (last)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.base = SpectrogramExtractor()
        self.hop_length = self.base.hop_length            # eval/validate read this
        self.n_fft = cfg.N_FFT
        self.win_length = cfg.WIN_LENGTH
        self.eps = eps
        self.register_buffer("window", torch.hann_window(self.win_length))
        base_ch = 4 if cfg.DUAL_RESOLUTION else 3
        self.n_out_channels = base_ch + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        base = self.base(audio)                           # (B, base_ch, F, T) == 13d
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=False, return_complex=True,
        )
        rawmag = stft.abs().unsqueeze(1)                  # (B, 1, F, T) raw magnitude
        return torch.cat([base, rawmag], dim=1)           # (B, base_ch+1, F, T)


if __name__ == "__main__":
    torch.manual_seed(0)
    sr = cfg.SAMPLE_RATE
    audio = torch.randn(2, sr * 30) * 0.02

    ext = RawMagChannelExtractor()
    out = ext(audio)
    base = SpectrogramExtractor()(audio)
    T_stft = (sr * 30 - cfg.N_FFT) // cfg.HOP_LENGTH + 1
    print(f"base channels : {tuple(base.shape)}")
    print(f"rawmag-aug out: {tuple(out.shape)}  (expect (2, {base.shape[1] + 1}, 129, {T_stft}))")
    assert out.shape[1] == base.shape[1] + 1 == ext.n_out_channels
    assert out.shape[-1] == T_stft

    # channels 0..base-1 must be byte-identical to 13d
    same = torch.allclose(out[:, :base.shape[1]], base, atol=0, rtol=0)
    print(f"base channels identical to 13d: {same}")
    assert same

    # last channel is the raw magnitude of a center=False STFT
    ref = torch.stft(audio, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                     win_length=cfg.WIN_LENGTH, window=torch.hann_window(cfg.WIN_LENGTH),
                     center=False, return_complex=True).abs()
    assert torch.allclose(out[:, -1], ref, atol=1e-5), "last channel != raw |STFT|"
    print(f"last channel == raw |STFT| : True")

    n_par = sum(p.numel() for p in ext.parameters() if p.requires_grad)
    print(f"trainable params = {n_par} (must be 0)")
    assert n_par == 0
    print("ALL RAW-MAG-CHANNEL SELF-TESTS PASSED")
