"""
FilterAugment + frequency warping for the Conformer final stack (phase 13f)
===========================================================================

Frequency-only, train-time spectrogram augmentation:

* **FilterAugment** (Nam et al., ICASSP 2022) — multiplies random frequency
  bands by random gains. Applied to the MAGNITUDE channel(s) only (channel 0,
  the long-window magnitude, and channel 3, the short-window magnitude under
  dual-resolution); the cos/sin phase channels are left untouched, since a gain
  on phase is meaningless. Reported +6.50% PSDS in the original SED study vs
  +2.13% for plain frequency masking.

* **Frequency warping** — a per-sample monotonic single-pivot warp of the
  frequency axis applied to ALL channels (it is a re-indexing of frequency, so
  phase warps consistently with magnitude). A frequency-dependent distortion in
  the spirit of FreDNet's frequency warp.

Both transforms touch only the frequency axis and never time, so the temporal
structure of the calls — the Z-call's unit order, the fin pulse rhythm — is
preserved, unlike time-domain or call-splice augmentation (the latter being a
documented negative result here, precisely because it broke that structure).

Train-time only: pass ``build_spec_augment(...)`` as the ``conformer_core``
``augment_fn``; the engine applies it to the spectrogram inside the training
loop and never at validation.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def magnitude_channels(num_ch: int) -> list[int]:
    """Indices of magnitude channels: 0 (long window) and, with the
    dual-resolution 4th channel, 3 (short window). Phase = channels 1, 2."""
    return [0, 3] if num_ch >= 4 else [0]


def filter_augment(spec: torch.Tensor, mag_ch: list[int], *,
                   db_range: float = 4.5, n_bands: tuple[int, int] = (2, 5),
                   p_linear: float = 0.5) -> torch.Tensor:
    """Random per-band gains on the magnitude channels (band edges shared across
    the batch, gains per-sample). Returns a new tensor; phase channels unchanged.

    With probability ``p_linear`` the gains are linearly interpolated between
    band edges ("linear" type), otherwise constant within each band ("step").
    """
    B, C, Fr, T = spec.shape
    dev = spec.device
    n = random.randint(*n_bands)
    if n <= 1 or Fr < 2:
        return spec
    n_cut = min(n - 1, Fr - 1)
    edges = [0] + sorted(random.sample(range(1, Fr), k=n_cut)) + [Fr]
    nb = len(edges) - 1

    gain_db = torch.empty(B, Fr, device=dev)
    if random.random() < p_linear:                       # linear: gains at edges
        ge = (torch.rand(B, nb + 1, device=dev) * 2 - 1) * db_range
        for i in range(nb):
            lo, hi = edges[i], edges[i + 1]
            t = torch.linspace(0, 1, hi - lo, device=dev)            # (w,)
            gain_db[:, lo:hi] = ge[:, i:i + 1] * (1 - t) + ge[:, i + 1:i + 2] * t
    else:                                                # step: one gain per band
        gb = (torch.rand(B, nb, device=dev) * 2 - 1) * db_range
        for i in range(nb):
            gain_db[:, edges[i]:edges[i + 1]] = gb[:, i:i + 1]

    gain = 10.0 ** (gain_db / 20.0)                       # (B, Fr) linear factor
    full = torch.ones(B, C, Fr, device=dev)               # 1.0 on phase channels
    for c in mag_ch:
        if c < C:
            full[:, c, :] = gain
    return spec * full.unsqueeze(-1)                      # broadcast over time


def frequency_warp(spec: torch.Tensor, *, max_warp: float = 0.10) -> torch.Tensor:
    """Per-sample monotonic single-pivot frequency warp (all channels), via
    grid_sample. ``max_warp`` is the maximum fractional shift of the pivot's
    source location; the time axis is left identity."""
    B, C, Fr, T = spec.shape
    dev = spec.device
    pivot = torch.rand(B, 1, device=dev) * 0.6 + 0.2                 # (B,1) in (0.2,0.8)
    shift = (torch.rand(B, 1, device=dev) * 2 - 1) * max_warp
    ps = (pivot + shift).clamp(0.05, 0.95)                           # warped pivot source
    u = torch.linspace(0, 1, Fr, device=dev).unsqueeze(0).expand(B, -1)   # (B, Fr)
    below = u <= pivot
    src = torch.where(below,
                      u / pivot * ps,
                      ps + (u - pivot) / (1 - pivot) * (1 - ps))      # (B, Fr) in [0,1]
    src_y = src.clamp(0, 1) * 2 - 1                                   # -> [-1, 1]
    xt = torch.linspace(-1, 1, T, device=dev).view(1, 1, T).expand(B, Fr, T)
    yt = src_y.unsqueeze(2).expand(B, Fr, T)
    grid = torch.stack([xt, yt], dim=-1)                             # (B, Fr, T, 2)
    return F.grid_sample(spec, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def build_spec_augment(*, db_range: float = 4.5, n_bands: tuple[int, int] = (2, 5),
                       p_linear: float = 0.5, freq_warp: bool = True,
                       max_warp: float = 0.10, p: float = 1.0):
    """Return a train-time ``augment_fn(spec) -> spec`` combining FilterAugment
    and (optionally) frequency warping. Magnitude channels are inferred from the
    spectrogram width, so it works for both the 3- and 4-channel inputs."""
    def augment(spec: torch.Tensor) -> torch.Tensor:
        if p < 1.0 and random.random() > p:
            return spec
        out = filter_augment(spec, magnitude_channels(spec.shape[1]),
                             db_range=db_range, n_bands=n_bands, p_linear=p_linear)
        if freq_warp:
            out = frequency_warp(out, max_warp=max_warp)
        return out
    return augment


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    for C in (3, 4):
        x = torch.randn(2, C, 129, 50).abs()
        mc = magnitude_channels(C)

        fa = filter_augment(x, mc, db_range=6.0)
        assert fa.shape == x.shape
        assert torch.equal(fa[:, 1:3], x[:, 1:3]), f"phase changed (C={C})"   # invariant
        assert not torch.equal(fa[:, 0], x[:, 0]), f"magnitude unchanged (C={C})"
        if C >= 4:
            assert not torch.equal(fa[:, 3], x[:, 3]), "short-window magnitude unchanged"

        fw = frequency_warp(x, max_warp=0.10)
        assert fw.shape == x.shape and torch.isfinite(fw).all()
        ident = frequency_warp(x, max_warp=0.0)
        assert torch.allclose(ident, x, atol=1e-4), "zero-warp is not identity"

        y = build_spec_augment(db_range=4.5, max_warp=0.10)(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        print(f"C={C}: shape preserved, phase untouched by FilterAugment, "
              f"warp shape + zero-warp identity OK")

    x = torch.randn(2, 3, 129, 40, requires_grad=True)
    build_spec_augment()(x).sum().backward()
    assert x.grad is not None
    print("augment is differentiable")
    print("ALL FILTERAUGMENT TESTS PASSED")
