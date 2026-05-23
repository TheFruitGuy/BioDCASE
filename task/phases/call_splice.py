"""
Call-Splice Augmentation
=========================

Drop-in augmentation that replaces a training sample with a synthetic
(unseen-site host + transplanted call) example. Follows the same
signature as the other audio-domain augmentations in
``augmentations.py`` so it plugs straight into
``phase1_baseline.run_phase1_training``.

The full audio-domain pipeline per sample (when splicing fires):

    1. Sample a host clip from the AADC pool (30 s, unseen site).
    2. Sample a call from the call bank.
    3. FFT-bandpass the call to ``[f_low - margin, f_high + margin]``
       with a cosine transition band (no Gibbs ringing).
    4. Apply a Hann fade-in/fade-out on the call's onset and offset.
    5. Choose a random time position inside the host where the call
       fits, leaving a 0.5 s margin from each edge.
    6. Compute the SNR scaling factor from call energy vs host energy.
    7. Add the scaled call into the host at the chosen position.
    8. Overwrite ``audio[b]`` with the synthetic mix.
    9. Zero ``targets[b]`` and set the planted-call frames to 1 for
       the call's 3-class label.

Why audio-domain (not spec-domain)
----------------------------------
The model's filterbank is learned and sits after this hook. Operating
in the spec domain would bypass the learned frontend's input
distribution, lose the phase channels, and require us to recover
phase artificially. The bandpass-and-add pipeline gives the same
time-frequency rectangle effect with no rectangle artefacts: the
filter's transition band is the frequency ramp, the Hann taper is
the time ramp.

Why replace rather than add
---------------------------
The intent is to expose the model to calls embedded in completely
unseen acoustic backgrounds. Adding a transplanted call on top of
the existing training sample keeps the training-site background
attached to the original call — defeats the point. Replacement is
the correct semantics.
"""

import math
import random
from dataclasses import dataclass
from typing import Any

import torch


# ----------------------------------------------------------------------
# Class index mapping (matches dataset.py's 3-class collapse)
# ----------------------------------------------------------------------

CLASS_NAMES_3 = ["bmabz", "d", "bp"]
CLASS_TO_IDX_3 = {name: i for i, name in enumerate(CLASS_NAMES_3)}


# ----------------------------------------------------------------------
# Lightweight pool wrappers
# ----------------------------------------------------------------------

@dataclass
class CallBank:
    """
    In-memory wrapper over ``call_bank.pt`` for fast augmentation sampling.

    The sampling policy is uniform-across-classes-then-uniform-within-class
    by default. This counteracts the heavy BmZ/BmA bias in the training
    set and gives BpD and other rare classes proportionally more splice
    opportunities. Pass ``balanced=False`` to sample uniformly across all
    calls instead.
    """

    calls:           list[dict]
    by_label_3class: dict[str, list[int]]
    by_label_7class: dict[str, list[int]]
    by_site:         dict[str, list[int]]
    balanced:        bool = True

    @classmethod
    def load(cls, path: str, balanced: bool = True) -> "CallBank":
        bank = torch.load(path, map_location="cpu")
        return cls(
            calls=bank["calls"],
            by_label_3class=bank["by_label_3class"],
            by_label_7class=bank["by_label_7class"],
            by_site=bank["by_site"],
            balanced=balanced,
        )

    def sample(self, rng: random.Random) -> dict:
        """Draw one call. See class docstring for the sampling policy."""
        if self.balanced:
            label = rng.choice(list(self.by_label_3class.keys()))
            idx = rng.choice(self.by_label_3class[label])
        else:
            idx = rng.randrange(len(self.calls))
        return self.calls[idx]


@dataclass
class SpliceHostPool:
    """
    In-memory wrapper over ``splice_host_pool.pt``.

    Sampling is uniform across donor sites (rather than uniform across
    clips) so a site with many clips doesn't dominate. This keeps the
    site-level variety high.
    """

    hosts:        list[dict]
    by_site:      dict[str, list[int]]
    duration_s:   float

    @classmethod
    def load(cls, path: str) -> "SpliceHostPool":
        pool = torch.load(path, map_location="cpu")
        return cls(
            hosts=pool["hosts"],
            by_site=pool["by_site"],
            duration_s=pool["config"]["duration_s"],
        )

    def sample(self, rng: random.Random) -> dict:
        site = rng.choice(list(self.by_site.keys()))
        idx = rng.choice(self.by_site[site])
        return self.hosts[idx]


@dataclass
class CallSpliceState:
    """State object passed through the augmentation hook."""
    bank: CallBank
    pool: SpliceHostPool
    rng:  random.Random


# ----------------------------------------------------------------------
# FFT bandpass with cosine transition band
# ----------------------------------------------------------------------

def fft_bandpass(
    audio: torch.Tensor,
    f_low_hz: float,
    f_high_hz: float,
    sample_rate: int,
    margin_hz: float = 4.0,
    transition_hz: float = 2.0,
) -> torch.Tensor:
    """
    Zero-phase bandpass via FFT with a cosine taper at the band edges.

    Parameters
    ----------
    audio : torch.Tensor, shape (n_samples,) or (..., n_samples)
        Input waveform.
    f_low_hz, f_high_hz : float
        Pass band in Hz. The actual flat region is
        ``[f_low - margin, f_high + margin]``; outside this, the
        magnitude tapers as a cosine to zero across
        ``transition_hz`` of additional bandwidth.
    sample_rate : int
    margin_hz : float
        Padding around the annotated bbox. Compensates for annotator
        variability and for the call's spectral tails.
    transition_hz : float
        Width of each cosine roll-off. Wider transitions are smoother
        but leak more out-of-band energy.

    Returns
    -------
    torch.Tensor with the same shape as ``audio``.
    """
    n = audio.shape[-1]
    spec = torch.fft.rfft(audio, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate).to(audio.device)

    lo_flat = max(0.5, f_low_hz - margin_hz)
    hi_flat = min(sample_rate / 2.0 - 0.5, f_high_hz + margin_hz)
    lo_zero = max(0.0, lo_flat - transition_hz)
    hi_zero = min(sample_rate / 2.0, hi_flat + transition_hz)

    mask = torch.zeros_like(freqs)
    # Flat pass band.
    flat = (freqs >= lo_flat) & (freqs <= hi_flat)
    mask = torch.where(flat, torch.ones_like(mask), mask)
    # Lower transition: cosine from 0 (at lo_zero) to 1 (at lo_flat).
    lo_tr = (freqs >= lo_zero) & (freqs < lo_flat)
    if (hi_flat - lo_flat) > 0 and transition_hz > 0:
        t = (freqs - lo_zero) / max(transition_hz, 1e-6)
        cos_in = 0.5 - 0.5 * torch.cos(math.pi * t.clamp(0, 1))
        mask = torch.where(lo_tr, cos_in, mask)
    # Upper transition: cosine from 1 (at hi_flat) to 0 (at hi_zero).
    hi_tr = (freqs > hi_flat) & (freqs <= hi_zero)
    if transition_hz > 0:
        t = (hi_zero - freqs) / max(transition_hz, 1e-6)
        cos_out = 0.5 - 0.5 * torch.cos(math.pi * t.clamp(0, 1))
        mask = torch.where(hi_tr, cos_out, mask)

    return torch.fft.irfft(spec * mask, n=n, dim=-1)


# ----------------------------------------------------------------------
# Hann taper on call edges
# ----------------------------------------------------------------------

def apply_edge_taper(audio: torch.Tensor, taper_samples: int) -> torch.Tensor:
    """
    Apply a half-Hann fade-in and fade-out to a 1-D waveform.

    A hard cut at the call's onset/offset creates a broadband click
    that the model could learn as a "splice marker." Tapering kills
    the click. The full call's interior energy is unchanged because
    the window is 1.0 in the middle.

    Parameters
    ----------
    audio : torch.Tensor, shape (n_samples,)
    taper_samples : int
        Length of each ramp (each side). Clamped to ``len(audio) // 4``
        so the ramps never meet for very short calls.
    """
    n = audio.shape[-1]
    if taper_samples <= 0 or n <= 1:
        return audio
    taper_samples = min(taper_samples, n // 4)
    if taper_samples < 1:
        return audio
    win = torch.ones(n, device=audio.device, dtype=audio.dtype)
    # Full Hann of length (2 * taper) → split into ramp-up and ramp-down.
    ramp = torch.hann_window(
        2 * taper_samples, periodic=False, dtype=audio.dtype, device=audio.device,
    )
    win[:taper_samples] = ramp[:taper_samples]
    win[-taper_samples:] = ramp[taper_samples:]
    return audio * win


# ----------------------------------------------------------------------
# The augmentation hook
# ----------------------------------------------------------------------

def apply_call_splice(
    *,
    spec: torch.Tensor | None,
    mask: torch.Tensor,
    audio: torch.Tensor,
    targets: torch.Tensor,
    metas: list[dict],
    state: Any,                 # CallSpliceState
    splice_prob: float = 0.5,
    snr_min_db: float = -6.0,
    snr_max_db: float = 6.0,
    taper_ms: float = 50.0,
    bandpass_margin_hz: float = 4.0,
    bandpass_transition_hz: float = 2.0,
    edge_guard_s: float = 0.5,
    sample_rate: int = 250,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample call-splice augmentation. Audio-domain hook.

    For each sample with probability ``splice_prob``:
      - replace its 30 s audio with an unseen-site host clip,
      - plant a bandpassed + tapered call into the host at random offset,
      - rewrite its targets to mark only the planted call's frames.

    The augmentation operates in-place on the input tensors and
    returns them for API symmetry with the other augmentations.

    Parameters
    ----------
    spec, mask, audio, targets, metas, state
        Standard augmentation-hook arguments. ``state`` must be a
        ``CallSpliceState`` instance carrying the call bank, splice
        host pool, and an RNG.
    splice_prob : float
        Per-sample probability of replacement.
    snr_min_db, snr_max_db : float
        SNR range for the planted call relative to host energy.
    taper_ms : float
        Hann fade-in / fade-out length on the call's edges.
    bandpass_margin_hz, bandpass_transition_hz : float
        Bandpass band-edge padding and roll-off width.
    edge_guard_s : float
        Keep at least this much margin between the planted call and
        the 30 s clip's edges, to avoid frames clipped by the STFT
        boundary.
    sample_rate : int
        Audio sample rate. Should match ``cfg.SAMPLE_RATE``.
    """
    B = audio.size(0)
    n_samples = audio.size(1)
    T_target = targets.size(1)
    n_classes = targets.size(2)
    device = audio.device
    rng = state.rng

    target_frame_rate = T_target / (n_samples / sample_rate)
    taper_samples = max(1, int(taper_ms * 1e-3 * sample_rate))

    # NOTE on mutation semantics: the dispatcher in
    # ``phase1_baseline.train_epoch`` reassigns only ``(spec, mask, audio)``
    # from our return value — it does NOT reassign ``targets``. So
    # target updates MUST be in-place on the tensor the caller passed.
    # We clone audio and mask (which we return) but leave ``targets``
    # as the caller's reference so our writes are visible.
    audio = audio.clone()
    mask = mask.clone()

    for b in range(B):
        if rng.random() >= splice_prob:
            continue

        call_entry = state.bank.sample(rng)
        host_entry = state.pool.sample(rng)

        synth_audio, info = splice_one_sample(
            call_entry=call_entry,
            host_entry=host_entry,
            rng=rng,
            n_samples=n_samples,
            sample_rate=sample_rate,
            target_frame_rate=target_frame_rate,
            taper_samples=taper_samples,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            bandpass_margin_hz=bandpass_margin_hz,
            bandpass_transition_hz=bandpass_transition_hz,
            edge_guard_s=edge_guard_s,
            device=device,
            dtype=audio.dtype,
        )

        audio[b] = synth_audio
        targets[b].zero_()
        # The synthetic clip is a clean 30 s of unseen-site audio with
        # only the planted call in it. Every frame is valid → mask=1.
        # Use int 1 instead of float 1.0 so this works whether the
        # caller's mask is bool (train.py) or float (phase1_baseline.py).
        mask[b].fill_(1)

        cls_idx = info["cls_idx"]
        if cls_idx is None or cls_idx >= n_classes:
            # Unrecognized label or model has fewer classes than
            # expected. Skip target marking — sample becomes a pure
            # negative, still useful training signal.
            continue
        f_start = max(0, min(info["f_start"], T_target - 1))
        f_end = max(f_start + 1, min(info["f_end"], T_target))
        targets[b, f_start:f_end, cls_idx] = 1.0

    return spec, mask, audio


# Tells the dispatcher in phase1_baseline.train_epoch to invoke this
# augmentation before the STFT step (it modifies audio, not spec).
apply_call_splice.DOMAIN = "audio"


# ----------------------------------------------------------------------
# Per-sample splice helper (shared by apply_call_splice and the
# offline example-dump utility in train_phase_splice.py).
# ----------------------------------------------------------------------

def splice_one_sample(
    *,
    call_entry: dict,
    host_entry: dict,
    rng: random.Random,
    n_samples: int,
    sample_rate: int,
    target_frame_rate: float,
    taper_samples: int,
    snr_min_db: float,
    snr_max_db: float,
    bandpass_margin_hz: float,
    bandpass_transition_hz: float,
    edge_guard_s: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """
    Build one synthetic (host + planted call) sample.

    Shared between ``apply_call_splice`` (per-batch, in-training) and
    ``dump_synthetic_examples`` (offline, for sanity inspection).
    Returns the synthetic audio plus an info dict describing where
    the planted call lives and which class it belongs to, so callers
    can mark targets or label saved files consistently.

    Returns
    -------
    synth_audio : torch.Tensor, shape (n_samples,)
        Host audio with the bandpassed + tapered call mixed in at the
        chosen position and SNR.
    info : dict
        Keys: ``cls_idx`` (int or None), ``f_start``, ``f_end`` (target
        frame range), ``t_call_start_s``, ``t_call_end_s`` (seconds),
        ``snr_db``, ``label_3class``, ``label_7class``,
        ``source_site``, ``host_site``, ``f_low_hz``, ``f_high_hz``,
        ``host_audio``, ``call_audio_bandpassed`` (raw building blocks
        useful for diagnostic plots).
    """
    # ---- 1) Host audio (already at sample_rate × 30 s) ----
    host_audio = host_entry["audio"].to(device=device, dtype=dtype)
    if host_audio.numel() != n_samples:
        if host_audio.numel() > n_samples:
            host_audio = host_audio[:n_samples]
        else:
            pad = n_samples - host_audio.numel()
            host_audio = torch.nn.functional.pad(host_audio, (0, pad))

    # ---- 2) Bandpassed + tapered call ----
    call_audio = call_entry["audio"].to(device=device, dtype=dtype)
    call_audio_bp = fft_bandpass(
        call_audio,
        f_low_hz=call_entry["f_low_hz"],
        f_high_hz=call_entry["f_high_hz"],
        sample_rate=sample_rate,
        margin_hz=bandpass_margin_hz,
        transition_hz=bandpass_transition_hz,
    )
    call_audio_t = apply_edge_taper(call_audio_bp, taper_samples=taper_samples)

    n_call = int(call_audio_t.numel())
    edge_guard_samples = int(edge_guard_s * sample_rate)
    max_start = n_samples - n_call - edge_guard_samples
    min_start = edge_guard_samples
    if max_start <= min_start:
        # Call too long for this host (with margins). Center it.
        start = max(0, (n_samples - n_call) // 2)
        n_call_eff = min(n_call, n_samples - start)
        call_audio_t = call_audio_t[:n_call_eff]
        n_call = n_call_eff
    else:
        start = rng.randint(min_start, max_start)
    end = start + n_call

    # ---- 3) SNR scaling ----
    call_energy = (call_audio_t ** 2).mean().clamp_min(1e-12)
    host_energy = (host_audio ** 2).mean().clamp_min(1e-12)
    target_snr_db = snr_min_db + rng.random() * (snr_max_db - snr_min_db)
    # SNR_dB = 10*log10(call/host) AFTER scale.
    # ⇒ scale = sqrt(host_energy * 10^(SNR/10) / call_energy)
    scale = math.sqrt(
        host_energy.item() * (10.0 ** (target_snr_db / 10.0)) / call_energy.item()
    )

    # ---- 4) Compose synthetic audio ----
    synth = host_audio.clone()
    synth[start:end] = synth[start:end] + scale * call_audio_t

    # ---- 5) Compute target frame range for the true call interior ----
    label_3 = call_entry["label_3class"]
    cls_idx = CLASS_TO_IDX_3.get(label_3)

    true_offset_samples = int(call_entry.get("true_start_offset_s", 0.0) * sample_rate)
    true_dur_samples = int(call_entry["duration_s"] * sample_rate)
    call_lo = max(0, min(start + true_offset_samples, n_samples - 1))
    call_hi = max(call_lo + 1, min(call_lo + true_dur_samples, n_samples))
    t_call_start_s = call_lo / sample_rate
    t_call_end_s = call_hi / sample_rate
    f_start = int(t_call_start_s * target_frame_rate)
    f_end = int(math.ceil(t_call_end_s * target_frame_rate))

    info = {
        "cls_idx":         cls_idx,
        "f_start":         f_start,
        "f_end":           f_end,
        "t_call_start_s":  t_call_start_s,
        "t_call_end_s":    t_call_end_s,
        "snr_db":          target_snr_db,
        "label_3class":    label_3,
        "label_7class":    call_entry.get("label_7class", "?"),
        "source_site":     call_entry["source_site"],
        "host_site":       host_entry["site"],
        "f_low_hz":        call_entry["f_low_hz"],
        "f_high_hz":       call_entry["f_high_hz"],
    }
    return synth, info
