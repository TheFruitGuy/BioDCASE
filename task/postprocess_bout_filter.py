"""
Post-Processing Bout & Frequency Filters
=========================================

Two domain-knowledge filters that sit AFTER `postprocess_predictions` (so
they consume a `list[Detection]`) and BEFORE the event-level scorer
`compute_metrics`. Eval-only — never touches probability arrays, never
re-trains anything.

1. Calling-bout temporal filter (operating on `d` and `bmabz` only)
   For every detection, slide a window of `window_size_sec` centred on
   that detection's start time. Keep the detection iff:
     - the window contains at least `min_calls` detections of the same
       class on the same (dataset, filename), AND
     - at least one of those detections has confidence
       >= `high_conf_threshold`.

2. Strict frequency guardrail (per-class biological band vs adaptive
   out-of-band reference; STFT energy ratio)
   For each candidate detection, average the linear-STFT magnitude over
   the event's time span, then compare mean power in the biological band
   to mean power in an off-band reference strip. If the ratio is below
   `noise_threshold_ratio` the detection is dropped — broadband
   "everywhere" energy is a hallmark of FP hallucinations on noise.

Both filters return a NEW `list[Detection]`. The originals are never
mutated. `bp` (fin 20 Hz) is untouched in the default config because its
calling pattern is too regular to benefit from a bout filter and its band
is narrow.

Audio bands (Antarctic, confirmed by user; cf. SCCE bands are too high)
-----------------------------------------------------------------------
  bmabz : 17–29 Hz biological,  40–90 Hz reference
  d     : 30–100 Hz biological, [5–15 ∪ 110–120] Hz reference (split)
  bp    : 15–28 Hz (no filter applied by default)

Sample rate / STFT
------------------
The model frontend operates on a 129-bin linear STFT at SR=250, N_FFT=256,
HOP_LENGTH=5 (~0.98 Hz/bin, 20 ms/frame). The guardrail computes its own
STFT with the same parameters so frame indexing aligns with detection
times: frame_idx = round(t * SR / HOP) = round(t * 50).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence, Callable
import bisect

import numpy as np

try:
    from scipy.signal import stft as _scipy_stft
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------
# Detection container — mirrors postprocess.Detection so this module is
# usable standalone (e.g. in tests) without pulling in the project deps.
# Production code will import `Detection` from `postprocess` directly;
# the duck-typed attribute access below works equally on either class.
# ---------------------------------------------------------------------
@dataclass
class _DetLike:
    dataset: str
    filename: str
    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0


# ---------------------------------------------------------------------
# Antarctic frequency bands. Edit these if your site-level analysis
# justifies tighter or looser ranges.
# ---------------------------------------------------------------------
BIOLOGICAL_BANDS_HZ: dict[str, tuple[float, float]] = {
    "bmabz": (17.0, 29.0),
    "d":     (30.0, 100.0),
    "bp":    (15.0, 28.0),  # only used if you explicitly enable bp guarding
}

# Adaptive out-of-band references. For each class, a list of (f_lo, f_hi)
# strips that are clearly outside the biological band but still within
# Nyquist (=125 Hz at SR=250).
NOISE_BANDS_HZ: dict[str, list[tuple[float, float]]] = {
    "bmabz": [(40.0, 90.0)],
    "d":     [(5.0, 15.0), (110.0, 120.0)],
    "bp":    [(40.0, 90.0)],
}


# =====================================================================
# Task 1.1 — ICI analysis
# =====================================================================

def _ascii_ici_histogram(
    hist: np.ndarray,
    edges: np.ndarray,
    mode_idx: int,
    median_s: float,
    intervals: np.ndarray,
    bar_width: int = 48,
    pad_trailing: int = 2,
) -> str:
    """
    Render a compact in-terminal histogram of inter-call intervals.

    Layout (one line per bin, trailing zero-rows truncated):

         lo - hi s │ █████████████  count [markers]

    `markers` ∈ {◄ mode, ◄ median, ◄ mode+median}. Bar widths scale to
    `bar_width`. The vertical separator is a single Unicode glyph (│),
    fine in any modern terminal but ASCII-safe alternatives (|, :) work
    too if your environment is hostile to Unicode.
    """
    if hist.sum() == 0:
        return "  (all intervals exceed the histogram range)"

    # Last bin with a non-zero count, plus a small zero-pad so the user
    # can see where the distribution actually trails off.
    nz = np.nonzero(hist)[0]
    last = int(nz.max()) + 1
    n_show = min(len(hist), last + pad_trailing)

    # Identify which bin contains the median, for the marker.
    median_idx = int(np.searchsorted(edges, median_s, side="right")) - 1
    median_idx = max(0, min(len(hist) - 1, median_idx))

    max_count = int(hist[:n_show].max())
    n_excluded = int(intervals.size - hist.sum())

    bar_full = "█"     # Unicode full block; falls back to '#' below if needed.
    lines = []

    # Compact header showing scale.
    lines.append(
        f"  ICI distribution  (bin={edges[1]-edges[0]:g}s, "
        f"bar_full={max_count}, shown 0–{edges[n_show]:g}s"
        + (f", {n_excluded} interval(s) > {edges[-1]:g}s excluded"
           if n_excluded else "")
        + ")"
    )

    for i in range(n_show):
        lo = edges[i]
        hi = edges[i + 1]
        c = int(hist[i])
        bar_len = int(round(c / max_count * bar_width)) if max_count else 0
        bar = bar_full * bar_len

        marker = ""
        if i == mode_idx and i == median_idx:
            marker = "  ◄ mode+median"
        elif i == mode_idx:
            marker = "  ◄ mode"
        elif i == median_idx:
            marker = "  ◄ median"

        lines.append(
            f"  {lo:5.1f}–{hi:5.1f}s │ {bar:<{bar_width}}  {c:4d}{marker}"
        )

    return "\n".join(lines)


def analyze_ici(
    detections: Sequence,
    target_class: str = "d",
    score_threshold: float = 0.2,
    bin_step_s: float = 2.0,
    bin_max_s: float = 120.0,
    show: bool = True,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compute and visualise inter-call intervals for one class.

    Operates on the list of `Detection` events produced by
    `postprocess_predictions`. For each (dataset, filename) it sorts the
    target-class events by start_s and takes consecutive deltas. The
    first detection of every file is skipped (no predecessor).

    Parameters
    ----------
    detections : sequence of Detection-like objects
        Must expose: dataset, filename, label, start_s, confidence.
    target_class : {"bmabz", "d", "bp"}
    score_threshold : float
        Pre-filter on confidence to remove the tail of low-quality events
        whose intervals would dominate the histogram.
    bin_step_s, bin_max_s : float
        Histogram parameters.
    show : bool
        If True, render the matplotlib figure interactively.
    save_path : str or None
        If set, save the figure to this path (PNG/PDF/etc).

    Returns
    -------
    dict with keys:
        intervals : np.ndarray of all ICIs in seconds
        mode_s    : the centre of the most populated bin
        median_s  : sample median
        count     : number of intervals
    """
    # Filter to target class above the noise floor, group by file.
    filtered = [d for d in detections
                if d.label == target_class and d.confidence >= score_threshold]

    by_file: dict[tuple[str, str], list] = {}
    for d in filtered:
        by_file.setdefault((d.dataset, d.filename), []).append(d)

    intervals = []
    for key, dets in by_file.items():
        dets.sort(key=lambda x: x.start_s)
        starts = [x.start_s for x in dets]
        for i in range(1, len(starts)):
            intervals.append(starts[i] - starts[i - 1])
    intervals = np.asarray(intervals, dtype=np.float64)

    summary = {
        "intervals": intervals,
        "mode_s":   float("nan"),
        "median_s": float("nan"),
        "count":    int(intervals.size),
    }

    if intervals.size == 0:
        print(f"[ICI] class={target_class}: no intervals found "
              f"(only {len(filtered)} events above score>={score_threshold:.2f}).")
        return summary

    edges = np.arange(0.0, bin_max_s + bin_step_s, bin_step_s)
    hist, _ = np.histogram(intervals, bins=edges)
    mode_idx = int(np.argmax(hist))
    mode_centre = (edges[mode_idx] + edges[mode_idx + 1]) / 2.0
    median_s = float(np.median(intervals))

    summary["mode_s"] = float(mode_centre)
    summary["median_s"] = median_s

    print(f"[ICI] class={target_class}  n={intervals.size}  "
          f"mode≈{mode_centre:.1f}s  median={median_s:.1f}s  "
          f"min={intervals.min():.1f}s  max={intervals.max():.1f}s")

    # ASCII histogram for at-a-glance terminal inspection. Width adapts
    # to the terminal but capped at 48 so we don't blow up small screens.
    try:
        import shutil
        term_cols = shutil.get_terminal_size((100, 24)).columns
    except Exception:
        term_cols = 100
    bar_w = max(20, min(48, term_cols - 32))
    print(_ascii_ici_histogram(
        hist, edges, mode_idx, median_s, intervals, bar_width=bar_w))

    if show or save_path:
        # Lazy import — matplotlib is only needed for the histogram path.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(intervals, bins=edges, edgecolor="black", linewidth=0.4)
        ax.axvline(mode_centre, color="C3", ls="--",
                   label=f"mode ≈ {mode_centre:.1f} s")
        ax.axvline(median_s, color="C2", ls=":",
                   label=f"median = {median_s:.1f} s")
        ax.set_xlabel("Inter-call interval (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"ICI distribution — class={target_class}  (n={intervals.size})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    return summary


# =====================================================================
# Task 1.2 — Sliding-window bout filter
# =====================================================================

def apply_temporal_bout_filter(
    detections: Sequence,
    target_classes: Iterable[str] = ("d", "bmabz"),
    window_size_sec: float = 3600.0,
    min_calls: int = 3,
    high_conf_threshold: float = 0.7,
) -> list:
    """
    Keep detections that sit inside a calling bout, drop isolates.

    For each detection of a target class:
        let half = window_size_sec / 2
        let W = { dets of same class on same (dataset, filename) whose
                  start_s ∈ [det.start_s - half, det.start_s + half] }
        keep det iff |W| >= min_calls
               AND   max(conf in W) >= high_conf_threshold

    Detections whose class is NOT in `target_classes` (e.g. `bp` by
    default) are passed through untouched.

    Parameters
    ----------
    detections : sequence of Detection-like objects
    target_classes : iterable of class names to filter
    window_size_sec : float
        Total window width. 3600 ⇒ ±0.5 h around each detection.
    min_calls : int
    high_conf_threshold : float

    Returns
    -------
    list of Detection (or whatever input type was)  -- a NEW list, never
    mutates the input. Filtered detections are simply absent; their
    surviving siblings are NOT modified in any way.
    """
    target_classes = set(target_classes)
    half = window_size_sec / 2.0

    # Partition: pass-through everything not in target_classes, filter
    # the rest per (dataset, filename, class) group.
    keep_passthrough = [d for d in detections if d.label not in target_classes]
    to_filter = [d for d in detections if d.label in target_classes]

    # Group detections-to-filter by (dataset, filename, class).
    groups: dict[tuple[str, str, str], list] = {}
    for d in to_filter:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    kept: list = []
    n_dropped_per_class: dict[str, int] = {c: 0 for c in target_classes}
    n_kept_per_class:    dict[str, int] = {c: 0 for c in target_classes}

    for (ds, fn, cls), dets in groups.items():
        # Sort once by start_s for O(log n) window lookups.
        dets.sort(key=lambda x: x.start_s)
        starts = [x.start_s for x in dets]
        confs = [x.confidence for x in dets]

        for i, det in enumerate(dets):
            lo = bisect.bisect_left(starts, det.start_s - half)
            hi = bisect.bisect_right(starts, det.start_s + half)
            n_in_window = hi - lo
            if n_in_window < min_calls:
                n_dropped_per_class[cls] += 1
                continue
            # Cheap max over the window slice. (Could maintain a deque
            # for amortised O(1) but N is small.)
            max_conf_in_window = max(confs[lo:hi])
            if max_conf_in_window < high_conf_threshold:
                n_dropped_per_class[cls] += 1
                continue
            kept.append(det)
            n_kept_per_class[cls] += 1

    print(f"[BoutFilter] window={window_size_sec:.0f}s "
          f"(±{half:.0f}s), min_calls={min_calls}, "
          f"high_conf≥{high_conf_threshold:.2f}")
    for cls in sorted(target_classes):
        k = n_kept_per_class.get(cls, 0)
        d = n_dropped_per_class.get(cls, 0)
        total = k + d
        rate = (d / total * 100) if total else 0.0
        print(f"            {cls:6} kept={k:5}  dropped={d:5}  "
              f"({rate:.1f}% dropped)")

    # Stable ordering: passthrough first, then filtered survivors,
    # then sort by (dataset, filename, start_s) so downstream code that
    # assumes some ordering still works.
    out = keep_passthrough + kept
    out.sort(key=lambda x: (x.dataset, x.filename, x.start_s))
    return out


# =====================================================================
# Task 2.1 — Linear STFT bin mapping
# =====================================================================

def get_linear_bin_indices(
    fmin_hz: float,
    fmax_hz: float,
    sr: int,
    n_fft: int,
) -> tuple[int, int]:
    """
    Map a frequency band [fmin, fmax] to a half-open slice [start, end)
    of linear-STFT bin indices.

    The model uses a real STFT with `n_fft=256` at SR=250 Hz, producing
    `n_fft/2 + 1 = 129` bins spaced ~0.977 Hz apart. Returning bin
    indices (not frequencies) so the caller can `power[..., start:end]`
    directly.

    Parameters
    ----------
    fmin_hz, fmax_hz : float
        Inclusive band edges. Must satisfy 0 <= fmin_hz <= fmax_hz <= SR/2.
    sr, n_fft : int

    Returns
    -------
    (bin_start, bin_end) : tuple of int
        Half-open: use as `arr[..., bin_start:bin_end]`. Guaranteed
        bin_end > bin_start (clipped to at least one bin).

    Raises
    ------
    ValueError if the band is empty or above Nyquist.
    """
    if fmin_hz < 0 or fmax_hz < fmin_hz:
        raise ValueError(f"invalid band [{fmin_hz}, {fmax_hz}]")
    nyquist = sr / 2.0
    if fmin_hz > nyquist:
        raise ValueError(f"fmin_hz={fmin_hz} above Nyquist={nyquist}")
    if fmax_hz > nyquist:
        # Soft-clamp: warn-and-clip rather than fail, since 'just at
        # Nyquist' is a normal request.
        fmax_hz = nyquist

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # length n_fft//2 + 1
    bin_start = int(np.searchsorted(freqs, fmin_hz, side="left"))
    bin_end = int(np.searchsorted(freqs, fmax_hz, side="right"))
    if bin_end <= bin_start:
        bin_end = bin_start + 1  # never return an empty slice
    return bin_start, bin_end


# =====================================================================
# Task 2.2 — Frequency guardrail
# =====================================================================

def _stft_power_numpy(
    audio: np.ndarray, sr: int, n_fft: int, hop_length: int,
) -> np.ndarray:
    """Real STFT → |X|^2. Shape (n_frames, n_bins). Hann window."""
    # Use scipy if available — it's faster and handles edge framing
    # consistently with librosa-style "center=True" by default.
    if _HAS_SCIPY:
        # scipy.signal.stft: returns f, t, Z complex. boundary=None and
        # padded=False to match a vanilla framewise STFT without
        # symmetric padding artefacts.
        f, t, Z = _scipy_stft(
            audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
            window="hann", boundary=None, padded=False,
            return_onesided=True,
        )
        # Z shape (n_bins, n_frames). Transpose to (n_frames, n_bins).
        return (np.abs(Z) ** 2).T.astype(np.float32)

    # Fallback: minimal numpy STFT. Slower but dependency-free.
    win = np.hanning(n_fft).astype(np.float32)
    n = audio.shape[0]
    if n < n_fft:
        # Pad short audio so we still emit one frame.
        audio = np.pad(audio, (0, n_fft - n))
        n = audio.shape[0]
    n_frames = 1 + (n - n_fft) // hop_length
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for k in range(n_frames):
        s = k * hop_length
        frame = audio[s:s + n_fft] * win
        spec = np.fft.rfft(frame, n=n_fft)
        out[k] = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return out


class FrequencyGuardrail:
    """
    Stateful guardrail with a per-file STFT cache.

    Usage pattern (designed for the threshold-tuning inner loop where
    detections are produced many times over the same files):

        guard = FrequencyGuardrail(audio_path_resolver, sr=250,
                                   n_fft=256, hop_length=5)
        kept_dets = guard.filter(detections, noise_threshold_ratio=1.5)
        # ... call .filter() repeatedly with different detection lists;
        #     STFTs are computed once per file and reused.
        guard.clear_cache()  # call when you're done with this val set

    The cache holds at most `max_cached_files` STFTs at a time (LRU
    eviction). With ~6 MB per file (1-h at SR=250) this is cheap.
    """

    def __init__(
        self,
        audio_path_resolver: Callable[[str, str], Optional[Path]],
        sr: int = 250,
        n_fft: int = 256,
        hop_length: int = 5,
        biological_bands: dict[str, tuple[float, float]] = None,
        noise_bands: dict[str, list[tuple[float, float]]] = None,
        target_classes: Iterable[str] = ("d", "bmabz"),
        max_cached_files: int = 16,
    ):
        self.resolve = audio_path_resolver
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bio = dict(biological_bands or BIOLOGICAL_BANDS_HZ)
        self.noise = dict(noise_bands or NOISE_BANDS_HZ)
        self.targets = set(target_classes)
        self.max_cached_files = max_cached_files

        # Precompute bin index slices for each target class.
        self._bio_slice: dict[str, tuple[int, int]] = {}
        self._noise_slices: dict[str, list[tuple[int, int]]] = {}
        for cls in self.targets:
            f0, f1 = self.bio[cls]
            self._bio_slice[cls] = get_linear_bin_indices(
                f0, f1, sr, n_fft)
            self._noise_slices[cls] = [
                get_linear_bin_indices(a, b, sr, n_fft)
                for (a, b) in self.noise[cls]
            ]

        # LRU cache: list of (key, power_array) — most-recent at end.
        self._cache: list[tuple[tuple[str, str], np.ndarray]] = []

    # ---- public API ----

    def filter(
        self,
        detections: Sequence,
        noise_threshold_ratio: float = 1.5,
        verbose: bool = True,
    ) -> list:
        """
        Drop detections whose biological-band energy doesn't sufficiently
        exceed the off-band reference. Pass-through any non-target class.
        """
        out = []
        n_dropped: dict[str, int] = {c: 0 for c in self.targets}
        n_kept:    dict[str, int] = {c: 0 for c in self.targets}
        n_missing_audio = 0

        for det in detections:
            if det.label not in self.targets:
                out.append(det)
                continue

            power = self._get_power(det.dataset, det.filename)
            if power is None:
                # If audio is unresolvable, do NOT drop — fail open.
                # Better to pass through a possible FP than to silently
                # delete real calls because of a path issue.
                n_missing_audio += 1
                out.append(det)
                continue

            ok = self._passes_guardrail(
                power, det.start_s, det.end_s, det.label,
                noise_threshold_ratio,
            )
            if ok:
                out.append(det)
                n_kept[det.label] += 1
            else:
                n_dropped[det.label] += 1

        if verbose:
            print(f"[FreqGuardrail] noise_ratio<{noise_threshold_ratio:.2f} ⇒ drop")
            for cls in sorted(self.targets):
                k = n_kept[cls]
                d = n_dropped[cls]
                total = k + d
                rate = (d / total * 100) if total else 0.0
                print(f"              {cls:6} kept={k:5}  dropped={d:5}  "
                      f"({rate:.1f}% dropped)")
            if n_missing_audio:
                print(f"              [warn] {n_missing_audio} detections "
                      f"with unresolvable audio path — passed through")

        return out

    def clear_cache(self):
        self._cache = []

    # ---- internals ----

    def _get_power(
        self, dataset: str, filename: str,
    ) -> Optional[np.ndarray]:
        key = (dataset, filename)
        for i, (k, _) in enumerate(self._cache):
            if k == key:
                # Move to MRU end.
                self._cache.append(self._cache.pop(i))
                return self._cache[-1][1]
        # Cache miss.
        path = self.resolve(dataset, filename)
        if path is None or not Path(path).exists():
            return None
        try:
            import soundfile as sf
            audio, file_sr = sf.read(str(path), dtype="float32",
                                     always_2d=False)
        except Exception as e:
            print(f"[FreqGuardrail] failed to read {path}: {e}")
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != self.sr:
            # We expect the dataset to be 250 Hz already (cfg.SAMPLE_RATE).
            # If not, resample lazily via numpy (good enough — guardrail
            # doesn't need bit-exact alignment with model input).
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(self.sr), int(file_sr))
                up, down = int(self.sr) // g, int(file_sr) // g
                audio = resample_poly(audio, up, down).astype(np.float32)
            except Exception as e:
                print(f"[FreqGuardrail] resample fail for {path}: {e}")
                return None
        power = _stft_power_numpy(
            audio, self.sr, self.n_fft, self.hop_length)
        # LRU insert + evict.
        self._cache.append((key, power))
        while len(self._cache) > self.max_cached_files:
            self._cache.pop(0)
        return power

    def _passes_guardrail(
        self,
        power: np.ndarray,         # (n_frames, n_bins)
        start_s: float,
        end_s: float,
        cls: str,
        noise_threshold_ratio: float,
    ) -> bool:
        n_frames = power.shape[0]
        f0 = int(round(start_s * self.sr / self.hop_length))
        f1 = int(round(end_s   * self.sr / self.hop_length))
        f0 = max(0, min(n_frames - 1, f0))
        f1 = max(f0 + 1, min(n_frames, f1))
        window = power[f0:f1]  # (n_w, n_bins)

        # Mean over time → per-bin power spectrum for this event.
        spec = window.mean(axis=0)

        bs, be = self._bio_slice[cls]
        bio_energy = float(spec[bs:be].mean())

        # Average over all reference strips, weighted by the number of
        # bins in each strip (so a wider strip contributes proportionally).
        total_bins = 0
        total_energy = 0.0
        for ns, ne in self._noise_slices[cls]:
            n = ne - ns
            total_bins += n
            total_energy += float(spec[ns:ne].sum())
        noise_energy = total_energy / max(total_bins, 1)

        # Guard against div-by-zero on perfectly silent reference bands.
        if noise_energy <= 0:
            return True  # fail open
        ratio = bio_energy / noise_energy
        return ratio >= noise_threshold_ratio


# =====================================================================
# Convenience: default audio path resolver for the BioDCASE 2026 layout
# =====================================================================

def make_default_audio_resolver(
    data_root,
    splits: Sequence[str] = ("validation", "train"),
) -> Callable[[str, str], Optional[Path]]:
    """Return a resolver that searches each split's audio folder."""
    data_root = Path(data_root)

    def _resolve(dataset: str, filename: str) -> Optional[Path]:
        for split in splits:
            p = data_root / split / "audio" / dataset / filename
            if p.exists():
                return p
        return None

    return _resolve


# =====================================================================
# Self-test on synthetic Detections
# =====================================================================

def _synth_dets():
    """Build a controlled detection list for the temporal-filter test."""
    D = _DetLike
    return [
        # File A, class 'd': a clear bout (5 close together) — keep all.
        D("siteA", "f1.wav", "d", 100, 105, 0.40),
        D("siteA", "f1.wav", "d", 130, 135, 0.55),
        D("siteA", "f1.wav", "d", 160, 165, 0.85),  # the anchor
        D("siteA", "f1.wav", "d", 190, 195, 0.50),
        D("siteA", "f1.wav", "d", 220, 225, 0.45),

        # File A, class 'd': isolated single — drop.
        D("siteA", "f1.wav", "d", 2000, 2005, 0.35),

        # File A, class 'd': pair, neither high-conf — drop both.
        D("siteA", "f1.wav", "d", 4000, 4005, 0.40),
        D("siteA", "f1.wav", "d", 4020, 4025, 0.45),

        # File A, class 'd': triplet with one anchor — keep all three.
        D("siteA", "f1.wav", "d", 6000, 6005, 0.30),
        D("siteA", "f1.wav", "d", 6010, 6015, 0.35),
        D("siteA", "f1.wav", "d", 6030, 6035, 0.75),

        # File B is a separate scope: an isolated event on B must NOT
        # be rescued by the bout on A.
        D("siteB", "f2.wav", "d", 160, 165, 0.85),  # isolated → drop

        # bmabz: same patterns, smaller scale.
        D("siteA", "f1.wav", "bmabz", 50, 55, 0.30),
        D("siteA", "f1.wav", "bmabz", 80, 85, 0.40),
        D("siteA", "f1.wav", "bmabz", 110, 115, 0.75),  # anchor → keep all 3
        D("siteA", "f1.wav", "bmabz", 3000, 3005, 0.80),  # isolated → drop

        # bp is in target list? No. It must always pass through.
        D("siteA", "f1.wav", "bp", 500, 502, 0.10),  # would be dropped if filtered
        D("siteA", "f1.wav", "bp", 700, 702, 0.05),
    ]


def _run_smoke_tests():
    import sys
    print("=" * 70)
    print("postprocess_bout_filter — self-test")
    print("=" * 70)

    dets = _synth_dets()
    n_in = len(dets)
    print(f"\nInput: {n_in} detections "
          f"({sum(1 for d in dets if d.label=='d')} d, "
          f"{sum(1 for d in dets if d.label=='bmabz')} bmabz, "
          f"{sum(1 for d in dets if d.label=='bp')} bp)")

    # ---- ICI on the d-class bout. Should see a peak around 30 s. ----
    print("\n--- ICI analysis (target_class='d') ---")
    summary = analyze_ici(dets, target_class="d",
                          score_threshold=0.2, bin_step_s=5.0,
                          bin_max_s=120.0, show=False)
    # Mode should fall in the 25–35 s range (most spacings are 30 s, or
    # 10–20 s within the late triplet).
    assert summary["count"] > 0, "ICI must produce >0 intervals"
    print(f"    (mode_s={summary['mode_s']:.1f} — expected ~10-30 s)")

    # ---- Temporal filter ----
    print("\n--- Temporal bout filter ---")
    out = apply_temporal_bout_filter(
        dets,
        target_classes=("d", "bmabz"),
        window_size_sec=3600.0,
        min_calls=3,
        high_conf_threshold=0.7,
    )

    # Expected survivors by index in the input list:
    #   d on siteA/f1: indices 0..4  (bout of 5) → all keep
    #                  indices 8..10 (triplet w/ 0.75 anchor) → all keep
    #   d on siteA/f1: index 5 (single, 0.35) → drop
    #   d on siteA/f1: indices 6,7 (pair, max 0.45) → drop
    #   d on siteB/f2: index 11 (isolated, 0.85) → drop (no neighbours)
    #   bmabz on siteA/f1: indices 12..14 (triplet w/ 0.75) → all keep
    #                      index 15 (isolated) → drop
    #   bp untouched: indices 16, 17 → all keep (pass-through)
    expected_d_kept = 8       # 5 + 3
    expected_bmabz_kept = 3
    expected_bp_kept = 2

    got_d = sum(1 for d in out if d.label == "d")
    got_bmabz = sum(1 for d in out if d.label == "bmabz")
    got_bp = sum(1 for d in out if d.label == "bp")

    print(f"\nResult counts: d={got_d} (expect {expected_d_kept}), "
          f"bmabz={got_bmabz} (expect {expected_bmabz_kept}), "
          f"bp={got_bp} (expect {expected_bp_kept})")

    ok = (got_d == expected_d_kept and
          got_bmabz == expected_bmabz_kept and
          got_bp == expected_bp_kept)
    if not ok:
        print("\nFAIL — surviving detections:")
        for d in out:
            print(f"  {d.label:6} {d.dataset}/{d.filename} "
                  f"t={d.start_s:.0f}s conf={d.confidence:.2f}")
        sys.exit(1)

    # Ensure originals weren't mutated.
    assert len(dets) == n_in, "input list mutated!"

    # ---- Linear bin indexing sanity ----
    print("\n--- get_linear_bin_indices sanity ---")
    bs, be = get_linear_bin_indices(17.0, 29.0, sr=250, n_fft=256)
    print(f"    bmabz 17-29 Hz at SR=250,N_FFT=256 → bins [{bs}, {be})  "
          f"= {be-bs} bins, ~{(be-bs)*250/256:.1f} Hz wide")
    assert 5 <= be - bs <= 20, "bin count for 17-29 Hz seems wrong"

    bs, be = get_linear_bin_indices(30.0, 100.0, sr=250, n_fft=256)
    print(f"    d     30-100 Hz                    → bins [{bs}, {be})  "
          f"= {be-bs} bins")
    assert 60 <= be - bs <= 80, "bin count for 30-100 Hz seems wrong"

    # Above-Nyquist soft-clamp
    bs, be = get_linear_bin_indices(100.0, 200.0, sr=250, n_fft=256)
    assert be <= 129, "must not return bins past Nyquist+1"
    print(f"    100-200 Hz (clamped at Nyquist)    → bins [{bs}, {be})")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _run_smoke_tests()
