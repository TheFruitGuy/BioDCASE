"""
Augmentations for contrastive pretraining on 250 Hz underwater audio.

Each augmentation is designed for the specific characteristics of this domain:
  - Low sample rate (250 Hz → 0-125 Hz bandwidth)
  - Whale calls at 15-120 Hz
  - Variable noise conditions (ice, ships, chorus)

Two augmented "views" of the same audio segment form a positive pair.
"""

import random
import torch
import torch.nn.functional as F
import numpy as np


class AudioAugmentor:
    """
    Produce two differently-augmented views of the same audio segment.
    
    Each view gets a random subset of augmentations applied,
    so the model must learn features invariant to these transformations.
    """
    def __init__(
        self,
        sample_rate: int = 250,
        # Probability of applying each augmentation
        p_noise: float = 0.5,
        p_gain: float = 0.8,
        p_time_mask: float = 0.5,
        p_freq_mask: float = 0.5,
        p_time_shift: float = 0.5,
        # Augmentation parameters
        noise_snr_range: tuple = (5, 30),      # dB
        gain_range: tuple = (-6, 6),            # dB
        time_mask_max_s: float = 2.0,           # max mask width in seconds
        n_time_masks: int = 2,
        freq_mask_max_bins: int = 20,           # max freq bins to mask
        n_freq_masks: int = 2,
        time_shift_max_s: float = 5.0,          # max circular shift
    ):
        self.sample_rate = sample_rate
        self.p_noise = p_noise
        self.p_gain = p_gain
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_time_shift = p_time_shift
        self.noise_snr_range = noise_snr_range
        self.gain_range = gain_range
        self.time_mask_max_samples = int(time_mask_max_s * sample_rate)
        self.n_time_masks = n_time_masks
        self.freq_mask_max_bins = freq_mask_max_bins
        self.n_freq_masks = n_freq_masks
        self.time_shift_max_samples = int(time_shift_max_s * sample_rate)

    def __call__(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: (T,) raw waveform
        Returns:
            view1, view2: two augmented versions of the same audio
        """
        view1 = self._augment_waveform(audio.clone())
        view2 = self._augment_waveform(audio.clone())
        return view1, view2

    def _augment_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random waveform-level augmentations."""
        # Time shift (circular)
        if random.random() < self.p_time_shift:
            shift = random.randint(-self.time_shift_max_samples,
                                    self.time_shift_max_samples)
            x = torch.roll(x, shifts=shift, dims=0)

        # Additive Gaussian noise
        if random.random() < self.p_noise:
            snr_db = random.uniform(*self.noise_snr_range)
            signal_power = x.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10) + 1e-8)
            noise = torch.randn_like(x) * noise_power.sqrt()
            x = x + noise

        # Random gain
        if random.random() < self.p_gain:
            gain_db = random.uniform(*self.gain_range)
            x = x * (10 ** (gain_db / 20))

        return x


class SpectrogramAugmentor:
    """
    Apply SpecAugment-style masking to spectrograms.
    Applied AFTER computing the spectrogram, before the encoder.
    """
    def __init__(
        self,
        p_time_mask: float = 0.5,
        p_freq_mask: float = 0.5,
        time_mask_max_frames: int = 50,    # ~1s at 20ms stride
        freq_mask_max_bins: int = 20,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
    ):
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.time_mask_max = time_mask_max_frames
        self.freq_mask_max = freq_mask_max_bins
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (C, F, T) spectrogram (C=3 for phase-aware: mag, cos, sin)
        Returns:
            masked spectrogram
        """
        spec = spec.clone()
        _, F, T = spec.shape

        # Time masks
        if random.random() < self.p_time_mask:
            for _ in range(self.n_time_masks):
                width = random.randint(1, min(self.time_mask_max, T // 4))
                start = random.randint(0, T - width)
                spec[:, :, start:start + width] = 0

        # Frequency masks
        if random.random() < self.p_freq_mask:
            for _ in range(self.n_freq_masks):
                width = random.randint(1, min(self.freq_mask_max, F // 4))
                start = random.randint(0, F - width)
                spec[:, start:start + width, :] = 0

        return spec
