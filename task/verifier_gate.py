"""
verifier_gate.py — optional verifier-gate for the MT consistency stream (_final)
================================================================================

Uses the trained stage-2 SupCon verifier (``model_verifier.WhaleVerifier``) to
decide whether to TRUST the teacher's POSITIVE pseudo-labels for a target class
(default: D) on the unlabeled AADC stream, before they enter the mean-teacher
consistency loss. Where the teacher fires the target class, the corresponding
audio window is cropped (in-memory, straight from the 30 s clip tensor — no
file IO), re-scored by the verifier, and only verifier-ACCEPTED events keep
their consistency signal; verifier-REJECTED events are masked out. Wherever the
teacher does NOT fire the target class the gate is 1 (untouched), so normal
negative consistency is preserved.

Why event-level and 3-class only
--------------------------------
The verifier is a per-EVENT binary re-scorer on a ~15 s crop plus the
detector's event confidence (see ``dataset_verifier.CandidateRecord`` /
``eval_verifier``). It has one binary head per class in ``CALL_TYPES_3`` space.
Gating a 7-class student head would need a 7->3->7 round trip, so this gate
refuses 7-class heads — use it with a 3-class MT run (the D-recovery story is
3-class anyway, and the verifier is 3-class). SAT + DA work for both heads;
only this optional gate is 3-class-only.

Module-boundary note (read once)
--------------------------------
The verifier was trained on the OLD module stack (``config`` + ``spectrogram``),
NOT ``config_final`` / ``spectrogram_final``. To feed it exactly what it saw in
training we build its spectrogram with the old ``spectrogram.SpectrogramExtractor``
and read SR / crop geometry from old ``config``. SAMPLE_RATE (250) and N_FFT
(256) match between the two configs, so the in-memory crop sample counts line
up with the _final 30 s clip tensor. The two ``config`` modules are distinct
objects, so the MT trainer flipping ``config_final.USE_3CLASS`` does not touch
the verifier side.

The ``stage1_score`` aux fed to the verifier here is the EMA TEACHER's mean
prob over the candidate span (the trainer's detector at this step), standing in
for the base-model event confidence used at verifier-training time. Close in
spirit; a minor aux-distribution shift, noted here for honesty.

Frame->sample mapping is downsample-agnostic: it uses ``stride = n_samples / T``
from the actual clip length and teacher frame count, so it is correct even if
the stage-1 model pools time internally (do NOT assume frame == spec hop).
"""

from __future__ import annotations

from typing import Iterator, Sequence

import torch

import config as vcfg                          # OLD config (verifier-native)
from model_verifier import WhaleVerifier
from spectrogram import SpectrogramExtractor as VerifierSpec


def _runs(mask_1d: torch.Tensor, min_len: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) frame spans of contiguous True runs of length >= min_len."""
    m = mask_1d.to(torch.int8).tolist()
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) >= min_len:
                yield i, j
            i = j
        else:
            i += 1


class VerifierGate:
    """
    Build a per-element accept/reject gate for the MT consistency stream.

    Parameters
    ----------
    checkpoint : str
        Path to a verifier ``best.pt`` (the trainer's checkpoint; ``args`` is
        read to recover ``crop_s`` / dropouts).
    device : torch.device
    gate_classes : sequence of str, default ("d",)
        Classes (CALL_TYPES_3 names) whose POSITIVE pseudo-labels are gated.
    crop_s : float, optional
        Override the crop length; default = checkpoint's training value (15 s).
    accept_threshold : float, default 0.5
        Verifier P(real call) below which a teacher event is masked out.
    fire_threshold : float, default 0.5
        Teacher prob above which a frame counts as a candidate-event frame.
    min_event_frames : int, default 2
        Minimum contiguous fired frames to treat as an event (drops 1-frame
        flickers).
    """

    def __init__(
        self,
        checkpoint: str,
        device,
        gate_classes: Sequence[str] = ("d",),
        crop_s: float | None = None,
        accept_threshold: float = 0.5,
        fire_threshold: float = 0.5,
        min_event_frames: int = 2,
    ):
        self.device = device
        self.accept = float(accept_threshold)
        self.fire = float(fire_threshold)
        self.min_event_frames = int(min_event_frames)

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        targs = ckpt.get("args", {}) or {}
        self.crop_s = float(crop_s if crop_s is not None else targs.get("crop_s", 15.0))
        self.crop_samples = int(round(self.crop_s * vcfg.SAMPLE_RATE))

        self.model = WhaleVerifier(
            n_classes=len(vcfg.CALL_TYPES_3),
            backbone_dropout=targs.get("backbone_dropout", 0.5),
            head_dropout=targs.get("head_dropout", 0.3),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.spec = VerifierSpec().to(device)

        # Gated class indices in CALL_TYPES_3 order (== _final 3-class order).
        self.gate_idx = [vcfg.CALL_TYPES_3.index(c) for c in gate_classes]

    @torch.no_grad()
    def gate_mask(self, audio_u: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        """
        Return a (B, T, C) gate in {0, 1}. It is 1 everywhere except inside
        verifier-REJECTED events of the gated classes.

        Parameters
        ----------
        audio_u : (B, n_samples) on ``device`` — the 30 s unlabeled clips.
        teacher_probs : (B, T, C) teacher sigmoid probs (3-class).
        """
        B, T, C = teacher_probs.shape
        gate = torch.ones(B, T, C, device=teacher_probs.device)
        n_samples = audio_u.size(1)
        stride = n_samples / float(max(T, 1))      # samples per frame (downsample-agnostic)
        half = self.crop_samples // 2

        crops: list[torch.Tensor] = []
        slots: list[tuple[int, int, int, int]] = []   # (b, c, f0, f1)
        cls_list: list[int] = []
        aux_list: list[float] = []

        for b in range(B):
            for c in self.gate_idx:
                pc = teacher_probs[b, :, c]
                fired = pc > self.fire
                for f0, f1 in _runs(fired, self.min_event_frames):
                    score = float(pc[f0:f1].mean().item())
                    mid = int(round(((f0 + f1) / 2.0) * stride))
                    s0 = mid - half
                    s0 = max(0, min(s0, n_samples - self.crop_samples))
                    s0 = max(0, s0)
                    s1 = min(n_samples, s0 + self.crop_samples)
                    crop = audio_u[b, s0:s1]
                    if crop.numel() < self.crop_samples:
                        crop = torch.nn.functional.pad(
                            crop, (0, self.crop_samples - crop.numel()))
                    crops.append(crop)
                    cls_list.append(c)
                    aux_list.append(score)
                    slots.append((b, c, f0, f1))

        if not crops:
            return gate

        crop_batch = torch.stack(crops, dim=0)                  # (N, crop_samples)
        spec = self.spec(crop_batch)                            # (N, 3, F, T')
        cls = torch.tensor(cls_list, device=self.device, dtype=torch.long)
        aux = torch.tensor(aux_list, device=self.device,
                           dtype=torch.float32).unsqueeze(1)    # (N, 1)
        p_real = torch.sigmoid(self.model(spec, cls, aux))      # (N,)

        for (b, c, f0, f1), pr in zip(slots, p_real.tolist()):
            if pr < self.accept:
                gate[b, f0:f1, c] = 0.0                         # reject -> mask out
        return gate
