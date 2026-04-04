"""
Whale-Conformer: Conformer-based Whale Vocalisation Activity Detection

Replaces the CNN-BiLSTM backbone of Whale-VAD (Geldenhuys et al., 2025)
with a Conformer encoder, while retaining key insights:
  - Spectral phase input (magnitude, cos θ, sin θ)
  - Learned filterbank front-end
  - Per-frame multi-label classification with sigmoid outputs
  - Focal loss + class weighting
  - Stochastic negative mini-batch undersampling

Target: BioDCASE 2026 Task 2 — Antarctic blue & fin whale call detection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    """Gated Linear Unit along the channel dimension."""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (added, not concatenated)."""
    def __init__(self, d_model: int, max_len: int = 20_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_out(out)


class ConvolutionModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → Swish → pointwise."""
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise expansion (×2 for GLU)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = GLU(dim=1)
        # Depthwise conv
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        # Pointwise projection
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)  # (B, T, D)


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module: LN → Linear → Swish → Dropout → Linear → Dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ConformerBlock(nn.Module):
    """
    Single Conformer block (Gulati et al., 2020):
        x = x + 0.5 * FFN(x)
        x = x + MHSA(x)
        x = x + Conv(x)
        x = x + 0.5 * FFN(x)
        x = LayerNorm(x)
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        residual = x
        x = self.attn_norm(x)
        x = residual + self.attn_dropout(self.attn(x, mask=mask))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Front-end: phase-aware spectrogram → learned filterbank → subsampling
# ---------------------------------------------------------------------------

class PhaseAwareFrontEnd(nn.Module):
    """
    Replicates the Whale-VAD insight: instead of just power spectrum,
    feed (magnitude, cos θ, sin θ) as a 3-channel input.
    Then apply a learned 1-D filterbank + CNN subsampling.
    """
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 5,       # 20 ms at 250 Hz
        win_length: int = 250,     # ~1 s at 250 Hz
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        n_freq = n_fft // 2 + 1  # 129 bins

        # 3-channel input: (mag, cos_phase, sin_phase)
        # Learned filterbank: 1-D conv across frequency axis per frame
        self.filterbank = nn.Conv2d(3, 64, kernel_size=(7, 1), stride=(3, 1), padding=(2, 0))
        self.bn0 = nn.BatchNorm2d(64)

        # Two conv blocks for further feature extraction & frequency reduction
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))

        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 1), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        # Projection to d_model — computed dynamically
        self._proj = None
        self._d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def _get_projection(self, freq_dim: int, device: torch.device) -> nn.Linear:
        """Lazily create projection layer once we know the frequency dimension."""
        if self._proj is None or self._proj.in_features != freq_dim * 128:
            self._proj = nn.Linear(freq_dim * 128, self._d_model).to(device)
        return self._proj

    def _compute_stft(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute STFT and return (magnitude, cos_phase, sin_phase), each (B, F, T)."""
        window = torch.hann_window(self.win_length, device=audio.device)
        # Pad the window to n_fft if win_length < n_fft
        if self.win_length < self.n_fft:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            window = F.pad(window, (pad_left, pad_right))

        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window,
            return_complex=True, center=True,
        )
        mag = stft.abs()
        phase = stft.angle()
        return mag, torch.cos(phase), torch.sin(phase)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, num_samples)
        Returns:
            features: (B, T_out, d_model)
        """
        mag, cos_ph, sin_ph = self._compute_stft(audio)
        # Stack to (B, 3, F, T)
        x = torch.stack([mag, cos_ph, sin_ph], dim=1)

        # Learned filterbank
        x = F.gelu(self.bn0(self.filterbank(x)))

        # Conv blocks — treat freq as "height", time as "width"
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # (B, C, F', T) → (B, T, C*F')
        B, C, F_out, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F_out)

        # Project to d_model
        proj = self._get_projection(F_out, x.device)
        x = self.dropout(proj(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class WhaleConformer(nn.Module):
    """
    Conformer-based whale call sound event detector.

    Pipeline:
        audio → PhaseAwareFrontEnd → PositionalEncoding → N × ConformerBlock → Linear → sigmoid

    Outputs per-frame probabilities for each call class.
    """
    def __init__(
        self,
        n_classes: int = 3,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        n_fft: int = 256,
        hop_length: int = 5,
        win_length: int = 250,
        sample_rate: int = 250,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_classes = n_classes

        self.frontend = PhaseAwareFrontEnd(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            d_model=d_model, dropout=dropout,
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                conv_kernel_size=conv_kernel_size, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self, audio: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            audio: (B, num_samples) raw waveform at 250 Hz
            mask:  optional (B, 1, T, T) attention mask

        Returns:
            logits: (B, T_frames, n_classes) — apply sigmoid for probabilities
        """
        x = self.frontend(audio)       # (B, T, d_model)
        x = self.pos_enc(x)

        for block in self.conformer_blocks:
            x = block(x, mask=mask)

        logits = self.classifier(x)    # (B, T, n_classes)
        return logits

    @classmethod
    def from_config(cls, cfg=None) -> "WhaleConformer":
        """Build model from the central Config object."""
        if cfg is None:
            from config import CFG
            cfg = CFG
        return cls(
            n_classes=cfg.model.n_classes,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            d_ff=cfg.model.d_ff,
            n_layers=cfg.model.n_layers,
            conv_kernel_size=cfg.model.conv_kernel_size,
            dropout=cfg.model.dropout,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            win_length=cfg.audio.win_length,
            sample_rate=cfg.audio.sample_rate,
        )

    def predict(self, audio: torch.Tensor, thresholds: torch.Tensor | None = None) -> torch.Tensor:
        """Convenience method: returns binary predictions after sigmoid + threshold."""
        logits = self.forward(audio)
        probs = torch.sigmoid(logits)
        if thresholds is None:
            thresholds = torch.tensor([0.5] * self.n_classes, device=probs.device)
        return (probs > thresholds.view(1, 1, -1)).long()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCEWithFocal(nn.Module):
    """Combined weighted BCE + focal loss, with per-class weights."""
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
        self.focal_weight = focal_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:       (B, T, C)
            targets:      (B, T, C) binary
            padding_mask: (B, T) — True where valid, False where padded
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.class_weights is not None:
            bce = bce * self.class_weights.view(1, 1, -1)

        focal = self.focal(logits, targets)
        loss = bce + self.focal_weight * focal  # (B, T, C)

        if padding_mask is not None:
            loss = loss * padding_mask.unsqueeze(-1).float()
            return loss.sum() / (padding_mask.sum() * logits.size(-1) + 1e-8)
        return loss.mean()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import config as cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhaleConformer(
        n_classes=cfg.n_classes(),
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS, d_ff=cfg.D_FF,
        n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL, dropout=cfg.DROPOUT,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.WIN_LENGTH,
        sample_rate=cfg.SAMPLE_RATE,
    ).to(device)

    # Simulate a 30s clip
    batch = torch.randn(2, cfg.SAMPLE_RATE * 30, device=device)
    logits = model(batch)
    print(f"Input:  {batch.shape}")
    print(f"Output: {logits.shape}  (B, T_frames, n_classes)")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")
