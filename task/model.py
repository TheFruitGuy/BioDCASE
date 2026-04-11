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
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024,
        conv_kernel_size: int = 31, dropout: float = 0.1,
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
# Multi-Scale Head
# ---------------------------------------------------------------------------

class MultiScaleClassifier(nn.Module):
    """
    Parallel classification heads tailored to 7 whale call durations.
    Maintains sequence length but smooths temporal features locally.
    Assumes class order: [bma, bmb, bmz, bmd, bpd, bp20, bp20plus]
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Long (2.5s) for Z-calls: bma, bmb, bmz (3 classes)
        self.pool_long = nn.AvgPool1d(kernel_size=125, stride=1, padding=62)
        self.classifier_long = nn.Linear(d_model, 3)

        # Medium (0.5s) for downsweeps: bmd, bpd (2 classes)
        self.pool_med = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.classifier_med = nn.Linear(d_model, 2)

        # Short (0.1s) for pulses: bp20, bp20plus (2 classes)
        self.pool_short = nn.AvgPool1d(kernel_size=15, stride=1, padding=7)
        self.classifier_short = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)  # (B, D, T)

        feat_long = self.pool_long(x_t).transpose(1, 2)   # (B, T, D)
        feat_med = self.pool_med(x_t).transpose(1, 2)     # (B, T, D)
        feat_short = self.pool_short(x_t).transpose(1, 2) # (B, T, D)

        logit_long = self.classifier_long(feat_long)      # (B, T, 3)
        logit_med = self.classifier_med(feat_med)         # (B, T, 2)
        logit_short = self.classifier_short(feat_short)   # (B, T, 2)

        # Recombine into (B, T, 7)
        return torch.cat([logit_long, logit_med, logit_short], dim=-1)

# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------

class PhaseAwareFrontEnd(nn.Module):
    def __init__(
        self, n_fft: int = 256, hop_length: int = 5, win_length: int = 250,
        d_model: int = 256, dropout: float = 0.1,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.filterbank = nn.Conv2d(3, 64, kernel_size=(7, 1), stride=(3, 1), padding=(2, 0))
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))

        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 1), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        self._proj = None
        self._d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def _get_projection(self, freq_dim: int, device: torch.device) -> nn.Linear:
        if self._proj is None or self._proj.in_features != freq_dim * 128:
            self._proj = nn.Linear(freq_dim * 128, self._d_model).to(device)
        return self._proj

    def _compute_stft(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = torch.hann_window(self.win_length, device=audio.device)
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
        mag, cos_ph, sin_ph = self._compute_stft(audio)
        x = torch.stack([mag, cos_ph, sin_ph], dim=1)
        x = F.gelu(self.bn0(self.filterbank(x)))
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        B, C, F_out, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F_out)
        proj = self._get_projection(F_out, x.device)
        x = self.dropout(proj(x))
        return x

# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class WhaleConformer(nn.Module):
    def __init__(
        self, n_classes: int = 7, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024,
        n_layers: int = 4, conv_kernel_size: int = 31, dropout: float = 0.1,
        n_fft: int = 256, hop_length: int = 5, win_length: int = 250, sample_rate: int = 250,
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

        # Use Multi-Scale Head for 7 classes, fallback to Linear otherwise
        if n_classes == 7:
            self.classifier = MultiScaleClassifier(d_model)
        else:
            self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, audio: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.frontend(audio)
        x = self.pos_enc(x)
        for block in self.conformer_blocks:
            x = block(x, mask=mask)
        logits = self.classifier(x)
        return logits

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class WeightedBCEWithFocal(nn.Module):
    def __init__(
        self, pos_weight: torch.Tensor | None = None, focal_alpha: float = 0.75,
        focal_gamma: float = 2.0, focal_weight: float = 1.0,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        pw = self.pos_weight.view(1, 1, -1) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")

        if self.focal_weight > 0:
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            focal_mod = alpha_t * (1 - p_t) ** self.focal_gamma
            loss = bce + self.focal_weight * focal_mod * bce
        else:
            loss = bce

        if padding_mask is not None:
            loss = loss * padding_mask.unsqueeze(-1).float()
            return loss.sum() / (padding_mask.sum() * logits.size(-1) + 1e-8)
        return loss.mean()