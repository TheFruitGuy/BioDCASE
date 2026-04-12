"""
Whale-Conformer v2: Conformer + BiLSTM hybrid

Key changes from v1, informed by Whale-VAD (Geldenhuys et al., 2025):

1. Added BiLSTM after Conformer blocks
   WHY: Whale-VAD showed BiLSTM produces coherent start/end boundaries
   that closely match human annotators. Pure attention struggles with
   maintaining high confidence over long call durations (the "endpointing"
   problem noted in Section 2.3 of their report).

2. Segment-based class weighting (their formula: w_c = N/P_c)
   WHY: Their report Section 2.6 found segment-count weighting works
   better than duration-based. Our frame-level pos_weight=50 caused NaN.

3. Kept phase-aware input (their biggest win: +30% F1)
   Already in our PhaseAwareFrontEnd.

4. Back to 3-class (their +15.2% improvement)
   Set USE_3CLASS=True in config.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks (unchanged from v1)
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RelativeMultiHeadAttention(nn.Module):
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

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_out(out)

class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                    padding=kernel_size // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)

class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x = self.attn_norm(x)
        x = residual + self.attn_dropout(self.attn(x, mask=mask))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Front-end (unchanged — phase-aware STFT)
# ---------------------------------------------------------------------------

class PhaseAwareFrontEnd(nn.Module):
    def __init__(self, n_fft=256, hop_length=5, win_length=250,
                 d_model=256, dropout=0.1):
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

    def _get_projection(self, freq_dim, device):
        if self._proj is None or self._proj.in_features != freq_dim * 128:
            self._proj = nn.Linear(freq_dim * 128, self._d_model).to(device)
        return self._proj

    def _compute_stft(self, audio):
        window = torch.hann_window(self.win_length, device=audio.device)
        if self.win_length < self.n_fft:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            window = F.pad(window, (pad_left, pad_right))
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.n_fft, window=window,
                          return_complex=True, center=True)
        mag = stft.abs()
        phase = stft.angle()
        return mag, torch.cos(phase), torch.sin(phase)

    def forward(self, audio):
        mag, cos_ph, sin_ph = self._compute_stft(audio)
        mag = mag - mag.mean(dim=-1, keepdim=True)
        cos_ph = cos_ph - cos_ph.mean(dim=-1, keepdim=True)
        sin_ph = sin_ph - sin_ph.mean(dim=-1, keepdim=True)
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
# NEW: BiLSTM temporal coherence layer
# Stolen from Whale-VAD Section 2.3/2.4:
# "the posterior call probabilities produced by these models aligned
# well with the call segments... both the start and end boundaries
# closely matched those of the human annotators"
# ---------------------------------------------------------------------------

class TemporalCoherenceLayer(nn.Module):
    """
    BiLSTM that smooths the Conformer output to produce coherent
    call boundaries. Addresses the "endpointing problem" where
    attention-based models produce flickering frame-level predictions.
    
    Residually connected so the Conformer features pass through
    directly, with the BiLSTM adding temporal coherence on top.
    """
    def __init__(self, d_model=256, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(2 * hidden_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        lstm_out, _ = self.lstm(x)
        projected = self.projection(lstm_out)
        return self.layer_norm(residual + projected)


# ---------------------------------------------------------------------------
# Full model: Conformer + BiLSTM hybrid
# ---------------------------------------------------------------------------

class WhaleConformer(nn.Module):
    """
    audio → PhaseAwareFrontEnd → PosEnc → N × ConformerBlock
          → BiLSTM (temporal coherence) → Linear → sigmoid
    """
    def __init__(
        self, n_classes=3, d_model=256, n_heads=4, d_ff=1024,
        n_layers=4, conv_kernel_size=31, dropout=0.1,
        n_fft=256, hop_length=5, win_length=250, sample_rate=250,
        lstm_hidden=128, lstm_layers=2, lstm_dropout=0.3,
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
            ConformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                          conv_kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.temporal_lstm = TemporalCoherenceLayer(
            d_model=d_model, hidden_dim=lstm_hidden,
            n_layers=lstm_layers, dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, audio, mask=None):
        x = self.frontend(audio)
        x = self.pos_enc(x)
        for block in self.conformer_blocks:
            x = block(x, mask=mask)
        x = self.temporal_lstm(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedBCEWithFocal(nn.Module):
    def __init__(self, pos_weight=None, focal_alpha=0.75,
                 focal_gamma=2.0, focal_weight=1.0):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight

    def forward(self, logits, targets, padding_mask=None):
        pw = self.pos_weight.view(1, 1, -1) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none")
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


# ---------------------------------------------------------------------------
# Segment-based class weighting (Whale-VAD formula: w_c = N / P_c)
# ---------------------------------------------------------------------------

def compute_segment_weights(n_classes: int = 3) -> torch.Tensor:
    """
    Whale-VAD Section 2.6: weight by NUMBER OF SEGMENTS, not duration.
    w_c = N_negative_segments / P_c_positive_segments
    """
    import config as cfg
    from dataset import load_annotations

    annotations = load_annotations(cfg.TRAIN_DATASETS)
    total_files = annotations.groupby(["dataset", "filename"]).ngroups
    class_names = cfg.class_names()

    weights = []
    for c_name in class_names:
        if cfg.USE_3CLASS:
            orig_labels = [k for k, v in cfg.COLLAPSE_MAP.items() if v == c_name]
            class_annots = annotations[annotations["annotation"].isin(orig_labels)]
        else:
            class_annots = annotations[annotations["annotation"] == c_name]

        n_pos = max(class_annots.groupby(["dataset", "filename"]).ngroups, 1)
        n_neg = total_files - n_pos
        w = min(n_neg / n_pos, 20.0)
        weights.append(w)

    result = torch.tensor(weights, dtype=torch.float32)
    result = result / result.mean()
    print(f"Segment-based class weights: {result}")
    return result


if __name__ == "__main__":
    import config as cfg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhaleConformer(
        n_classes=cfg.n_classes(),
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS, d_ff=cfg.D_FF,
        n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=cfg.DROPOUT, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, sample_rate=cfg.SAMPLE_RATE,
    ).to(device)
    batch = torch.randn(2, cfg.SAMPLE_RATE * 30, device=device)
    logits = model(batch)
    print(f"Input:  {batch.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
