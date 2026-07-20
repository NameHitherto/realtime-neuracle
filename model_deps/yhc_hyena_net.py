import math
from typing import Sequence

import torch
import torch.nn as nn


class SensorDropout(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        if x.dim() == 4:
            mask_shape = (x.shape[0], 1, x.shape[2], 1)
        else:
            mask_shape = (x.shape[0], x.shape[1], 1)
        mask = torch.empty(mask_shape, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob


class FixedFFTFilterBank(nn.Module):
    """Non-learned MI frequency bands implemented on GPU with FFT masks."""

    def __init__(self, bands: Sequence[tuple[float, float]], sfreq: int = 250):
        super().__init__()
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.sfreq = sfreq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == len(self.bands):
            return x
        if x.dim() == 4:
            x = x.squeeze(1)
        samples = x.shape[-1]
        freq = torch.fft.rfftfreq(samples, d=1.0 / self.sfreq).to(x.device)
        x_f = torch.fft.rfft(x, dim=-1)

        masks = []
        for low, high in self.bands:
            masks.append((freq >= low) & (freq <= high))
        mask = torch.stack(masks, dim=0).to(x_f.dtype)
        band_f = x_f.unsqueeze(1) * mask.view(1, len(self.bands), 1, -1)
        return torch.fft.irfft(band_f, n=samples, dim=-1).to(x.dtype)


class SpatialSpectralCalibration(nn.Module):
    """Conservative band-channel calibration before tokenization."""

    def __init__(
        self,
        n_bands: int,
        n_chans: int,
        reduction: int = 4,
        scale_init: float = 0.05,
        dynamic: bool = False,
    ):
        super().__init__()
        in_dim = n_bands * n_chans
        hidden = max(in_dim // max(reduction, 1), 16)
        self.n_bands = n_bands
        self.n_chans = n_chans
        self.dynamic = dynamic
        self.static_gate = nn.Parameter(torch.zeros(1, n_bands, n_chans, 1))
        self.net = (
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, in_dim),
            )
            if dynamic
            else None
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.static_gate
        if self.net is not None:
            summary = x.var(dim=-1, unbiased=False).clamp_min(1e-6).log().flatten(1)
            weights = weights + self.net(summary).view(x.shape[0], self.n_bands, self.n_chans, 1)
        return x * (1.0 + self.scale * torch.tanh(weights))


class EEGTokenStem(nn.Module):
    def __init__(self, n_chans: int = 22, emb_dim: int = 64, temporal_filters: int = 16, pool_stride: int = 8, dropout: float = 0.35):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(temporal_filters, emb_dim, kernel_size=(n_chans, 1), bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool_stride), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv2d(emb_dim, emb_dim, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.proj(x)
        return x.squeeze(2).transpose(1, 2)


class PyramidEEGTokenStem(nn.Module):
    """Multi-scale temporal CNN tokenizer for MI filter-bank signals."""

    def __init__(
        self,
        n_chans: int = 22,
        emb_dim: int = 64,
        temporal_filters: int = 12,
        kernels: Sequence[int] = (15, 31, 63),
        pool_stride: int = 8,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for kernel in kernels:
            if kernel % 2 == 0:
                kernel += 1
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(1, temporal_filters, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
                    nn.BatchNorm2d(temporal_filters),
                    nn.ELU(inplace=True),
                )
            )
        in_filters = temporal_filters * len(kernels)
        self.spatial = nn.Sequential(
            nn.Conv2d(in_filters, emb_dim, kernel_size=(n_chans, 1), bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool_stride), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(emb_dim, max(emb_dim // 4, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(emb_dim // 4, 8), emb_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(emb_dim, emb_dim, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.spatial(x)
        x = x * self.channel_gate(x)
        x = self.proj(x)
        return x.squeeze(2).transpose(1, 2)


class TokenMixStyle(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or not self.training or torch.rand((), device=x.device) > self.p:
            return x
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        x_norm = (x - mean) / std
        perm = torch.randperm(x.shape[0], device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.shape[0], 1, 1)).to(x.device, x.dtype)
        mixed_mean = lam * mean + (1.0 - lam) * mean[perm]
        mixed_std = lam * std + (1.0 - lam) * std[perm]
        return x_norm * mixed_std + mixed_mean


class AdaptiveFourierMixer(nn.Module):
    def __init__(self, dim: int, modes: int = 32, dropout: float = 0.1):
        super().__init__()
        self.modes = modes
        self.norm = nn.LayerNorm(dim)
        self.real = nn.Parameter(torch.zeros(modes, dim))
        self.imag = nn.Parameter(torch.zeros(modes, dim))
        nn.init.normal_(self.real, std=0.02)
        nn.init.normal_(self.imag, std=0.02)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_freq = torch.fft.rfft(x, dim=1)
        modes = min(self.modes, x_freq.shape[1])
        weight = torch.complex(self.real[:modes], self.imag[:modes]).unsqueeze(0)
        x_freq = torch.cat([x_freq[:, :modes, :] * (1.0 + weight), x_freq[:, modes:, :]], dim=1)
        x = torch.fft.irfft(x_freq, n=x.shape[1], dim=1)
        return residual + self.out(x)


class ImplicitLongConv(nn.Module):
    def __init__(self, dim: int, pos_bands: int = 16, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pos_bands = pos_bands
        in_dim = 1 + 2 * pos_bands
        self.filter_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def position_features(self, length: int, device, dtype) -> torch.Tensor:
        t = torch.linspace(0, 1, length, device=device, dtype=dtype).unsqueeze(-1)
        bands = torch.arange(1, self.pos_bands + 1, device=device, dtype=dtype).unsqueeze(0)
        angles = 2 * math.pi * t * bands
        return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            length = x.shape[1]
            features = self.position_features(length, x.device, torch.float32)
            filt = torch.tanh(self.filter_mlp(features)).transpose(0, 1) / math.sqrt(length)

            x_t = x.transpose(1, 2)
            fft_len = 2 * length
            x_f = torch.fft.rfft(x_t, n=fft_len)
            h_f = torch.fft.rfft(filt, n=fft_len).unsqueeze(0)
            y = torch.fft.irfft(x_f * h_f, n=fft_len)[..., :length]
            y = y.transpose(1, 2).to(out_dtype)
        return self.dropout(y)


class StaticLongConv(nn.Module):
    """Fast Hyena-style long depthwise convolution for short MI token sequences."""

    def __init__(self, dim: int, kernel_size: int = 63, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(y)


class HyenaEEGBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2, dropout: float = 0.1, long_conv: str = "static", long_kernel: int = 63):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 3)
        self.short_conv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        if long_conv == "implicit":
            self.long_conv = ImplicitLongConv(dim, dropout=dropout)
        elif long_conv == "static":
            self.long_conv = StaticLongConv(dim, kernel_size=long_kernel, dropout=dropout)
        else:
            raise ValueError(f"Unknown long_conv type: {long_conv}")
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        local, long_input, gate = self.in_proj(z).chunk(3, dim=-1)
        local = self.short_conv(local.transpose(1, 2)).transpose(1, 2)
        long_out = self.long_conv(long_input)
        mixed = (local + long_out) * torch.sigmoid(gate)
        x = x + self.out_proj(mixed)
        return x + self.ffn(x)


class BandGate(nn.Module):
    def __init__(self, dim: int, n_bands: int):
        super().__init__()
        self.n_bands = n_bands
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        summary = torch.cat([tokens.mean(dim=2), tokens.amax(dim=2)], dim=-1)
        logits = self.net(summary).squeeze(-1)
        return torch.softmax(logits, dim=1)


class BandFusion(nn.Module):
    def __init__(self, dim: int, n_bands: int, mode: str = "hybrid", dropout: float = 0.1, band_dropout: float = 0.0):
        super().__init__()
        if mode not in {"sum", "concat", "hybrid"}:
            raise ValueError(f"Unknown band fusion mode: {mode}")
        self.mode = mode
        self.band_dropout = band_dropout
        self.concat_proj = nn.Sequential(
            nn.LayerNorm(dim * n_bands),
            nn.Linear(dim * n_bands, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.hybrid_scale = nn.Parameter(torch.tensor(0.0))

    def apply_band_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.band_dropout <= 0 or not self.training:
            return tokens
        keep_prob = 1.0 - self.band_dropout
        mask = torch.empty(tokens.shape[0], tokens.shape[1], 1, 1, device=tokens.device, dtype=tokens.dtype).bernoulli_(keep_prob)
        counts = mask.sum(dim=1, keepdim=True)
        mask = torch.where(counts == 0, torch.ones_like(mask), mask / keep_prob)
        return tokens * mask

    def forward(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        tokens = self.apply_band_dropout(tokens)
        weighted = (tokens * weights[:, :, None, None]).sum(dim=1)
        if self.mode == "sum":
            return weighted

        batch, n_bands, length, dim = tokens.shape
        concat = tokens.permute(0, 2, 1, 3).reshape(batch, length, n_bands * dim)
        concat = self.concat_proj(concat)
        if self.mode == "concat":
            return concat
        return weighted + torch.sigmoid(self.hybrid_scale) * concat


class CovarianceBranch(nn.Module):
    """Compact correlation feature branch for subject-specific spatial patterns."""

    def __init__(self, n_bands: int, n_chans: int, emb_dim: int, dropout: float = 0.35):
        super().__init__()
        in_dim = n_bands * n_chans * n_chans
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            x = x - x.mean(dim=-1, keepdim=True)
            cov = x @ x.transpose(-1, -2)
            cov = cov / max(x.shape[-1] - 1, 1)
            diag = cov.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
            denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2)).clamp_min(1e-6)
            corr = torch.nan_to_num(cov / denom, nan=0.0, posinf=0.0, neginf=0.0).to(out_dtype)
        return self.proj(corr.flatten(1))


class TemporalLSTMRefiner(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or max(dim // 2, 1)
        self.norm = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * 2
        self.proj = nn.Linear(out_dim, dim) if out_dim != dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.norm(x))
        return x + self.dropout(self.proj(y))


class DilatedTemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = dilation * (kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim, bias=False)
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1, bias=False),
            nn.GLU(dim=1),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = self.pointwise(y).transpose(1, 2)
        return x + self.scale * y


class TemporalTCNRefiner(nn.Module):
    """Small dilated TCN residual refiner after Hyena token mixing."""

    def __init__(self, dim: int, layers: int = 2, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DilatedTemporalConvBlock(dim, kernel_size=kernel_size, dilation=2**idx, dropout=dropout)
                for idx in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AttentiveTemporalPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, feature_dim: int, emb_dim: int, n_classes: int, dropout: float = 0.35, scale: float = 16.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.kaiming_normal_(self.weight)
        self.logit_scale = nn.Parameter(torch.tensor(float(scale)).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = nn.functional.normalize(self.embed(x), dim=-1)
        w = nn.functional.normalize(self.weight, dim=-1)
        return self.logit_scale.exp().clamp(max=64.0) * (z @ w.t())


class YHCHyenaNet(nn.Module):
    def __init__(
        self,
        n_chans: int = 22,
        n_classes: int = 4,
        sfreq: int = 250,
        bands: Sequence[tuple[float, float]] = ((4, 8), (8, 13), (13, 30), (30, 40)),
        emb_dim: int = 64,
        band_depth: int = 2,
        global_depth: int = 1,
        dropout: float = 0.35,
        pool_stride: int = 8,
        fourier_modes: int = 32,
        mixstyle_p: float = 0.2,
        sensor_dropout: float = 0.05,
        stem_type: str = "compact",
        stem_kernels: Sequence[int] = (15, 31, 63),
        shared_band_hyena: bool = True,
        long_conv: str = "implicit",
        long_kernel: int = 63,
        use_logvar_branch: bool = True,
        transformer_layers: int = 0,
        transformer_heads: int = 4,
        tcn_layers: int = 0,
        tcn_kernel: int = 5,
        lstm_layers: int = 1,
        lstm_hidden: int | None = None,
        band_fusion: str = "hybrid",
        band_dropout: float = 0.1,
        ss_calibration: bool = True,
        ss_reduction: int = 4,
        ss_scale_init: float = 0.05,
        ss_dynamic: bool = False,
        use_cov_branch: bool = False,
        use_attn_pool: bool = False,
        head_type: str = "linear",
    ):
        super().__init__()
        if stem_type not in {"compact", "pyramid"}:
            raise ValueError(f"Unknown stem_type: {stem_type}")
        if head_type not in {"linear", "cosine"}:
            raise ValueError(f"Unknown head_type: {head_type}")
        self.n_bands = len(bands)
        self.sensor_dropout = SensorDropout(sensor_dropout)
        self.filter_bank = FixedFFTFilterBank(bands=bands, sfreq=sfreq)
        self.ss_calibration = (
            SpatialSpectralCalibration(
                self.n_bands,
                n_chans,
                reduction=ss_reduction,
                scale_init=ss_scale_init,
                dynamic=ss_dynamic,
            )
            if ss_calibration
            else nn.Identity()
        )
        self.use_logvar_branch = use_logvar_branch
        self.use_cov_branch = use_cov_branch
        if stem_type == "pyramid":
            self.stem = PyramidEEGTokenStem(
                n_chans=n_chans,
                emb_dim=emb_dim,
                kernels=stem_kernels,
                pool_stride=pool_stride,
                dropout=dropout,
            )
        else:
            self.stem = EEGTokenStem(n_chans=n_chans, emb_dim=emb_dim, pool_stride=pool_stride, dropout=dropout)
        self.band_embed = nn.Parameter(torch.zeros(1, self.n_bands, 1, emb_dim))
        nn.init.normal_(self.band_embed, std=0.02)
        self.mixstyle = TokenMixStyle(p=mixstyle_p)
        self.shared_band_hyena = shared_band_hyena
        if shared_band_hyena:
            self.band_blocks = nn.ModuleList(
                [HyenaEEGBlock(emb_dim, dropout=dropout * 0.5, long_conv=long_conv, long_kernel=long_kernel) for _ in range(band_depth)]
            )
        else:
            self.band_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [HyenaEEGBlock(emb_dim, dropout=dropout * 0.5, long_conv=long_conv, long_kernel=long_kernel) for _ in range(self.n_bands)]
                    )
                    for _ in range(band_depth)
                ]
            )
        self.band_gate = BandGate(emb_dim, self.n_bands)
        self.band_fusion = BandFusion(emb_dim, self.n_bands, mode=band_fusion, dropout=dropout * 0.5, band_dropout=band_dropout)
        self.fourier_mixer = AdaptiveFourierMixer(emb_dim, modes=fourier_modes, dropout=dropout * 0.5) if fourier_modes > 0 else nn.Identity()
        self.global_blocks = nn.ModuleList(
            [HyenaEEGBlock(emb_dim, dropout=dropout * 0.5, long_conv=long_conv, long_kernel=long_kernel) for _ in range(global_depth)]
        )
        self.tcn_refine = (
            TemporalTCNRefiner(emb_dim, layers=tcn_layers, kernel_size=tcn_kernel, dropout=dropout * 0.5)
            if tcn_layers > 0
            else nn.Identity()
        )
        self.lstm_refine = (
            TemporalLSTMRefiner(emb_dim, hidden=lstm_hidden, layers=lstm_layers, dropout=dropout * 0.5)
            if lstm_layers > 0
            else nn.Identity()
        )
        if transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=transformer_heads,
                dim_feedforward=emb_dim * 2,
                dropout=dropout * 0.5,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_refine = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        else:
            self.transformer_refine = nn.Identity()
        self.logvar_proj = nn.Sequential(
            nn.LayerNorm(self.n_bands * n_chans),
            nn.Linear(self.n_bands * n_chans, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cov_proj = CovarianceBranch(self.n_bands, n_chans, emb_dim, dropout=dropout) if use_cov_branch else None
        self.attn_pool = AttentiveTemporalPool(emb_dim, dropout=dropout * 0.5) if use_attn_pool else None
        branch_count = 2 + int(use_attn_pool) + int(use_logvar_branch) + int(use_cov_branch)
        self.feature_dim = emb_dim * branch_count
        if head_type == "cosine":
            self.head = CosineClassifier(self.feature_dim, emb_dim, n_classes, dropout=dropout)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, n_classes),
            )

    def forward_features(self, x: torch.Tensor, return_band_weights: bool = False):
        x = self.sensor_dropout(x)
        x = self.filter_bank(x)
        x = self.ss_calibration(x)
        logvar_feature = None
        if self.use_logvar_branch:
            logvar = x.var(dim=-1).clamp_min(1e-6).log().flatten(1)
            logvar_feature = self.logvar_proj(logvar)
        cov_feature = self.cov_proj(x) if self.cov_proj is not None else None
        batch, n_bands, n_chans, samples = x.shape
        tokens = self.stem(x.reshape(batch * n_bands, n_chans, samples))
        length, dim = tokens.shape[1], tokens.shape[2]
        tokens = tokens.reshape(batch, n_bands, length, dim) + self.band_embed

        for layer in self.band_blocks:
            tokens = tokens.reshape(batch * n_bands, length, dim)
            tokens = self.mixstyle(tokens)
            tokens = tokens.reshape(batch, n_bands, length, dim)
            if self.shared_band_hyena:
                tokens = layer(tokens.reshape(batch * n_bands, length, dim)).reshape(batch, n_bands, length, dim)
            else:
                band_outputs = [block(tokens[:, idx]) for idx, block in enumerate(layer)]
                tokens = torch.stack(band_outputs, dim=1)

        weights = self.band_gate(tokens)
        x = self.band_fusion(tokens, weights)
        x = self.fourier_mixer(x)
        for block in self.global_blocks:
            x = block(x)
        x = self.tcn_refine(x)
        x = self.lstm_refine(x)
        x = self.transformer_refine(x)

        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        features = torch.cat([mean_pool, max_pool], dim=-1)
        if self.attn_pool is not None:
            features = torch.cat([features, self.attn_pool(x)], dim=-1)
        if logvar_feature is not None:
            features = torch.cat([features, logvar_feature], dim=-1)
        if cov_feature is not None:
            features = torch.cat([features, cov_feature], dim=-1)
        if return_band_weights:
            return features, weights
        return features

    def forward(self, x: torch.Tensor, return_features: bool = False, return_band_weights: bool = False):
        if return_band_weights:
            features, weights = self.forward_features(x, return_band_weights=True)
            logits = self.head(features)
            return logits, features, weights
        features = self.forward_features(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits
