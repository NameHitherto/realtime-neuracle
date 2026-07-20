import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SST-DPN-main"))
sys.path.insert(0, str(ROOT / "YHC-Hyena"))

from model_utils import SSA, LightweightConv1d, Mixer1D  # noqa: E402
from yhc_hyena_net import HyenaEEGBlock  # noqa: E402


class ActiveHyenaAdapter(nn.Module):
    """Bounded non-zero Hyena residual adapter.

    The adapter keeps the SST-DPN feature distribution mostly intact, while
    guaranteeing a small Hyena contribution so gradients can reach the Hyena
    block from the beginning of training.
    """

    def __init__(
        self,
        dim: int,
        layers: int = 1,
        dropout: float = 0.1,
        long_conv: str = "implicit",
        long_kernel: int = 63,
        min_scale: float = 0.005,
        max_extra_scale: float = 0.05,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HyenaEEGBlock(
                    dim,
                    dropout=dropout,
                    long_conv=long_conv,
                    long_kernel=long_kernel,
                )
                for _ in range(layers)
            ]
        )
        self.min_scale = float(min_scale)
        self.max_extra_scale = float(max_extra_scale)
        self.scale_logit = nn.Parameter(torch.tensor(-4.0))

    def adapter_scale(self) -> torch.Tensor:
        return self.min_scale + self.max_extra_scale * torch.sigmoid(self.scale_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        y = y.transpose(1, 2)
        return residual + self.adapter_scale() * (y - residual)


class EfficientEncoderHyenaActive(nn.Module):
    def __init__(
        self,
        samples,
        chans,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=(50, 100, 200),
        hyena_layers: int = 1,
        hyena_dropout: float = 0.1,
        hyena_long_conv: str = "implicit",
        hyena_long_kernel: int = 63,
        hyena_min_scale: float = 0.005,
        hyena_max_extra_scale: float = 0.05,
    ):
        super().__init__()
        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)
        self.chan_conv = nn.Sequential(
            nn.Conv1d(chans * F1, F2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )
        self.hyena = ActiveHyenaAdapter(
            F2,
            layers=hyena_layers,
            dropout=hyena_dropout,
            long_conv=hyena_long_conv,
            long_kernel=hyena_long_kernel,
            min_scale=hyena_min_scale,
            max_extra_scale=hyena_max_extra_scale,
        )
        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):
        x = self.time_conv(x)
        x, _ = self.ssa(x)
        x = self.chan_conv(x)
        x = self.hyena(x)
        return self.mixer(x)


class SST_DPN_Hyena_Active(nn.Module):
    def __init__(
        self,
        chans,
        samples,
        num_classes=4,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=(50, 100, 200),
        device=None,
        hyena_layers: int = 1,
        hyena_dropout: float = 0.1,
        hyena_long_conv: str = "implicit",
        hyena_long_kernel: int = 63,
        hyena_min_scale: float = 0.005,
        hyena_max_extra_scale: float = 0.05,
    ):
        super().__init__()
        self.encoder = EfficientEncoderHyenaActive(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
            hyena_layers=hyena_layers,
            hyena_dropout=hyena_dropout,
            hyena_long_conv=hyena_long_conv,
            hyena_long_kernel=hyena_long_kernel,
            hyena_min_scale=hyena_min_scale,
            hyena_max_extra_scale=hyena_max_extra_scale,
        )
        self.features = None

        probe_device = torch.device(device) if device is not None else torch.device("cpu")
        self.encoder.to(probe_device)
        with torch.no_grad():
            x = torch.ones((1, chans, samples), device=probe_device)
            out = self.encoder(x)
        feat_dim = out.shape[-1]

        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim, device=probe_device), requires_grad=True)
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim, device=probe_device), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)

    def get_features(self):
        if self.features is None:
            raise RuntimeError("No features available. Run forward() first.")
        return self.features

    def forward(self, x):
        features = self.encoder(x)
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        return torch.einsum("bd,cd->bc", features, self.isp)
