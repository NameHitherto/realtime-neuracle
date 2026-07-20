import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SST-DPN-main"))
sys.path.insert(0, str(ROOT / "YHC-Hyena"))

from model_utils import LightweightConv1d, Mixer1D  # noqa: E402
from model_sstdpn_hyena_active import ActiveHyenaAdapter  # noqa: E402
from model_sstdpn_hyena_rhythm import RhythmFourierAdapter  # noqa: E402


class DualStatisticSSA(nn.Module):
    """SSA variant using both feature power and temporal-difference energy."""

    def __init__(self, num_channels: int, epsilon: float = 1e-5, gate_scale: float = 0.5):
        super().__init__()
        self.epsilon = epsilon
        self.gate_scale = float(gate_scale)
        self.power_gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.diff_gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

    def normalize_stat(self, stat: torch.Tensor) -> torch.Tensor:
        denom = stat.pow(2).mean(dim=1, keepdim=True).add(self.epsilon).sqrt()
        return stat / denom

    def forward(self, x):
        power = x.pow(2).mean(dim=-1, keepdim=True).add(self.epsilon).sqrt()
        diff = x[:, :, 1:] - x[:, :, :-1]
        diff_power = diff.pow(2).mean(dim=-1, keepdim=True).add(self.epsilon).sqrt()

        power = self.normalize_stat(power)
        diff_power = self.normalize_stat(diff_power)
        gate_input = power * self.power_gamma + diff_power * self.diff_gamma + self.beta
        gate = 1.0 + self.gate_scale * torch.tanh(gate_input)
        return x * gate, gate


class EfficientEncoderHyenaRhythmDSSA(nn.Module):
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
        rhythm_modes: int = 40,
        rhythm_dropout: float = 0.05,
        rhythm_min_scale: float = 0.002,
        rhythm_max_extra_scale: float = 0.03,
        ssa_gate_scale: float = 0.5,
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
        self.ssa = DualStatisticSSA(chans * F1, gate_scale=ssa_gate_scale)
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
        self.rhythm = RhythmFourierAdapter(
            F2,
            modes=rhythm_modes,
            dropout=rhythm_dropout,
            min_scale=rhythm_min_scale,
            max_extra_scale=rhythm_max_extra_scale,
        )
        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):
        x = self.time_conv(x)
        x, _ = self.ssa(x)
        x = self.chan_conv(x)
        x = self.hyena(x)
        x = self.rhythm(x)
        return self.mixer(x)


class SST_DPN_Hyena_Rhythm_DSSA(nn.Module):
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
        rhythm_modes: int = 40,
        rhythm_dropout: float = 0.05,
        rhythm_min_scale: float = 0.002,
        rhythm_max_extra_scale: float = 0.03,
        ssa_gate_scale: float = 0.5,
    ):
        super().__init__()
        self.encoder = EfficientEncoderHyenaRhythmDSSA(
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
            rhythm_modes=rhythm_modes,
            rhythm_dropout=rhythm_dropout,
            rhythm_min_scale=rhythm_min_scale,
            rhythm_max_extra_scale=rhythm_max_extra_scale,
            ssa_gate_scale=ssa_gate_scale,
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
