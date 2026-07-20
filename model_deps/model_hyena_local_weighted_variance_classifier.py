import math

import torch
import torch.nn.functional as F
from torch import nn

from model_sstdpn_hyena_rhythm_dssa import EfficientEncoderHyenaRhythmDSSA
from model_hyena_ordered_variance_classifier import pairwise_discriminative_loss


class LocalWeightedVarianceCompression(nn.Module):
    """Learn ordered local pooling without changing variance amplitude units."""

    def __init__(
        self,
        channels: int,
        samples: int,
        kernel_sizes=(50, 100, 200),
        output_bins=(16, 10, 6),
        locality: float = 1.25,
        max_logit_offset: float = 2.0,
    ):
        super().__init__()
        if channels % len(kernel_sizes) != 0:
            raise ValueError("channels must be divisible by the number of scales")
        if len(kernel_sizes) != len(output_bins):
            raise ValueError("kernel_sizes and output_bins must have equal length")

        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        self.output_bins = tuple(int(value) for value in output_bins)
        self.channels_per_scale = channels // len(self.kernel_sizes)
        self.output_dim = self.channels_per_scale * sum(self.output_bins)
        self.max_logit_offset = float(max_logit_offset)
        self.weight_offsets = nn.ParameterList()

        for kernel_size, bins in zip(self.kernel_sizes, self.output_bins):
            stride = max(kernel_size // 2, 1)
            input_length = math.floor((samples - kernel_size) / stride + 1)
            input_positions = torch.linspace(0.0, 1.0, input_length)
            output_positions = torch.linspace(0.0, 1.0, bins)
            bin_spacing = 1.0 / max(bins - 1, 1)
            sigma = locality * bin_spacing
            squared_distance = (
                output_positions[:, None] - input_positions[None, :]
            ).square()
            log_prior = -0.5 * squared_distance / max(sigma**2, 1e-6)
            self.register_buffer(
                f"log_prior_{len(self.weight_offsets)}",
                log_prior,
            )
            self.weight_offsets.append(nn.Parameter(torch.zeros_like(log_prior)))

    @staticmethod
    def local_variance(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        stride = max(kernel_size // 2, 1)
        mean_square = F.avg_pool1d(
            x.square(),
            kernel_size=kernel_size,
            stride=stride,
        )
        square_mean = F.avg_pool1d(
            x,
            kernel_size=kernel_size,
            stride=stride,
        ).square()
        return (mean_square - square_mean).clamp_min(0.0)

    def pooling_weights(self, index: int) -> torch.Tensor:
        prior = getattr(self, f"log_prior_{index}")
        offset = self.max_logit_offset * torch.tanh(self.weight_offsets[index])
        return torch.softmax(prior + offset, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = torch.split(x, self.channels_per_scale, dim=1)
        compressed = []
        for index, (group, kernel_size) in enumerate(
            zip(groups, self.kernel_sizes)
        ):
            variance = self.local_variance(group, kernel_size)
            weights = self.pooling_weights(index)
            ordered_bins = torch.einsum("bcl,ol->bco", variance, weights)
            compressed.append(ordered_bins.flatten(start_dim=1))
        return torch.cat(compressed, dim=-1)

    def smoothness_loss(self) -> torch.Tensor:
        losses = []
        for index in range(len(self.weight_offsets)):
            weights = self.pooling_weights(index)
            if weights.shape[0] > 1:
                losses.append((weights[1:] - weights[:-1]).square().mean())
        return torch.stack(losses).mean()


class HyenaLocalWeightedVarianceClassifierNet(nn.Module):
    """Hyena EEG encoder with local learnable ordered variance compression."""

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
        output_bins=(16, 10, 6),
        locality: float = 1.25,
        max_logit_offset: float = 2.0,
        classifier_dropout: float = 0.1,
        classifier_maxnorm: float = 1.0,
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
        self.encoder.mixer = nn.Identity()
        self.compression = LocalWeightedVarianceCompression(
            channels=F2,
            samples=samples,
            kernel_sizes=pool_kernels,
            output_bins=output_bins,
            locality=locality,
            max_logit_offset=max_logit_offset,
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(
            self.compression.output_dim,
            num_classes,
            bias=True,
        )
        self.classifier_maxnorm = float(classifier_maxnorm)
        self.embedding = None

        module_device = torch.device(device) if device is not None else torch.device("cpu")
        self.to(module_device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.compression(self.encoder(x))
        self.embedding = embedding
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(x)
        with torch.no_grad():
            self.classifier.weight.renorm_(
                p=2,
                dim=0,
                maxnorm=self.classifier_maxnorm,
            )
        return self.classifier(self.dropout(embedding))


__all__ = [
    "HyenaLocalWeightedVarianceClassifierNet",
    "LocalWeightedVarianceCompression",
    "pairwise_discriminative_loss",
]
