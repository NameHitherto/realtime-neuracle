import torch
import torch.nn.functional as F
from torch import nn

from model_sstdpn_hyena_rhythm_dssa import EfficientEncoderHyenaRhythmDSSA


class OrderedVarianceCompression(nn.Module):
    """Retain ordered local variance bins at three temporal scales."""

    def __init__(
        self,
        channels: int,
        kernel_sizes=(50, 100, 200),
        output_bins=(16, 10, 6),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = torch.split(x, self.channels_per_scale, dim=1)
        compressed = []
        for group, kernel_size, bins in zip(
            groups,
            self.kernel_sizes,
            self.output_bins,
        ):
            variance = self.local_variance(group, kernel_size)
            ordered_bins = F.adaptive_avg_pool1d(variance, bins)
            compressed.append(ordered_bins.flatten(start_dim=1))
        return torch.cat(compressed, dim=-1)


class HyenaOrderedVarianceClassifierNet(nn.Module):
    """Hyena EEG encoder with ordered variance compression and no prototypes."""

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
        self.compression = OrderedVarianceCompression(
            channels=F2,
            kernel_sizes=pool_kernels,
            output_bins=output_bins,
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


def pairwise_discriminative_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    cosine_margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact same-class samples and separate different-class directions."""

    sample_count = embedding.shape[0]
    if sample_count < 2:
        zero = embedding.new_zeros(())
        return zero, zero

    same_class = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(
        sample_count,
        device=embedding.device,
        dtype=torch.bool,
    )
    same_class = same_class & ~diagonal
    different_class = ~labels[:, None].eq(labels[None, :])

    absolute_difference = (
        embedding[:, None, :] - embedding[None, :, :]
    ).abs()
    pairwise_huber = torch.where(
        absolute_difference < 1.0,
        0.5 * absolute_difference.square(),
        absolute_difference - 0.5,
    ).mean(dim=-1)
    compactness = (
        pairwise_huber[same_class].mean()
        if same_class.any()
        else embedding.new_zeros(())
    )

    directions = F.normalize(embedding, p=2, dim=-1)
    cosine = directions @ directions.transpose(0, 1)
    separation = (
        F.relu(cosine[different_class] - cosine_margin).mean()
        if different_class.any()
        else embedding.new_zeros(())
    )
    return compactness, separation
