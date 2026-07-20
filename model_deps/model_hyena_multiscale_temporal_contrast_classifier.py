import torch
import torch.nn.functional as F
from torch import nn

from model_hyena_input_motor_rhythm_enhanced_classifier import (
    HyenaInputMotorRhythmEnhancedClassifierNet,
    pairwise_discriminative_loss,
)


class MultiScaleTemporalContrastEnhancer(nn.Module):
    """CT-MIFNet-inspired multi-scale local temporal contrast on raw EEG."""

    def __init__(
        self,
        channels: int,
        kernel_sizes=(20, 40, 80, 160),
        max_scale: float = 0.12,
    ):
        super().__init__()
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        self.max_scale = float(max_scale)
        self.scale_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes), channels, 1))
        self.mix_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes)))

    def scales(self):
        return self.max_scale * torch.tanh(self.scale_logits)

    def mix_weights(self):
        return torch.softmax(self.mix_logits, dim=0)

    @staticmethod
    def moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        left = kernel_size // 2
        right = kernel_size - 1 - left
        padded = F.pad(x, (left, right), mode="reflect")
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.zeros_like(x)
        mix = self.mix_weights()
        scales = self.scales()
        for index, kernel_size in enumerate(self.kernel_sizes):
            local_mean = self.moving_average(x, kernel_size)
            local_contrast = x - local_mean
            residual = residual + mix[index] * scales[index] * local_contrast
        return x + residual


class CompositeInputEnhancer(nn.Module):
    def __init__(self, rhythm_enhancer: nn.Module, temporal_enhancer: nn.Module):
        super().__init__()
        self.rhythm_enhancer = rhythm_enhancer
        self.temporal_enhancer = temporal_enhancer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal_enhancer(self.rhythm_enhancer(x))


class HyenaMultiScaleTemporalContrastClassifierNet(
    HyenaInputMotorRhythmEnhancedClassifierNet
):
    """Hyena rhythm model with CT-MIFNet-inspired raw-signal multi-scale contrast."""

    def __init__(
        self,
        *args,
        contrast_kernel_sizes=(20, 40, 80, 160),
        contrast_max_scale: float = 0.12,
        device=None,
        **kwargs,
    ):
        super().__init__(*args, device=device, **kwargs)
        channels = int(kwargs.get("chans", 22))
        module_device = torch.device(device) if device is not None else torch.device("cpu")
        rhythm_enhancer = self.encoder.time_conv.enhancer
        temporal_enhancer = MultiScaleTemporalContrastEnhancer(
            channels=channels,
            kernel_sizes=contrast_kernel_sizes,
            max_scale=contrast_max_scale,
        )
        self.encoder.time_conv.enhancer = CompositeInputEnhancer(
            rhythm_enhancer,
            temporal_enhancer,
        ).to(module_device)


__all__ = [
    "MultiScaleTemporalContrastEnhancer",
    "HyenaMultiScaleTemporalContrastClassifierNet",
    "pairwise_discriminative_loss",
]
