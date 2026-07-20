import torch
from torch import nn

from model_hyena_local_weighted_variance_classifier import (
    HyenaLocalWeightedVarianceClassifierNet,
    pairwise_discriminative_loss,
)


class InputMotorRhythmEnhancer(nn.Module):
    """Identity-initialized per-electrode mu/beta residual enhancement."""

    def __init__(
        self,
        channels: int,
        samples: int,
        sampling_rate: float = 250.0,
        max_scale: float = 0.25,
    ):
        super().__init__()
        frequencies = torch.fft.rfftfreq(samples, d=1.0 / sampling_rate)
        self.samples = int(samples)
        self.max_scale = float(max_scale)
        self.register_buffer(
            "mu_mask",
            ((frequencies >= 8.0) & (frequencies < 13.0)).float(),
        )
        self.register_buffer(
            "beta_mask",
            ((frequencies >= 13.0) & (frequencies <= 30.0)).float(),
        )
        self.mu_logits = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta_logits = nn.Parameter(torch.zeros(1, channels, 1))

    def scales(self):
        return (
            self.max_scale * torch.tanh(self.mu_logits),
            self.max_scale * torch.tanh(self.beta_logits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.samples:
            raise ValueError(f"Expected {self.samples} samples, got {x.shape[-1]}")

        output_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=-1)
            mu = torch.fft.irfft(
                spectrum * self.mu_mask[None, None, :],
                n=self.samples,
                dim=-1,
            )
            beta = torch.fft.irfft(
                spectrum * self.beta_mask[None, None, :],
                n=self.samples,
                dim=-1,
            )
        mu_scale, beta_scale = self.scales()
        return x + mu_scale * mu.to(output_dtype) + beta_scale * beta.to(output_dtype)


class RhythmEnhancedTimeConv(nn.Module):
    def __init__(self, enhancer: nn.Module, time_conv: nn.Module):
        super().__init__()
        self.enhancer = enhancer
        self.time_conv = time_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.time_conv(self.enhancer(x))


class HyenaInputMotorRhythmEnhancedClassifierNet(
    HyenaLocalWeightedVarianceClassifierNet
):
    """Current Hyena classifier with raw-channel mu/beta residual enhancement."""

    def __init__(
        self,
        *args,
        sampling_rate: float = 250.0,
        rhythm_max_scale: float = 0.25,
        device=None,
        **kwargs,
    ):
        super().__init__(*args, device=device, **kwargs)
        channels = int(kwargs.get("chans", 22))
        samples = int(kwargs.get("samples", 1000))
        module_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.encoder.time_conv = RhythmEnhancedTimeConv(
            InputMotorRhythmEnhancer(
                channels=channels,
                samples=samples,
                sampling_rate=sampling_rate,
                max_scale=rhythm_max_scale,
            ),
            self.encoder.time_conv,
        ).to(module_device)


__all__ = [
    "InputMotorRhythmEnhancer",
    "HyenaInputMotorRhythmEnhancedClassifierNet",
    "pairwise_discriminative_loss",
]
