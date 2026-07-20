import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SST-DPN-Hyena"))
sys.path.insert(0, str(ROOT / "YHC-Hyena"))

from model_hyena_input_motor_rhythm_enhanced_classifier import (
    HyenaInputMotorRhythmEnhancedClassifierNet,
    pairwise_discriminative_loss,
)
from model_hyena_multiscale_temporal_contrast_classifier import (
    HyenaMultiScaleTemporalContrastClassifierNet,
)


class BandpowerFeatureEncoder(nn.Module):
    """Log-bandpower side branch for HGD-style broad-band MI decoding."""

    def __init__(
        self,
        channels: int,
        samples: int,
        sampling_rate: float,
        bands=((8.0, 13.0), (13.0, 30.0), (30.0, 60.0), (60.0, 100.0), (100.0, 122.0)),
        feature_dim: int = 128,
        dropout: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.samples = int(samples)
        self.sampling_rate = float(sampling_rate)
        self.bands = tuple((float(lo), float(hi)) for lo, hi in bands)
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)

        frequencies = torch.fft.rfftfreq(self.samples, d=1.0 / self.sampling_rate)
        masks = []
        for low, high in self.bands:
            masks.append(((frequencies >= low) & (frequencies < high)).float())
        self.register_buffer("band_masks", torch.stack(masks, dim=0))

        input_dim = self.channels * len(self.bands)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.samples:
            raise ValueError(f"Expected {self.samples} samples, got {x.shape[-1]}")

        with torch.amp.autocast("cuda", enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=-1)
            band_features = []
            for mask in self.band_masks:
                filtered = torch.fft.irfft(
                    spectrum * mask[None, None, :],
                    n=self.samples,
                    dim=-1,
                )
                band_features.append(torch.log(filtered.var(dim=-1) + self.eps))
            features = torch.cat(band_features, dim=1)
        return self.net(features.to(x.dtype))


class FrequencyConditionedReadoutMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_frequency_conditioned_readout(
        self,
        *,
        channels: int,
        samples: int,
        sampling_rate: float,
        num_classes: int,
        bandpower_feature_dim: int,
        bandpower_dropout: float,
        classifier_dropout: float,
        classifier_maxnorm: float,
    ):
        self.bandpower = BandpowerFeatureEncoder(
            channels=channels,
            samples=samples,
            sampling_rate=sampling_rate,
            feature_dim=bandpower_feature_dim,
            dropout=bandpower_dropout,
        )
        self.embedding_dim = int(self.compression.output_dim + bandpower_feature_dim)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes, bias=True)
        self.classifier_maxnorm = float(classifier_maxnorm)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        temporal_embedding = self.compression(self.encoder(x))
        spectral_embedding = self.bandpower(x)
        embedding = torch.cat([temporal_embedding, spectral_embedding], dim=1)
        self.embedding = embedding
        return embedding


class HGDFrequencyConditionedRhythmHyenaNet(
    FrequencyConditionedReadoutMixin,
    HyenaInputMotorRhythmEnhancedClassifierNet,
):
    """Rhythm-Hyena with an explicit high-gamma bandpower readout branch."""

    def __init__(
        self,
        *args,
        bandpower_feature_dim: int = 128,
        bandpower_dropout: float = 0.2,
        sampling_rate: float = 250.0,
        classifier_dropout: float = 0.1,
        classifier_maxnorm: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            sampling_rate=sampling_rate,
            classifier_dropout=classifier_dropout,
            classifier_maxnorm=classifier_maxnorm,
            **kwargs,
        )
        self._init_frequency_conditioned_readout(
            channels=int(kwargs.get("chans", 22)),
            samples=int(kwargs.get("samples", 1000)),
            sampling_rate=sampling_rate,
            num_classes=int(kwargs.get("num_classes", 4)),
            bandpower_feature_dim=bandpower_feature_dim,
            bandpower_dropout=bandpower_dropout,
            classifier_dropout=classifier_dropout,
            classifier_maxnorm=classifier_maxnorm,
        )


class HGDFrequencyConditionedTemporalHyenaNet(
    FrequencyConditionedReadoutMixin,
    HyenaMultiScaleTemporalContrastClassifierNet,
):
    """Temporal-contrast Hyena with an explicit high-gamma bandpower branch."""

    def __init__(
        self,
        *args,
        bandpower_feature_dim: int = 128,
        bandpower_dropout: float = 0.2,
        sampling_rate: float = 250.0,
        classifier_dropout: float = 0.1,
        classifier_maxnorm: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            sampling_rate=sampling_rate,
            classifier_dropout=classifier_dropout,
            classifier_maxnorm=classifier_maxnorm,
            **kwargs,
        )
        self._init_frequency_conditioned_readout(
            channels=int(kwargs.get("chans", 22)),
            samples=int(kwargs.get("samples", 1000)),
            sampling_rate=sampling_rate,
            num_classes=int(kwargs.get("num_classes", 4)),
            bandpower_feature_dim=bandpower_feature_dim,
            bandpower_dropout=bandpower_dropout,
            classifier_dropout=classifier_dropout,
            classifier_maxnorm=classifier_maxnorm,
        )


__all__ = [
    "BandpowerFeatureEncoder",
    "HGDFrequencyConditionedRhythmHyenaNet",
    "HGDFrequencyConditionedTemporalHyenaNet",
    "pairwise_discriminative_loss",
]
