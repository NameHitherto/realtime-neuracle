import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SST-DPN-Hyena"))
sys.path.insert(0, str(ROOT / "YHC-Hyena"))

from model_hgd_frequency_conditioned_hyena import (
    HGDFrequencyConditionedRhythmHyenaNet,
    HGDFrequencyConditionedTemporalHyenaNet,
)
from model_hyena_input_motor_rhythm_enhanced_classifier import (
    HyenaInputMotorRhythmEnhancedClassifierNet,
    pairwise_discriminative_loss,
)
from model_hyena_multiscale_temporal_contrast_classifier import (
    HyenaMultiScaleTemporalContrastClassifierNet,
)


class CosineClassifier(nn.Module):
    """Scale-controlled cosine classifier for session-robust EEG readout."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        scale_init: float = 16.0,
        scale_max: float = 40.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale_max = float(scale_max)
        init_ratio = min(max(float(scale_init) / self.scale_max, 1e-4), 1.0 - 1e-4)
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1.0 - init_ratio)))
        )
        self.eps = float(eps)

    def scale(self) -> torch.Tensor:
        return self.scale_max * torch.sigmoid(self.log_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1, eps=self.eps)
        weight = F.normalize(self.weight, dim=1, eps=self.eps)
        return self.scale() * F.linear(x, weight)


class ResidualCosineClassifier(nn.Module):
    """Linear evidence with a learnable residual cosine-prototype correction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        scale_init: float = 16.0,
        scale_max: float = 40.0,
        residual_max: float = 1.0,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=True)
        self.cosine = CosineClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            scale_init=scale_init,
            scale_max=scale_max,
        )
        self.residual_logit = nn.Parameter(torch.zeros(()))
        self.residual_max = float(residual_max)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.linear.weight

    def residual_ratio(self) -> torch.Tensor:
        return self.residual_max * torch.tanh(self.residual_logit)

    def scale(self) -> torch.Tensor:
        return self.cosine.scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_ratio() * self.cosine(x)


class CosineReadoutMixin:
    def _replace_classifier_with_cosine(
        self,
        *,
        input_dim: int,
        num_classes: int,
        scale_init: float,
        scale_max: float,
    ) -> None:
        self.embedding_dim = int(input_dim)
        self.classifier = CosineClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            scale_init=scale_init,
            scale_max=scale_max,
        )


class ResidualCosineReadoutMixin:
    def _replace_classifier_with_residual_cosine(
        self,
        *,
        input_dim: int,
        num_classes: int,
        scale_init: float,
        scale_max: float,
        residual_max: float,
    ) -> None:
        self.embedding_dim = int(input_dim)
        self.classifier = ResidualCosineClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            scale_init=scale_init,
            scale_max=scale_max,
            residual_max=residual_max,
        )


class HyenaCosineRhythmNet(
    CosineReadoutMixin,
    HyenaInputMotorRhythmEnhancedClassifierNet,
):
    """Rhythm-Hyena with normalized cosine class prototypes."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_cosine(
            input_dim=self.compression.output_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
        )


class HyenaCosineTemporalNet(
    CosineReadoutMixin,
    HyenaMultiScaleTemporalContrastClassifierNet,
):
    """Temporal-contrast Hyena with normalized cosine class prototypes."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_cosine(
            input_dim=self.compression.output_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
        )


class HyenaCosineFrequencyRhythmNet(
    CosineReadoutMixin,
    HGDFrequencyConditionedRhythmHyenaNet,
):
    """Frequency-conditioned rhythm Hyena with a cosine classifier."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_cosine(
            input_dim=self.embedding_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
        )


class HyenaCosineFrequencyTemporalNet(
    CosineReadoutMixin,
    HGDFrequencyConditionedTemporalHyenaNet,
):
    """Frequency-conditioned temporal Hyena with a cosine classifier."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_cosine(
            input_dim=self.embedding_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
        )


class HyenaResidualCosineRhythmNet(
    ResidualCosineReadoutMixin,
    HyenaInputMotorRhythmEnhancedClassifierNet,
):
    """Rhythm-Hyena with a gated residual cosine correction to the linear head."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        residual_cosine_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_residual_cosine(
            input_dim=self.compression.output_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
            residual_max=residual_cosine_max,
        )


class HyenaResidualCosineTemporalNet(
    ResidualCosineReadoutMixin,
    HyenaMultiScaleTemporalContrastClassifierNet,
):
    """Temporal-contrast Hyena with a gated residual cosine correction."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        residual_cosine_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_residual_cosine(
            input_dim=self.compression.output_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
            residual_max=residual_cosine_max,
        )


class HyenaResidualCosineFrequencyTemporalNet(
    ResidualCosineReadoutMixin,
    HGDFrequencyConditionedTemporalHyenaNet,
):
    """Frequency-conditioned temporal Hyena with a residual cosine correction."""

    def __init__(
        self,
        *args,
        cosine_scale_init: float = 16.0,
        cosine_scale_max: float = 40.0,
        residual_cosine_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._replace_classifier_with_residual_cosine(
            input_dim=self.embedding_dim,
            num_classes=int(kwargs.get("num_classes", 4)),
            scale_init=cosine_scale_init,
            scale_max=cosine_scale_max,
            residual_max=residual_cosine_max,
        )


__all__ = [
    "CosineClassifier",
    "ResidualCosineClassifier",
    "HyenaCosineRhythmNet",
    "HyenaCosineTemporalNet",
    "HyenaCosineFrequencyRhythmNet",
    "HyenaCosineFrequencyTemporalNet",
    "HyenaResidualCosineRhythmNet",
    "HyenaResidualCosineTemporalNet",
    "HyenaResidualCosineFrequencyTemporalNet",
    "pairwise_discriminative_loss",
]
