"""CNN model definitions for the VLM-guided grasp regression experiments."""

from __future__ import annotations

import torch
from torch import nn


def _backbone_layers() -> list[nn.Module]:
    return [
        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # The feature map is always 7x7 for the fixed 224x224 input.
        # Fixed average pooling is mathematically equivalent here and has a
        # deterministic CUDA backward implementation.
        nn.AvgPool2d(kernel_size=7),
        nn.Flatten(),
    ]


def _regression_layers(output_size: int) -> list[nn.Module]:
    return [
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, output_size),
    ]


class GraspFeatureBackbone(nn.Module):
    """Four convolution blocks followed by global average pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(*_backbone_layers())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class RegressionHead(nn.Module):
    """Small MLP used by each grasp parameter group."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        if input_size != 256:
            raise ValueError("the current regression head expects 256 features")
        self.layers = nn.Sequential(*_regression_layers(output_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SingleHeadCNNGraspRegressor(nn.Module):
    """Legacy-compatible six-value grasp regressor."""

    def __init__(self) -> None:
        super().__init__()
        # Keep one Sequential container so legacy ``model.<index>`` state keys
        # remain loadable by the original experiment script.
        self.model = nn.Sequential(
            *_backbone_layers(),
            *_regression_layers(output_size=6),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class MultiHeadCNNGraspRegressor(nn.Module):
    """Shared visual backbone with centre, size, and orientation heads."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = GraspFeatureBackbone()
        self.centre_head = RegressionHead(256, 2)
        self.size_head = RegressionHead(256, 2)
        self.orientation_head = RegressionHead(256, 2)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(inputs)
        return {
            "centre": self.centre_head(features),
            "size": self.size_head(features),
            "orientation": self.orientation_head(features),
        }


def compute_multi_head_loss(
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    orientation_norm_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return the total multi-head loss and each auditable component."""

    smooth_l1 = nn.SmoothL1Loss()
    centre = smooth_l1(predictions["centre"], targets[:, 0:2])
    size = smooth_l1(predictions["size"], targets[:, 2:4])
    orientation = smooth_l1(predictions["orientation"], targets[:, 4:6])
    norms = torch.linalg.vector_norm(predictions["orientation"], dim=1)
    unit_norm = torch.mean((norms - 1.0) ** 2)
    total = centre + size + orientation + orientation_norm_weight * unit_norm
    return {
        "total": total,
        "centre": centre,
        "size": size,
        "orientation": orientation,
        "unit_norm": unit_norm,
    }
