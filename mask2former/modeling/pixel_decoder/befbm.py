# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
from torch.nn import functional as F


class SobelBoundaryExtractor(nn.Module):
    """
    Compute a single-channel boundary magnitude map using Sobel gradients.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], dtype=torch.float32
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]], dtype=torch.float32
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Convert RGB to grayscale for gradient extraction.
        gray = images.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)


class FeatureBridgeUnit(nn.Module):
    """
    Pair-wise adaptive bridge:
    F_bridge = alpha * Zi + (1 - alpha) * Zj
    alpha = sigmoid(W1 * GAP(Zi) + W2 * GAP(Zj))
    """

    def __init__(self, channels: int):
        super().__init__()
        self.w1 = nn.Linear(channels, channels)
        self.w2 = nn.Linear(channels, channels)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_j.shape[-2:] != z_i.shape[-2:]:
            z_j = F.interpolate(z_j, size=z_i.shape[-2:], mode="bilinear", align_corners=False)

        gap_i = F.adaptive_avg_pool2d(z_i, output_size=1).flatten(1)
        gap_j = F.adaptive_avg_pool2d(z_j, output_size=1).flatten(1)
        alpha = torch.sigmoid(self.w1(gap_i) + self.w2(gap_j)).unsqueeze(-1).unsqueeze(-1)
        return alpha * z_i + (1.0 - alpha) * z_j


class BoundaryEnhancedFeatureBridgingModule(nn.Module):
    """
    Apply pair-wise bridging across adjacent feature levels.
    """

    def __init__(self, channels: int, num_feature_levels: int):
        super().__init__()
        self.bridges = nn.ModuleList(
            [FeatureBridgeUnit(channels) for _ in range(max(0, num_feature_levels - 1))]
        )

    def forward(self, features):
        if len(features) <= 1:
            return features

        fused = list(features)
        for i, bridge in enumerate(self.bridges):
            fused[i] = bridge(features[i], features[i + 1])
        return fused
