# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SobelBoundaryExtractor(nn.Module):
    """
    Build boundary maps with Sobel gradients:
        E = sqrt((dI/dx)^2 + (dI/dy)^2)
    """

    def __init__(
        self,
        to_grayscale: bool = True,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.to_grayscale = to_grayscale
        self.normalize = normalize
        self.eps = eps

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], dtype=torch.float32
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]], dtype=torch.float32
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _rgb_to_gray(self, images: torch.Tensor) -> torch.Tensor:
        # ITU-R BT.601 luma transform.
        weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (images * weights).sum(dim=1, keepdim=True)

    def forward(
        self,
        images: torch.Tensor,
        target_sizes: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        if images.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {tuple(images.shape)}.")
        if images.shape[1] not in (1, 3):
            raise ValueError(
                f"Expected 1 or 3 channels for SobelBoundaryExtractor, got {images.shape[1]}."
            )

        if self.to_grayscale:
            if images.shape[1] == 3:
                gray = self._rgb_to_gray(images)
            else:
                gray = images
        else:
            gray = images.mean(dim=1, keepdim=True) if images.shape[1] > 1 else images

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        boundary = torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

        if self.normalize:
            b_min = boundary.amin(dim=(2, 3), keepdim=True)
            b_max = boundary.amax(dim=(2, 3), keepdim=True)
            boundary = (boundary - b_min) / (b_max - b_min + self.eps)

        if target_sizes is None:
            return boundary

        multiscale_boundaries: List[torch.Tensor] = []
        for size in target_sizes:
            multiscale_boundaries.append(
                F.interpolate(boundary, size=size, mode="bilinear", align_corners=False)
            )
        return multiscale_boundaries


class FeatureBridgingModule(nn.Module):
    """
    Feature bridging between two scales:
        alpha = sigmoid(W1 * GAP(Z_i) + W2 * GAP(Z_j))
        F_bridge = alpha * Z_i + (1 - alpha) * Z_j
    """

    def __init__(
        self,
        in_channels_i: int,
        in_channels_j: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        if out_channels is None:
            if in_channels_i != in_channels_j:
                raise ValueError(
                    "out_channels must be set when input channels differ: "
                    f"{in_channels_i} vs {in_channels_j}."
                )
            out_channels = in_channels_i

        self.out_channels = out_channels
        self.align_i = (
            nn.Identity()
            if in_channels_i == out_channels
            else nn.Conv2d(in_channels_i, out_channels, kernel_size=1)
        )
        self.align_j = (
            nn.Identity()
            if in_channels_j == out_channels
            else nn.Conv2d(in_channels_j, out_channels, kernel_size=1)
        )
        self.w1 = nn.Linear(out_channels, out_channels)
        self.w2 = nn.Linear(out_channels, out_channels)

    def _align_spatial(self, z_i: torch.Tensor, z_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_h = max(z_i.shape[-2], z_j.shape[-2])
        target_w = max(z_i.shape[-1], z_j.shape[-1])
        if z_i.shape[-2:] != (target_h, target_w):
            z_i = F.interpolate(z_i, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if z_j.shape[-2:] != (target_h, target_w):
            z_j = F.interpolate(z_j, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return z_i, z_j

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, return_alpha: bool = False):
        if z_i.dim() != 4 or z_j.dim() != 4:
            raise ValueError("FeatureBridgingModule expects 4D feature maps.")
        if z_i.shape[0] != z_j.shape[0]:
            raise ValueError(
                f"Mismatched batch sizes for feature bridging: {z_i.shape[0]} vs {z_j.shape[0]}."
            )

        z_i = self.align_i(z_i)
        z_j = self.align_j(z_j)
        z_i, z_j = self._align_spatial(z_i, z_j)

        gap_i = F.adaptive_avg_pool2d(z_i, output_size=1).flatten(1)
        gap_j = F.adaptive_avg_pool2d(z_j, output_size=1).flatten(1)
        alpha = torch.sigmoid(self.w1(gap_i) + self.w2(gap_j)).view(z_i.shape[0], self.out_channels, 1, 1)

        fused = alpha * z_i + (1.0 - alpha) * z_j
        if return_alpha:
            return fused, alpha
        return fused
