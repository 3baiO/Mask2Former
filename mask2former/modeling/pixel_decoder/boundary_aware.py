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


class BoundaryPropagationGate(nn.Module):
    """
    Convert boundary logits to propagation confidences:
        p = 1 - beta * sigmoid(alpha * b - gamma)
    """

    def __init__(self, alpha: float = 20.0, gamma: float = 4.0, beta_init: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, boundary_logits: torch.Tensor) -> torch.Tensor:
        if boundary_logits.dim() != 4 or boundary_logits.shape[1] != 1:
            raise ValueError(
                "BoundaryPropagationGate expects boundary logits with shape [B, 1, H, W], "
                f"got {tuple(boundary_logits.shape)}."
            )

        boundary_scores = boundary_logits.sigmoid()
        propagation = 1.0 - self.beta * torch.sigmoid(self.alpha * boundary_scores - self.gamma)
        return propagation.clamp_(0.0, 1.0)


class DirectionalPropagation1D(nn.Module):
    """
    Boundary-controlled directional propagation along width or height.
    """

    def __init__(self, channels: int, kernel_size: int = 1):
        super().__init__()
        padding = max(0, kernel_size // 2)
        self.input_proj = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.state_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=False)

    def _propagate_sequence(self, inputs: torch.Tensor, confidence: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            inputs = torch.flip(inputs, dims=[-1])
            confidence = torch.flip(confidence, dims=[-1])

        projected_inputs = self.input_proj(inputs)
        outputs: List[torch.Tensor] = []
        prev_state = inputs.new_zeros(inputs.shape[0], inputs.shape[1], 1)
        for idx in range(inputs.shape[-1]):
            gated_prev = prev_state * confidence[..., idx : idx + 1]
            recurrent = self.state_proj(gated_prev)
            current = projected_inputs[..., idx : idx + 1] + recurrent + self.bias.view(1, -1, 1)
            prev_state = self.activation(current)
            outputs.append(prev_state)

        outputs = torch.cat(outputs, dim=-1)
        if reverse:
            outputs = torch.flip(outputs, dims=[-1])
        return outputs

    def forward(self, feature: torch.Tensor, confidence: torch.Tensor, direction: str) -> torch.Tensor:
        if feature.dim() != 4:
            raise ValueError(f"DirectionalPropagation1D expects 4D features, got {tuple(feature.shape)}.")
        if confidence.shape != (feature.shape[0], 1, feature.shape[2], feature.shape[3]):
            raise ValueError(
                "Propagation confidence must match feature spatial size, "
                f"got feature {tuple(feature.shape)} and confidence {tuple(confidence.shape)}."
            )

        if direction in ("lr", "rl"):
            batch, channels, height, width = feature.shape
            seq = feature.permute(0, 2, 1, 3).reshape(batch * height, channels, width)
            gate = confidence.permute(0, 2, 1, 3).reshape(batch * height, 1, width)
            propagated = self._propagate_sequence(seq, gate, reverse=direction == "rl")
            return propagated.reshape(batch, height, channels, width).permute(0, 2, 1, 3)

        if direction in ("tb", "bt"):
            batch, channels, height, width = feature.shape
            seq = feature.permute(0, 3, 1, 2).reshape(batch * width, channels, height)
            gate = confidence.permute(0, 3, 1, 2).reshape(batch * width, 1, height)
            propagated = self._propagate_sequence(seq, gate, reverse=direction == "bt")
            return propagated.reshape(batch, width, channels, height).permute(0, 2, 3, 1)

        raise ValueError(f"Unsupported propagation direction: {direction}")


def _build_bfp_norm(norm: str, channels: int) -> nn.Module:
    if norm in ("", "none", None):
        return nn.Identity()
    if norm == "GN":
        groups = 32 if channels % 32 == 0 else 1
        return nn.GroupNorm(groups, channels)
    if norm == "BN":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"Unsupported BFP norm: {norm}")


class BoundaryFeaturePropagation(nn.Module):
    """
    Four-direction boundary-controlled propagation on a single feature level.
    """

    def __init__(
        self,
        channels: int,
        directions: Sequence[str],
        kernel_size: int = 1,
        alpha: float = 20.0,
        gamma: float = 4.0,
        beta_init: float = 1.0,
        fuse: str = "sum",
        residual: bool = True,
        norm: str = "GN",
    ):
        super().__init__()
        if not directions:
            raise ValueError("BoundaryFeaturePropagation requires at least one direction.")
        self.directions = list(directions)
        self.fuse = fuse
        self.residual = residual
        self.gate = BoundaryPropagationGate(alpha=alpha, gamma=gamma, beta_init=beta_init)
        self.propagators = nn.ModuleDict(
            {
                direction: DirectionalPropagation1D(channels=channels, kernel_size=kernel_size)
                for direction in self.directions
            }
        )
        self.out_norm = _build_bfp_norm(norm, channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        feature: torch.Tensor,
        boundary_logits: torch.Tensor,
        return_confidence: bool = False,
    ):
        confidence = self.gate(boundary_logits)
        propagated = [self.propagators[direction](feature, confidence, direction) for direction in self.directions]

        if self.fuse == "mean":
            fused = torch.stack(propagated, dim=0).mean(dim=0)
        elif self.fuse == "sum":
            fused = torch.stack(propagated, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unsupported BFP fuse mode: {self.fuse}")

        fused = self.out_proj(fused)
        if self.residual:
            fused = fused + feature
        fused = self.out_norm(fused)

        if return_confidence:
            return fused, confidence
        return fused
