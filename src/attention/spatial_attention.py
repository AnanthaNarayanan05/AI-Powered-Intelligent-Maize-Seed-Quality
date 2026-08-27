"""CBAM-style spatial attention.
Reference: Woo et al., "CBAM: Convolutional Block Attention Module" — used together
with SqueezeExcitation (channel_attention.py) to form the project's cognitive
attention module."""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(pooled))
        return x * attn
