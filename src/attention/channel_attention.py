"""Squeeze-and-Excitation channel attention.
Reference: Hu et al., "Squeeze-and-Excitation Networks" — standard channel-attention
building block, used here as one half of the project's cognitive attention module
(see docs/02_LITERATURE_SURVEY_ANALYSIS.md, CBAM precedent rows 43/46)."""
import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
