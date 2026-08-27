"""Cognitive attention module: channel attention (SE) followed by spatial attention
(CBAM-style), applied to a backbone's intermediate feature maps. This is the
"cognitive attention" block referenced throughout docs/05_ARCHITECTURE.md — it is a
combination of two well-established, literature-precedented mechanisms
(docs/02_LITERATURE_SURVEY_ANALYSIS.md rows 43, 46), not a new theoretical
contribution; the project's novelty is in *combining* this with contrastive
pretraining (see src/contrastive/simclr.py), evaluated empirically via the Phase 7
ablation experiments in src/training/train_variety.py.
"""
import torch.nn as nn

from src.attention.channel_attention import SqueezeExcitation
from src.attention.spatial_attention import SpatialAttention


class CognitiveAttention(nn.Module):
    def __init__(self, channels: int, use_channel: bool = True, use_spatial: bool = True, reduction: int = 16):
        super().__init__()
        self.channel_attn = SqueezeExcitation(channels, reduction) if use_channel else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_spatial else nn.Identity()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MultiScaleFusion(nn.Module):
    """Fuses feature maps from multiple backbone stages (different spatial
    resolutions) into one representation by global-pooling each stage and
    concatenating — used only when configs/config.yaml -> attention.use_multiscale
    is true, per docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md (item 14, CONDITIONALLY
    SUPPORTED)."""

    def __init__(self, in_channels_list: list[int], out_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        total = sum(in_channels_list)
        self.project = nn.Sequential(
            nn.Linear(total, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, feature_maps: list):
        pooled = [self.pool(f).flatten(1) for f in feature_maps]
        import torch

        cat = torch.cat(pooled, dim=1)
        return self.project(cat)
