"""Pixel-level defect segmentation network.

Architecture: the project's own EfficientNet-B0 trunk (src/models/backbone.py) used
as a U-Net encoder, Cognitive Attention on the bottleneck, and a U-Net decoder that
restores full input resolution. Nothing here is a new architectural claim — U-Net
(Ronneberger 2015) with a pretrained classification encoder is the standard recipe;
the point is that it is the SAME trunk the variety and quality heads already use, so
the contrastive encoder this project trained (src/contrastive/simclr.py) can be
loaded straight into it and the segmentation head becomes one more task on the shared
representation rather than a bolted-on second model.

Output channels are independent sigmoids, not a softmax. Two reasons:
  * a kernel can be cracked AND mouldy in the same place, so the classes are not
    mutually exclusive;
  * the seed-body channel deliberately OVERLAPS every defect channel, because a
    defect is by definition part of the seed. Defect coverage % is measured against
    that body (docs: VISIBLE DEFECT AREA %), so the denominator has to be predicted,
    not assumed.

Channel semantics are carried in the checkpoint, never hard-coded downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.cognitive_attention import CognitiveAttention

# EfficientNet-B0 `features` stage boundaries. At 224x224 input the spatial sizes are
# 112, 56, 28, 14, 7; we tap the last stage at each resolution and use the 1280-channel
# head as the bottleneck. Indices are into torchvision's efficientnet_b0().features.
_EFFNET_B0_TAPS = [1, 2, 3, 5, 8]
_EFFNET_B0_TAP_CHANNELS = [16, 24, 40, 112, 1280]


class _DecoderBlock(nn.Module):
    """Upsample, concatenate the encoder skip, then two 3x3 convs.

    Bilinear upsampling rather than a transposed conv: transposed convs put a regular
    checkerboard into the logits, and a checkerboard in a mask is a checkerboard in
    the defect-area number that gets read off it.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        target = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DefectSegmenter(nn.Module):
    def __init__(
        self,
        channel_names: list[str],
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        use_attention: bool = True,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        if backbone_name != "efficientnet_b0":
            raise ValueError(
                f"{backbone_name!r} has no skip-tap map here yet; only efficientnet_b0 "
                "is wired for segmentation (add its stage indices to _EFFNET_B0_TAPS)."
            )
        from src.models.backbone import build_backbone

        self.backbone, bottleneck_dim = build_backbone(backbone_name, pretrained)
        self.channel_names = list(channel_names)
        self.taps = _EFFNET_B0_TAPS
        skips = _EFFNET_B0_TAP_CHANNELS[:-1][::-1] + [0]   # 112, 40, 24, 16, then none

        self.use_attention = use_attention
        self.attention = (
            CognitiveAttention(bottleneck_dim) if use_attention else nn.Identity()
        )

        blocks = []
        in_ch = bottleneck_dim
        for out_ch, skip_ch in zip(decoder_channels, skips):
            blocks.append(_DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, len(self.channel_names), kernel_size=1)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        out = x
        for i, stage in enumerate(self.backbone):
            out = stage(out)
            if i in self.taps:
                feats.append(out)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, (B, C, H, W), at the input resolution.

        Logits, not probabilities: the loss needs them numerically stable and every
        caller that wants a mask has to make its own thresholding decision explicit.
        """
        in_hw = x.shape[-2:]
        feats = self.encode(x)
        skips = feats[:-1][::-1] + [None]
        out = self.attention(feats[-1])
        for block, skip in zip(self.decoder, skips):
            out = block(out, skip)
        logits = self.head(out)
        if logits.shape[-2:] != in_hw:
            logits = F.interpolate(logits, size=in_hw, mode="bilinear", align_corners=False)
        return logits

    def load_contrastive_encoder(self, state_dict: dict) -> tuple[int, int]:
        """Warm-start the encoder from a SimCLR checkpoint trained by this project.

        Returns (matched, total) so the caller can log how much actually transferred
        instead of assuming it did -- a silently-empty load looks exactly like a
        successful one until the metrics come in low.
        """
        own = self.backbone.state_dict()
        src = {}
        for k, v in state_dict.items():
            key = k[len("backbone."):] if k.startswith("backbone.") else k
            if key in own and own[key].shape == v.shape:
                src[key] = v
        self.backbone.load_state_dict(src, strict=False)
        return len(src), len(own)
