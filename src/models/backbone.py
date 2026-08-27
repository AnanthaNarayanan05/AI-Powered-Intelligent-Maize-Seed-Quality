"""Backbone factory. EfficientNet-B0 (default) or ResNet50, both ImageNet-pretrained
via torchvision, as specified in configs/config.yaml -> backbone and justified by
docs/02_LITERATURE_SURVEY_ANALYSIS.md (EfficientNet/ResNet are the most consistently
strong backbones across the surveyed maize seed papers, rows 4, 18)."""
import torch.nn as nn
from torchvision import models


def build_backbone(name: str = "efficientnet_b0", pretrained: bool = True):
    """Returns (feature_extractor, feature_dim). The classifier head is stripped so
    callers can attach their own attention + classification head."""
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[1].in_features
        features = net.features  # conv stages, output (B, feature_dim, H', W')
        return features, feature_dim
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        feature_dim = net.fc.in_features
        features = nn.Sequential(*list(net.children())[:-2])  # drop avgpool+fc
        return features, feature_dim
    else:
        raise ValueError(f"Unsupported backbone: {name}")
