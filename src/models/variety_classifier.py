"""The proposed model: backbone -> cognitive attention -> GAP -> classification
head, with an optional projection head for contrastive pretraining (removed for
downstream fine-tuning, per configs/config.yaml and docs/05_ARCHITECTURE.md).

Used for BOTH Dataset A and Dataset B variety classifiers (two independently trained
instances — see docs/03_DATASET_AUDIT.md cross-dataset compatibility analysis for why
they are not merged), and for the synthetic defect classifier
(docs/06_SYNTHETIC_DEFECT_POLICY.md) via the same architecture with a different head.
"""
import torch
import torch.nn as nn

from src.models.backbone import build_backbone
from src.attention.cognitive_attention import CognitiveAttention
from src.contrastive.simclr import ProjectionHead


class CognitiveAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        use_attention: bool = True,
        use_channel: bool = True,
        use_spatial: bool = True,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        self.use_attention = use_attention
        self.attention = (
            CognitiveAttention(feature_dim, use_channel, use_spatial) if use_attention else nn.Identity()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
        )
        self.projection_head = ProjectionHead(feature_dim, out_dim=projection_dim)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.attention(feats)
        pooled = self.pool(feats).flatten(1)
        return pooled

    def forward(self, x: torch.Tensor, mode: str = "classify"):
        embedding = self.extract_embedding(x)
        if mode == "embedding":
            return embedding
        if mode == "contrastive":
            return self.projection_head(embedding)
        logits = self.classifier(embedding)
        return logits

    def forward_with_features(self, x: torch.Tensor):
        """Used by Grad-CAM: returns logits AND the last conv feature map (with
        gradient hooks left to the caller in src/explainability/gradcam.py)."""
        feats = self.backbone(x)
        feats = self.attention(feats)
        pooled = self.pool(feats).flatten(1)
        logits = self.classifier(pooled)
        return logits, feats
