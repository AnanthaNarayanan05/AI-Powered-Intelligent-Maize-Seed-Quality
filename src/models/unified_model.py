"""The unified seed model: one shared trunk, two task heads.

Why two heads rather than one merged softmax
--------------------------------------------
The request was a single efficient model holding all the trained data. The naive
reading -- concatenate the label sets into one 8-class softmax over
{3 Dataset A varieties, 3 Dataset B varieties, Good, Bad} -- is wrong, because
those labels are not mutually exclusive. A kernel is a variety *and* a quality
grade; a softmax would force the model to choose between "Rugosa" and "Good".

Worse, it would require inventing labels. Datasets A and B carry no quality
annotation and the Mendeley quality set carries no variety annotation. Merging
them into one label space would mean asserting a quality grade for 18,759 images
nobody ever graded, which is exactly the fabrication the project rules forbid.

So: one EfficientNet-B0 trunk with cognitive attention, shared by everything, and
two linear heads on top of the pooled embedding. Every image trains the trunk.
Each head is trained only by the images that actually carry its label, enforced by
CrossEntropyLoss(ignore_index=-1) in the training loop. The result is a single
checkpoint and a single forward pass at serve time that returns both predictions,
with the shared representation learned from all 23,605 images.

Appending new data later means adding rows to the unified manifest and, if the new
data introduces classes, widening the relevant head -- the trunk and the other head
are untouched.

The third head (Phase 5)
------------------------
GrainSpace M600 supplies 1,260 kernel crops carrying expert visible-condition
labels, which is a third label space that is again not mutually exclusive with
the other two -- a kernel is a variety, a quality grade *and* a visible
condition. So it becomes a third head on the same trunk, by exactly the argument
above, rather than being folded into either existing softmax.

``symptom_classes`` is optional and defaults to None. A model built without it is
byte-identical to the two-head model that shipped, holds no extra parameters, and
loads every existing checkpoint unchanged; ``forward()`` keeps returning the same
two-tuple it always has, because the serving pipeline and the test suite are
written against that contract. Callers that want all available heads ask for them
explicitly via ``forward_heads()``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbone import build_backbone
from src.attention.cognitive_attention import CognitiveAttention
from src.contrastive.simclr import ProjectionHead


class UnifiedSeedModel(nn.Module):
    def __init__(
        self,
        variety_classes: list[str],
        quality_classes: list[str],
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        use_attention: bool = True,
        use_channel: bool = True,
        use_spatial: bool = True,
        projection_dim: int = 128,
        symptom_classes: list[str] | None = None,
    ):
        super().__init__()
        self.variety_classes = list(variety_classes)
        self.quality_classes = list(quality_classes)
        self.symptom_classes = list(symptom_classes) if symptom_classes else []

        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        self.use_attention = use_attention
        self.attention = (
            CognitiveAttention(feature_dim, use_channel, use_spatial)
            if use_attention else nn.Identity()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = feature_dim

        self.variety_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_dim, len(self.variety_classes))
        )
        self.quality_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_dim, len(self.quality_classes))
        )
        # Only constructed when symptom classes were asked for, so a two-head
        # model has no unused parameters and its state_dict keys are unchanged.
        self.symptom_head = (
            nn.Sequential(nn.Dropout(0.3), nn.Linear(feature_dim, len(self.symptom_classes)))
            if self.symptom_classes else None
        )
        # Retained so the unified trunk can be contrastively pretrained with the
        # same SimCLR code path as the single-task models.
        self.projection_head = ProjectionHead(feature_dim, out_dim=projection_dim)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.attention(feats)
        return self.pool(feats).flatten(1)

    def forward(self, x: torch.Tensor, mode: str = "classify"):
        embedding = self.extract_embedding(x)
        if mode == "embedding":
            return embedding
        if mode == "contrastive":
            return self.projection_head(embedding)
        return self.variety_head(embedding), self.quality_head(embedding)

    def forward_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every head this model actually has, from one shared forward pass.

        The symptom key is present only when the model was built with symptom
        classes -- absent, not zero-filled, because a model with no symptom head
        has no opinion about symptoms and pretending otherwise is the failure
        mode this whole codebase is written against.
        """
        embedding = self.extract_embedding(x)
        out = {
            "variety": self.variety_head(embedding),
            "quality": self.quality_head(embedding),
        }
        if self.symptom_head is not None:
            out["symptom"] = self.symptom_head(embedding)
        return out

    def head_module(self, head: str) -> nn.Module:
        """Resolve a head name to its module, raising rather than falling back.

        Silently substituting the variety head for an absent symptom head would
        label one model's decision with another model's name.
        """
        if head == "variety":
            return self.variety_head
        if head == "quality":
            return self.quality_head
        if head == "symptom":
            if self.symptom_head is None:
                raise ValueError(
                    "this checkpoint has no symptom head; it was trained before the "
                    "visible-symptom head existed or was built without symptom classes"
                )
            return self.symptom_head
        raise ValueError(f"unknown head {head!r}")

    def classes_for(self, head: str) -> list[str]:
        return {
            "variety": self.variety_classes,
            "quality": self.quality_classes,
            "symptom": self.symptom_classes,
        }[head]

    def attach_symptom_head(self, symptom_classes: list[str]) -> None:
        """Add a symptom head to an already-loaded two-head model.

        Used by the stage-1 trainer: the shipped unified checkpoint is loaded
        exactly as it is (so no key is missing and none is ignored), and only
        then is the new head grown on top. Doing it this way rather than with
        ``load_state_dict(..., strict=False)`` means a genuinely corrupt
        checkpoint still raises instead of being quietly tolerated.
        """
        self.symptom_classes = list(symptom_classes)
        self.symptom_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.feature_dim, len(self.symptom_classes))
        ).to(next(self.parameters()).device)

    def forward_with_features(self, x: torch.Tensor, head: str = "variety"):
        """Grad-CAM entry point. Returns (logits, feature_map) for one named head,
        since a CAM is only meaningful with respect to a single task's logit."""
        feats = self.backbone(x)
        feats = self.attention(feats)
        pooled = self.pool(feats).flatten(1)
        logits = self.head_module(head)(pooled)
        return logits, feats

    def load_pretrained_trunk(self, path: str, map_location=None) -> None:
        """Load a contrastively pretrained backbone, leaving both heads random."""
        state = torch.load(path, map_location=map_location)
        self.backbone.load_state_dict(state["backbone_state_dict"])
