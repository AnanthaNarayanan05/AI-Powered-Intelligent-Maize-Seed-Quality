"""Pydantic request/response schemas. Field names deliberately keep "synthetic_"
prefixes and never introduce fields implying real-world defect certification,
purity, or foreign-object detection — see
docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md.

The one segmentation-shaped field here, `SegmentationStatus`, carries no mask and
no measurement: it reports whether pixel-level analysis was possible and, when it
was not, which of the two independent reasons applied. Saying "unavailable" out
loud is the opposite of implying the capability."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] in pixel coordinates")
    confidence: float


class VarietyPrediction(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    model: str


class SyntheticDefectPrediction(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    model: str
    is_synthetic_model: bool = True
    disclaimer: str


class SimilarityMatch(BaseModel):
    """One gallery neighbour. VISUAL SIMILARITY only.

    The labels describe the neighbour image, not the query, and they are null when
    the neighbour's source dataset never carried that label — the quality-only
    subset of the unified gallery has no variety, and none is inferred for it.
    """

    path: str
    variety_label: Optional[str] = None
    quality_label: Optional[str] = None
    source: Optional[str] = None
    label: Optional[str] = None  # legacy alias for variety_label
    distance: float  # L2 in the encoder's feature space; lower = nearer
    similarity_score: float  # 1/(1+distance): monotone display rescaling, not a probability


class ResolutionSummary(BaseModel):
    """How big the kernels in this image actually are, against the measured floor.

    Every number is a measurement of the submitted image or a threshold read from
    the file that established it (`source`), so it stays meaningful even when no
    pixel-level model ran — which is exactly when a reader most needs it.
    """

    min_kernel_px: Optional[int] = None
    source: Optional[str] = None
    seeds_measured: int = 0
    seeds_below_floor: int = 0
    kernel_px_median: Optional[int] = None
    kernel_px_min: Optional[int] = None
    kernel_px_max: Optional[int] = None


class SegmentationStatus(BaseModel):
    """Whether defect segmentation could run, and if not, why not.

    `reason` separates two failures that must never be shown as one:
    "no_served_model" (the registry allows no model to answer this task at all)
    and "insufficient_resolution" (a model could, but these kernels are below the
    measured floor). Only the second is fixable by re-photographing the sample.
    `status` is "available", "partial" (some seeds clear the floor, some do not)
    or "unavailable".
    """

    status: str
    reason: Optional[str] = None
    message: Optional[str] = None
    model: Optional[str] = None
    resolution: ResolutionSummary


class SeedAnalysisResult(BaseModel):
    seed_index: int
    bbox: list[float]
    detection_confidence: float
    # short side of the detection box: the same quantity the resolution floor was
    # measured in, so the two can be compared without a conversion
    kernel_px: Optional[int] = None
    variety_prediction: Optional[VarietyPrediction] = None
    synthetic_defect_prediction: Optional[SyntheticDefectPrediction] = None
    similarity_results: list[SimilarityMatch] = []


class ImageAnalysisResponse(BaseModel):
    analysis_id: str
    seed_count: int
    seeds: list[SeedAnalysisResult]
    warnings: list[str] = []
    segmentation: Optional[SegmentationStatus] = None


class BatchAnalysisResponse(BaseModel):
    batch_id: str
    total_images: int
    successful_images: int
    failed_images: int
    aggregate_stats: dict
    analysis_ids: list[str]


class DetectionResponse(BaseModel):
    seed_count: int
    detections: list[BoundingBox]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
