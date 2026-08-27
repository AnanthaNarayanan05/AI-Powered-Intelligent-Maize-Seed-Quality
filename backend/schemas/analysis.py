"""Pydantic request/response schemas. Field names deliberately keep "synthetic_"
prefixes and never introduce fields implying real-world defect certification,
segmentation, purity, or foreign-object detection — see
docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md."""
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
    path: str
    label: str
    distance: float
    similarity_score: float


class SeedAnalysisResult(BaseModel):
    seed_index: int
    bbox: list[float]
    detection_confidence: float
    variety_prediction: Optional[VarietyPrediction] = None
    synthetic_defect_prediction: Optional[SyntheticDefectPrediction] = None
    similarity_results: list[SimilarityMatch] = []


class ImageAnalysisResponse(BaseModel):
    analysis_id: str
    seed_count: int
    seeds: list[SeedAnalysisResult]
    warnings: list[str] = []


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
