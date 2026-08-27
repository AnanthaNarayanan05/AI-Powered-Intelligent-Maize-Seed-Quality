"""Wraps src/pipeline/unified_pipeline.py as a singleton so model weights load once
per process, and translates pipeline exceptions into clear, safe API-facing errors
(never leaking stack traces, per Phase 18)."""
from __future__ import annotations

from src.pipeline.unified_pipeline import AnalysisPipeline, ModelNotAvailableError, InvalidImageError  # noqa: F401
from src.utils.logging_utils import get_logger

logger = get_logger("pipeline_service")

_pipeline_instance: AnalysisPipeline | None = None


def get_pipeline() -> AnalysisPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AnalysisPipeline()
    return _pipeline_instance
