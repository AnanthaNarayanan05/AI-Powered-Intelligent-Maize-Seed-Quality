"""One orchestrator per process, sharing the one pipeline.

The run cache only means anything if it survives between requests, and the model
weights only load once if the orchestrator borrows the pipeline the rest of the
API already built. Constructing an Orchestrator per request would do neither.
"""
from __future__ import annotations

from backend.services.pipeline_service import get_pipeline
from src.pipeline.orchestrator import Orchestrator, OrchestratorError  # noqa: F401

_orchestrator_instance: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator(pipeline=get_pipeline())
    return _orchestrator_instance
