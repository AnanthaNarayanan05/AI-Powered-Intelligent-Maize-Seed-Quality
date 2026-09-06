"""Phase 6 response contract.

The envelope is typed because it is a promise: every orchestrated answer says
which question it understood, which stages it planned, which of those actually
ran, and what it could not do. `answer` and `seeds` stay free-form dicts on
purpose -- their shape is the intent's, and forcing seven intents through one
flattened model would either drop fields or invent optional ones that are
meaningless for six of them.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class IntentResolution(BaseModel):
    """Which question was understood, and how that was decided.

    `resolved_by` is "explicit" (the caller named an intent), "pattern" (a phrase
    matched), "default" (nothing matched and the general analysis was run) or
    "no_question". The last two are said out loud rather than passed off as
    comprehension.
    """

    requested: Optional[str] = None
    resolved: str
    canonical: str
    question: Optional[str] = None
    resolved_by: str
    matched: Optional[str] = None


class PlannedStage(BaseModel):
    stage: str
    tasks: list[str]
    expensive: bool
    describe: str


class StageReport(BaseModel):
    """What happened to one stage.

    `status` is "ran", "reused" (this question needed it, the cache had it),
    "cached" (an earlier question about this image ran it; this one did not need
    it), "skipped" (a prerequisite produced nothing), "unavailable" (an expected,
    explainable refusal -- see `reason`) or "failed".
    """

    stage: str
    status: str
    model: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    duration_ms: Optional[int] = None
    describe: str


class UnavailableItem(BaseModel):
    """Something this question asked for and did not get, with the reason.

    `reason` is "no_served_model" (the registry allows nothing to answer this --
    a better photograph will not help), "insufficient_resolution" (a model could,
    but not about this image -- a better photograph will help), "not_implemented"
    (the capability has not been built; the answer names the phase that owns it)
    or "stage_error".
    """

    capability: Optional[str] = None
    stage: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None


class OrchestratedResponse(BaseModel):
    orchestrator_version: int
    analysis_id: str
    image_digest: str
    intent: IntentResolution
    plan: list[PlannedStage]
    stages: list[StageReport]
    executed: list[str]
    reused: list[str]
    seed_count: int
    seeds: list[dict[str, Any]] = []
    segmentation: Optional[dict[str, Any]] = None
    answer: dict[str, Any]
    unavailable: list[UnavailableItem] = []
    warnings: list[str] = []


class CapabilityReport(BaseModel):
    """Derived from the live registry on every call, never written down.

    A capability list kept as a constant is a claim that goes stale the moment a
    model is declared or retired.
    """

    orchestrator_version: int
    tasks: dict[str, Any]
    intents: list[dict[str, Any]]
    note: str
