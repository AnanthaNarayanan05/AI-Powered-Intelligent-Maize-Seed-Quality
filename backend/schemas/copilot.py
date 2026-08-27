from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CopilotChatRequest(BaseModel):
    analysis_id: str
    question: str


class CopilotChatResponse(BaseModel):
    answer: str
    ai_available: bool
    label: str = "AI-generated explanation based on model analysis"


class ExplainAnalysisRequest(BaseModel):
    analysis_id: str


class SummarizeBatchRequest(BaseModel):
    batch_id: str


class CompareRequest(BaseModel):
    analysis_ids: list[str]


class CopilotResponse(BaseModel):
    text: Optional[str] = None
    ai_available: bool
    label: str = "AI-generated explanation based on model analysis"
    fallback_reason: Optional[str] = None
