"""Phase 19-21: Gemini copilot endpoints. Every prompt here is built from verified,
already-stored structured results (never re-invented), per
backend/services/gemini_service.py's LIMITATION_CONTEXT."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.schemas.copilot import (
    CopilotChatRequest, ExplainAnalysisRequest, SummarizeBatchRequest, CompareRequest, CopilotResponse,
)
from backend.services.history_service import get_analysis, get_batch
from backend.services.gemini_service import generate_text

router = APIRouter(prefix="/api/copilot", tags=["copilot"])


def _respond(text: str | None, reason: str | None) -> CopilotResponse:
    return CopilotResponse(text=text, ai_available=text is not None, fallback_reason=reason)


@router.post("/chat", response_model=CopilotResponse)
async def chat(req: CopilotChatRequest):
    analysis = get_analysis(req.analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    prompt = (
        f"Here is a verified maize seed analysis result (JSON):\n{json.dumps(analysis, indent=2)}\n\n"
        f"User question: {req.question}\n"
        f"Answer using ONLY the data above. If the question asks about something this "
        f"system does not compute (e.g. real disease, exact defect area, purity), say so plainly."
    )
    text, reason = generate_text(prompt)
    return _respond(text, reason)


@router.post("/explain-analysis", response_model=CopilotResponse)
async def explain_analysis(req: ExplainAnalysisRequest):
    analysis = get_analysis(req.analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    prompt = (
        f"Explain this verified maize seed analysis result to a student in clear, simple "
        f"language. Cover: how many seeds were detected, what varieties/synthetic-defect "
        f"classes were predicted and with what confidence, and what limitations apply. "
        f"Data (JSON):\n{json.dumps(analysis, indent=2)}"
    )
    text, reason = generate_text(prompt)
    return _respond(text, reason)


@router.post("/summarize-batch", response_model=CopilotResponse)
async def summarize_batch(req: SummarizeBatchRequest):
    batch = get_batch(req.batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    prompt = (
        f"Summarize this batch analysis in natural language, using only the real "
        f"statistics given — do not invent trends beyond what the numbers show. "
        f"Data (JSON):\n{json.dumps(batch, indent=2)}"
    )
    text, reason = generate_text(prompt)
    return _respond(text, reason)


@router.post("/compare", response_model=CopilotResponse)
async def compare(req: CompareRequest):
    if len(req.analysis_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 analysis_ids to compare")

    analyses = []
    for aid in req.analysis_ids:
        a = get_analysis(aid)
        if a is None:
            raise HTTPException(status_code=404, detail=f"Analysis not found: {aid}")
        analyses.append(a)

    prompt = (
        f"Compare these {len(analyses)} verified maize seed analyses: seed counts, "
        f"predicted varieties, synthetic-defect predictions, and confidence levels. "
        f"Highlight real differences only. Data (JSON):\n{json.dumps(analyses, indent=2)}"
    )
    text, reason = generate_text(prompt)
    return _respond(text, reason)
