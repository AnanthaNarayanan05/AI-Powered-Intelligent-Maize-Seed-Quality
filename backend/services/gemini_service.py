"""Gemini intelligence layer (Phase 19-22). Gemini only ever receives verified,
already-computed structured ML results plus the user's question — it never invents
predictions, and every prompt explicitly states which capabilities are NOT
supported (per docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md) so it cannot claim them.

Failure handling: if GEMINI_API_KEY is missing, the API times out, rate-limits, or
returns an invalid response, this module NEVER raises up to crash the request — it
returns (None, reason) and the caller falls back to showing ML results with an
"AI explanation temporarily unavailable" notice. No secret is ever logged.
"""
from __future__ import annotations

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, gemini_is_configured
from src.utils.logging_utils import get_logger

logger = get_logger("gemini_service")

LIMITATION_CONTEXT = """
You are explaining outputs of a maize seed computer-vision system to a student/user.
Ground rules you MUST follow:
- Only describe what is in the verified structured JSON results given to you.
- Never claim exact defect area, segmentation, real fungal/insect/disease diagnosis,
  foreign-object detection, or purity/seed-lot certification — this system does not
  support any of those (no real dataset exists for them).
- Any "synthetic_defect_prediction" field comes from a model trained on procedurally
  generated synthetic damage images, NOT verified real-world plant pathology data.
  Always describe it as a simulated/demonstration prediction, never as certain or as
  a real diagnosis.
- Grad-CAM/attention visualizations show which pixels influenced the model, not a
  precise defect boundary or mask.
- Similarity search results are visual/feature similarity only, not a certification
  of variety purity.
- Be honest about uncertainty: low confidence scores should be described as such,
  not smoothed over.
""".strip()


def _get_client():
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL, system_instruction=LIMITATION_CONTEXT)


def generate_text(prompt: str, timeout_seconds: int = 20) -> tuple[str | None, str | None]:
    """Returns (text, fallback_reason). Exactly one of these is non-None."""
    if not gemini_is_configured():
        return None, "GEMINI_API_KEY is not configured on the server."

    try:
        model = _get_client()
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout_seconds},
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text, None
    except Exception as e:  # noqa: BLE001 — Gemini failures must never crash the app
        # Log only the exception type/message, never the API key or full prompt contents
        # that might include sensitive request context.
        logger.warning(f"Gemini request failed: {type(e).__name__}")
        return None, "The AI explanation service is temporarily unavailable. Please try again."
