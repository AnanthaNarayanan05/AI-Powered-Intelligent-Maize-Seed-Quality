"""Gemini intelligence layer (Phase 19-22). Gemini only ever receives verified,
already-computed structured ML results plus the user's question — it never invents
predictions, and every prompt explicitly states which capabilities are NOT
supported (per docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md) so it cannot claim them.

Failure handling: if GEMINI_API_KEY is missing, the API times out, rate-limits, or
returns an invalid response, this module NEVER raises up to crash the request — it
returns (None, reason) and the caller falls back to showing ML results with an
"AI explanation temporarily unavailable" notice. No secret is ever logged.

SDK: google-genai (`from google import genai`). It replaces google-generativeai,
which is retired. Three differences the migration had to get right rather than
translate mechanically:

  * Timeouts are milliseconds here and were seconds before. Copying the number
    across would have turned a 20-second budget into 20 milliseconds, and every
    call would have failed as a timeout that looked like an outage.
  * The system instruction belongs to the request config, not to a model object,
    so LIMITATION_CONTEXT is attached per call. There is no longer any way for a
    reused model handle to carry a stale set of ground rules.
  * A client owns a connection pool, so one is built once and reused. The old
    module reconfigured a global and constructed a model on every request.
  * generate_content now runs an automatic function-calling loop by default.
    This module passes no tools, so that loop is switched off explicitly.
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

# Built on first use and reused. Module-level rather than lru_cache so that the
# reload the tests perform to swap the key also discards the client holding the
# old one -- a cached client keyed on nothing would outlive the configuration
# that created it.
_client = None


def _get_client():
    """The shared genai client. Imported lazily so that an environment without the
    SDK installed degrades to the same clean fallback as a missing key, instead of
    breaking every import of the backend."""
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _refusal_reason(response) -> str:
    """Why a response carried no text.

    Worth distinguishing: a prompt the safety filters blocked is a different
    problem from a model that ran out of output budget, and reporting both as
    "empty response" hides which one happened.
    """
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return "Gemini declined to answer this prompt."
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    if finish is not None and "MAX_TOKENS" in str(finish):
        return "Gemini's response was cut off before it produced any text."
    return "Gemini returned an empty response."


def generate_text(prompt: str, timeout_seconds: int = 20) -> tuple[str | None, str | None]:
    """Returns (text, fallback_reason). Exactly one of these is non-None."""
    if not gemini_is_configured():
        return None, "GEMINI_API_KEY is not configured on the server."

    try:
        from google.genai import types

        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=LIMITATION_CONTEXT,
                # Milliseconds. See the module docstring.
                http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
                # This module passes no tools, so the SDK's automatic
                # function-calling loop has nothing to call. Left on, it wraps
                # every request in a retry loop and logs a warning advising a
                # chat session we do not want.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            return None, _refusal_reason(response)
        return text, None
    except Exception as e:  # noqa: BLE001 — Gemini failures must never crash the app
        # Log only the exception type, never the API key, the response body, or the
        # prompt, which carries the user's question and their analysis results.
        logger.warning(f"Gemini request failed: {type(e).__name__}")
        return None, "The AI explanation service is temporarily unavailable. Please try again."
