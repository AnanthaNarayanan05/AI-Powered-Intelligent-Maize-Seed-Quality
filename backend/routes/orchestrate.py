"""Phase 6: ask a question, get an answer and an account of what produced it.

There is deliberately no model parameter on these endpoints. The caller names a
question; the orchestrator names the stages; the registry names the models. That
is the whole point of the phase -- an operator should not have to know which
checkpoint answers "how severe is this", and letting them pick one would make
every honest refusal optional.
"""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.routes.analyze import _save_upload
from backend.schemas.orchestrator import CapabilityReport, OrchestratedResponse
from backend.services.history_service import save_analysis_result
from backend.services.orchestrator_service import get_orchestrator, OrchestratorError
from backend.services.pipeline_service import InvalidImageError
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/api/analyze", tags=["orchestrator"])
logger = get_logger("route_orchestrate")


@router.get("/capabilities", response_model=CapabilityReport)
async def capabilities():
    """Which questions can be answered right now, and what blocks the rest."""
    return get_orchestrator().capabilities()


@router.post("/ask", response_model=OrchestratedResponse)
async def ask(
    file: UploadFile | None = File(None),
    question: str | None = Form(None),
    intent: str | None = Form(None),
    analysis_id: str | None = Form(None),
    top_k: int = Form(5),
):
    """Answer one question about one image.

    Send a file, or an `analysis_id` from an earlier answer to ask a follow-up
    with no inference at all. `intent` overrides phrase matching for callers that
    already know which question they are asking; it selects a QUESTION, never a
    model.
    """
    if file is None and not analysis_id:
        raise HTTPException(
            status_code=400,
            detail="Send an image file, or the analysis_id of an earlier orchestrated run.",
        )

    path = _save_upload(file) if file is not None else None
    orchestrator = get_orchestrator()
    try:
        result = orchestrator.run(
            image_path=path, question=question, intent=intent,
            analysis_id=analysis_id, top_k=top_k,
        )
    except OrchestratorError as e:
        # An unknown intent or an expired analysis_id is the caller's to fix, and
        # the message says which.
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during orchestrated analysis")
        raise HTTPException(status_code=500, detail="Internal error during analysis. Please try again.")

    # Persisted once per image, on the run that actually detected it. Later
    # questions reuse that same analysis_id, so writing again would be a duplicate
    # row rather than a second analysis.
    if "detect" in result["executed"]:
        try:
            save_analysis_result(
                result, "unified",
                image_filename=file.filename if file is not None else None,
                image_path=result.get("image_path"),
            )
        except Exception:  # noqa: BLE001
            # History is a convenience; losing it must not lose the analysis.
            logger.exception("Could not persist orchestrated analysis %s", result["analysis_id"])
            result["warnings"].append("This analysis could not be saved to history.")

    return result
