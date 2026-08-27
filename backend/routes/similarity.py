from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.routes.analyze import _save_upload
from backend.services.pipeline_service import get_pipeline, ModelNotAvailableError, InvalidImageError
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/api", tags=["similarity"])
logger = get_logger("route_similarity")


@router.post("/similarity")
async def similarity(file: UploadFile = File(...), variety_dataset: str = Form("a"), top_k: int = Form(5)):
    if variety_dataset not in ("a", "b"):
        raise HTTPException(status_code=400, detail="variety_dataset must be 'a' or 'b'")

    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        image = pipeline.load_and_validate_image(path)
        results = pipeline.embed_and_search(image, variety_dataset, top_k)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected similarity search error")
        raise HTTPException(status_code=500, detail="Internal error during similarity search.")

    return {"query_note": "VISUAL/FEATURE SIMILARITY only — not a variety certification.", "results": results}
