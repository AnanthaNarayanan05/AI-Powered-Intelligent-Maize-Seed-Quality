from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.routes.analyze import _save_upload
from backend.services.pipeline_service import get_pipeline, ModelNotAvailableError, InvalidImageError
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/api", tags=["detect"])
logger = get_logger("route_detect")


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        pipeline.load_and_validate_image(path)
        detections = pipeline.detect_seeds(path)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected detection error")
        raise HTTPException(status_code=500, detail="Internal error during detection.")

    return {"seed_count": len(detections), "detections": detections}
