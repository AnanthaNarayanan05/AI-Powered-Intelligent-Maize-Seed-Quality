from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from backend.routes.analyze import _save_upload
from backend.services.pipeline_service import get_pipeline, ModelNotAvailableError, InvalidImageError
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/api", tags=["classify"])
logger = get_logger("route_classify")


@router.post("/classify/variety")
async def classify_variety(file: UploadFile = File(...), variety_dataset: str = Form("a")):
    if variety_dataset not in ("a", "b"):
        raise HTTPException(status_code=400, detail="variety_dataset must be 'a' or 'b'")

    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        image = pipeline.load_and_validate_image(path)
        prediction = pipeline.classify_variety(image, variety_dataset)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected variety classification error")
        raise HTTPException(status_code=500, detail="Internal error during classification.")

    return prediction


@router.post("/classify/synthetic-defect")
async def classify_synthetic_defect(file: UploadFile = File(...)):
    """NOTE: 'quality' in the original master prompt is renamed 'synthetic-defect'
    here since no real quality/defect dataset exists — see
    docs/06_SYNTHETIC_DEFECT_POLICY.md. The response is always clearly labeled
    synthetic."""
    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        image = pipeline.load_and_validate_image(path)
        prediction = pipeline.classify_synthetic_defect(image)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected synthetic defect classification error")
        raise HTTPException(status_code=500, detail="Internal error during classification.")

    return prediction


@router.post("/explain/gradcam")
async def explain_gradcam(file: UploadFile = File(...), variety_dataset: str = Form("a")):
    """Grad-CAM overlay for the variety prediction, returned as a PNG.

    This is an EXPLAINABILITY VISUALIZATION, not a segmentation mask and not a defect
    area — see docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md items 18/28/30. The response
    header `X-Explainability-Note` restates that for any client that renders the image
    without the surrounding UI copy.
    """
    if variety_dataset not in ("a", "b"):
        raise HTTPException(status_code=400, detail="variety_dataset must be 'a' or 'b'")

    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        from PIL import Image

        from src.explainability.gradcam import overlay_heatmap

        image = pipeline.load_and_validate_image(path)
        cam, _class_idx = pipeline.generate_gradcam(image, variety_dataset)
        resized = image.resize((cam.shape[1], cam.shape[0]))
        overlaid = overlay_heatmap(np.array(resized), cam)

        buf = io.BytesIO()
        Image.fromarray(overlaid).save(buf, format="PNG")
        buf.seek(0)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected Grad-CAM error")
        raise HTTPException(status_code=500, detail="Internal error generating Grad-CAM.")

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Explainability-Note": (
                "Grad-CAM attention heatmap - shows which pixels influenced the variety "
                "prediction. NOT a segmentation mask and NOT a defect area measurement."
            )
        },
    )
