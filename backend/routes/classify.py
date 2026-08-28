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


GRADCAM_DATASETS = ("unified", "a", "b")


def _parse_bbox(raw: str, size: tuple[int, int]) -> list[float]:
    """Parse "x1,y1,x2,y2" in ORIGINAL image pixels, as returned by /api/analyze."""
    parts = [p for p in raw.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bbox must be 'x1,y1,x2,y2'")
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox values must be numbers")
    w, h = size
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="bbox must have x2>x1 and y2>y1")
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise HTTPException(
            status_code=400,
            detail=f"bbox {parts} falls outside the {w}x{h} image it was sent with",
        )
    return [x1, y1, x2, y2]


@router.post("/explain/gradcam")
async def explain_gradcam(
    file: UploadFile = File(...),
    variety_dataset: str = Form("unified"),
    head: str = Form("variety"),
    bbox: str = Form(""),
):
    """Grad-CAM overlay for a served prediction, returned as a PNG.

    This is an EXPLAINABILITY VISUALIZATION, not a segmentation mask and not a defect
    area — see docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md items 18/28/30. The response
    header `X-Explainability-Note` restates that for any client that renders the image
    without the surrounding UI copy.

    `variety_dataset` defaults to "unified" because that is the model /api/analyze
    serves; explaining variety_a_full_best.pt while the UI displays a unified-model
    prediction would be captioning one model's decision with another's evidence.
    "a"/"b" remain available for the ablation figures.

    `head` picks which logit is explained — the two heads of the unified model do not
    attend to the same pixels, and a CAM is undefined without one. `bbox` (in original
    image pixels, exactly as /api/analyze returns it) explains ONE detected seed rather
    than the whole photograph; without it the heatmap covers everything in frame,
    including kernels other than the one whose result is on screen.
    """
    if variety_dataset not in GRADCAM_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"variety_dataset must be one of {list(GRADCAM_DATASETS)}",
        )

    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        from PIL import Image

        from src.explainability.gradcam import overlay_heatmap

        image = pipeline.load_and_validate_image(path)
        region = "whole_image"
        if bbox:
            box = _parse_bbox(bbox, image.size)
            image = pipeline.crop_seed(image, box)
            region = "seed_bbox"

        source = np.array(image)
        cam, meta = pipeline.generate_gradcam(
            image, variety_dataset, head=head, out_size=source.shape[:2]
        )
        overlaid = overlay_heatmap(source, cam)

        buf = io.BytesIO()
        Image.fromarray(overlaid).save(buf, format="PNG")
        buf.seek(0)
    except HTTPException:
        raise
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        # Raised by the pipeline for an unsupported head/dataset combination. That is
        # a bad request, not a server fault, and the message already says which.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected Grad-CAM error")
        raise HTTPException(status_code=500, detail="Internal error generating Grad-CAM.")

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Explainability-Note": (
                f"Grad-CAM attention heatmap over the {meta['target_layer']} feature map "
                f"- shows which pixels influenced the {meta['head']} prediction. "
                "NOT a segmentation mask and NOT a defect area measurement."
            ),
            "X-Gradcam-Model": meta["model"],
            "X-Gradcam-Head": meta["head"],
            "X-Gradcam-Target-Layer": meta["target_layer"],
            "X-Gradcam-Predicted-Class": meta["predicted_class"],
            "X-Gradcam-Region": region,
        },
    )
