from __future__ import annotations

import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import UPLOAD_DIR, MAX_UPLOAD_MB, ALLOWED_IMAGE_TYPES
from backend.services.pipeline_service import get_pipeline, ModelNotAvailableError, InvalidImageError
from backend.services.history_service import save_analysis_result, save_batch_result
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/api", tags=["analyze"])
logger = get_logger("route_analyze")


def _save_upload(file: UploadFile) -> str:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        os.remove(dest_path)
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB > {MAX_UPLOAD_MB} MB limit)")

    return dest_path


@router.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    variety_dataset: str = Form("unified"),
    run_similarity: bool = Form(True),
    run_synthetic_defect: bool = Form(False),
):
    # "unified" is what the platform serves: one model, six varieties, real quality.
    # "a"/"b" remain accepted so the per-dataset ablation models stay reachable.
    if variety_dataset not in ("unified", "a", "b"):
        raise HTTPException(status_code=400, detail="variety_dataset must be 'unified', 'a' or 'b'")

    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        result = pipeline.analyze_image(
            path, variety_dataset=variety_dataset,
            run_similarity=run_similarity, run_synthetic_defect=run_synthetic_defect,
        )
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during image analysis")
        raise HTTPException(status_code=500, detail="Internal error during analysis. Please try again.")

    save_analysis_result(result, variety_dataset, image_filename=file.filename,
                         image_path=result.get("image_path"))
    return result


@router.post("/analyze/batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    variety_dataset: str = Form("unified"),
):
    # "unified" is what the platform serves: one model, six varieties, real quality.
    # "a"/"b" remain accepted so the per-dataset ablation models stay reachable.
    if variety_dataset not in ("unified", "a", "b"):
        raise HTTPException(status_code=400, detail="variety_dataset must be 'unified', 'a' or 'b'")

    pipeline = get_pipeline()
    analysis_ids, variety_dist, quality_dist, confidences = [], {}, {}, []
    successful, failed = 0, 0

    for file in files:
        try:
            path = _save_upload(file)
            result = pipeline.analyze_image(path, variety_dataset=variety_dataset)
            analysis_id = save_analysis_result(result, variety_dataset, image_filename=file.filename,
                                               image_path=result.get("image_path"))
            analysis_ids.append(analysis_id)
            successful += 1
            for seed in result.get("seeds", []):
                if seed.get("variety_prediction"):
                    vp = seed["variety_prediction"]
                    variety_dist[vp["predicted_class"]] = variety_dist.get(vp["predicted_class"], 0) + 1
                    confidences.append(vp["confidence"])
                if seed.get("quality_prediction"):
                    qp = seed["quality_prediction"]
                    key = qp["predicted_class"]
                    if qp.get("out_of_distribution"):
                        key = f"{key} (unverified)"
                    quality_dist[key] = quality_dist.get(key, 0) + 1
                if seed.get("synthetic_defect_prediction"):
                    sp = seed["synthetic_defect_prediction"]
                    quality_dist[sp["predicted_class"]] = quality_dist.get(sp["predicted_class"], 0) + 1
        except (InvalidImageError, ModelNotAvailableError) as e:
            failed += 1
            logger.warning(f"Batch item failed: {e}")
        except Exception:  # noqa: BLE001
            failed += 1
            logger.exception("Unexpected batch item error")

    total_seeds = sum(variety_dist.values())
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
    stats = {
        "total_seeds_detected": total_seeds,
        "variety_distribution": variety_dist,
        "quality_distribution": quality_dist,
        "average_confidence": avg_conf,
        "low_confidence_count": sum(1 for c in confidences if c < 0.6),
    }

    batch_id = str(uuid.uuid4())
    save_batch_result(batch_id, len(files), successful, failed, stats, analysis_ids)

    return {
        "batch_id": batch_id,
        "total_images": len(files),
        "successful_images": successful,
        "failed_images": failed,
        "aggregate_stats": stats,
        "analysis_ids": analysis_ids,
    }
