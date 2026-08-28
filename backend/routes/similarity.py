from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.routes.analyze import _save_upload
from backend.services.pipeline_service import get_pipeline, ModelNotAvailableError, InvalidImageError
from src.similarity.embedding_index import IndexEncoderMismatch
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/api", tags=["similarity"])
logger = get_logger("route_similarity")

# "unified" is the platform model; a/b remain reachable so the ablation figures in
# the report stay reproducible against their own galleries.
SIMILARITY_DATASETS = ("unified", "a", "b")

QUERY_NOTE = (
    "VISUAL/FEATURE SIMILARITY only — the nearest gallery images in this model's "
    "feature space. Labels shown belong to the neighbour images, not to the query, "
    "and this is not a variety certification."
)


@router.post("/similarity")
async def similarity(
    file: UploadFile = File(...),
    variety_dataset: str = Form("unified"),
    top_k: int = Form(5),
):
    if variety_dataset not in SIMILARITY_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"variety_dataset must be one of {', '.join(SIMILARITY_DATASETS)}",
        )
    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    path = _save_upload(file)
    pipeline = get_pipeline()
    try:
        image = pipeline.load_and_validate_image(path)
        results = pipeline.embed_and_search(image, variety_dataset, top_k)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except IndexEncoderMismatch as e:
        # The index on disk was built by a different model than the one that would
        # answer this query. Returning neighbours anyway would look normal and mean
        # nothing, so this is surfaced as unavailable instead.
        logger.error("Similarity index/encoder mismatch: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected similarity search error")
        raise HTTPException(status_code=500, detail="Internal error during similarity search.")

    return {
        "query_note": QUERY_NOTE,
        "encoder": pipeline._get_faiss_index(variety_dataset).info.get("encoder"),
        "gallery_size": pipeline._get_faiss_index(variety_dataset).index.ntotal,
        "results": results,
    }
