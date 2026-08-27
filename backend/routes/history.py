from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.services.history_service import list_analyses, get_analysis, get_batch

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
async def history(limit: int = Query(20, le=100), offset: int = Query(0, ge=0)):
    return {"analyses": list_analyses(limit=limit, offset=offset), "limit": limit, "offset": offset}


@router.get("/history/{analysis_id}")
async def history_detail(analysis_id: str):
    result = get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@router.get("/history/batch/{batch_id}")
async def batch_detail(batch_id: str):
    result = get_batch(batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return result
