from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.services.history_service import (
    delete_analysis, export_history_csv, list_analyses, get_analysis, get_batch,
)

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
async def history(limit: int = Query(20, le=100), offset: int = Query(0, ge=0)):
    return {"analyses": list_analyses(limit=limit, offset=offset), "limit": limit, "offset": offset}


# Declared before /history/{analysis_id} -- FastAPI matches routes in order, and
# a dynamic path segment would otherwise swallow the literal "export" as an id.
@router.get("/history/export")
async def history_export():
    csv_text = export_history_csv()
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=maize_analysis_history.csv"},
    )


@router.get("/history/{analysis_id}")
async def history_detail(analysis_id: str):
    result = get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@router.delete("/history/{analysis_id}")
async def history_delete(analysis_id: str):
    deleted = delete_analysis(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"deleted": True, "analysis_id": analysis_id}


@router.get("/history/batch/{batch_id}")
async def batch_detail(batch_id: str):
    result = get_batch(batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return result
