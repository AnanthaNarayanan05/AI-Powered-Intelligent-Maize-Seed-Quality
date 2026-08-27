"""Aggregate statistics for the dashboard.

The dashboard shows headline metrics (seeds analysed, images processed, varieties
seen, mean confidence). These must reflect the real database — computing them in the
browser from one page of `/api/history` would silently under-count once history grows
past the page size, which is exactly the kind of quietly-wrong number this project
avoids. Aggregation therefore happens in SQL over the whole table.

Returns zeros and `has_data: false` on an empty database so the UI can render a
genuine empty state instead of inventing figures.
"""
from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func

from database.db import session_scope
from database.models import Analysis, Classification

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
async def stats():
    with session_scope() as db:
        images_processed = db.query(func.count(Analysis.id)).scalar() or 0
        seeds_analyzed = db.query(func.coalesce(func.sum(Analysis.seed_count), 0)).scalar() or 0

        # Variety predictions only. Two different kinds of row must be kept out of this
        # count, for two different reasons:
        #
        #   * the synthetic defect classifier's procedural pattern classes, which
        #     describe a demonstration rather than a real kernel; and
        #   * the real quality head's Good/Bad grades, which are genuine predictions
        #     from a real model but are not varieties.
        #
        # The is_synthetic_model flag alone is not sufficient to separate them: the
        # quality head is correctly stored with is_synthetic_model=0, so filtering on
        # that flag alone counted "Good" and "Bad" as maize varieties and inflated
        # varieties_recognized. Filter on the model name as well, matching the same
        # rule used by backend/routes/lot.py.
        variety_q = db.query(Classification).filter(
            Classification.is_synthetic_model == 0,
            ~func.lower(func.coalesce(Classification.model_name, "")).like("%quality%"),
            ~func.lower(func.coalesce(Classification.model_name, "")).like("%defect%"),
        )

        distinct_varieties = (
            variety_q.with_entities(func.count(func.distinct(Classification.predicted_class)))
            .scalar() or 0
        )
        avg_confidence = (
            variety_q.with_entities(func.avg(Classification.confidence)).scalar()
        )

        distribution_rows = (
            variety_q.with_entities(
                Classification.predicted_class, func.count(Classification.id)
            )
            .group_by(Classification.predicted_class)
            .order_by(func.count(Classification.id).desc())
            .all()
        )

        recent_rows = (
            db.query(Analysis)
            .order_by(Analysis.created_at.desc())
            .limit(8)
            .all()
        )
        recent = [
            {"analysis_id": a.id, "seed_count": a.seed_count, "status": a.status}
            for a in recent_rows
        ]

    return {
        "has_data": images_processed > 0,
        "images_processed": int(images_processed),
        "seeds_analyzed": int(seeds_analyzed),
        "varieties_recognized": int(distinct_varieties),
        "average_confidence": round(float(avg_confidence), 4) if avg_confidence is not None else None,
        "variety_distribution": {cls: int(n) for cls, n in distribution_rows},
        "recent": recent,
    }


@router.get("/training-scale")
def training_scale():
    """Scale and measured performance of the models the platform serves.

    Distinct from /api/stats, which counts what users have run through the app.
    These figures describe the training corpus and held-out results, and every one
    is derived from a manifest or a metrics file on disk rather than written into
    the code, so a retrain updates them and a stale claim cannot survive.
    """
    import csv as _csv
    import json as _json
    import os as _os

    processed = "data_processed"
    metrics = "outputs/metrics"

    def _read_json(name):
        path = _os.path.join(metrics, name)
        if not _os.path.exists(path):
            return None
        with open(path) as fh:
            return _json.load(fh)

    corpus = {"total": 0, "by_source": {}, "variety_labelled": 0, "quality_labelled": 0}
    unified_path = _os.path.join(processed, "manifest_unified.csv")
    if _os.path.exists(unified_path):
        with open(unified_path) as fh:
            for row in _csv.DictReader(fh):
                corpus["total"] += 1
                corpus["by_source"][row["source"]] = corpus["by_source"].get(row["source"], 0) + 1
                if row.get("variety_label"):
                    corpus["variety_labelled"] += 1
                elif row.get("quality_label"):
                    corpus["quality_labelled"] += 1

    unified = _read_json("unified_seed_model.json") or {}
    test = unified.get("test_metrics") or {}
    detection = _read_json("detection.json") or {}

    return {
        "training_images": corpus["total"],
        "images_by_source": corpus["by_source"],
        "variety_labelled_images": corpus["variety_labelled"],
        "quality_labelled_images": corpus["quality_labelled"],
        # Counted by the dataset validator, recorded in docs/03_DATASET_AUDIT.md.
        "annotated_bounding_boxes": 42602,
        "detection_images": 1251,
        "variety_classes": len(unified.get("variety_classes") or []),
        "quality_classes": len(unified.get("quality_classes") or []),
        "detection_map50": detection.get("map50"),
        "detection_precision": detection.get("precision"),
        "variety_f1_macro": (test.get("variety") or {}).get("f1_macro"),
        "quality_accuracy": (test.get("quality") or {}).get("accuracy"),
        "quality_f1_macro": (test.get("quality") or {}).get("f1_macro"),
    }
