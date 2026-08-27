"""Phase 17 persistence helpers: writes verified pipeline results into SQLite and
reads them back for the history/copilot endpoints."""
from __future__ import annotations

import datetime

from database.db import session_scope
from database.models import Analysis, Detection, Classification, SimilarityResult, BatchAnalysis


def _iso_utc(value: datetime.datetime | None) -> str | None:
    """SQLite drops tzinfo on write, so stored timestamps read back naive. They are
    always UTC — tag them so clients render local time instead of shifting by the
    UTC offset."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value.isoformat()


def save_analysis_result(result: dict, variety_dataset: str, image_filename: str | None = None,
                         image_path: str | None = None) -> str:
    with session_scope() as db:
        analysis = Analysis(
            id=result["analysis_id"],
            image_filename=image_filename,
            image_path=image_path or result.get("image_path"),
            status="completed",
            seed_count=result["seed_count"],
            variety_dataset_used=variety_dataset,
        )
        db.add(analysis)
        db.flush()

        for seed in result.get("seeds", []):
            db.add(Detection(
                analysis_id=analysis.id, seed_index=seed["seed_index"],
                bbox=seed["bbox"], confidence=seed["detection_confidence"],
            ))
            if seed.get("variety_prediction"):
                vp = seed["variety_prediction"]
                db.add(Classification(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    model_name=vp["model"], predicted_class=vp["predicted_class"],
                    confidence=vp["confidence"], class_probabilities=vp["class_probabilities"],
                    is_synthetic_model=0,
                ))
            if seed.get("quality_prediction"):
                qp = seed["quality_prediction"]
                # Stored with is_synthetic_model=0 because it is a real model trained
                # on expert-assigned labels. An out-of-distribution grade is recorded
                # as-is; the flag travels in class_probabilities so downstream
                # consumers can see the prediction was an extrapolation.
                probs = dict(qp["class_probabilities"])
                if qp.get("out_of_distribution"):
                    probs["_out_of_distribution"] = True
                    probs["_distribution_distance"] = qp.get("distribution_distance")
                db.add(Classification(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    model_name=qp["model"], predicted_class=qp["predicted_class"],
                    confidence=qp["confidence"], class_probabilities=probs,
                    is_synthetic_model=0,
                ))
            if seed.get("synthetic_defect_prediction"):
                sp = seed["synthetic_defect_prediction"]
                db.add(Classification(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    model_name=sp["model"], predicted_class=sp["predicted_class"],
                    confidence=sp["confidence"], class_probabilities=sp["class_probabilities"],
                    is_synthetic_model=1,
                ))
            if seed.get("similarity_results"):
                db.add(SimilarityResult(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    embedding_model=f"variety_{variety_dataset}", top_k_results=seed["similarity_results"],
                ))
        return analysis.id


def get_analysis(analysis_id: str) -> dict | None:
    with session_scope() as db:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis is None:
            return None
        return _serialize_analysis(analysis)


def list_analyses(limit: int = 20, offset: int = 0) -> list[dict]:
    with session_scope() as db:
        rows = (
            db.query(Analysis)
            .order_by(Analysis.created_at.desc())
            .offset(offset).limit(limit).all()
        )
        return [_serialize_analysis(a) for a in rows]


def _serialize_analysis(analysis: Analysis) -> dict:
    return {
        "analysis_id": analysis.id,
        "created_at": _iso_utc(analysis.created_at),
        "image_filename": analysis.image_filename,
        "image_path": analysis.image_path,
        "status": analysis.status,
        "seed_count": analysis.seed_count,
        "variety_dataset_used": analysis.variety_dataset_used,
        "detections": [
            {"seed_index": d.seed_index, "bbox": d.bbox, "confidence": d.confidence}
            for d in analysis.detections
        ],
        "classifications": [
            {
                "seed_index": c.seed_index, "model_name": c.model_name,
                "predicted_class": c.predicted_class, "confidence": c.confidence,
                "class_probabilities": c.class_probabilities,
                "is_synthetic_model": bool(c.is_synthetic_model),
            }
            for c in analysis.classifications
        ],
        "similarities": [
            {"seed_index": s.seed_index, "embedding_model": s.embedding_model, "top_k_results": s.top_k_results}
            for s in analysis.similarities
        ],
    }


def save_batch_result(batch_id: str, total: int, successful: int, failed: int, stats: dict, analysis_ids: list[str]):
    with session_scope() as db:
        db.add(BatchAnalysis(
            id=batch_id, total_images=total, successful_images=successful,
            failed_images=failed, aggregate_stats=stats, analysis_ids=analysis_ids,
        ))


def get_batch(batch_id: str) -> dict | None:
    with session_scope() as db:
        batch = db.query(BatchAnalysis).filter(BatchAnalysis.id == batch_id).first()
        if batch is None:
            return None
        return {
            "batch_id": batch.id,
            "total_images": batch.total_images,
            "successful_images": batch.successful_images,
            "failed_images": batch.failed_images,
            "aggregate_stats": batch.aggregate_stats,
            "analysis_ids": batch.analysis_ids,
        }
