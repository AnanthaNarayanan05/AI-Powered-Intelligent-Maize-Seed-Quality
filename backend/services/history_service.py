"""Phase 17 persistence helpers: writes verified pipeline results into SQLite and
reads them back for the history/copilot endpoints.

Phase 9 widened what gets kept. The rule applied throughout is that a field is
written only when a model actually produced it: `kernel_px` is stored because the
pipeline measures it, `defect_coverage_percent` is left null because nothing has
measured one yet, and neither is guessed from the other. Where the pipeline says a
thing was unavailable, the reason travels with it, so a limitation is stored as a
limitation rather than as a missing value that later reads like an oversight.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import os
import re

from database.db import session_scope
from database.models import (
    Analysis, Detection, Classification, SimilarityResult, BatchAnalysis,
    SeedSegmentation, SeedAssessment,
)
from src.registry import get_registry, UnknownModelError
from src.similarity.embedding_index import encoder_id


def _iso_utc(value: datetime.datetime | None) -> str | None:
    """SQLite drops tzinfo on write, so stored timestamps read back naive. They are
    always UTC — tag them so clients render local time instead of shifting by the
    UTC offset."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value.isoformat()


def _model_version(model_name: str | None) -> str | None:
    """Version of the registry entry behind a pipeline model name.

    The pipeline names heads, not checkpoints: one `unified_seed_model` answers to
    both `unified_seed_model_variety` and `unified_seed_model_quality`. Matching on
    the longest registry key that prefixes the name resolves that without a second
    hand-maintained mapping to fall out of date. A name the registry does not know
    returns None rather than raising -- history is a convenience and must not be
    the thing that fails an otherwise good analysis.
    """
    if not model_name:
        return None
    try:
        registry = get_registry()
    except Exception:  # noqa: BLE001
        return None
    candidates = [k for k in (m.key for m in registry.all()) if model_name.startswith(k)]
    if not candidates:
        return None
    try:
        return registry.get(max(candidates, key=len)).version
    except UnknownModelError:
        return None


def _derive_stage(result: dict) -> str:
    """How far this analysis actually got.

    Read backwards from the strongest evidence present in the result, so the answer
    describes what happened rather than what was planned. An analysis that asked for
    segmentation and was refused by the resolution gate reports "classified": the
    stage it reached, not the stage it wanted.
    """
    seeds = result.get("seeds") or []
    if result.get("status") == "failed":
        return "failed"
    segmentations = [s.get("segmentation") or {} for s in seeds]
    measurements = [m for seg in segmentations
                    for m in (seg.get("measurements") or {}).values()]
    if any(m.get("defect_coverage_percent") is not None for m in measurements):
        return "measured"
    if any(seg.get("available") for seg in segmentations):
        return "segmented"
    if any(s.get("variety_prediction") or s.get("quality_prediction") for s in seeds):
        return "classified"
    return "detected"


def _model_versions_used(result: dict) -> dict:
    """{registry key: version} for every model that contributed a value here."""
    names = set()
    for seed in result.get("seeds") or []:
        for key in ("variety_prediction", "quality_prediction", "synthetic_defect_prediction"):
            prediction = seed.get(key)
            if prediction and prediction.get("model"):
                names.add(prediction["model"])
        segmentation = seed.get("segmentation") or {}
        if segmentation.get("model"):
            names.add(segmentation["model"])
    versions = {}
    for name in names:
        version = _model_version(name)
        if version is not None:
            versions[name] = version
    return versions


def _file_sha256(path: str | None) -> str | None:
    """Digest of a mask on disk, so a stored area stays bound to the exact pixels
    it was measured from. Returns None for a path that is not there rather than
    inventing a digest for bytes nobody can check."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_segmentation_rows(db, analysis_id: str, seed: dict) -> None:
    """Record what the segmenter said about one seed, per channel.

    Called only when the pipeline attached a `segmentation` entry to the seed, i.e.
    when a segmenter was genuinely put to this kernel. A seed that no segmenter ever
    saw gets no row at all -- the image-level `segmentation_status` on the analysis
    already says why, and inventing per-seed refusals would fill the table with rows
    describing an attempt that never happened.
    """
    segmentation = seed.get("segmentation")
    if not segmentation:
        return

    common = {
        "analysis_id": analysis_id,
        "seed_index": seed["seed_index"],
        "model_key": segmentation.get("model"),
        "model_version": _model_version(segmentation.get("model")),
        "is_synthetic_model": 1 if segmentation.get("is_synthetic_model") else 0,
        "kernel_px": seed.get("kernel_px"),
    }

    if not segmentation.get("available"):
        # One row, no channel: the refusal applies to the whole seed, and there is
        # no per-channel result to attribute it to.
        db.add(SeedSegmentation(
            channel="", status="unavailable", reason=segmentation.get("reason"), **common,
        ))
        return

    measurements = segmentation.get("measurements") or {}
    for channel in segmentation.get("channels") or []:
        m = measurements.get(channel) or {}
        # A severity band is stored only alongside the basis that derived it. Phase 3
        # forbids a band with no documented derivation, and dropping it here means
        # such a band cannot reach the database even if some future caller offers one.
        band, basis = m.get("severity_band"), m.get("severity_basis")
        if not basis:
            band = None
        mask_path = m.get("mask_path")
        db.add(SeedSegmentation(
            channel=channel,
            status="measured",
            reason=None,
            mask_path=mask_path,
            mask_sha256=m.get("mask_sha256") or _file_sha256(mask_path),
            seed_area_px=m.get("seed_area_px"),
            defect_area_px=m.get("defect_area_px"),
            defect_coverage_percent=m.get("defect_coverage_percent"),
            severity_band=band,
            severity_basis=basis,
            **common,
        ))


def _add_assessment_row(db, analysis_id: str, seed: dict) -> None:
    """Store the distribution gate's and foreign-object gate's numbers for one seed.

    These are measured on every kernel already, but until now they travelled inside
    a classification's probability dict, where nothing could query them.

    Since Phase 5 the visible-symptom verdict is stored here too, including the
    refusals. ``symptom_class`` is null on a withheld verdict by construction --
    it is read straight off the payload, which already nulls it -- so no query can
    turn an abstention into a category. ``symptom_argmax_class`` keeps what the
    model would have said unguarded, under a name that says so.

    ``foreign_object_status`` is only ever ``known_maize``,
    ``possible_foreign_object`` or ``unavailable`` -- the last for an object the
    gate declined to score, which a later reader must not fold into either of the
    other two. It is never a material name: the gate measures
    distance from known maize, and this project holds no labels that would let it
    say a flagged object is a stone rather than a husk. The column records
    flagging, and the basis column records which gate did the flagging so a stored
    verdict can be traced back to the operating point that produced it.
    """
    quality = seed.get("quality_prediction") or {}
    foreign = seed.get("foreign_object") or {}
    symptom = seed.get("symptom_prediction") or {}
    if "distribution_distance" not in quality and not foreign and not symptom:
        return
    db.add(SeedAssessment(
        analysis_id=analysis_id,
        seed_index=seed["seed_index"],
        in_distribution=(None if "distribution_distance" not in quality
                         else (0 if quality.get("out_of_distribution") else 1)),
        distribution_distance=quality.get("distribution_distance"),
        distribution_threshold=quality.get("distribution_threshold"),
        distribution_model=quality.get("model"),
        foreign_object_status=foreign.get("status"),
        foreign_object_basis=foreign.get("basis"),
        symptom_status=symptom.get("status"),
        symptom_reason=symptom.get("reason"),
        # Already null on the payload when the verdict was withheld. Read through
        # rather than reconstructed, so the column can never disagree with the
        # response the caller was given.
        symptom_class=symptom.get("predicted_class"),
        symptom_confidence=symptom.get("confidence"),
        symptom_argmax_class=symptom.get("argmax_class_before_gate"),
        symptom_argmax_confidence=symptom.get("argmax_confidence"),
        symptom_threshold=symptom.get("confidence_threshold"),
        symptom_model=symptom.get("basis"),
    ))


def save_analysis_result(result: dict, variety_dataset: str, image_filename: str | None = None,
                         image_path: str | None = None) -> str:
    with session_scope() as db:
        routing = result.get("intent") or {}
        analysis = Analysis(
            id=result["analysis_id"],
            image_filename=image_filename,
            image_path=image_path or result.get("image_path"),
            status="completed",
            seed_count=result["seed_count"],
            variety_dataset_used=variety_dataset,
            analysis_stage=_derive_stage(result),
            # Null for the legacy /api/analyze/image route, which asks no question.
            intent=routing.get("resolved") if isinstance(routing, dict) else None,
            stages={
                "executed": result.get("executed") or [],
                "reused": result.get("reused") or [],
                "unavailable": result.get("unavailable") or [],
                "report": result.get("stages"),
            } if "executed" in result else None,
            model_versions=_model_versions_used(result) or None,
            segmentation_status=result.get("segmentation"),
        )
        db.add(analysis)
        db.flush()

        for seed in result.get("seeds", []):
            db.add(Detection(
                analysis_id=analysis.id, seed_index=seed["seed_index"],
                bbox=seed["bbox"], confidence=seed["detection_confidence"],
                kernel_px=seed.get("kernel_px"),
            ))
            if seed.get("variety_prediction"):
                vp = seed["variety_prediction"]
                db.add(Classification(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    model_name=vp["model"], model_version=_model_version(vp["model"]),
                    predicted_class=vp["predicted_class"],
                    confidence=vp["confidence"], class_probabilities=vp["class_probabilities"],
                    is_synthetic_model=0,
                    # Phase 17. .get(), not [], because a row written before this
                    # field existed has no key to read -- None is the honest value
                    # for "nobody recorded whether this one was calibrated",
                    # distinct from False ("measured and found uncalibrated").
                    confidence_calibrated=vp.get("confidence_calibrated"),
                ))
            if seed.get("quality_prediction"):
                qp = seed["quality_prediction"]
                # Stored with is_synthetic_model=0 because it is a real model trained
                # on expert-assigned labels. An out-of-distribution grade is recorded
                # as-is; the flag travels in class_probabilities so downstream
                # consumers can see the prediction was an extrapolation. Since Phase 9
                # the same numbers also live in seed_assessments, where they are
                # queryable; this copy stays for the callers already reading it.
                probs = dict(qp["class_probabilities"])
                if qp.get("out_of_distribution"):
                    probs["_out_of_distribution"] = True
                    probs["_distribution_distance"] = qp.get("distribution_distance")
                db.add(Classification(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    model_name=qp["model"], model_version=_model_version(qp["model"]),
                    predicted_class=qp["predicted_class"],
                    confidence=qp["confidence"], class_probabilities=probs,
                    is_synthetic_model=0,
                    confidence_calibrated=qp.get("confidence_calibrated"),
                ))
            if seed.get("synthetic_defect_prediction"):
                sp = seed["synthetic_defect_prediction"]
                db.add(Classification(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    model_name=sp["model"], model_version=_model_version(sp["model"]),
                    predicted_class=sp["predicted_class"],
                    confidence=sp["confidence"], class_probabilities=sp["class_probabilities"],
                    is_synthetic_model=1,
                ))
            if seed.get("similarity_results"):
                db.add(SimilarityResult(
                    analysis_id=analysis.id, seed_index=seed["seed_index"],
                    # record WHICH encoder produced these neighbours, so a stored
                    # result stays interpretable after the index is rebuilt
                    embedding_model=encoder_id(variety_dataset),
                    top_k_results=seed["similarity_results"],
                ))
            _add_segmentation_rows(db, analysis.id, seed)
            _add_assessment_row(db, analysis.id, seed)
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
        "analysis_stage": analysis.analysis_stage,
        "intent": analysis.intent,
        "stages": analysis.stages,
        "model_versions": analysis.model_versions,
        "segmentation_status": analysis.segmentation_status,
        "detections": [
            {"seed_index": d.seed_index, "bbox": d.bbox, "confidence": d.confidence,
             "kernel_px": d.kernel_px}
            for d in analysis.detections
        ],
        "classifications": [
            {
                "seed_index": c.seed_index, "model_name": c.model_name,
                "model_version": c.model_version,
                "predicted_class": c.predicted_class, "confidence": c.confidence,
                "class_probabilities": c.class_probabilities,
                "is_synthetic_model": bool(c.is_synthetic_model),
                # Phase 17. None survives as None here (not coerced to False): a
                # row from before calibration existed must keep reading as
                # "never measured", the same distinction the column itself keeps.
                "confidence_calibrated": c.confidence_calibrated,
            }
            for c in analysis.classifications
        ],
        "similarities": [
            {"seed_index": s.seed_index, "embedding_model": s.embedding_model, "top_k_results": s.top_k_results}
            for s in analysis.similarities
        ],
        "segmentations": [
            {
                "seed_index": s.seed_index, "channel": s.channel,
                "status": s.status, "reason": s.reason,
                "mask_path": s.mask_path, "mask_sha256": s.mask_sha256,
                "seed_area_px": s.seed_area_px, "defect_area_px": s.defect_area_px,
                "defect_coverage_percent": s.defect_coverage_percent,
                "severity_band": s.severity_band, "severity_basis": s.severity_basis,
                "model_key": s.model_key, "model_version": s.model_version,
                "is_synthetic_model": bool(s.is_synthetic_model),
                "kernel_px": s.kernel_px,
            }
            for s in analysis.segmentations
        ],
        "assessments": [
            {
                "seed_index": a.seed_index,
                "in_distribution": None if a.in_distribution is None else bool(a.in_distribution),
                "distribution_distance": a.distribution_distance,
                "distribution_threshold": a.distribution_threshold,
                "distribution_model": a.distribution_model,
                "foreign_object_status": a.foreign_object_status,
                "foreign_object_basis": a.foreign_object_basis,
                "symptom_status": a.symptom_status,
                "symptom_reason": a.symptom_reason,
                "symptom_class": a.symptom_class,
                "symptom_confidence": a.symptom_confidence,
                "symptom_argmax_class": a.symptom_argmax_class,
                "symptom_argmax_confidence": a.symptom_argmax_confidence,
                "symptom_threshold": a.symptom_threshold,
                "symptom_model": a.symptom_model,
            }
            for a in analysis.assessments
        ],
    }


def save_batch_result(batch_id: str, total: int, successful: int, failed: int, stats: dict, analysis_ids: list[str]):
    with session_scope() as db:
        db.add(BatchAnalysis(
            id=batch_id, total_images=total, successful_images=successful,
            failed_images=failed, aggregate_stats=stats, analysis_ids=analysis_ids,
        ))


def delete_analysis(analysis_id: str) -> bool:
    """Hard-deletes one analysis and everything the cascade in database/models.py
    owns for it (detections, classifications, similarities, segmentations,
    assessments). Returns False rather than raising when there is no such row,
    so the caller can tell "nothing to delete" apart from "deleted".

    Deliberately does not touch the uploaded file at `image_path` on disk: a
    history row and the bytes it points at are two different things to own, and
    deleting a file a moment after deleting the row that named it is a second,
    separate destructive action this function does not take silently.
    """
    with session_scope() as db:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis is None:
            return False
        db.delete(analysis)
        return True


_QUALITY_MODEL_RE = re.compile(r"(quality|defect)", re.IGNORECASE)


def _is_quality_row(c: Classification) -> bool:
    return not c.is_synthetic_model and bool(_QUALITY_MODEL_RE.search(c.model_name or ""))


def _is_variety_row(c: Classification) -> bool:
    return not c.is_synthetic_model and not _QUALITY_MODEL_RE.search(c.model_name or "")


_EXPORT_COLUMNS = [
    "analysis_id", "created_at", "image_filename", "seed_count",
    "variety_dataset_used", "analysis_stage", "intent",
    "top_variety", "top_variety_confidence",
    "quality_grade", "quality_confidence",
]


def export_history_csv() -> str:
    """One row per analysis, summarising the same fields the History page's card
    view already shows for it -- not a per-seed dump. A multi-seed batch analysis
    reports only its first matching seed's variety/quality here, exactly the
    simplification History.jsx's card view already makes; the full per-seed
    detail remains available one row at a time from GET /api/history/{id}.
    """
    with session_scope() as db:
        rows = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_EXPORT_COLUMNS)
        for a in rows:
            variety = next((c for c in a.classifications if _is_variety_row(c)), None)
            quality = next((c for c in a.classifications if _is_quality_row(c)), None)
            writer.writerow([
                a.id,
                _iso_utc(a.created_at) or "",
                a.image_filename or "",
                a.seed_count,
                a.variety_dataset_used or "",
                a.analysis_stage or "",
                a.intent or "",
                variety.predicted_class if variety else "",
                f"{variety.confidence:.4f}" if variety and variety.confidence is not None else "",
                quality.predicted_class if quality else "",
                f"{quality.confidence:.4f}" if quality and quality.confidence is not None else "",
            ])
        return buf.getvalue()


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
