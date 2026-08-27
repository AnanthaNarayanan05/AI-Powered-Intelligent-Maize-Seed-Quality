"""Seed-lot composition report.

Aggregates per-seed variety and quality predictions across many analyses into the
figures a seed grader actually asks for: what varieties are in this lot, in what
proportion, and how many kernels are sound.

Deliberate limits, enforced here rather than left to the UI:

  * This is NOT a certification. Certification is a legal determination made by an
    accredited laboratory following a prescribed sampling protocol (ISTA/AOSA),
    including physical purity by weight and a minimum working-sample size. Nothing
    computed from photographs can confer it. Every response carries
    `is_certification: false` and a caveat list that says so.

  * Composition is measured over MODEL PREDICTIONS, not ground truth. A variety the
    classifier has never seen cannot appear in the breakdown: the model is
    closed-set and will assign such a kernel to its nearest known class. The report
    therefore states the low-confidence share explicitly, because that is the only
    available signal that the lot may contain something off-menu.

  * Synthetic-model outputs are excluded from every figure. Procedural pattern
    classes describe a demonstration of the attention architecture, not a real
    kernel, and must never contribute to a soundness rate.
"""
from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import selectinload

from database.db import session_scope
from database.models import Analysis
from src.utils.logging_utils import get_logger

logger = get_logger("lot")
router = APIRouter(prefix="/api/lot", tags=["lot"])

# Matches the frontend's HEALTH_THRESHOLD in frontend/src/lib/seedHealth.js.
LOW_CONFIDENCE = 0.65
UNSOUND_PREFIXES = ("bad", "defect", "damaged", "unhealthy", "poor", "rotten", "broken")


class LotRequest(BaseModel):
    analysis_ids: list[str] = Field(..., min_length=1)
    declared_variety: str | None = Field(
        None,
        description="Optional. The variety the lot is sold as. When given, off-type "
                    "rate is measured against it instead of against the observed "
                    "majority, which is what a purity check actually means.",
    )
    lot_reference: str | None = None


def _shannon_evenness(counts: list[int]) -> float | None:
    """Pielou's evenness: 0 = one variety dominates entirely, 1 = perfectly mixed.
    Undefined for a single observed class, where there is nothing to be even about."""
    total = sum(counts)
    if total == 0 or len(counts) < 2:
        return None
    h = -sum((c / total) * math.log(c / total) for c in counts if c)
    return round(h / math.log(len(counts)), 4)


@router.post("/report")
async def lot_report(req: LotRequest):
    with session_scope() as db:
        analyses = (
            db.query(Analysis)
            .options(selectinload(Analysis.classifications))
            .filter(Analysis.id.in_(req.analysis_ids))
            .all()
        )
        if not analyses:
            raise HTTPException(status_code=404, detail="No analyses found for the supplied ids.")

        found = {a.id for a in analyses}
        missing = [i for i in req.analysis_ids if i not in found]

        variety_counts: dict[str, int] = {}
        variety_conf: dict[str, list[float]] = {}
        quality_counts: dict[str, int] = {}
        low_conf = 0
        seeds_with_variety = 0
        seeds_with_quality = 0
        quality_ood = 0
        synthetic_seen = 0
        datasets_used = set()

        for a in analyses:
            if a.variety_dataset_used:
                datasets_used.add(a.variety_dataset_used)
            for c in a.classifications:
                if c.is_synthetic_model:
                    synthetic_seen += 1
                    continue
                model = (c.model_name or "").lower()
                if "quality" in model or "defect" in model:
                    # A grade the model extrapolated cannot contribute to a soundness
                    # rate: it was produced for a kernel unlike anything the quality
                    # head was validated on. Counted separately, never averaged in.
                    if (c.class_probabilities or {}).get("_out_of_distribution"):
                        quality_ood += 1
                        continue
                    seeds_with_quality += 1
                    quality_counts[c.predicted_class] = quality_counts.get(c.predicted_class, 0) + 1
                else:
                    seeds_with_variety += 1
                    variety_counts[c.predicted_class] = variety_counts.get(c.predicted_class, 0) + 1
                    variety_conf.setdefault(c.predicted_class, []).append(c.confidence or 0.0)
                    if (c.confidence or 0.0) < LOW_CONFIDENCE:
                        low_conf += 1

        n_analyses = len(analyses)

    if seeds_with_variety == 0:
        raise HTTPException(
            status_code=422,
            detail="The selected analyses contain no non-synthetic variety predictions, "
                   "so no lot composition can be computed.",
        )

    composition = sorted(
        (
            {
                "variety": k,
                "count": v,
                "percent": round(v / seeds_with_variety * 100, 2),
                "mean_confidence": round(sum(variety_conf[k]) / len(variety_conf[k]), 4),
            }
            for k, v in variety_counts.items()
        ),
        key=lambda d: d["count"],
        reverse=True,
    )

    dominant = composition[0]
    reference = req.declared_variety or dominant["variety"]
    matching = variety_counts.get(reference, 0)
    off_type = seeds_with_variety - matching

    # Soundness exists only if a real quality model actually ran.
    sound_rate = None
    if seeds_with_quality:
        unsound = sum(
            n for cls, n in quality_counts.items()
            if cls.lower().startswith(UNSOUND_PREFIXES)
        )
        sound_rate = round((seeds_with_quality - unsound) / seeds_with_quality * 100, 2)

    caveats = [
        "This report is not a seed certification. Certification requires an accredited "
        "laboratory, a prescribed working-sample size, and physical purity by weight.",
        "Composition is measured over model predictions, not verified ground truth.",
        "The variety classifier is closed-set: it recognises only the classes it was "
        "trained on and will assign an unfamiliar kernel to the nearest one it knows. "
        "Treat the low-confidence share as the signal for that.",
    ]
    if sound_rate is None:
        caveats.append(
            "No soundness rate is reported: no real quality model contributed to these "
            "analyses. Any defect output present came from the synthetic demonstration "
            "model and was excluded."
        )
    if synthetic_seen:
        caveats.append(
            f"{synthetic_seen} synthetic-model predictions were excluded from all figures."
        )
    if quality_ood:
        caveats.append(
            f"{quality_ood} kernel(s) were graded outside the quality model's validated "
            f"image distribution. Those grades are extrapolations and were excluded from "
            f"the soundness rate rather than averaged into it."
        )
    if len(datasets_used) > 1:
        caveats.append(
            f"These analyses span more than one variety model ({', '.join(sorted(datasets_used))}), "
            f"whose class lists differ. Percentages mix two label spaces and are not "
            f"directly comparable."
        )
    if missing:
        caveats.append(f"{len(missing)} requested analysis id(s) were not found and are excluded.")
    if seeds_with_variety < 100:
        caveats.append(
            f"Only {seeds_with_variety} kernels were assessed. Seed-testing standards call "
            f"for working samples in the hundreds to thousands; treat these percentages as "
            f"indicative."
        )

    return {
        "lot_reference": req.lot_reference,
        "is_certification": False,
        "images_analysed": n_analyses,
        "kernels_assessed": seeds_with_variety,
        "declared_variety": req.declared_variety,
        "reference_variety": reference,
        "reference_is_declared": req.declared_variety is not None,
        "composition": composition,
        "dominant_variety": dominant["variety"],
        "dominant_percent": dominant["percent"],
        "off_type_count": off_type,
        "off_type_percent": round(off_type / seeds_with_variety * 100, 2),
        "purity_percent": round(matching / seeds_with_variety * 100, 2),
        "evenness": _shannon_evenness([d["count"] for d in composition]),
        "low_confidence_count": low_conf,
        "low_confidence_percent": round(low_conf / seeds_with_variety * 100, 2),
        "low_confidence_threshold": LOW_CONFIDENCE,
        "soundness": {
            "available": sound_rate is not None,
            "sound_percent": sound_rate,
            "kernels_graded": seeds_with_quality,
            "kernels_out_of_distribution": quality_ood,
            "distribution": quality_counts or None,
        },
        "caveats": caveats,
    }
