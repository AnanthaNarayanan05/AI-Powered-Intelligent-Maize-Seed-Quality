"""SQLAlchemy ORM models for analysis history. SQLite for local dev, per
the master prompt. Never stores secrets (GEMINI_API_KEY is never written here).

PHASE 9 EXTENSION. The pixel-level phases (2 defect area, 3 severity, 4 foreign
objects, 5 visible symptoms) each produce a per-seed measurement that had nowhere
to live: history stored a bounding box, a label and a set of neighbours, and threw
away everything the pipeline knew about resolution and segmentation. The tables
below add that room. Three rules shaped them.

MASKS LIVE ON DISK. `SeedSegmentation.mask_path` is a path and `mask_sha256` is
the digest of the file at that path. Nothing here holds mask bytes: a SQLite row
per kernel per channel at native resolution would grow the file faster than every
other table combined, and the same bytes already exist as a PNG that the media
route can serve directly. The digest is what makes the path trustworthy -- it
binds a stored area measurement to the exact mask it was measured from, the same
scheme the model registry uses to bind metrics to a checkpoint.

AN ABSENT ROW AND AN UNAVAILABLE ROW MEAN DIFFERENT THINGS. A seed with no
SeedSegmentation row was never put to a segmenter. A seed with a row whose
status is "unavailable" was, and the `reason` says what stopped it -- no served
model, or a kernel below the measured resolution floor. Storing the refusal is
the point: an empty result that reads as "not attempted" is how a limitation
quietly becomes invisible.

SYMPTOMS ARE CLASSIFICATIONS, NOT ASSESSMENTS. Phase 5's visible-symptom head
writes to `classifications` like every other head, because that is exactly what
it produces -- a predicted class, a confidence and a probability vector. Giving
symptoms their own column somewhere else would put two answers to "what did the
models say about this seed" in two places, and the second one always goes stale.
"""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# How far an analysis actually got, in order. Derived from what produced a value,
# never from what was requested -- an analysis that planned segmentation and was
# refused by the resolution gate did not reach "segmented".
STAGE_ORDER = ("failed", "detected", "classified", "segmented", "measured")


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    image_filename = Column(String, nullable=True)  # original client-side filename
    # Server-side path of the stored upload, so the UI can render a thumbnail of the
    # analysed image in History. Nullable: rows written before this column existed
    # have no path and the UI shows a placeholder rather than a broken image.
    image_path = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | completed | failed
    seed_count = Column(Integer, default=0)
    variety_dataset_used = Column(String, nullable=True)  # "a" | "b"
    error_message = Column(Text, nullable=True)

    # --- Phase 9 -------------------------------------------------------------
    # Furthest stage that produced a value; one of STAGE_ORDER. A single column
    # rather than a JSON probe because lot analytics filters on it.
    analysis_stage = Column(String, nullable=True)
    # Which question was asked, when the request came through the orchestrator.
    # Null for the legacy /api/analyze/image route, which asks nothing.
    intent = Column(String, nullable=True)
    # The orchestrator's own account of the run: which stages ran, which were
    # reused from an earlier question about the same image, which were refused
    # and why. Stored whole because its shape is the orchestrator's to change.
    stages = Column(JSON, nullable=True)
    # {model_key: version} for every model that actually contributed a value here.
    # Without it a stored result cannot be reproduced after a model is retrained:
    # the row would still say "Indurata, 0.94" and no longer say what said it.
    model_versions = Column(JSON, nullable=True)
    # The image-level segmentation verdict the pipeline already computes and, until
    # now, discarded on write: status, reason, message, and the measured kernel
    # pixel sizes the resolution gate judged.
    segmentation_status = Column(JSON, nullable=True)

    detections = relationship("Detection", back_populates="analysis", cascade="all, delete-orphan")
    classifications = relationship("Classification", back_populates="analysis", cascade="all, delete-orphan")
    similarities = relationship("SimilarityResult", back_populates="analysis", cascade="all, delete-orphan")
    segmentations = relationship("SeedSegmentation", back_populates="analysis", cascade="all, delete-orphan")
    assessments = relationship("SeedAssessment", back_populates="analysis", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)
    bbox = Column(JSON)  # [x1,y1,x2,y2]
    confidence = Column(Float)
    # Phase 9. The kernel's measured short side in pixels, by the same rule the
    # resolution floor was measured with. Decides per seed which pixel-level
    # answers were available for it, so a stored refusal can be re-justified
    # later without re-opening the image.
    kernel_px = Column(Integer, nullable=True)

    analysis = relationship("Analysis", back_populates="detections")


class Classification(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)
    model_name = Column(String)  # e.g. "variety_a_full" or "synthetic_defect_classifier"
    # Phase 9. model_name says which model; this says which build of it. Two rows
    # naming the same model with different versions are not comparable, and
    # before this column there was no way to notice.
    model_version = Column(String, nullable=True)
    predicted_class = Column(String)
    confidence = Column(Float)
    class_probabilities = Column(JSON)
    is_synthetic_model = Column(Integer, default=0)  # 0/1 boolean flag, always explicit

    analysis = relationship("Analysis", back_populates="classifications")


class SimilarityResult(Base):
    __tablename__ = "similarity_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)
    embedding_model = Column(String)
    # list of {path,variety_label,quality_label,source,distance,similarity_score};
    # labels belong to the neighbour images, never to the analysed seed
    top_k_results = Column(JSON)

    analysis = relationship("Analysis", back_populates="similarities")


class SeedSegmentation(Base):
    """One row per (seed, defect channel): what the segmenter said, or why nothing.

    Per channel rather than per seed because the segmenter emits one mask plane per
    condition, and a kernel can be both cracked and mouldy over different pixels.
    `seed_area_px` repeats across a seed's channels; that denormalisation is
    deliberate, so a coverage figure can be re-checked from its own row without a
    second lookup for the denominator it was divided by.
    """

    __tablename__ = "seed_segmentations"
    __table_args__ = (UniqueConstraint("analysis_id", "seed_index", "channel",
                                       name="uq_seed_segmentation"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)
    channel = Column(String)  # cracked | discolored_mold | insect_damaged | seed_body

    # measured | unavailable. Never null: a row exists precisely to record one or
    # the other, and a third silent state would defeat the distinction.
    status = Column(String, default="unavailable")
    # null when measured; otherwise no_served_model | insufficient_resolution.
    reason = Column(String, nullable=True)

    # Path on disk to the PNG, and the digest of the bytes at that path. Bytes are
    # deliberately not stored here -- see the module docstring.
    mask_path = Column(String, nullable=True)
    mask_sha256 = Column(String, nullable=True)

    # Phase 2. valid seed-body pixels, defect pixels, and the percentage the second
    # is of the first. Null while the phase is unbuilt; a zero would read as a
    # measurement of no defect, which is a different claim from no measurement.
    seed_area_px = Column(Integer, nullable=True)
    defect_area_px = Column(Integer, nullable=True)
    defect_coverage_percent = Column(Float, nullable=True)

    # Phase 3. A band is only ever written together with the basis that justifies
    # its thresholds; the pair is enforced in history_service, because a severity
    # label with no derivation behind it is the exact thing Phase 3 forbids.
    severity_band = Column(String, nullable=True)
    severity_basis = Column(String, nullable=True)

    model_key = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    is_synthetic_model = Column(Integer, default=0)
    # Repeated from the detection so a refusal carries its own evidence.
    kernel_px = Column(Integer, nullable=True)

    analysis = relationship("Analysis", back_populates="segmentations")


class SeedAssessment(Base):
    """One row per seed: verdicts that are neither a classification nor a mask.

    Today that is the distribution gate -- the cosine distance from the quality
    reference and the threshold it was compared against. Those numbers are measured
    on every seed already, but they used to travel smuggled inside a
    classification's probability dict, where nothing could query them.

    The foreign-object columns are Phase 4's to fill and stay null until then. The
    gate can honestly say a kernel is unlike anything it was trained on; calling
    that a stone, a husk or a cob fragment needs labelled data this project does
    not yet have, so `foreign_object_status` records flagging and never identity.

    The symptom columns are Phase 5's, and carry the same restraint one step
    further: they record the visible condition category an expert grader would
    assign, together with the gate's decision about whether it may be asserted at
    all. A withheld verdict is stored as a withheld verdict, with its reason, and
    never collapsed into a null that a later reader would take for "no symptom".
    """

    __tablename__ = "seed_assessments"
    __table_args__ = (UniqueConstraint("analysis_id", "seed_index",
                                       name="uq_seed_assessment"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)

    in_distribution = Column(Integer, nullable=True)  # 0/1, null when not gated
    distribution_distance = Column(Float, nullable=True)
    distribution_threshold = Column(Float, nullable=True)
    distribution_model = Column(String, nullable=True)

    # Phase 4. known_maize | possible_foreign_object. Never a material name.
    foreign_object_status = Column(String, nullable=True)
    foreign_object_basis = Column(String, nullable=True)

    # Phase 5. A visual grading category, never a diagnosis: no column here names
    # a pathogen, a toxin or a species, and none may be added.
    #
    # symptom_class is NULL whenever symptom_status is "withheld", so no query can
    # read a category off a kernel the gate refused to categorise. The class that
    # won the argmax is kept in its own column under a name that cannot be
    # mistaken for the verdict, because "the model nearly said MY and was stopped"
    # is worth recovering later and is not the same claim.
    symptom_status = Column(String, nullable=True)   # reported | withheld
    symptom_reason = Column(String, nullable=True)   # class_not_validated | low_confidence
    symptom_class = Column(String, nullable=True)
    symptom_confidence = Column(Float, nullable=True)
    symptom_argmax_class = Column(String, nullable=True)
    symptom_argmax_confidence = Column(Float, nullable=True)
    symptom_threshold = Column(Float, nullable=True)
    symptom_model = Column(String, nullable=True)

    analysis = relationship("Analysis", back_populates="assessments")


class BatchAnalysis(Base):
    __tablename__ = "batch_analyses"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    total_images = Column(Integer, default=0)
    successful_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    aggregate_stats = Column(JSON, nullable=True)
    analysis_ids = Column(JSON, nullable=True)  # list of Analysis.id
