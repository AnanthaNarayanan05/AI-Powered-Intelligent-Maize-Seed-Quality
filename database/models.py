"""SQLAlchemy ORM models for Phase 17 (analysis history). SQLite for local dev, per
the master prompt. Never stores secrets (GEMINI_API_KEY is never written here)."""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


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

    detections = relationship("Detection", back_populates="analysis", cascade="all, delete-orphan")
    classifications = relationship("Classification", back_populates="analysis", cascade="all, delete-orphan")
    similarities = relationship("SimilarityResult", back_populates="analysis", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)
    bbox = Column(JSON)  # [x1,y1,x2,y2]
    confidence = Column(Float)

    analysis = relationship("Analysis", back_populates="detections")


class Classification(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    seed_index = Column(Integer)
    model_name = Column(String)  # e.g. "variety_a_full" or "synthetic_defect_classifier"
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


class BatchAnalysis(Base):
    __tablename__ = "batch_analyses"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    total_images = Column(Integer, default=0)
    successful_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    aggregate_stats = Column(JSON, nullable=True)
    analysis_ids = Column(JSON, nullable=True)  # list of Analysis.id
