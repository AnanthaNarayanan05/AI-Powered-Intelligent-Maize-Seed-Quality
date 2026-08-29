"""SQLite engine/session setup. Path comes from configs/config.yaml -> paths.database."""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.utils.config import load_config
from database.models import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = load_config()
        db_path = cfg["paths"]["database"]
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(_engine)
        _apply_additive_migrations(_engine)
    return _engine


def _apply_additive_migrations(engine) -> None:
    """Add nullable columns introduced after a database was first created.

    `create_all` only creates missing *tables*, never missing columns, so an existing
    analyses.db keeps its old schema and every query naming a new column fails. These
    migrations are additive and nullable only — no data is rewritten or dropped.
    """
    from sqlalchemy import inspect, text

    # Phase 9 widened three existing tables. Every entry is nullable with no
    # default, so rows written before the column existed read back as NULL --
    # which is the honest value: nobody measured that kernel's pixel width, and
    # backfilling one now would be inventing it.
    additive = {
        "analyses": {
            "image_path": "VARCHAR",
            "analysis_stage": "VARCHAR",
            "intent": "VARCHAR",
            "stages": "JSON",
            "model_versions": "JSON",
            "segmentation_status": "JSON",
        },
        "detections": {"kernel_px": "INTEGER"},
        "classifications": {"model_version": "VARCHAR"},
    }

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in additive.items():
            if table not in inspector.get_table_names():
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


@contextmanager
def session_scope():
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
