"""What the database is allowed to remember, and what it must refuse to invent.

Phase 9 widened the schema so pixel-level results have somewhere to live. The risk
in widening a schema is not that a column is missing -- that fails loudly. It is
that a column exists and gets filled with something plausible: a zero standing in
for an unmeasured area, a severity band with no derivation behind it, a row that
says "unavailable" about an attempt nobody made. Every test here defends one of
those distinctions.

The migration tests matter for a different reason. There is a real database with
151 analyses in it that predates all of this, and a schema change that silently
drops it would be the single most expensive mistake available in this phase.
"""
from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import database.db as dbmod
from database.db import _apply_additive_migrations
from database.models import (
    STAGE_ORDER, Analysis, Base, Classification, Detection, SeedAssessment, SeedSegmentation,
)
from backend.services.history_service import (
    _derive_stage, _model_version, delete_analysis, export_history_csv,
    get_analysis, save_analysis_result,
)


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point the whole persistence layer at a throwaway file for one test.

    A file rather than :memory: so the same bytes can be reopened with raw sqlite3
    and inspected without the ORM's interpretation in the way.
    """
    path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(dbmod, "_engine", engine)
    monkeypatch.setattr(dbmod, "_SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False))
    return path


def _result(analysis_id="a1", **overrides):
    """A minimal pipeline result of the shape unified_pipeline actually emits."""
    seed = {
        "seed_index": 0,
        "bbox": [10, 10, 60, 70],
        "detection_confidence": 0.91,
        "kernel_px": 50,
        "variety_prediction": {
            # The variety head reports the checkpoint name; the quality head below
            # reports a composite. Both shapes occur, and both must resolve.
            "model": "unified_seed_model", "predicted_class": "Indurata",
            "confidence": 0.82, "class_probabilities": {"Indurata": 0.82},
        },
        "quality_prediction": {
            "model": "unified_seed_model_quality", "predicted_class": "Good",
            "confidence": 0.77, "class_probabilities": {"Good": 0.77, "Bad": 0.23},
            "distribution_distance": 0.31, "distribution_threshold": 0.2056,
            "out_of_distribution": True,
        },
    }
    result = {
        "analysis_id": analysis_id, "seed_count": 1, "seeds": [seed],
        "segmentation": {"status": "unavailable", "reason": "no_served_model",
                         "message": "Segmentation unavailable."},
    }
    result.update(overrides)
    return result


# ------------------------------------------------------------------- migration
def test_additive_migration_adds_columns_without_touching_rows(tmp_path):
    """The pre-Phase-9 schema, carrying data, must survive the widening intact."""
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE analyses (id VARCHAR PRIMARY KEY, seed_count INTEGER)")
    con.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY, seed_index INTEGER)")
    con.execute("CREATE TABLE classifications (id INTEGER PRIMARY KEY, model_name VARCHAR)")
    con.executemany("INSERT INTO analyses VALUES (?,?)", [("old-1", 12), ("old-2", 7)])
    con.execute("INSERT INTO detections VALUES (1, 0)")
    con.commit()
    con.close()

    engine = create_engine(f"sqlite:///{path}")
    _apply_additive_migrations(engine)

    columns = {c["name"] for c in inspect(engine).get_columns("analyses")}
    assert {"analysis_stage", "intent", "stages", "model_versions",
            "segmentation_status", "image_path"} <= columns
    assert "kernel_px" in {c["name"] for c in inspect(engine).get_columns("detections")}
    classification_cols = {c["name"] for c in inspect(engine).get_columns("classifications")}
    assert {"model_version", "confidence_calibrated"} <= classification_cols

    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 2
    assert con.execute("SELECT seed_count FROM analyses WHERE id='old-1'").fetchone()[0] == 12
    # Not backfilled: nobody measured that kernel, and a number here would be invented.
    assert con.execute("SELECT kernel_px FROM detections WHERE id=1").fetchone()[0] is None
    con.close()


def test_additive_migration_is_idempotent(tmp_path):
    path = tmp_path / "twice.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    _apply_additive_migrations(engine)
    _apply_additive_migrations(engine)  # must not raise "duplicate column name"
    names = [c["name"] for c in inspect(engine).get_columns("analyses")]
    assert len(names) == len(set(names))


def test_migration_never_drops_or_rewrites(tmp_path):
    """Guards the property that makes the migration safe to run on the real file:
    it only ever appends columns, so no existing column can lose its data."""
    path = tmp_path / "shape.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    before = {t: [c["name"] for c in inspect(engine).get_columns(t)]
              for t in inspect(engine).get_table_names()}
    _apply_additive_migrations(engine)
    after = {t: [c["name"] for c in inspect(engine).get_columns(t)]
             for t in inspect(engine).get_table_names()}
    assert set(before) == set(after)
    for table, columns in before.items():
        assert after[table][:len(columns)] == columns


# ----------------------------------------------------------------- stage
def test_stage_order_is_the_only_vocabulary():
    assert STAGE_ORDER == ("failed", "detected", "classified", "segmented", "measured")


@pytest.mark.parametrize("result,expected", [
    ({"status": "failed", "seeds": []}, "failed"),
    ({"seeds": [{"seed_index": 0}]}, "detected"),
    ({"seeds": [{"variety_prediction": {"predicted_class": "Indurata"}}]}, "classified"),
    ({"seeds": [{"segmentation": {"available": True}}]}, "segmented"),
    ({"seeds": [{"segmentation": {"available": True, "measurements": {
        "cracked": {"defect_coverage_percent": 3.2}}}}]}, "measured"),
])
def test_stage_is_derived_from_evidence(result, expected):
    assert _derive_stage(result) == expected
    assert _derive_stage(result) in STAGE_ORDER


def test_refused_segmentation_does_not_reach_the_segmented_stage():
    """The gate refusing a kernel is not the same as a segmenter having run on it."""
    refused = {"seeds": [{
        "variety_prediction": {"predicted_class": "Indurata"},
        "segmentation": {"available": False, "reason": "insufficient_resolution"},
    }]}
    assert _derive_stage(refused) == "classified"


# ----------------------------------------------------------------- model version
def test_composite_head_name_resolves_to_its_checkpoint_version():
    """The pipeline names heads; the registry names checkpoints. One unified model
    answers to both head names and must report the same version for each."""
    variety = _model_version("unified_seed_model_variety")
    quality = _model_version("unified_seed_model_quality")
    assert variety is not None and variety == quality


def test_unknown_model_reports_no_version_rather_than_guessing():
    assert _model_version("model_that_does_not_exist") is None
    assert _model_version(None) is None


# ----------------------------------------------------------------- persistence
def test_save_records_stage_versions_and_segmentation_verdict(temp_db):
    save_analysis_result(_result(), variety_dataset="a", image_filename="x.jpg")
    stored = get_analysis("a1")

    assert stored["analysis_stage"] == "classified"
    assert stored["segmentation_status"]["reason"] == "no_served_model"
    assert stored["model_versions"]["unified_seed_model"]
    assert stored["model_versions"]["unified_seed_model_quality"]
    assert stored["detections"][0]["kernel_px"] == 50
    assert all(c["model_version"] for c in stored["classifications"])


def test_distribution_numbers_become_queryable_rows(temp_db):
    """They were already measured; before Phase 9 they were buried in a JSON blob
    inside a probability dict, where no query could reach them."""
    save_analysis_result(_result(), variety_dataset="a")
    with dbmod.session_scope() as db:
        row = db.query(SeedAssessment).one()
        assert row.distribution_distance == pytest.approx(0.31)
        assert row.distribution_threshold == pytest.approx(0.2056)
        assert row.in_distribution == 0  # distance exceeded the threshold
        # Phase 4's columns. The gate flags unfamiliarity; it cannot name a material.
        assert row.foreign_object_status is None


def test_legacy_out_of_distribution_keys_are_preserved(temp_db):
    """Existing frontend and copilot code reads these. Moving the numbers to their
    own table must not break the readers that already found them."""
    save_analysis_result(_result(), variety_dataset="a")
    stored = get_analysis("a1")
    quality = [c for c in stored["classifications"] if c["model_name"].endswith("quality")][0]
    assert quality["class_probabilities"]["_out_of_distribution"] is True


def test_no_assessment_row_when_nothing_gated_the_seed(temp_db):
    result = _result()
    for key in ("distribution_distance", "distribution_threshold", "out_of_distribution"):
        result["seeds"][0]["quality_prediction"].pop(key)
    save_analysis_result(result, variety_dataset="a")
    with dbmod.session_scope() as db:
        assert db.query(SeedAssessment).count() == 0


def test_unattempted_segmentation_writes_no_row(temp_db):
    """An absent row means no segmenter saw this seed. Writing "unavailable" rows
    for seeds nobody attempted would erase the distinction the schema exists for --
    and today, with no served segmenter, that is every seed in the database."""
    save_analysis_result(_result(), variety_dataset="a")
    with dbmod.session_scope() as db:
        assert db.query(SeedSegmentation).count() == 0
    # The refusal is still recorded once, at the image level, where it happened.
    assert get_analysis("a1")["segmentation_status"]["status"] == "unavailable"


def test_refusal_on_an_attempted_seed_is_stored_with_its_reason(temp_db):
    result = _result()
    result["seeds"][0]["segmentation"] = {
        "available": False, "reason": "insufficient_resolution",
        "model": "defect_segmenter_synthetic", "is_synthetic_model": True,
    }
    save_analysis_result(result, variety_dataset="a")
    with dbmod.session_scope() as db:
        row = db.query(SeedSegmentation).one()
        assert row.status == "unavailable"
        assert row.reason == "insufficient_resolution"
        assert row.is_synthetic_model == 1
        assert row.kernel_px == 50
        # No measurement was made, so none is recorded. Zero would be a claim.
        assert row.defect_area_px is None
        assert row.defect_coverage_percent is None


def test_measured_channels_store_area_and_a_digest(temp_db, tmp_path):
    mask = tmp_path / "cracked.png"
    mask.write_bytes(b"not-really-a-png-but-real-bytes")
    result = _result()
    result["seeds"][0]["segmentation"] = {
        "available": True, "model": "defect_segmenter_synthetic", "is_synthetic_model": True,
        "channels": ["cracked", "seed_body"],
        "measurements": {
            "cracked": {"mask_path": str(mask), "seed_area_px": 2000,
                        "defect_area_px": 100, "defect_coverage_percent": 5.0},
            "seed_body": {"seed_area_px": 2000},
        },
    }
    save_analysis_result(result, variety_dataset="a")

    with dbmod.session_scope() as db:
        rows = {r.channel: r for r in db.query(SeedSegmentation).all()}
        assert set(rows) == {"cracked", "seed_body"}
        cracked = rows["cracked"]
        assert cracked.status == "measured" and cracked.reason is None
        assert cracked.defect_coverage_percent == pytest.approx(5.0)
        # Computed from the file on disk, binding the area to the pixels behind it.
        assert cracked.mask_sha256 and len(cracked.mask_sha256) == 64
        assert cracked.mask_path == str(mask)


def test_masks_are_never_stored_as_bytes(temp_db):
    """Phase 9's explicit constraint. No column may hold mask pixel data."""
    columns = {c.name: c for c in SeedSegmentation.__table__.columns}
    assert "mask_path" in columns and "mask_sha256" in columns
    for name, column in columns.items():
        assert "BLOB" not in str(column.type).upper(), f"{name} could hold mask bytes"


def test_severity_band_without_a_basis_is_refused(temp_db):
    """Phase 3 forbids severity thresholds with no documented derivation. A caller
    offering a band and no basis gets neither -- the pair is inseparable."""
    result = _result()
    result["seeds"][0]["segmentation"] = {
        "available": True, "model": "defect_segmenter_synthetic",
        "channels": ["cracked"],
        "measurements": {"cracked": {"defect_coverage_percent": 40.0,
                                     "severity_band": "HIGH"}},  # no severity_basis
    }
    save_analysis_result(result, variety_dataset="a")
    with dbmod.session_scope() as db:
        row = db.query(SeedSegmentation).one()
        assert row.severity_band is None
        assert row.defect_coverage_percent == pytest.approx(40.0)  # the measurement survives


def test_severity_band_with_a_basis_is_kept(temp_db):
    result = _result()
    result["seeds"][0]["segmentation"] = {
        "available": True, "model": "defect_segmenter_synthetic", "channels": ["cracked"],
        "measurements": {"cracked": {"defect_coverage_percent": 40.0, "severity_band": "HIGH",
                                     "severity_basis": "coverage_percentile_v1"}},
    }
    save_analysis_result(result, variety_dataset="a")
    with dbmod.session_scope() as db:
        row = db.query(SeedSegmentation).one()
        assert (row.severity_band, row.severity_basis) == ("HIGH", "coverage_percentile_v1")


def test_orchestrated_run_records_the_question_and_what_ran(temp_db):
    result = _result(intent={"resolved": "assess_quality"},
                     executed=["detect", "classify"], reused=[], unavailable=["segment"],
                     stages={"segment": {"status": "unavailable"}})
    save_analysis_result(result, variety_dataset="a")
    stored = get_analysis("a1")
    assert stored["intent"] == "assess_quality"
    assert stored["stages"]["executed"] == ["detect", "classify"]
    assert stored["stages"]["unavailable"] == ["segment"]


def test_legacy_route_stores_no_intent(temp_db):
    """/api/analyze/image asks no question, so nothing may be recorded as its intent."""
    save_analysis_result(_result(), variety_dataset="a")
    stored = get_analysis("a1")
    assert stored["intent"] is None
    assert stored["stages"] is None


def test_existing_history_shape_is_unchanged(temp_db):
    """Every key the frontend and copilot already read must still be present."""
    save_analysis_result(_result(), variety_dataset="a", image_filename="x.jpg")
    stored = get_analysis("a1")
    for key in ("analysis_id", "created_at", "image_filename", "image_path", "status",
                "seed_count", "variety_dataset_used", "detections", "classifications",
                "similarities"):
        assert key in stored
    assert stored["seed_count"] == 1
    assert len(stored["classifications"]) == 2


def test_rows_written_before_phase_9_read_back_as_unknown(temp_db):
    """A pre-Phase-9 analysis has no stage, no versions and no resolution verdict.
    It must read as unknown rather than as a default that looks like a finding."""
    with dbmod.session_scope() as db:
        db.add(Analysis(id="legacy", seed_count=3, status="completed"))
        db.flush()
        db.add(Detection(analysis_id="legacy", seed_index=0, bbox=[0, 0, 1, 1], confidence=0.5))
        db.add(Classification(analysis_id="legacy", seed_index=0, model_name="old_model",
                              predicted_class="Good", confidence=0.6, class_probabilities={}))
    stored = get_analysis("legacy")
    assert stored["analysis_stage"] is None
    assert stored["model_versions"] is None
    assert stored["segmentation_status"] is None
    assert stored["detections"][0]["kernel_px"] is None
    assert stored["classifications"][0]["model_version"] is None
    assert stored["segmentations"] == [] and stored["assessments"] == []
    assert stored["classifications"][0]["confidence_calibrated"] is None


# ------------------------------------------------------------ calibration flag
def test_confidence_calibrated_round_trips_true_and_false(temp_db):
    """Phase 17: whichever of variety/quality a run's calibration accepted must
    come back exactly as recorded per head, not collapsed to one flag for the
    whole analysis."""
    result = _result()
    result["seeds"][0]["variety_prediction"]["confidence_calibrated"] = True
    result["seeds"][0]["quality_prediction"]["confidence_calibrated"] = False
    save_analysis_result(result, variety_dataset="a", image_filename="x.jpg")
    stored = get_analysis("a1")
    by_model = {c["model_name"]: c for c in stored["classifications"]}
    assert by_model["unified_seed_model"]["confidence_calibrated"] is True
    assert by_model["unified_seed_model_quality"]["confidence_calibrated"] is False


def test_confidence_calibrated_is_unknown_not_false_when_never_recorded(temp_db):
    """_result() carries no confidence_calibrated key at all -- the shape every
    analysis written before Phase 17 has. That must read back as None, the same
    'nobody measured this' value the rest of this file already defends for
    kernel_px and model_version."""
    save_analysis_result(_result(), variety_dataset="a", image_filename="x.jpg")
    stored = get_analysis("a1")
    for c in stored["classifications"]:
        assert c["confidence_calibrated"] is None


# ------------------------------------------------------------ delete / export
def test_deleting_an_analysis_removes_its_children_too(temp_db):
    """The cascade in database/models.py is the thing under test here, not just
    the row deletion API on top of it."""
    save_analysis_result(_result(), variety_dataset="a")
    assert delete_analysis("a1") is True
    assert get_analysis("a1") is None
    with dbmod.session_scope() as db:
        assert db.query(Detection).filter(Detection.analysis_id == "a1").count() == 0
        assert db.query(Classification).filter(Classification.analysis_id == "a1").count() == 0


def test_deleting_an_unknown_analysis_id_reports_nothing_to_delete(temp_db):
    assert delete_analysis("does-not-exist") is False


def test_csv_export_carries_one_row_per_analysis_with_its_top_classifications(temp_db):
    save_analysis_result(_result(analysis_id="a1"), variety_dataset="a", image_filename="k1.jpg")
    save_analysis_result(_result(analysis_id="a2"), variety_dataset="a", image_filename="k2.jpg")
    csv_text = export_history_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0] == (
        "analysis_id,created_at,image_filename,seed_count,variety_dataset_used,"
        "analysis_stage,intent,top_variety,top_variety_confidence,quality_grade,quality_confidence"
    )
    # Newest first, same ordering list_analyses already uses.
    assert lines[1].startswith("a2,")
    body = "\n".join(lines[1:])
    assert "Indurata" in body and "Good" in body
    assert "k1.jpg" in body and "k2.jpg" in body


def test_csv_export_is_empty_but_headed_with_no_history(temp_db):
    csv_text = export_history_csv()
    lines = csv_text.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("analysis_id,")


# --------------------------------------------------- HTTP layer (Phase 7's routes)
# The tests above cover the service functions the routes call; these confirm the
# routes themselves -- status codes, the JSON/CSV shape a client actually receives,
# and the ordering that keeps "/history/export" from being swallowed by the
# "/history/{analysis_id}" dynamic segment. Uses temp_db so no request here can
# ever touch the real database/app.db -- session_scope() re-reads the module-level
# engine on every call, so monkeypatching it before each request is enough to
# retarget the live FastAPI app at the throwaway file for the duration of the test.
def test_delete_route_deletes_once_then_reports_404(temp_db):
    from fastapi.testclient import TestClient
    from backend.main import app

    save_analysis_result(_result(), variety_dataset="a")
    client = TestClient(app)

    r = client.delete("/api/history/a1")
    assert r.status_code == 200
    assert r.json() == {"deleted": True, "analysis_id": "a1"}

    assert client.get("/api/history/a1").status_code == 404
    # Deleting the same id again finds nothing left to delete.
    assert client.delete("/api/history/a1").status_code == 404


def test_delete_route_404s_for_an_id_that_was_never_stored(temp_db):
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.delete("/api/history/does-not-exist")
    assert r.status_code == 404


def test_export_route_serves_a_downloadable_csv_not_swallowed_by_the_id_route(temp_db):
    from fastapi.testclient import TestClient
    from backend.main import app

    save_analysis_result(_result(analysis_id="a1"), variety_dataset="a", image_filename="k1.jpg")
    client = TestClient(app)

    r = client.get("/api/history/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="maize_analysis_history.csv"' in r.headers["content-disposition"] \
        or "attachment; filename=maize_analysis_history.csv" in r.headers["content-disposition"]
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("analysis_id,")
    assert "k1.jpg" in r.text and "Indurata" in r.text
