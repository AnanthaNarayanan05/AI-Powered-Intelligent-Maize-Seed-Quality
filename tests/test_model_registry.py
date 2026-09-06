"""The registry is the platform's answer to "which model serves this task?".

These tests defend three things: that the declaration is internally coherent,
that what it reports about an artefact matches the artefact, and -- most
importantly -- that it refuses rather than substitutes. A registry that quietly
answered a defect question with a synthetic-defect model would be worse than no
registry at all.
"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from src.registry import (
    TaskUnavailableError,
    UnknownModelError,
    checkpoint_sha256,
    get_registry,
    load_registry,
)
from src.registry.model_registry import REGISTRY_PATH

registry = get_registry()

# Models whose labels this project generated. They exist, they are evaluated, and
# they must never be reachable through resolve().
SYNTHETIC_KEYS = ("synthetic_defect_classifier", "defect_segmenter_synthetic")


# ------------------------------------------------------------- the declaration
def test_registry_yaml_declares_every_task_its_models_claim():
    spec = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    declared = set(spec["tasks"])
    for model in spec["models"]:
        assert model["task"] in declared, f"{model['key']} claims undeclared task"
        assert set(model["serves"]) <= declared, f"{model['key']} serves undeclared task"


def test_exactly_one_model_serves_each_served_task():
    """Two models claiming one task would make the choice depend on iteration
    order, which is a coin flip dressed up as a decision."""
    for task in registry.served_tasks():
        assert len(registry.for_task(task)) == 1, f"'{task}' has multiple servers"


def test_every_model_key_and_checkpoint_is_distinct():
    keys = [m.key for m in registry.all()]
    assert len(keys) == len(set(keys))
    paths = [m.checkpoint for m in registry.all()]
    assert len(paths) == len(set(paths)), "two entries point at one checkpoint"


# ---------------------------------------------------------------------- refusal
@pytest.mark.parametrize("task", ["defect_segmentation", "defect_pattern"])
def test_resolve_refuses_a_task_no_model_is_allowed_to_serve(task):
    """The core guarantee. Both tasks HAVE a trained model sitting on disk; both
    were trained on defects this project painted, so both declare serves: [] and
    the task must come back unavailable rather than answered with them."""
    trained = [m for m in registry.all() if m.task == task]
    assert trained, f"fixture precondition: a model for '{task}' should be declared"

    with pytest.raises(TaskUnavailableError) as excinfo:
        registry.resolve(task)
    message = str(excinfo.value)
    assert task in message
    # the refusal must not name a substitute
    for model in trained:
        assert model.key not in message


def test_resolve_rejects_a_task_the_registry_does_not_declare():
    """Distinct from the refusal above, and deliberately so. A task with no
    server is a real capability the platform cannot currently answer; a task the
    registry never declared is a caller bug -- a misspelling or an invented
    capability -- and gets a different exception so it cannot be caught and
    reported to a user as "temporarily unavailable"."""
    with pytest.raises(UnknownModelError):
        registry.resolve("diagnose_pathogen")


def test_get_rejects_an_unknown_key():
    with pytest.raises(UnknownModelError):
        registry.get("model_that_does_not_exist")


def test_synthetic_models_are_declared_unserved_and_carry_their_provenance():
    for key in SYNTHETIC_KEYS:
        record = registry.get(key)
        assert record.serves == [], f"{key} must not be servable"
        assert record.served is False
        assert record.available is False, "unserved, even with the checkpoint present"
        assert record.label_provenance == "synthetic"


# --------------------------------------------------------- declared vs measured
def test_status_reflects_what_is_actually_on_this_machine():
    for record in registry.all():
        assert record.present == os.path.exists(record.checkpoint)
        if not record.present:
            assert record.status == "missing"
        elif record.served:
            assert record.status == "ready"
        else:
            assert record.status == "available"
        # `available` is the conjunction, never inferred from either half alone
        assert record.available == (record.served and record.present)


def test_present_models_report_a_fingerprint_and_a_size():
    for record in registry.all():
        if not record.present:
            assert record.fingerprint is None
            continue
        assert record.fingerprint and len(record.fingerprint) == 64
        assert record.size_bytes == os.path.getsize(record.checkpoint)
        assert record.trained_at and record.trained_at.endswith("+00:00")


def test_fingerprint_is_the_real_hash_of_the_file_and_is_stable():
    record = registry.resolve("variety")
    assert record.fingerprint == checkpoint_sha256(record.checkpoint)
    # a second load must agree; a cached hash that drifts is worse than none
    assert load_registry().get(record.key).fingerprint == record.fingerprint


def test_class_lists_come_from_the_checkpoint_not_from_the_yaml():
    spec = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert not any("classes" in m for m in spec["models"]), (
        "class lists must not be declared; a retrain would silently invalidate them"
    )
    record = registry.resolve("variety")
    if record.present:
        assert record.classes, "the unified checkpoint carries both label sets"
        assert any(c.startswith("variety: ") for c in record.classes)
        assert any(c.startswith("quality: ") for c in record.classes)


# ------------------------------------------------------------- metrics binding
def test_metrics_binding_states_are_honest_about_what_they_prove():
    for record in registry.all():
        binding = record.metrics_binding
        assert binding in {"verified", "stale", "unrecorded", "missing"}
        if not record.metrics or not record.present:
            assert binding == "missing"
            continue
        recorded = record.metrics.get("checkpoint_sha256")
        if not recorded:
            # explicitly NOT "verified": no hash says nothing either way
            assert binding == "unrecorded"
        else:
            assert binding == ("verified" if recorded == record.fingerprint else "stale")


def test_a_mismatched_recorded_hash_reads_stale_rather_than_verified():
    """The failure this whole scheme exists to catch: a model retrained without
    being re-evaluated must not go on reporting the previous run's score."""
    from src.registry.model_registry import _binding

    record = registry.resolve("variety")
    if not record.present:
        pytest.skip("unified model not trained")
    original = record.metrics
    try:
        record.metrics = {"checkpoint_sha256": record.fingerprint}
        assert _binding(record) == "verified"
        record.metrics = {"checkpoint_sha256": "0" * 64}
        assert _binding(record) == "stale"
        record.metrics = {"accuracy": 0.99}
        assert _binding(record) == "unrecorded"
    finally:
        record.metrics = original


def test_training_scripts_record_the_fingerprint_they_evaluated():
    """Every metrics writer must stamp the checkpoint hash, or new runs land back
    in the unrecorded state this was built to leave behind."""
    writers = [
        "src/training/train_unified.py",
        "src/training/train_variety.py",
        "src/training/train_detection.py",
        "src/training/train_synthetic_defect.py",
        "src/segmentation/train_seg.py",
    ]
    for path in writers:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        assert "from src.registry.model_registry import checkpoint_sha256" in source, path
        assert '"checkpoint_sha256": checkpoint_sha256(' in source, path


# -------------------------------------------------------------------- consumers
def test_the_pipeline_takes_its_checkpoints_from_the_registry():
    """Regression: checkpoint paths used to be spelled out in the pipeline, so
    the registry could be edited without changing which model actually ran."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    assert pipeline._checkpoint_for("detect") == registry.resolve("detect").checkpoint
    assert pipeline._checkpoint_for("variety") == registry.resolve("variety").checkpoint

    with open("src/pipeline/unified_pipeline.py", encoding="utf-8") as f:
        source = f.read()
    for literal in ("unified_seed_model_best.pt", "quality_reference.npz", "detection_corn"):
        assert literal not in source, f"{literal} is hard-coded in the pipeline again"


def test_the_pipeline_reports_an_unservable_task_as_unavailable_not_as_a_guess():
    from src.pipeline.unified_pipeline import AnalysisPipeline, ModelNotAvailableError

    pipeline = AnalysisPipeline()
    with pytest.raises(ModelNotAvailableError) as excinfo:
        pipeline._checkpoint_for("defect_segmentation")
    assert "defect_segmentation" in str(excinfo.value)


def test_system_info_describes_the_registry_and_not_a_copy_of_it():
    from backend.services.system_service import system_report

    rows = system_report()["models"]
    assert {r["key"] for r in rows} == {m.key for m in registry.all()}
    for row in rows:
        record = registry.get(row["key"])
        assert row["status"] == record.status
        assert row["served"] == record.served
        assert row["fingerprint"] == record.fingerprint
        assert row["metrics_binding"] == record.metrics_binding
        assert isinstance(row["metrics"], list)

    with open("backend/services/system_service.py", encoding="utf-8") as f:
        assert "_MODEL_SPECS" not in f.read(), "the duplicate model table is back"


def test_unverified_metrics_are_flagged_to_the_reader():
    """A number the server cannot tie to the checkpoint on disk must not reach
    the page looking like one it can."""
    from backend.services.system_service import system_report

    for row in system_report()["models"]:
        if row["metrics_binding"] in ("stale", "unrecorded") and row["metrics"]:
            assert row["metrics_note"], f"{row['key']} shows unqualified metrics"
        if row["metrics_binding"] == "verified":
            assert row["metrics_note"] is None


def test_the_resolution_floor_is_declared_from_a_measurement():
    """Phase 8 gates serving on this number, so it must trace to the file that
    measured it rather than to a guess typed into the YAML."""
    checked = 0
    for record in registry.all():
        if not record.requires_resolution:
            continue
        source = record.requires_resolution["source"]
        assert os.path.exists(source), source
        with open(source, encoding="utf-8") as f:
            measured = json.load(f)
        assert (
            record.requires_resolution["min_kernel_px"]
            == measured["resolution_floor_kernel_px"]
        )
        checked += 1
    assert checked, "the segmenter should declare the measured floor"
