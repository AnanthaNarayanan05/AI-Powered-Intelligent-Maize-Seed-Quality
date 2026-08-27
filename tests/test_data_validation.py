"""Run with: pytest tests/test_data_validation.py
Requires the real datasets extracted locally at the configured paths — these tests
are skipped automatically if the datasets aren't present (e.g. in CI without data)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.data.validate_dataset import audit_classification_dataset
from src.data.validate_detection_dataset import validate_yolo_dataset
from src.utils.config import load_config

cfg = load_config()


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip(f"Dataset not present at {path} — place it locally before running this test")


def test_dataset_a_audit():
    _skip_if_missing(cfg["paths"]["dataset_a"])
    report = audit_classification_dataset(cfg["paths"]["dataset_a"])
    assert report["total_images"] == cfg["dataset_a"]["num_images"]
    assert set(report["classes"]) == set(cfg["dataset_a"]["classes"])
    assert len(report["corrupted_files"]) == 0


def test_dataset_b_audit():
    _skip_if_missing(cfg["paths"]["dataset_b"])
    report = audit_classification_dataset(cfg["paths"]["dataset_b"])
    assert report["total_images"] == cfg["dataset_b"]["num_images"]
    assert set(report["classes"]) == set(cfg["dataset_b"]["classes"])


def test_dataset_c_yolo_validity():
    _skip_if_missing(cfg["paths"]["dataset_c"])
    report = validate_yolo_dataset(cfg["paths"]["dataset_c"])
    assert report["data_yaml"]["names"] == cfg["dataset_c"]["classes"]
    assert len(report["missing_labels"]) == 0
    assert len(report["empty_labels"]) == 0
