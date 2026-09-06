"""Phase 28: the machine-readable data-quality audit runs as part of the suite,
not only when someone remembers to invoke it by hand before a retrain.

Slower than most of this suite (it hashes every file every manifest names), which is
the deliberate trade-off: a leakage bug here is exactly the kind of defect that
otherwise surfaces months later as an implausibly high accuracy (docs/09 section 6.9).

    python -m pytest tests/test_data_audit.py -v
"""
from __future__ import annotations

import os

import pytest

from src.data.audit_manifests import audit
from src.utils.config import load_config

cfg = load_config()
PROCESSED = cfg["paths"]["processed"]

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PROCESSED), reason="data_processed/ not present on this machine"
)


def test_no_manifest_has_group_split_leakage():
    """Every group-bearing manifest keeps each group inside one split."""
    report = audit(hash_content=False)
    offenders = {
        name: entry["group_split_leakage"]
        for name, entry in report["manifests"].items()
        if entry["group_split_leakage"].get("applicable")
        and not entry["group_split_leakage"]["passed"]
    }
    assert not offenders, f"group/split leakage found: {offenders}"


def test_no_manifest_has_content_hash_leakage():
    """Byte-identical files never span train/val/test, independent of grouping."""
    report = audit(hash_content=True)
    offenders = {
        name: entry["content_hash_leakage"]
        for name, entry in report["manifests"].items()
        if entry.get("content_hash_leakage", {}).get("applicable")
        and not entry["content_hash_leakage"]["passed"]
    }
    assert not offenders, f"content-hash leakage found: {offenders}"


def test_declared_independent_manifests_share_no_files():
    """Datasets this project treats as independently-sourced (docs/03's cross-dataset
    compatibility table) share zero byte-identical images."""
    report = audit(hash_content=True)
    offenders = {
        pair: result
        for pair, result in report["cross_manifest_overlap"].items()
        if result.get("applicable") and not result["passed"]
    }
    assert not offenders, f"unexpected cross-dataset overlap: {offenders}"
