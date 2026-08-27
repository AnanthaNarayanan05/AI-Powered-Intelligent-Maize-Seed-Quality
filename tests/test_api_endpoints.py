"""Backend API tests that run WITHOUT any trained model or GPU present — verifying
Phase 29's required error handling (missing models, invalid images, missing Gemini
key, 404s) never crashes the app."""
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "gemini_configured" in r.json()


def test_system_info_lists_unsupported_capabilities():
    """The capabilities the data genuinely cannot support must stay declared.

    Purity moved off this list on 2026-08-26 when /api/lot/report was added: lot
    composition is aggregation over verified per-seed predictions and is real. What
    remains impossible is *certification*, which is a legal determination requiring an
    accredited laboratory, so that must still be disclaimed — as must severity,
    segmentation, and pathogen diagnosis, none of which any dataset here can back.
    """
    r = client.get("/api/system-info")
    assert r.status_code == 200
    body = r.json()
    assert "explicitly_not_supported" in body

    unsupported = " ".join(body["explicitly_not_supported"]).lower()
    for claim in ("certif", "severity", "segmentation", "diagnosis"):
        assert claim in unsupported, f"{claim!r} is no longer declared unsupported"


def test_system_info_never_claims_synthetic_defects_as_real():
    """The synthetic classifier is superseded and off by default. It must not be
    advertised as a supported capability, and quality claims must be tied to the
    real expert labels rather than to procedural patterns."""
    body = client.get("/api/system-info").json()
    supported = " ".join(body["supported_capabilities"]).lower()

    assert "synthetic" not in supported, "synthetic defect model is advertised as supported"
    assert "quality" in supported
    # the accompanying limit must travel with the capability
    assert "unverified" in supported or "out-of-distribution" in supported


def test_detect_rejects_non_image_file():
    r = client.post("/api/detect", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_detect_returns_503_without_trained_model():
    fake_jpg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"0" * 100)  # minimal JPEG-like header, likely invalid but typed correctly
    r = client.post("/api/detect", files={"file": ("test.jpg", fake_jpg, "image/jpeg")})
    assert r.status_code in (400, 503)  # either "invalid image" or "model not available" — never a 500 crash


def test_history_not_found():
    r = client.get("/api/history/does-not-exist")
    assert r.status_code == 404


def test_history_list_empty_ok():
    r = client.get("/api/history")
    assert r.status_code == 200
    assert "analyses" in r.json()


def test_copilot_chat_missing_analysis_returns_404_not_crash():
    r = client.post("/api/copilot/chat", json={"analysis_id": "missing", "question": "test?"})
    assert r.status_code == 404


def test_classify_variety_rejects_invalid_dataset_param():
    fake_jpg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"0" * 100)
    r = client.post(
        "/api/classify/variety",
        files={"file": ("test.jpg", fake_jpg, "image/jpeg")},
        data={"variety_dataset": "z"},
    )
    assert r.status_code == 400
