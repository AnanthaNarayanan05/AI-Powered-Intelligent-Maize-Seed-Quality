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


def test_system_info_is_read_from_disk_not_hard_coded():
    """Phase 14: the page used to recite a stored description of the platform, which
    kept advertising "Dataset A: 3 classes" long after one 6-variety model replaced
    both. Every model row must now come from an artefact that is actually here."""
    import os

    body = client.get("/api/system-info").json()
    models = {m["key"]: m for m in body["models"]}
    assert models, "no models reported"

    for m in models.values():
        assert m["status"] in ("ready", "available", "missing")
        # status is a claim about this machine, so it must match this machine
        on_disk = os.path.exists(m["checkpoint"])
        assert (m["status"] != "missing") == on_disk, (
            f"{m['key']} reported {m['status']} but exists={on_disk}"
        )
        if m["status"] == "missing":
            assert m["metrics"] == [] or m["trained_at"] is None

    # the served stack is exactly the models the pipeline actually runs. Phase 5
    # added the fifth: the visible-symptom classifier ships as its own checkpoint
    # rather than as a head on the unified model, because the fine-tune that made
    # the symptom head usable measurably wrecked variety and quality in the same
    # weights.
    assert {k for k, m in models.items() if m["served"]} == {
        "detection", "unified_seed_model", "quality_gate", "maize_identity_gate",
        "visible_symptom_classifier",
    }


def test_system_info_reports_only_one_route_and_no_stale_dataset_claims():
    """Two @app.get("/api/system-info") handlers were declared; Starlette matched the
    first and silently dropped the second, so the file held two contradictory
    descriptions with nothing to reveal the conflict."""
    routes = [r for r in app.routes if getattr(r, "path", None) == "/api/system-info"]
    assert len(routes) == 1, f"{len(routes)} handlers registered for /api/system-info"

    text = " ".join(client.get("/api/system-info").json()["supported_capabilities"]).lower()
    assert "dataset a" not in text and "dataset b" not in text, (
        "the retired per-dataset variety models are still advertised as the stack"
    )


def test_system_info_never_leaks_the_gemini_key():
    """The key is read from the environment and must never reach a response body."""
    from backend.config import GEMINI_API_KEY

    raw = client.get("/api/system-info").text
    if GEMINI_API_KEY:
        assert GEMINI_API_KEY not in raw
    body = client.get("/api/system-info").json()
    assert set(body["gemini"]) == {"configured", "model"}
    assert isinstance(body["gemini"]["configured"], bool)


def test_system_info_runtime_and_database_are_probed():
    body = client.get("/api/system-info").json()

    runtime = body["runtime"]
    assert runtime["device"] in ("cuda", "cpu")
    # a GPU name is reported only when CUDA genuinely answered
    assert (runtime["gpu"] is not None) == runtime["cuda_available"]

    db = body["database"]
    assert db["status"] in ("connected", "not created", "unavailable")
    if db["status"] == "connected":
        assert set(db["tables"]) >= {"analyses", "detections"}
        assert all(isinstance(v, int) for v in db["tables"].values())


def test_system_info_binds_each_gallery_to_the_encoder_that_built_it():
    """Phase 13 provenance has to be visible on the page, not only in the sidecar."""
    for idx in client.get("/api/system-info").json()["similarity_indices"]:
        assert idx["status"] in ("built", "not built")
        if idx["status"] == "built":
            assert idx["encoder"], f"gallery {idx['dataset']} names no encoder"
            # an evaluation image in the gallery would make retrieval self-confirming
            assert idx["gallery_split"] == "train"
            assert idx["gallery_size"] > 0


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
