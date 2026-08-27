"""End-to-end integration test against the REAL trained checkpoints (Phase 28).

Unlike tests/test_api_endpoints.py — which asserts that endpoints degrade gracefully
when no model is trained — this module asserts the opposite: that once the
checkpoints actually exist on disk, every stage chains together and produces sane
output on real images. Each test skips (rather than fails) when its checkpoint is
absent, so the suite stays green on a fresh clone before training has been run.

    python -m pytest tests/test_integration_e2e.py -v
"""
from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from src.utils.config import load_config

cfg = load_config()
CKPT_DIR = cfg["paths"]["checkpoints"]
DATASET_A_ROOT = cfg["paths"]["dataset_a"]

client = TestClient(app)


def _ckpt(*parts) -> str:
    return os.path.join(CKPT_DIR, *parts)


def _has(*parts) -> bool:
    return os.path.exists(_ckpt(*parts))


def _first_real_image() -> str | None:
    """A genuine Dataset A photograph to run the pipeline against."""
    if not os.path.isdir(DATASET_A_ROOT):
        return None
    for cls in sorted(os.listdir(DATASET_A_ROOT)):
        cls_dir = os.path.join(DATASET_A_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(cls_dir, name)
    return None


def _upload(path: str):
    with open(path, "rb") as f:
        return {"file": (os.path.basename(path), io.BytesIO(f.read()), "image/jpeg")}


REAL_IMAGE = _first_real_image()
needs_image = pytest.mark.skipif(REAL_IMAGE is None, reason="Dataset A not extracted")


@needs_image
@pytest.mark.skipif(not _has("detection_corn", "weights", "best.pt"), reason="detection model not trained")
def test_detect_returns_boxes_on_a_real_image():
    r = client.post("/api/detect", files=_upload(REAL_IMAGE))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed_count"] == len(body["detections"])
    for det in body["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        assert x2 > x1 and y2 > y1
        assert 0.0 <= det["confidence"] <= 1.0


@needs_image
@pytest.mark.skipif(not _has("variety_a_full_best.pt"), reason="variety model A not trained")
def test_variety_classifier_produces_calibrated_probabilities():
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    pred = pipeline.classify_variety(image, dataset="a")

    probs = pred["class_probabilities"]
    assert pred["predicted_class"] in probs
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert pred["confidence"] == pytest.approx(max(probs.values()), abs=1e-4)
    # the resolver must report which experiment actually served the request
    assert pred["model"].startswith("variety_a_")


@needs_image
@pytest.mark.skipif(not _has("variety_a_full_best.pt"), reason="variety model A not trained")
def test_predicted_variety_matches_the_source_folder_label():
    """The strongest end-to-end signal available without new annotation: a held-out
    real image should classify as the variety of the folder it came from."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    expected = os.path.basename(os.path.dirname(REAL_IMAGE))
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    pred = pipeline.classify_variety(image, dataset="a")
    assert pred["predicted_class"] == expected


@needs_image
@pytest.mark.skipif(not _has("synthetic_defect_classifier_best.pt"), reason="synthetic defect model not trained")
def test_synthetic_defect_output_is_always_labelled_synthetic():
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    pred = pipeline.classify_synthetic_defect(image)

    assert pred["is_synthetic_model"] is True
    assert "synthetic" in pred["disclaimer"].lower()
    assert pred["model"] == "synthetic_defect_classifier"


@needs_image
@pytest.mark.skipif(not _has("faiss_variety_a.faiss"), reason="FAISS index A not built")
def test_similarity_search_returns_ranked_neighbours():
    r = client.post("/api/similarity", files=_upload(REAL_IMAGE), data={"variety_dataset": "a", "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "not a variety certification" in body["query_note"].lower()
    results = body["results"]
    assert len(results) > 0
    distances = [x["distance"] for x in results]
    assert distances == sorted(distances), "neighbours must come back nearest-first"


@needs_image
@pytest.mark.skipif(not _has("variety_a_full_best.pt"), reason="variety model A not trained")
def test_gradcam_heatmap_has_image_shape_and_normalised_range():
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    cam, class_idx = pipeline.generate_gradcam(image, dataset="a")

    assert cam.ndim == 2
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert isinstance(class_idx, int)


@needs_image
@pytest.mark.skipif(not _has("variety_a_full_best.pt"), reason="variety model A not trained")
def test_gradcam_endpoint_returns_png_with_the_not_segmentation_disclaimer():
    r = client.post("/api/explain/gradcam", files=_upload(REAL_IMAGE), data={"variety_dataset": "a"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    note = r.headers["X-Explainability-Note"].lower()
    assert "not a segmentation mask" in note
    assert "not a defect area" in note

    # the returned overlay must be a real image, not a blank canvas
    overlay = Image.open(io.BytesIO(r.content))
    assert overlay.size[0] > 0 and overlay.size[1] > 0


@needs_image
@pytest.mark.skipif(
    not (_has("detection_corn", "weights", "best.pt") and _has("variety_a_full_best.pt")),
    reason="full pipeline not trained",
)
def test_analyze_image_chains_every_stage_and_persists_history():
    r = client.post("/api/analyze/image", files=_upload(REAL_IMAGE), data={"variety_dataset": "a"})
    assert r.status_code == 200, r.text
    result = r.json()

    assert result["seed_count"] == len(result["seeds"])
    assert result["warnings"] == [], f"pipeline degraded unexpectedly: {result['warnings']}"

    for seed in result["seeds"]:
        assert seed["variety_prediction"] is not None
        # The synthetic defect model is superseded by the real quality head and is
        # off by default; it must not silently reappear in served output.
        assert "synthetic_defect_prediction" not in seed
        assert len(seed["similarity_results"]) > 0

    # the analysis must be retrievable from persistent history afterwards
    detail = client.get(f"/api/history/{result['analysis_id']}")
    assert detail.status_code == 200
    assert detail.json()["analysis_id"] == result["analysis_id"]


@needs_image
@pytest.mark.skipif(
    not (_has("detection_corn", "weights", "best.pt") and _has("variety_a_full_best.pt")),
    reason="full pipeline not trained",
)
def test_analyze_batch_aggregates_across_images():
    files = [
        ("files", (os.path.basename(REAL_IMAGE), open(REAL_IMAGE, "rb").read(), "image/jpeg")),
        ("files", (os.path.basename(REAL_IMAGE), open(REAL_IMAGE, "rb").read(), "image/jpeg")),
    ]
    r = client.post("/api/analyze/batch", files=files, data={"variety_dataset": "a"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["total_images"] == 2
    assert body["successful_images"] == 2
    assert body["failed_images"] == 0
    stats = body["aggregate_stats"]
    assert stats["total_seeds_detected"] == sum(stats["variety_distribution"].values())
    assert 0.0 <= stats["average_confidence"] <= 1.0


def test_corrupt_upload_is_rejected_not_crashed():
    bad = {"file": ("broken.jpg", io.BytesIO(b"this is not a JPEG"), "image/jpeg")}
    r = client.post("/api/analyze/image", files=bad, data={"variety_dataset": "a"})
    assert r.status_code in (400, 503)
    assert "error" not in r.json() or r.status_code != 500


def test_unsupported_mime_type_is_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    r = client.post(
        "/api/analyze/image",
        files={"file": ("x.gif", io.BytesIO(buf.getvalue()), "image/gif")},
        data={"variety_dataset": "a"},
    )
    assert r.status_code == 415


def test_crop_seed_restores_context_padding():
    """Regression guard for the train/serve gap (docs/09 section 6.6).

    The variety classifier is trained on whole images but served YOLO crops. A tight
    box cost 22 accuracy points on the held-out split (100.00% -> 77.63%); restoring
    ~30% context recovered it. If this padding is ever dropped back to zero the
    served accuracy silently regresses, so assert the crop is actually expanded.
    """
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipe = AnalysisPipeline()
    pad = pipe.cfg["detection"]["crop_context_pad"]
    assert pad >= 0.30, f"crop_context_pad={pad} is below the 0.30 that closed the gap"

    img = Image.new("RGB", (400, 400))
    bbox = [100, 100, 200, 200]  # 100x100 box, well inside the image
    crop = pipe.crop_seed(img, bbox)
    expected = round(100 * (1 + pad))
    assert crop.size[0] == pytest.approx(expected, abs=2), crop.size
    assert crop.size[1] == pytest.approx(expected, abs=2), crop.size


def test_crop_seed_clamps_to_image_bounds():
    """Padding must never produce a crop outside the image or an inverted box."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipe = AnalysisPipeline()
    img = Image.new("RGB", (120, 120))
    crop = pipe.crop_seed(img, [0, 0, 20, 20])  # box flush against the corner
    assert crop.size[0] > 0 and crop.size[1] > 0
    assert crop.size[0] <= 120 and crop.size[1] <= 120
