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


needs_unified_model = pytest.mark.skipif(
    not _has("unified_seed_model_best.pt"), reason="unified model not trained"
)
needs_index_a = pytest.mark.skipif(
    not _has("faiss_variety_a.faiss"), reason="FAISS index A not built"
)
needs_index_unified = pytest.mark.skipif(
    not _has("faiss_seed_unified.faiss"), reason="unified FAISS index not built"
)


@needs_image
@needs_index_a
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
@needs_index_unified
def test_similarity_endpoint_serves_the_unified_gallery():
    """Regression: the endpoint used to 400 on the platform's own default model."""
    r = client.post(
        "/api/similarity", files=_upload(REAL_IMAGE), data={"variety_dataset": "unified", "top_k": 4}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["encoder"] == "unified_seed_model"
    assert body["gallery_size"] > 0
    assert len(body["results"]) == 4, "top_k must be honoured exactly"

    for n in body["results"]:
        # every neighbour carries its own provenance, and a label the source dataset
        # never had stays null instead of being filled in
        assert set(n) >= {"path", "variety_label", "quality_label", "source", "distance"}
        assert n["variety_label"] is None or isinstance(n["variety_label"], str)
        assert n["quality_label"] in (None, "Good", "Bad")


@needs_image
def test_similarity_endpoint_rejects_an_unknown_gallery_rather_than_substituting_one():
    for data, expected in [
        ({"variety_dataset": "zzz"}, "variety_dataset"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": 999}, "top_k"),
    ]:
        r = client.post("/api/similarity", files=_upload(REAL_IMAGE), data=data)
        assert r.status_code == 400, f"{data} -> {r.status_code} {r.text}"
        assert expected in r.json()["detail"]


@needs_index_a
@needs_index_unified
def test_index_refuses_to_be_queried_by_a_different_encoder():
    """The core Phase 13 defect: A's index and the unified model are BOTH 1280-d, so
    a mismatch produced confident nonsense instead of an error."""
    from src.similarity.embedding_index import EmbeddingIndex, IndexEncoderMismatch, index_path

    a_path = index_path(cfg, "a")
    # same width, different feature space -> must be refused on the name, not the dim
    with pytest.raises(IndexEncoderMismatch):
        EmbeddingIndex.load(a_path, dim=1280, encoder="unified_seed_model")
    with pytest.raises(IndexEncoderMismatch):
        EmbeddingIndex.load(a_path, dim=512, encoder="variety_a_full")

    ok = EmbeddingIndex.load(a_path, dim=1280, encoder="variety_a_full")
    assert ok.index.ntotal == len(ok.metadata), "vector/metadata mapping must be 1:1"


@needs_index_unified
def test_unified_gallery_is_the_train_split_and_labels_are_never_invented():
    """The gallery must not contain evaluation images, and the quality-only subset
    must stay variety-null rather than borrowing a label."""
    import csv

    from src.similarity.embedding_index import EmbeddingIndex, index_path

    idx = EmbeddingIndex.load(index_path(cfg, "unified"), dim=1280, encoder="unified_seed_model")
    assert idx.info["gallery_split"] == "train"

    with open(idx.info["gallery_manifest"]) as f:
        rows = list(csv.DictReader(f))
    truth = {r["filepath"]: r for r in rows}
    held_out = {r["filepath"] for r in rows if r["split"] in ("val", "test")}

    assert len(idx.metadata) == sum(1 for r in rows if r["split"] == "train")
    for m in idx.metadata:
        assert m["path"] not in held_out, f"evaluation image in the gallery: {m['path']}"
        row = truth[m["path"]]
        # every stored label is the manifest's, verbatim; blanks stay None
        assert m["variety_label"] == (row["variety_label"].strip() or None)
        assert m["quality_label"] == (row["quality_label"].strip() or None)

    assert any(m["variety_label"] is None for m in idx.metadata), (
        "the quality-only subset should be present and variety-null"
    )


@needs_image
@needs_unified_model
@needs_index_unified
def test_pipeline_searches_the_gallery_of_the_model_that_made_the_prediction():
    """Regression: analyze_image used to answer unified queries from dataset A's
    3-variety gallery, so a seed predicted WangDataa got neighbours from a gallery
    that structurally could not contain WangDataa."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    result = pipeline.analyze_image(REAL_IMAGE, variety_dataset="unified", top_k=3)
    assert result["warnings"] == [], result["warnings"]
    assert result["seeds"]

    gallery = pipeline._get_faiss_index("unified")
    assert gallery.info["encoder"] == "unified_seed_model"

    a_paths = {m["path"] for m in pipeline._get_faiss_index("a").metadata}
    unified_paths = {m["path"] for m in gallery.metadata}
    assert unified_paths - a_paths, "fixture precondition: the two galleries must differ"

    for seed in result["seeds"]:
        assert len(seed["similarity_results"]) == 3
        d = [n["distance"] for n in seed["similarity_results"]]
        assert d == sorted(d)
        for n in seed["similarity_results"]:
            # drawn from the unified gallery, never from A's 3-variety one
            assert n["path"] in unified_paths

    # Which of those images a given photograph actually retrieves is a property of
    # the photograph, so it is not asserted. What must hold is that there is no
    # fallback left: a gallery that does not exist is reported, never substituted.
    with pytest.raises(Exception):
        pipeline._get_faiss_index("no_such_dataset")


@needs_image
@needs_unified_model
@needs_index_unified
def test_similarity_never_writes_a_neighbour_label_onto_the_query():
    """Neighbour labels are evidence about the neighbours. The seed's own prediction
    must come from the classifier alone."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    result = pipeline.analyze_image(REAL_IMAGE, variety_dataset="unified", top_k=5)
    for seed in result["seeds"]:
        assert seed["variety_prediction"]["model"] == "unified_seed_model"
        assert "similarity" not in seed["variety_prediction"]
        assert "neighbour" not in str(seed["variety_prediction"]).lower()


needs_unified = needs_unified_model
needs_variety_a = pytest.mark.skipif(
    not _has("variety_a_full_best.pt"), reason="variety model A not trained"
)


@needs_image
@needs_unified
def test_gradcam_heatmap_has_image_shape_and_normalised_range():
    """The CAM itself: right shape, right range, and honest metadata."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    cam, meta = pipeline.generate_gradcam(image)

    assert cam.ndim == 2
    assert cam.min() >= 0.0 and cam.max() <= 1.0

    assert meta["model"] == "unified_seed_model"
    assert meta["head"] == "variety"
    # The CAM is taken after the cognitive-attention block, which is the part of the
    # architecture the project actually makes a claim about.
    assert meta["target_layer"] == "attention"
    assert isinstance(meta["class_index"], int)
    assert meta["is_segmentation_mask"] is False


@needs_image
@needs_unified
def test_gradcam_out_size_upsamples_the_cam_rather_than_shrinking_the_image():
    """The overlay must come back at the resolution the user uploaded."""
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    w, h = image.size
    cam, _meta = pipeline.generate_gradcam(image, out_size=(h, w))
    assert cam.shape == (h, w)


@needs_image
@needs_unified
def test_gradcam_heads_explain_different_pixels():
    """A CAM is defined per-logit: the two heads must not return the same map.

    If they did, the head selector in the UI would be decorative and the caption
    "explains the quality prediction" would be false.
    """
    import numpy as np

    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    variety_cam, variety_meta = pipeline.generate_gradcam(image, head="variety")
    quality_cam, quality_meta = pipeline.generate_gradcam(image, head="quality")

    assert quality_meta["head"] == "quality"
    assert quality_meta["model"] != variety_meta["model"]
    assert float(np.abs(variety_cam - quality_cam).mean()) > 1e-6


@needs_image
@needs_unified
def test_gradcam_leaves_no_hooks_and_no_mode_change_on_the_cached_model():
    """Regression: the old hook-based implementation never released its hooks.

    The pipeline caches model instances, so a leak here degraded ordinary
    classification for every later request in the process.
    """
    from src.pipeline.unified_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    image = pipeline.load_and_validate_image(REAL_IMAGE)
    pipeline.generate_gradcam(image)

    model, _v, _q = pipeline._get_unified_model()

    def hook_count():
        return sum(
            len(m._forward_hooks) + len(m._backward_hooks) + len(m._forward_pre_hooks)
            for m in model.modules()
        )

    before = hook_count()
    for _ in range(3):
        pipeline.generate_gradcam(image)
    assert hook_count() == before
    assert model.training is False


@needs_image
@needs_unified
def test_gradcam_endpoint_returns_png_with_the_not_segmentation_disclaimer():
    r = client.post("/api/explain/gradcam", files=_upload(REAL_IMAGE))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    note = r.headers["X-Explainability-Note"].lower()
    assert "not a segmentation mask" in note
    assert "not a defect area" in note

    # which model / head / layer / region produced this figure must be reported, not
    # left for the caller to assume
    assert r.headers["X-Gradcam-Model"] == "unified_seed_model"
    assert r.headers["X-Gradcam-Head"] == "variety"
    assert r.headers["X-Gradcam-Target-Layer"] == "attention"
    assert r.headers["X-Gradcam-Region"] == "whole_image"
    assert r.headers["X-Gradcam-Predicted-Class"]

    # the overlay keeps the uploaded resolution rather than the CAM's coarse grid
    with open(REAL_IMAGE, "rb") as f:
        original = Image.open(io.BytesIO(f.read())).size
    overlay = Image.open(io.BytesIO(r.content))
    assert overlay.size == original


@needs_image
@needs_unified
def test_gradcam_endpoint_explains_the_quality_head_when_asked():
    r = client.post(
        "/api/explain/gradcam", files=_upload(REAL_IMAGE), data={"head": "quality"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["X-Gradcam-Head"] == "quality"
    assert r.headers["X-Gradcam-Model"] == "unified_seed_model_quality"


@needs_image
@needs_variety_a
def test_gradcam_endpoint_still_serves_the_ablation_checkpoints():
    """dataset a/b stay reachable so the ablation figures remain reproducible."""
    r = client.post(
        "/api/explain/gradcam", files=_upload(REAL_IMAGE), data={"variety_dataset": "a"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["X-Gradcam-Model"].startswith("variety_a_")

    # ...but they have no quality head, and asking for one is a bad request rather
    # than a silent fall back to variety
    r = client.post(
        "/api/explain/gradcam",
        files=_upload(REAL_IMAGE),
        data={"variety_dataset": "a", "head": "quality"},
    )
    assert r.status_code == 400, r.text
    assert "quality" in r.json()["detail"]


@needs_image
@needs_unified
def test_gradcam_endpoint_scopes_the_heatmap_to_a_seed_bbox():
    """With a bbox the CAM explains ONE kernel, not everything in frame."""
    with open(REAL_IMAGE, "rb") as f:
        w, h = Image.open(io.BytesIO(f.read())).size

    box = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
    r = client.post(
        "/api/explain/gradcam",
        files=_upload(REAL_IMAGE),
        data={"bbox": ",".join(str(v) for v in box)},
    )
    assert r.status_code == 200, r.text
    assert r.headers["X-Gradcam-Region"] == "seed_bbox"
    # the crop (plus the classifier's context padding) is smaller than the full frame
    assert Image.open(io.BytesIO(r.content)).size[0] < w


@needs_image
def test_gradcam_endpoint_rejects_bad_requests_instead_of_guessing():
    cases = [
        ({"variety_dataset": "zzz"}, "variety_dataset"),
        ({"head": "zzz"}, "head"),
        ({"bbox": "1,2,3"}, "bbox"),
        ({"bbox": "0,0,999999,999999"}, "outside"),
        ({"bbox": "300,300,10,10"}, "x2>x1"),
    ]
    for data, expected in cases:
        r = client.post("/api/explain/gradcam", files=_upload(REAL_IMAGE), data=data)
        assert r.status_code == 400, f"{data} -> {r.status_code} {r.text}"
        assert expected in r.json()["detail"], f"{data} -> {r.json()['detail']}"


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
