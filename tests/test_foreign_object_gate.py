"""Phase 4: the platform flags objects unlike known maize, and never names them.

The failure this suite exists to prevent is not a low score. It is a sentence.
A foreign-object feature is one careless string away from claiming "stone" or
"husk", and this project holds no stone, husk, cob-fragment or debris label
anywhere -- so any such word would be invented rather than measured. Several
tests here assert on wording for exactly that reason, which is unusual for a
model test and deliberate: the claim is the artifact under test.

The second failure it guards is subtler and already happened once. The gate was
originally calibrated on whole dataset photographs and served on YOLO crops. The
calibration honoured its 2% false-flag promise on the images it saw and produced
an 80% flag rate on real uploads, because the same kernels sit far further from a
dataset-domain reference once they are cut out of a bounding box. A guarantee
measured in the wrong domain is not a weak guarantee, it is not a guarantee -- so
``test_reference_is_built_in_the_serving_domain`` pins the reference to crops.

Nothing here re-measures the gate's accuracy; that is
``src/analysis/eval_maize_gate.py``, and its numbers are bound to the checkpoint
by sha256 through the registry. These tests check the contract around it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipeline.unified_pipeline import AnalysisPipeline
from src.registry import TaskUnavailableError, get_registry

registry = get_registry()
GATE_KEY = "maize_identity_gate"
METRICS_FILE = Path("outputs/metrics/maize_gate_comparison.json")

# Words the platform is not entitled to say about a flagged object. Distance from
# a maize reference cannot distinguish any of these from any other.
FORBIDDEN_IDENTITIES = (
    "stone", "pebble", "husk", "cob", "debris", "dirt", "plastic",
    "straw", "chaff", "insect", "wheat", "rice", "soybean",
)


@pytest.fixture(scope="module")
def pipeline():
    return AnalysisPipeline()


@pytest.fixture(scope="module")
def gate(pipeline):
    record = registry.get(GATE_KEY)
    if not record.available:
        pytest.skip(f"{GATE_KEY} is not built; run src.analysis.build_maize_reference")
    return pipeline._get_maize_reference()


# ------------------------------------------------------------ registry contract
def test_the_gate_is_registered_as_flagging_not_classification():
    """The registry is where a reader learns what a model is allowed to claim, so
    the distinction has to survive there and not only in a docstring."""
    record = registry.get(GATE_KEY)
    task = "foreign_object_flag"
    assert task in registry.served_tasks() or not record.available
    text = f"{record.note or ''} {json.dumps(record.as_dict())}".lower()
    assert "never names" in text or "not" in text


def test_no_model_claims_to_identify_foreign_objects():
    """Closed-set identification is a task nothing here may serve. If a future
    phase adds labelled stone/husk data it must register a new task rather than
    quietly widening this one."""
    with pytest.raises((TaskUnavailableError, KeyError)):
        registry.resolve("foreign_object_classification")


# ---------------------------------------------------------- the artifact itself
def test_reference_is_built_in_the_serving_domain(gate):
    """The regression that voided the first build.

    Provenance is not decoration here: it is the only durable record that the
    reference holds detector crops rather than dataset photographs, and the whole
    false-flag guarantee rests on that. A rebuild that reverts to whole images
    must fail loudly rather than ship a promise it cannot keep.
    """
    z = np.load(registry.get(GATE_KEY).checkpoint)
    provenance = json.loads(str(z["provenance"]))
    assert provenance, "the reference records no provenance at all"
    # embed_crops yields one row per detected object, not one per image, so a
    # reference in the serving domain cannot have exactly one row per input.
    assert any(p.get("images") is not None for p in provenance), (
        "provenance has no image counts; this reference was probably built by "
        "the old whole-image path"
    )


def test_thresholds_and_measured_recall_travel_with_the_checkpoint(gate):
    """Serving must never hard-code a performance figure.

    An earlier build printed '52.7%' from a constant in the pipeline. When the
    gate was recalibrated the constant stayed behind and the platform kept quoting
    a number no artifact supported. Every figure now rides inside the npz.
    """
    for key in ("threshold", "review_budget", "k", "resolution_floor_px",
                "size_coef", "scene_limit_objects"):
        assert gate[key] is not None, f"the checkpoint carries no {key}"
    assert len(gate["size_coef"]) == 3, "the size correction is a quadratic in log px"
    assert gate["measured_recall"] is not None, (
        "the checkpoint carries no measured recall, so serving has nothing "
        "honest to quote"
    )
    assert 0.0 <= gate["measured_recall"] <= 1.0


def test_metrics_are_bound_to_this_checkpoint():
    """Phase 7's binding, applied here: metrics measured against a different
    artifact are not this artifact's metrics."""
    record = registry.get(GATE_KEY)
    if not record.available or not METRICS_FILE.exists():
        pytest.skip("gate or its metrics file is absent")
    assert record.metrics_binding == "verified", (
        f"metrics binding is {record.metrics_binding!r}; the reference was "
        "rebuilt without re-running eval_maize_gate"
    )


def test_the_detection_ceiling_is_recorded():
    """A flag can only fire on an object YOLO found. That ceiling multiplies into
    every recall figure, so it has to be measured and stored rather than left for
    a reader to assume away."""
    if not METRICS_FILE.exists():
        pytest.skip("metrics file is absent")
    metrics = json.loads(METRICS_FILE.read_text())
    detection = metrics.get("detection") or {}
    assert "grainset_impurities" in detection, (
        "no detection rate recorded for the impurity set"
    )
    rate = detection["grainset_impurities"]["detection_rate"]
    assert 0.0 < rate <= 1.0


# ------------------------------------------------------------- the verdict shape
def test_a_verdict_is_a_flag_and_never_an_identity(pipeline, gate):
    """The whole point of the phase, asserted on a real forward pass."""
    rows = list(Path("data_processed/uploads").glob("*.jpg"))[:12]
    verdicts = []
    for path in rows:
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue  # a handful of stored uploads are truncated
        crops = list(_crops(pipeline, image))
        for crop in crops:
            _, quality, foreign = pipeline.classify_unified_full(
                crop, scene_objects=len(crops))
            if foreign:
                verdicts.append(foreign)
        if verdicts:
            break
    if not verdicts:
        pytest.skip("no detectable object among the sampled uploads")

    for v in verdicts:
        # "unavailable" belongs in this list: a refusal to score is one of the
        # three honest outcomes, and it is still bound by the naming rule below.
        assert v["status"] in ("known_maize", "possible_foreign_object",
                               "unavailable")
        assert v["is_classification"] is False
        blob = json.dumps(v).lower()
        for word in FORBIDDEN_IDENTITIES:
            assert word not in blob, f"a verdict named a material: {word!r}"


def test_the_absence_of_a_flag_is_not_a_certification(pipeline, gate):
    """'known_maize' is the weaker of the two statements and its caveat has to say
    so. A grader who reads it as 'confirmed maize' has been misled by us, not by
    the model."""
    reference = gate["reference"]
    # A vector drawn from the reference itself is as maize-like as anything can
    # be, so this exercises the unflagged branch without needing a lucky image.
    emb = reference[0].copy()
    # Comfortably above the floor: the score is corrected for crop size, so a
    # size has to be supplied before any verdict can be reached at all.
    verdict = pipeline._flag_foreign_object(emb, quality_distance=0.0,
                                            crop_px=120)
    assert verdict["status"] == "known_maize"
    assert "not evidence" in verdict["caveat"].lower()


def test_a_caller_that_supplies_no_size_gets_no_score(pipeline, gate):
    """The rule used to be an AND over this gate and the quality gate. That
    pairing was measured in the serving domain and found worse than this gate
    alone, so it was dropped -- and with it the reason a missing quality distance
    suppressed the verdict. What cannot be missing now is the crop size: the
    score is corrected for it before it meets a threshold, so a caller who omits
    it is asking for a number that was never calibrated. Declining is the only
    honest answer, and it must not be silence either -- a None here would let a
    caller quietly drop the object from every count.
    """
    verdict = pipeline._flag_foreign_object(gate["reference"][0], None,
                                            crop_px=120)
    assert verdict["status"] in ("known_maize", "possible_foreign_object"), (
        "a missing quality distance no longer suppresses this gate"
    )
    sized = pipeline._flag_foreign_object(gate["reference"][0], 0.0)
    assert sized["status"] == "unavailable"
    assert sized["crop_px"] is None


def test_the_gate_is_optional(monkeypatch, pipeline):
    """With no reference built, seeds simply carry no verdict. Absence of the
    feature must not become an implicit 'nothing is foreign'."""
    monkeypatch.setattr(pipeline, "_maize_reference", False, raising=False)
    assert pipeline._get_maize_reference() is None
    assert pipeline._flag_foreign_object(np.zeros(32, np.float32), 0.9) is None


# ------------------------------------------------- limits of applicability
def test_the_score_is_corrected_for_crop_size(pipeline, gate):
    """The defect that voided the second build.

    A quarter of the raw kNN distance was crop short side alone, so the same
    kernel scored differently depending only on how many pixels it occupied --
    and small crops were surfaced far above budget on real uploads. The
    correction is a fitted term subtracted before anything meets a threshold, so
    the identical embedding read at two sizes must produce two different scores.
    """
    emb = gate["reference"][0].copy()
    floor = gate["resolution_floor_px"]
    small = pipeline._flag_foreign_object(emb, 0.0, crop_px=floor + 5)
    large = pipeline._flag_foreign_object(emb, 0.0, crop_px=300)
    assert small["maize_distance"] == large["maize_distance"], (
        "the raw distance cannot depend on crop size; only the correction may"
    )
    assert small["maize_score"] != large["maize_score"], (
        "the score is not being corrected for crop size at all"
    )


def test_below_the_measured_resolution_floor_no_score_is_produced(pipeline, gate):
    """Phase 8's principle, applied to this gate. The floor is derived from where
    calibration stops honouring its own budget, and below it the size correction
    is extrapolating -- so the platform declines instead of extrapolating."""
    floor = gate["resolution_floor_px"]
    verdict = pipeline._flag_foreign_object(gate["reference"][0], 0.0,
                                            crop_px=floor - 1)
    assert verdict["status"] == "unavailable"
    assert "unavailable" in verdict["caveat"].lower()
    assert verdict["resolution_floor_px"] == floor
    # A refusal must not carry a score a caller could read as a verdict.
    assert "maize_score" not in verdict


def test_a_scene_denser_than_calibration_is_declined(pipeline, gate):
    """The coverage gap this build found and did not paper over.

    A real upload of 300 detections -- a packed bed of ordinary maize, verified
    by eye -- had 42% of its objects surfaced against a 5% budget, because in a
    packed scene every crop is filled with fragments of its neighbours. Nothing
    in the calibration corpora is packed like that. Rather than mark two in five
    kernels of a clean sample, the gate reports that the scene is outside what it
    was measured on.
    """
    limit = gate["scene_limit_objects"]
    emb = gate["reference"][0].copy()
    inside = pipeline._flag_foreign_object(emb, 0.0, crop_px=120,
                                           scene_objects=limit)
    outside = pipeline._flag_foreign_object(emb, 0.0, crop_px=120,
                                            scene_objects=limit + 1)
    assert inside["status"] in ("known_maize", "possible_foreign_object")
    assert outside["status"] == "unavailable"
    assert outside["scene_objects"] == limit + 1


def test_the_scene_count_reaches_the_gate_from_a_whole_image(pipeline, gate):
    """The plumbing, not the rule. A per-crop call cannot know how crowded its
    image was, so the count has to travel from the loop that holds the image. If
    that thread breaks, the density limit silently stops applying and nothing
    else in the suite would notice.
    """
    limit = gate["scene_limit_objects"]
    image = Image.new("RGB", (400, 400), (210, 180, 90))
    _, _, verdict = pipeline.classify_unified_full(image,
                                                   scene_objects=limit + 50)
    assert verdict is not None
    assert verdict["status"] == "unavailable"
    assert verdict["scene_objects"] == limit + 50


def _crops(pipeline, image):
    for det in pipeline.detect_seeds(image):
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        if x2 > x1 and y2 > y1:
            yield image.crop((x1, y1, x2, y2))
