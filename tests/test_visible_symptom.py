"""Phase 5: the platform names a visible condition, or says why it would not.

The failure this suite exists to prevent is a sentence, not a score. A feature
that outputs "FM" is one careless label away from reading as a fusarium
diagnosis, and this project holds no pathogen, toxin or species label anywhere.
The labels are GrainSpace's expert GRADING vocabulary -- a human looked at a
kernel and filed it under a category by appearance -- so every layer carries
``is_diagnosis: False`` as a field rather than only as prose, and several tests
here assert on wording. That is unusual for a model test and deliberate: the
claim is the artifact under test.

The second failure is quieter and is the one that would actually reach a user.
Five of the seven classes may be asserted; HD holds 9 validation crops and SD
holds 2, which is below the support at which an operating point can be measured
at all, so when either wins the argmax the answer is withheld. A softmax floor
withholds the rest. That means a large share of kernels come back with no
category -- and a withheld kernel is NOT a clean kernel. Anything that folds an
abstention into NOR converts a silence into a clean result, which is the single
most damaging thing this feature could do to a grader's decision. Tests here
chase that one distinction through the classifier, the orchestrator's counts and
the database row.

The third is the reason the model ships as its own checkpoint. The fine-tune
that made the symptom head usable measurably wrecked variety (0.9316 -> 0.5048)
and quality (0.9721 -> 0.5800) in the same trunk, so the served symptom weights
are separate and ``unified_seed_model_best.pt`` is untouched. If a later change
quietly points both tasks at one file, two shipped capabilities regress with
nothing to announce it.

Nothing here re-measures accuracy; that is ``src.analysis.calibrate_symptom``,
and its numbers are bound to the checkpoint by sha256 through the registry.
These tests check the contract around it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.registry import TaskUnavailableError, get_registry

registry = get_registry()
KEY = "visible_symptom_classifier"
TASK = "visible_symptom"
GATE_FILE = Path("outputs/checkpoints/symptom_gate.json")
SERVING_DOMAIN_FILE = Path("outputs/metrics/symptom_serving_domain.json")
MANIFEST = Path("data_processed/manifest_symptom.csv")

# Claims this capability is not entitled to make. An image-level grading label
# cannot support any of them, and unlike the grader category names ("fusarium &
# mildew" is the name of a bucket a human ticked) none of these words has an
# honest use in a verdict about appearance.
FORBIDDEN_CLAIMS = (
    "aflatoxin", "mycotoxin", "infected", "infection", "diagnosed",
    "contaminat", "viability", "germination rate", "confirmed presence",
)


@pytest.fixture(scope="module")
def gate():
    if not GATE_FILE.exists():
        pytest.skip("symptom gate is not built; run src.analysis.calibrate_symptom")
    return json.loads(GATE_FILE.read_text())


@pytest.fixture(scope="module")
def pipeline():
    from src.pipeline.unified_pipeline import AnalysisPipeline
    return AnalysisPipeline()


@pytest.fixture(scope="module")
def verdicts(pipeline):
    """Real forward passes over the held-out split of the labelled manifest.

    GrainSpace crops rather than upload crops on purpose: these are the images
    the operating point was measured on, so a reported verdict here is one the
    gate is entitled to produce. The serving-domain shift is a separate concern
    and is measured in its own artifact, asserted further down.
    """
    if not MANIFEST.exists():
        pytest.skip("symptom manifest is absent; run src.data.build_symptom_manifest")
    rows = [r for r in csv.DictReader(MANIFEST.open()) if r["split"] == "test"]
    out = []
    # Enough to reach both branches of the gate without turning the suite into a
    # second evaluation run; the split is class-ordered, so stride rather than
    # slice or every sample would come from one category.
    for row in rows[::4]:
        path = Path(row["filepath"])
        if not path.exists():
            continue
        with Image.open(path) as im:
            out.append((row["symptom_label"], pipeline.classify_symptom(im.convert("RGB"))))
    if not out:
        pytest.skip("no readable crops in the held-out split")
    return out


# ------------------------------------------------------------ registry contract
def test_the_symptom_model_is_registered_and_served():
    record = registry.get(KEY)
    if not record.available:
        pytest.skip(f"{KEY} is not built; run src.training.train_symptom")
    assert TASK in registry.served_tasks()
    assert registry.resolve(TASK).key == KEY


def test_the_symptom_weights_are_not_the_unified_model_weights():
    """The whole reason this is a separate checkpoint.

    The joint fine-tune that made the symptom head usable cost variety 43 points
    of macro-F1 and quality 39 points of the same trunk. Serving both tasks from
    one file would mean shipping that trade silently, so the two records must
    never converge on the same artifact.
    """
    symptom = registry.get(KEY)
    unified = registry.get("unified_seed_model")
    assert Path(symptom.checkpoint) != Path(unified.checkpoint), (
        "the symptom model and the unified model now point at one checkpoint; "
        "one of the two capabilities has silently been retrained away"
    )


def test_the_registry_publishes_the_classes_that_are_never_asserted():
    """A served model's silences are part of its specification. A reader who sees
    seven classes and no note has been told the model can assert seven."""
    record = registry.get(KEY)
    if not record.available:
        pytest.skip(f"{KEY} is not built")
    declared = record.as_dict().get("withheld_classes")
    assert declared, "the registry lists no withheld classes for a gated model"
    assert set(declared) == {"HD", "SD"}


def test_metrics_are_bound_to_this_checkpoint():
    """Phase 7's binding: metrics measured against different weights are not this
    model's metrics."""
    record = registry.get(KEY)
    if not record.available:
        pytest.skip(f"{KEY} is not built")
    assert record.metrics_binding == "verified", (
        f"metrics binding is {record.metrics_binding!r}; the checkpoint was "
        "retrained without re-running the evaluation that describes it"
    )


def test_no_task_offers_disease_diagnosis():
    """Closed-set aetiology is a task nothing here may serve. If a later phase
    ever obtains pathogen-confirmed labels it must register a new task rather
    than quietly widen this one."""
    for task in ("disease_diagnosis", "pathogen_identification", "mycotoxin_risk"):
        with pytest.raises((TaskUnavailableError, KeyError)):
            registry.resolve(task)


# ---------------------------------------------------------------- the gate file
def test_the_gate_asserts_a_subset_of_the_classes_it_holds(gate):
    classes = set(gate["classes"])
    validated = set(gate["validated_classes"])
    withheld = set(gate["withheld_classes"])
    assert validated < classes, "every class is asserted; the gate is not gating"
    assert not (validated & withheld), "a class is both asserted and withheld"
    assert validated | withheld == classes, (
        "a class is neither asserted nor explicitly withheld, so nothing records "
        "what happens when it wins the argmax"
    )


def test_every_withheld_class_earned_its_silence(gate):
    """The gate declares a rule; this checks it applied that rule rather than a
    hand-picked list. A class is withheld for want of support or of measured
    quality, and the stored reason has to name which."""
    per_class = gate["validation"]["per_class"]
    for name, reason in gate["withheld_classes"].items():
        support = per_class[name]["support"]
        f1 = per_class[name]["f1"]
        assert support < 10 or f1 < 0.40, (
            f"{name} is withheld but has {support} validation crops at F1 {f1:.2f}, "
            "which clears the published rule; the withheld list is out of step "
            "with the rule it claims to follow"
        )
        assert str(support) in reason, (
            f"the recorded reason for {name} does not state its support"
        )


def test_the_threshold_was_chosen_by_the_published_rule(gate):
    """0.55 is not a round number someone liked. It is the smallest floor whose
    retained validation accuracy reaches 0.85, and the sweep that produced it
    ships beside it so the choice can be re-derived rather than trusted."""
    tau = gate["confidence_threshold"]
    sweep = gate["validation"]["sweep"]
    target = 0.85
    reachable = [s for s in sweep if s["accuracy"] >= target]
    assert reachable, "no floor in the sweep reaches the target accuracy"
    assert tau == min(s["threshold"] for s in reachable), (
        "the served threshold is not the smallest one the published selection "
        "rule would have chosen"
    )
    assert str(target) in gate["selection_rule"]


def test_the_served_accuracy_is_the_held_out_one(gate):
    """The threshold was chosen on validation, so validation accuracy at that
    threshold flatters it by construction. Serving must quote the split the
    choice was not made on."""
    served = gate["test_at_threshold"]
    val = gate["validation"]["at_threshold"]
    assert served["threshold"] == val["threshold"] == gate["confidence_threshold"]
    transfer = gate["calibration_transfer"]
    assert transfer["test_accuracy"] == served["accuracy"]
    assert transfer["optimism"] == pytest.approx(
        val["accuracy"] - served["accuracy"], abs=1e-9)


def test_the_optimism_gap_travels_with_the_accuracy(pipeline, gate):
    """A coverage-and-accuracy pair quoted without the gap beside it reads as a
    property of the model rather than of the split it was tuned on."""
    status = pipeline.symptom_status()
    if status["status"] != "available":
        pytest.skip(status["message"])
    assert status["held_out_accuracy"] is not None
    assert status["calibration_optimism"] is not None
    assert status["calibration_optimism"] == gate["calibration_transfer"]["optimism"]


# ---------------------------------------------------------------- what it says
def test_the_capability_never_claims_to_diagnose(pipeline):
    """The flag is a field, not a sentence, so a consumer cannot drop it by
    reformatting the prose."""
    status = pipeline.symptom_status()
    if status["status"] != "available":
        pytest.skip(status["message"])
    assert status["is_diagnosis"] is False
    blob = json.dumps(status).lower()
    for word in FORBIDDEN_CLAIMS:
        assert word not in blob, f"the capability statement makes a claim it cannot: {word!r}"
    # The denial has to be present, not merely the absence of an assertion.
    assert "not disease diagnosis" in status["note"].lower()


def test_the_grader_vocabulary_is_labelled_as_appearance(pipeline):
    """FM is the name of a bucket a human ticked. The description may use the
    grader's words -- inventing new ones would misreport the label set -- but it
    must be framed as appearance, and the payload must still deny diagnosis."""
    status = pipeline.symptom_status()
    if status["status"] != "available":
        pytest.skip(status["message"])
    descriptions = status["class_descriptions"]
    assert set(descriptions) == set(status["classes"]), (
        "a served class has no description, so the UI would show a bare code"
    )
    fm = descriptions["FM"].lower()
    assert "fusarium" in fm, "the grader's own category name has been replaced"
    assert "discolouration" in fm or "discoloration" in fm or "type" in fm, (
        "FM is described as a finding rather than as an appearance category"
    )
    assert status["is_diagnosis"] is False


# -------------------------------------------------------------- verdict shape
def test_a_withheld_verdict_carries_no_category(verdicts):
    """The property the whole gate exists to guarantee, asserted on real forward
    passes rather than on a constructed payload."""
    seen = set()
    for _, v in verdicts:
        assert v["status"] in ("reported", "withheld")
        assert v["is_diagnosis"] is False
        seen.add(v["status"])
        if v["status"] == "withheld":
            assert v["predicted_class"] is None
            assert v["description"] is None
            assert v["confidence"] is None
            assert v["reason"] in ("class_not_validated", "low_confidence")
            # Not silence either: a caller has to be able to tell "the model
            # looked and would not commit" from "nothing ran".
            assert v["caveat"]
    assert "withheld" in seen, (
        "no crop in the held-out split was withheld; the gate is not applying"
    )


def test_a_reported_verdict_cleared_both_halves_of_the_gate(verdicts, gate):
    tau = gate["confidence_threshold"]
    validated = set(gate["validated_classes"])
    reported = [v for _, v in verdicts if v["status"] == "reported"]
    assert reported, "nothing was reported; the gate is refusing everything"
    for v in reported:
        assert v["predicted_class"] in validated, (
            f"{v['predicted_class']} was asserted but is not a validated class"
        )
        assert v["confidence"] >= tau
        assert v["predicted_class"] == v["argmax_class_before_gate"]


def test_the_ungated_argmax_is_kept_under_a_name_that_is_not_the_answer(verdicts):
    """Shown so an operator can see the system was not idle. It must never be
    reachable under a key a consumer would render as the verdict."""
    for _, v in verdicts:
        assert "argmax_class_before_gate" in v
        assert v["argmax_class_before_gate"] is not None
        if v["status"] == "withheld":
            assert v["argmax_class_before_gate"] not in (v.get("predicted_class"),)
            # The confidence of a prediction that was not made must not be
            # served under the key that means "confidence in the verdict".
            assert v["confidence"] is None
            assert v["argmax_confidence"] is not None


def test_every_verdict_carries_its_operating_point(verdicts, gate):
    """A caller summarising an image reads the operating point off whichever
    verdict it happens to hold, so it rides on all of them."""
    for _, v in verdicts:
        assert v["confidence_threshold"] == pytest.approx(gate["confidence_threshold"])
        assert set(v["validated_classes"]) == set(gate["validated_classes"])
        assert v["measured_accuracy"] == gate["test_at_threshold"]["accuracy"]
        assert v["measured_coverage"] == gate["test_at_threshold"]["coverage"]
        assert "grainspace" in v["label_source"].lower()


def test_no_verdict_makes_a_claim_the_labels_cannot_support(verdicts):
    for label, v in verdicts:
        blob = json.dumps(v).lower()
        for word in FORBIDDEN_CLAIMS:
            assert word not in blob, (
                f"a verdict on a {label} crop made a claim it cannot: {word!r}"
            )


# ------------------------------------------------------- counting the silences
def test_an_abstention_is_never_counted_as_a_clean_kernel(pipeline):
    """The orchestrator's tally, on constructed verdicts so the arithmetic is
    checkable. Adding a withheld kernel to NOR would turn "the model would not
    commit" into "this kernel is fine", which is the failure with the shortest
    path to a bad grading decision.
    """
    from src.pipeline.orchestrator import Orchestrator, RunState, StageOutput

    def verdict(status, cls=None, reason=None, argmax="NOR"):
        return {"status": status, "predicted_class": cls, "reason": reason,
                "argmax_class_before_gate": argmax}

    state = RunState(digest="d", analysis_id="a", image_path="x.jpg")
    state.stages["symptom"] = StageOutput(stage="symptom", status="ran", value={
        0: verdict("reported", "NOR"),
        1: verdict("reported", "BN"),
        2: verdict("withheld", reason="low_confidence"),
        3: verdict("withheld", reason="class_not_validated", argmax="HD"),
        4: verdict("withheld", reason="low_confidence"),
    })

    answer = Orchestrator(pipeline=pipeline)._answer_visible_symptom(state, [])
    assert answer["kernels_scored"] == 5
    assert answer["categorised"] == 2
    assert answer["withheld"] == 3
    assert answer["category_counts"] == {"NOR": 1, "BN": 1}, (
        "a withheld kernel reached a category count"
    )
    assert answer["categorised"] + answer["withheld"] == answer["kernels_scored"]
    assert answer["coverage"] == pytest.approx(0.4)
    assert sorted(answer["withheld_indices"]["low_confidence"]) == [2, 4]
    assert answer["withheld_indices"]["class_not_validated"] == [3]
    assert answer["is_diagnosis"] is False
    assert "not a healthy kernel" in answer["what_it_does_not_mean"]


def test_the_database_row_cannot_disagree_with_the_response():
    """``symptom_class`` is read straight off the payload, which already nulls it
    on a withheld verdict. Reconstructing it here -- from the argmax, from the
    status -- is how a stored column drifts away from what the caller was told.
    """
    from backend.services.history_service import _add_assessment_row

    class Collector:
        def __init__(self):
            self.rows = []

        def add(self, row):
            self.rows.append(row)

    db = Collector()
    _add_assessment_row(db, "analysis-1", {
        "seed_index": 0,
        "symptom_prediction": {
            "status": "withheld", "reason": "class_not_validated",
            "predicted_class": None, "confidence": None,
            "argmax_class_before_gate": "HD", "argmax_confidence": 0.53,
            "confidence_threshold": 0.55, "basis": "visible_symptom_classifier",
        },
    })
    _add_assessment_row(db, "analysis-1", {
        "seed_index": 1,
        "symptom_prediction": {
            "status": "reported", "reason": None,
            "predicted_class": "NOR", "confidence": 0.90,
            "argmax_class_before_gate": "NOR", "argmax_confidence": 0.90,
            "confidence_threshold": 0.55, "basis": "visible_symptom_classifier",
        },
    })
    withheld, reported = db.rows
    assert withheld.symptom_status == "withheld"
    assert withheld.symptom_class is None, "an abstention was stored as a category"
    assert withheld.symptom_confidence is None
    # The near-miss is kept, but only under the name that says what it is.
    assert withheld.symptom_argmax_class == "HD"
    assert reported.symptom_class == "NOR"
    assert reported.symptom_reason is None


def test_a_seed_with_no_symptom_verdict_stores_no_symptom_row():
    """Absence of the capability must not be written down as a result. A row
    saying "withheld" about an attempt nobody made is an invented fact."""
    from backend.services.history_service import _add_assessment_row

    class Collector:
        def __init__(self):
            self.rows = []

        def add(self, row):
            self.rows.append(row)

    db = Collector()
    _add_assessment_row(db, "analysis-1", {"seed_index": 0, "quality_prediction": {}})
    assert db.rows == []


# ------------------------------------------------- limits of applicability
def test_the_serving_domain_was_measured_and_its_accuracy_left_unknown():
    """The labels are GrainSpace crops; the platform is fed YOLO crops from user
    uploads. Coverage transfers as a measurement because it needs no labels.
    Accuracy does not, and no estimate stands in for it -- an accuracy invented
    for the domain the model actually serves would be the most useful-looking
    fabrication available in this phase.
    """
    if not SERVING_DOMAIN_FILE.exists():
        pytest.skip("serving-domain measurement is absent; run "
                    "src.analysis.symptom_serving_domain")
    m = json.loads(SERVING_DOMAIN_FILE.read_text())
    assert m["kernels_scored"] > 0
    assert m["accuracy"] is None, (
        "an accuracy has appeared for a domain that carries no labels"
    )
    assert m["accuracy_note"], "the null accuracy is unexplained"
    assert m["domain"] != m["calibration_domain"], (
        "this file is meant to record the gap between the two domains"
    )
    assert 0.0 <= m["coverage"] <= 1.0
    # Both figures kept side by side: the difference between them is the point.
    assert m["held_out_coverage"] is not None
