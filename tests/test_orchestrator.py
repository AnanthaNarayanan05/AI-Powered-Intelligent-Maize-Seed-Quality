"""Phase 6: the orchestrator picks the models, and admits what it could not do.

Two things are under test here and they pull in opposite directions. One is that a
question selects only the stages it needs and reuses everything already known about
the image -- the cheapness. The other is that when a question names a capability this
project has not built, the answer says so and names the phase that owns it, instead
of reaching for the nearest model that would produce a number -- the honesty. A cheap
orchestrator that quietly substituted Grad-CAM for a defect mask would pass every
test in the first group and be worse than useless.

Three refusals are kept apart throughout, because the action each implies for the
reader differs: no_served_model (nothing may answer this), insufficient_resolution
(something may, but not about this image) and not_implemented (it has not been built
yet).
"""
from __future__ import annotations

import glob
import json

import pytest

from src.pipeline.orchestrator import (
    INTENTS_BY_NAME,
    LOW_CONFIDENCE,
    STAGES,
    UNBUILT,
    Orchestrator,
    OrchestratorError,
    StageOutput,
    plan_for,
    resolve_intent,
)
from src.registry import get_registry

registry = get_registry()

# The seven phrasings Phase 6 names, and the intent each must reach.
PHASE_6_QUESTIONS = {
    "Analyze these seeds": "analyze",
    "Find defective seeds": "find_defects",
    "Show me where the defect is": "localize_defect",
    "Measure the defect": "measure_defect",
    "How severe is the defect?": "severity",
    "Check for foreign objects": "foreign_objects",
    "Explain this result": "explain",
}

UPLOADS = sorted(glob.glob("data_processed/uploads/*.jpg"))
needs_image = pytest.mark.skipif(not UPLOADS, reason="no sample upload on disk")


@pytest.fixture(scope="module")
def run():
    """One orchestrator, one image, questions asked in sequence.

    Deliberately module-scoped: the run cache is the feature, and a fresh instance
    per test would make "reuse previous results" untestable.
    """
    return Orchestrator(), UPLOADS[0]


# ------------------------------------------------------- understanding the ask
@pytest.mark.parametrize("question,expected", PHASE_6_QUESTIONS.items())
def test_every_phase_6_phrasing_reaches_its_own_intent(question, expected):
    """Seven distinct questions must not collapse into one general analysis -- the
    whole point is that "measure the defect" is answered differently from "find
    defective seeds"."""
    assert resolve_intent(question)["resolved"] == expected


def test_a_specific_phrasing_beats_a_generic_keyword():
    """"Show me where the defect is" contains the word "defect", which would send it
    to find_defects if the table were matched in any other order."""
    assert resolve_intent("show me where the defect is")["resolved"] == "localize_defect"
    assert resolve_intent("find the defective ones")["resolved"] == "find_defects"


def test_an_unrecognised_question_says_it_fell_back_rather_than_claiming_it_understood():
    routing = resolve_intent("what is the weather like")
    assert routing["resolved"] == "analyze"
    assert routing["resolved_by"] == "default"


def test_an_explicit_intent_is_recorded_as_explicit_and_an_unknown_one_is_refused():
    assert resolve_intent(None, "severity")["resolved_by"] == "explicit"
    with pytest.raises(OrchestratorError):
        resolve_intent(None, "diagnose_pathogen")


def test_the_route_taken_is_auditable():
    """A surprising route should be debuggable by whoever it surprised, not only by
    whoever wrote the regex table."""
    assert resolve_intent("how severe is this")["matched"] is not None


# -------------------------------------------------------------------- planning
def test_a_plan_lists_prerequisites_before_the_stages_that_need_them():
    plan = plan_for("localize_defect")
    assert plan.index("detect") < plan.index("classify")
    assert plan.index("resolution") < plan.index("segment")


def test_a_plan_never_repeats_a_stage():
    plan = plan_for("analyze")
    assert len(plan) == len(set(plan))


def test_only_the_broad_question_pays_for_similarity_search():
    """Selective execution has to be visible somewhere or it is not real: asking for
    defective seeds must not run the gallery search that a broad analysis does."""
    assert "similarity" in plan_for("analyze")
    assert "similarity" not in plan_for("find_defects")
    assert "similarity" not in plan_for("severity")


def test_explaining_a_result_plans_no_inference_at_all():
    """An explanation that re-ran the models could report something the result it is
    explaining never said."""
    assert plan_for("explain") == []


def test_measuring_the_defect_does_not_plan_the_classifier():
    """Nothing about a Good/Bad probability contributes to an area measurement, so
    loading that model to answer this question would be work done for show."""
    assert "classify" not in plan_for("measure_defect")


# ------------------------------------------- stage reports stay JSON-safe
def test_a_stage_report_never_carries_the_stage_payload():
    """Segmentation values are numpy mask planes -- neither JSON nor small. Keeping
    `value` out of as_dict() makes the response serialisable by construction rather
    than by remembering to strip it at each call site."""
    output = StageOutput(stage="segment", status="ran", value={"masks": object()})
    assert "value" not in output.as_dict()
    json.dumps(output.as_dict())


# ------------------------------------------------------ capabilities, read live
def test_capabilities_are_derived_from_the_registry_not_written_down():
    """Retiring or declaring a model must change this report without anyone editing
    a sentence in the orchestrator."""
    report = Orchestrator().capabilities()
    for task, state in report["tasks"].items():
        assert state["served"] == registry.can_serve(task), task


def test_the_questions_this_project_cannot_answer_are_listed_as_such():
    report = Orchestrator().capabilities()
    by_intent = {entry["intent"]: entry for entry in report["intents"]}
    assert by_intent["analyze"]["answerable"] is True
    assert by_intent["find_defects"]["answerable"] is True
    # Built in Phase 4, and answerable -- as flagging. The intent stays in this
    # test because what it must never become is a classification: a capability
    # that answers is a capability that can over-answer.
    assert by_intent["foreign_objects"]["answerable"] is True
    # Not built yet, and each names the phase that owns it rather than going quiet.
    for name in ("measure_defect", "severity"):
        assert by_intent[name]["answerable"] is False
        assert by_intent[name]["not_implemented"]["owner"]
    # Built, but nothing is allowed to serve it -- a different problem, and the
    # report names the blocking task rather than blaming a phase.
    assert by_intent["localize_defect"]["answerable"] is False
    assert by_intent["localize_defect"]["not_implemented"] is None
    assert by_intent["localize_defect"]["blocked_by_tasks"] == ["defect_segmentation"]


def test_every_unbuilt_capability_names_an_owning_phase_and_explains_the_refusal():
    for capability in UNBUILT.values():
        assert "Phase" in capability.owner
        assert len(capability.note) > 80, capability.name


def test_the_low_confidence_threshold_matches_the_one_the_api_already_reports():
    """Two definitions of "low confidence" in one product would be worse than
    either; backend/routes/analyze.py already counts below 0.6."""
    assert LOW_CONFIDENCE == 0.60


# ------------------------------------------------------------ running for real
@needs_image
def test_a_broad_analysis_runs_its_whole_plan_and_says_so(run):
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Analyze these seeds")
    assert result["intent"]["resolved"] == "analyze"
    assert result["executed"] == plan_for("analyze")
    assert result["reused"] == []
    json.dumps(result)  # nothing numpy escaped into the response


@needs_image
def test_a_follow_up_question_about_the_same_image_re_runs_nothing(run):
    """Cheapness with receipts: the stages this question needed came back reused, and
    none of them ran again."""
    orchestrator, image = run
    orchestrator.run(image_path=image, question="Analyze these seeds")
    result = orchestrator.run(image_path=image, question="Find defective seeds")
    assert result["executed"] == []
    assert set(result["reused"]) == set(plan_for("find_defects"))


@needs_image
def test_a_stage_an_earlier_question_ran_is_not_reported_as_work_this_one_did(run):
    """"Which stages ran" has to mean this question. Similarity ran for the broad
    analysis; a later question that never planned it must show as cached."""
    orchestrator, image = run
    orchestrator.run(image_path=image, question="Analyze these seeds")
    result = orchestrator.run(image_path=image, question="Find defective seeds")
    by_stage = {s["stage"]: s["status"] for s in result["stages"]}
    assert by_stage["similarity"] == "cached"
    assert "similarity" not in result["executed"]
    assert "similarity" not in result["reused"]


@needs_image
def test_a_refusal_from_an_earlier_question_is_not_reported_against_this_one(run):
    """Asking where the defect is produces an honest segmentation refusal. Asking
    something unrelated afterwards must not inherit it -- an unavailable list is
    read as "what this answer is missing"."""
    orchestrator, image = run
    orchestrator.run(image_path=image, question="Show me where the defect is")
    result = orchestrator.run(image_path=image, question="Explain this result")
    assert result["unavailable"] == []


@needs_image
def test_the_same_image_under_a_new_filename_still_hits_the_cache(run):
    """Every upload lands under a fresh uuid, so a path-keyed cache would never hit
    and "reuse previous results" would be a claim with no mechanism behind it."""
    import shutil
    import tempfile

    orchestrator, image = run
    orchestrator.run(image_path=image, question="Analyze these seeds")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        shutil.copyfile(image, tmp.name)
    result = orchestrator.run(image_path=tmp.name, question="Analyze these seeds")
    assert result["executed"] == []


@needs_image
def test_an_unserved_task_is_refused_without_loading_anything(run):
    """The registry is consulted before a checkpoint is touched, which is what makes
    "select only the necessary models" true rather than aspirational."""
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Show me where the defect is")
    segment = next(s for s in result["stages"] if s["stage"] == "segment")
    assert segment["status"] == "unavailable"
    assert segment["reason"] == "no_served_model"
    # A capability problem, not an image problem: a better photograph will not help.
    assert "resolution" not in (segment["message"] or "").lower()


@needs_image
def test_asking_where_the_defect_is_rules_out_the_two_lookalikes(run):
    """The things that most resemble an answer here are a heatmap and a bounding box.
    Saying so at the point of refusal is the difference between an honest gap and an
    invitation to misread the next screenshot."""
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Show me where the defect is")
    answer = result["answer"]
    assert answer["available"] is False
    assert "not a defect" in answer["not_a_substitute"]["grad_cam"].lower()
    assert answer["not_a_substitute"]["detection_box"]


@needs_image
def test_measuring_the_defect_is_unavailable_and_carries_no_number(run):
    """The failure mode this guards is a plausible percentage derived from a bounding
    box. There must be no area and no coverage anywhere in the answer."""
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Measure the defect")
    answer = result["answer"]
    assert answer["available"] is False
    assert answer["reason"] == "not_implemented"
    assert "Phase 2" in answer["owner"]
    blob = json.dumps(answer).lower()
    assert "coverage_percent" not in blob
    assert "defect_area_px" not in blob


@needs_image
def test_severity_offers_the_binary_grade_only_with_its_limitation_attached(run):
    """Handing back Good/Bad as if it answered "how severe" would put a scale on a
    quantity nothing in this project measured."""
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="How severe is the defect?")
    answer = result["answer"]
    assert answer["available"] is False
    assert "not a severity score" in json.dumps(answer["nearest_supported"]).lower()


@needs_image
def test_foreign_objects_answers_with_flags_and_refusals_and_never_an_identity(run):
    """Phase 4 turned this from a refusal into an answer, which is the more
    dangerous of the two states. What the answer may contain is a count of flags,
    a count of objects the gate declined to score, and no name for anything.

    The three counts are asserted to reconcile because the tempting bug is to let
    a declined object fall into "known maize" -- which converts a refusal to
    examine an object into a clean bill of health for it.
    """
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Check for foreign objects")
    answer = result["answer"]
    assert answer["available"] is True
    assert answer["is_classification"] is False
    assert answer["capability"] == "foreign_object_flagging"
    assert (answer["known_maize"] + answer["possible_foreign_objects"]
            == answer["objects_scored"])
    assert (answer["objects_scored"] + answer["not_scored"]
            == answer["objects_detected"])
    assert answer["what_it_means"] and answer["what_it_does_not_mean"]
    # The answer must state its own limit. Scanning it for material names would
    # be the wrong test here -- the disclaimer earns the right to say "stone" by
    # saying the system cannot recognise one -- so the assertion is positive.
    # Verdicts themselves are held to the naming rule in test_foreign_object_gate.
    assert "not an identification" in json.dumps(answer["what_it_does_not_mean"]).lower()


@needs_image
def test_an_explanation_is_assembled_from_the_run_and_runs_nothing(run):
    """Every line is read back out of a recorded stage output, which is what makes it
    an explanation of this analysis rather than a plausible story about one."""
    orchestrator, image = run
    orchestrator.run(image_path=image, question="Analyze these seeds")
    result = orchestrator.run(image_path=image, question="Explain this result")
    assert result["executed"] == []
    answer = result["answer"]
    assert answer["findings"]
    assert {m["key"] for m in answer["models_used"]} <= {r.key for r in registry.all()}


@needs_image
def test_a_follow_up_can_be_asked_by_analysis_id_with_no_image_at_all(run):
    orchestrator, image = run
    first = orchestrator.run(image_path=image, question="Analyze these seeds")
    second = orchestrator.run(
        analysis_id=first["analysis_id"], question="Explain this result"
    )
    assert second["analysis_id"] == first["analysis_id"]
    assert second["executed"] == []


def test_an_unknown_analysis_id_is_refused_rather_than_answered_emptily():
    """Returning an empty analysis for an id the cache evicted would look exactly
    like an image with no seeds in it."""
    with pytest.raises(OrchestratorError):
        Orchestrator().run(analysis_id="does-not-exist", question="Explain this result")


@needs_image
def test_a_seed_carries_a_measurement_but_no_placeholder_grade(run):
    """An absent field means the stage that fills it did not run, and the stage
    report says which. No default class, no zero confidence."""
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Measure the defect")
    for seed in result["seeds"]:
        assert "kernel_px" in seed  # measured from the detection box
        assert "variety" not in seed  # classify was not planned for this question


@needs_image
def test_out_of_distribution_kernels_are_never_counted_as_defective(run):
    """Folding them in would let imagery the quality model was never validated on
    inflate a number people act on."""
    orchestrator, image = run
    result = orchestrator.run(image_path=image, question="Find defective seeds")
    counts = result["answer"]["counts"]
    assert "unverified_out_of_distribution" in counts
    assert sum(counts.values()) == result["seed_count"]


@needs_image
def test_a_failing_stage_does_not_take_the_rest_of_the_analysis_with_it(run):
    """One broken model must cost its own stage and nothing else, and what reaches a
    caller must not be a stack trace."""
    orchestrator, image = run
    orchestrator._runs.clear()
    orchestrator._by_id.clear()
    original = orchestrator._run_similarity

    def boom(*args, **kwargs):
        raise RuntimeError("boom C:/secret/path")

    orchestrator._run_similarity = boom
    try:
        result = orchestrator.run(image_path=image, question="Analyze these seeds")
    finally:
        orchestrator._run_similarity = original
        orchestrator._runs.clear()
        orchestrator._by_id.clear()

    similarity = next(s for s in result["stages"] if s["stage"] == "similarity")
    assert similarity["status"] == "failed"
    assert "secret" not in (similarity["message"] or "")
    assert "detect" in result["executed"]
    assert result["answer"]["available"] is True


@needs_image
def test_every_planned_stage_appears_in_the_report_with_a_known_status(run):
    """"Clearly report which stages ran" means no planned stage is missing -- silence
    about a stage is indistinguishable from it having succeeded."""
    orchestrator, image = run
    known = {"ran", "reused", "cached", "skipped", "unavailable", "failed"}
    for question in PHASE_6_QUESTIONS:
        result = orchestrator.run(image_path=image, question=question)
        reported = {s["stage"] for s in result["stages"]}
        planned = {p["stage"] for p in result["plan"]}
        assert planned <= reported, question
        for stage in result["stages"]:
            assert stage["status"] in known, (question, stage)


@needs_image
def test_no_answer_to_any_of_the_seven_questions_smuggles_a_mask_into_json(run):
    orchestrator, image = run
    for question in PHASE_6_QUESTIONS:
        json.dumps(orchestrator.run(image_path=image, question=question))


# ------------------------------------------------------------- table integrity
def test_every_stage_declares_the_tasks_it_may_not_run_without():
    """A stage with no declared task bypasses the registry pre-check entirely -- only
    `resolution` may, because it loads nothing and merely compares measured kernel
    sizes against a recorded floor."""
    for name, stage in STAGES.items():
        if not stage.tasks:
            assert name == "resolution"
        for task in stage.tasks:
            assert task in registry.tasks, f"{name} declares undeclared task {task}"


def test_every_intent_plans_only_known_stages_and_names_a_real_capability():
    for intent in INTENTS_BY_NAME.values():
        for stage in intent.stages:
            assert stage in STAGES
        if intent.capability:
            assert intent.capability in UNBUILT
