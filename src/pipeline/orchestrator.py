"""Phase 6: one question in, a plan out, and an honest record of what actually ran.

The user asks "find the defective seeds" or "measure the defect". They do not
choose a checkpoint, and they should not have to know that one of those questions
is answerable today and the other is not. This module turns the question into a
plan of stages, runs only the stages that plan needs, and returns -- alongside the
answer -- the list of what ran, what was reused, and what was refused and why.

Three refusals live here and they are deliberately kept apart, because the action
they imply for the person reading them is different in each case:

  no_served_model          The registry allows no model to answer this task. A
                           better photograph will not help. (Phase 7.)
  insufficient_resolution  A model could answer it, but not about this image. A
                           better photograph WILL help. (Phase 8.)
  not_implemented          The capability has not been built yet, and the phase
                           that owns it is named. Nothing about the input helps.

The fourth possible outcome, "we approximated it with a model trained for
something else", is not implemented and must not be. A question about defect area
is answered "unavailable, Phase 2 owns it" rather than with a bounding-box
estimate or a thresholded Grad-CAM, because a fabricated measurement is worse than
a missing one: the missing one is obviously missing.

Nothing here loads a model of its own. It composes AnalysisPipeline's stages,
resolves every one of them through the registry before touching a checkpoint, and
caches per image so that a follow-up question about an image already analysed
costs no inference at all.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from src.pipeline import resolution_gate
from src.pipeline.unified_pipeline import (
    AnalysisPipeline,
    InvalidImageError,  # noqa: F401  (re-exported for the route layer)
    ModelNotAvailableError,
)
from src.registry import TaskUnavailableError
from src.similarity.embedding_index import IndexEncoderMismatch
from src.utils.logging_utils import get_logger

logger = get_logger("orchestrator")

ORCHESTRATOR_VERSION = 1

# Below this a prediction is reported but held out of the headline count. Not a
# new threshold: backend/routes/analyze.py already reports low_confidence_count at
# the same 0.60, and two different definitions of "low confidence" in one product
# would be worse than either.
LOW_CONFIDENCE = 0.60


class OrchestratorError(Exception):
    """A request the orchestrator cannot even plan for (unknown intent, no image)."""


class StageUnavailable(Exception):
    """A stage that cannot run on this input, with the reason a reader needs.

    Distinct from a crash: this is an expected, explainable outcome, and it is
    reported in the result rather than raised out of the run.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


# --------------------------------------------------------------------------
# Capabilities that do not exist yet
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UnbuiltCapability:
    """A question the platform understands but cannot answer yet.

    `owner` names the phase that will build it, so the refusal is a roadmap entry
    rather than a dead end, and `requires_task` lets the refusal quote the live
    registry instead of a sentence written down here that will go stale the day a
    real model lands.
    """

    name: str
    label: str
    owner: str
    requires_task: str | None
    note: str


UNBUILT: dict[str, UnbuiltCapability] = {
    "defect_area": UnbuiltCapability(
        name="defect_area",
        label="Visible defect area / coverage %",
        owner="Phase 2 - defect area measurement",
        requires_task="defect_segmentation",
        note=(
            "Coverage is defect_pixels / valid_seed_pixels, so it cannot exist before a "
            "served segmenter produces those pixels. Estimating it from a bounding box, "
            "a class probability or a thresholded Grad-CAM heatmap would be a fabricated "
            "measurement wearing a number's clothes."
        ),
    ),
    "defect_severity": UnbuiltCapability(
        name="defect_severity",
        label="Defect severity",
        owner="Phase 3 - defect severity",
        requires_task="defect_segmentation",
        note=(
            "This project holds no severity labels. Severity will be derived from measured "
            "coverage once Phase 2 exists, and any LOW/MODERATE/HIGH banding will have to "
            "carry its own documented derivation. Inventing bands over a Good/Bad "
            "probability would put a scale on a quantity nothing measured."
        ),
    ),
    "foreign_object": UnbuiltCapability(
        name="foreign_object",
        label="Foreign object detection",
        owner="Phase 4 - foreign object detection",
        requires_task=None,
        note=(
            "The distribution gate already flags kernels that do not resemble the quality "
            "model's training data, and Phase 4 will build on it -- but it has never been "
            "evaluated against labelled foreign objects, so reading its score as one today "
            "would be claiming a detector this project has not tested."
        ),
    ),
}


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """One unit of work, and the registry tasks it is not allowed to run without.

    `tasks` is checked against the registry BEFORE anything is loaded, which is
    what makes "select only the necessary models" true rather than aspirational:
    an unserved task never reaches a checkpoint at all.
    """

    name: str
    tasks: tuple[str, ...]
    needs: tuple[str, ...]
    expensive: bool
    describe: str


STAGES: dict[str, Stage] = {
    "detect": Stage(
        name="detect",
        tasks=("detect",),
        needs=(),
        expensive=True,
        describe="Locate and count individual kernels.",
    ),
    "classify": Stage(
        name="classify",
        # Three tasks, one forward pass. The unified model serves all three, so
        # asking for the grade alone selects exactly the same weights as asking
        # for the variety -- splitting this into separate stages would suggest a
        # saving that does not exist.
        tasks=("variety", "quality", "distribution_gate"),
        needs=("detect",),
        expensive=True,
        describe="Variety, Good/Bad grade and the distribution gate, in one pass.",
    ),
    "resolution": Stage(
        name="resolution",
        tasks=(),
        needs=("detect",),
        expensive=False,
        describe="Measure each kernel against the measured segmentation floor.",
    ),
    "segment": Stage(
        name="segment",
        tasks=("defect_segmentation",),
        needs=("detect", "resolution"),
        expensive=True,
        describe="Pixel mask of the defective region on each kernel.",
    ),
    "similarity": Stage(
        name="similarity",
        tasks=("embedding",),
        needs=("detect",),
        expensive=True,
        describe="Nearest gallery kernels in the serving encoder's feature space.",
    ),
}


# --------------------------------------------------------------------------
# Intents
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Intent:
    name: str
    canonical: str
    stages: tuple[str, ...]
    capability: str | None
    patterns: tuple[str, ...]


# Order is the matching order, most specific first: "show me where the defect is"
# has to reach localize_defect before the bare word "defect" hands it to
# find_defects. The matcher is a plain regex table on purpose -- intent routing is
# part of the core ML path, and core ML must work with no language model available
# (project rule 26). Gemini explains what this decided; it does not decide it.
INTENTS: tuple[Intent, ...] = (
    Intent(
        name="localize_defect",
        canonical="Show me where the defect is",
        stages=("detect", "classify", "resolution", "segment"),
        capability=None,
        patterns=(
            r"\bshow me where\b",
            r"\bwhere\b[^?]*\bdefect",
            r"\blocali[sz]",
            r"\b(defect|damage)\s+mask\b",
            r"\bpinpoint\b",
            r"\bsegment(ation)?\b",
        ),
    ),
    Intent(
        name="measure_defect",
        canonical="Measure the defect",
        stages=("detect", "resolution"),
        capability="defect_area",
        patterns=(
            r"\bmeasure\b",
            r"\bdefect area\b",
            r"\bcoverage\b",
            r"\bhow (much|large|big)\b",
            r"\bwhat (percent|percentage|%)",
            r"\bmm2\b|\bmm\^?2\b",
        ),
    ),
    Intent(
        name="severity",
        canonical="How severe is the defect?",
        stages=("detect", "classify", "resolution"),
        capability="defect_severity",
        patterns=(r"\bsever", r"\bhow bad\b", r"\bgrade the (damage|defect)\b"),
    ),
    Intent(
        name="foreign_objects",
        canonical="Check for foreign objects",
        stages=("detect", "classify"),
        capability="foreign_object",
        patterns=(
            r"\bforeign\b",
            r"\bcontaminan",
            r"\bstone|\bdebris|\bhusk|\bcob\b",
            r"\bnon[- ]?(seed|maize|kernel)\b",
            r"\bimpurit",
        ),
    ),
    Intent(
        name="find_defects",
        canonical="Find defective seeds",
        stages=("detect", "classify", "resolution"),
        capability=None,
        patterns=(
            r"\bdefect",
            r"\bdamaged?\b",
            r"\bbad (seeds?|kernels?)\b",
            r"\bwhich .*\b(bad|spoil|reject)",
            r"\bmould|\bmold|\bfungus|\bfungal|\binsect|\bcrack",
        ),
    ),
    Intent(
        name="explain",
        canonical="Explain this result",
        # Deliberately empty: explaining a result must not change it. Every stage
        # comes back "reused" and no inference runs at all.
        stages=(),
        capability=None,
        patterns=(
            r"\bexplain\b",
            r"\bwhy\b",
            r"\bwhat does (this|that) mean\b",
            r"\binterpret\b",
            r"\bhow did you\b",
        ),
    ),
    Intent(
        name="analyze",
        canonical="Analyze these seeds",
        stages=("detect", "classify", "resolution", "similarity"),
        capability=None,
        patterns=(
            r"\banal[iy][sz]e\b",
            # deliberately not "what is the ...": that phrasing swallows questions
            # this table has no business claiming to understand, and an invented route
            # is worse than an admitted fallback.
            r"\bwhat (is|are) (this|these)\b",
            r"\bidentify\b",
            r"\bcount\b",
            r"\bvariety\b",
            r"\bcheck (these|this|my)\b",
        ),
    ),
)

INTENTS_BY_NAME = {i.name: i for i in INTENTS}
DEFAULT_INTENT = "analyze"


def resolve_intent(question: str | None = None, intent: str | None = None) -> dict:
    """Which question is being asked, and how that was decided.

    `resolved_by` and `matched` travel with the answer so a surprising route is
    debuggable by the person it surprised, rather than only by whoever wrote the
    regex table.
    """
    if intent:
        if intent not in INTENTS_BY_NAME:
            raise OrchestratorError(
                f"'{intent}' is not a known intent. Known intents: {sorted(INTENTS_BY_NAME)}"
            )
        chosen = INTENTS_BY_NAME[intent]
        return {
            "requested": intent,
            "resolved": chosen.name,
            "canonical": chosen.canonical,
            "question": question,
            "resolved_by": "explicit",
            "matched": None,
        }

    text = (question or "").strip().lower()
    if text:
        for candidate in INTENTS:
            for pattern in candidate.patterns:
                if re.search(pattern, text):
                    return {
                        "requested": None,
                        "resolved": candidate.name,
                        "canonical": candidate.canonical,
                        "question": question,
                        "resolved_by": "pattern",
                        "matched": pattern,
                    }

    fallback = INTENTS_BY_NAME[DEFAULT_INTENT]
    return {
        "requested": None,
        "resolved": fallback.name,
        "canonical": fallback.canonical,
        "question": question,
        # Said out loud rather than passed off as understanding: an unrecognised
        # question got the general analysis, and the reader should know that is
        # what happened.
        "resolved_by": "default" if text else "no_question",
        "matched": None,
    }


def plan_for(intent_name: str) -> list[str]:
    """The stages an intent needs, dependencies first, each listed once."""
    intent = INTENTS_BY_NAME[intent_name]
    ordered: list[str] = []

    def add(name: str) -> None:
        if name in ordered:
            return
        for prerequisite in STAGES[name].needs:
            add(prerequisite)
        ordered.append(name)

    for stage_name in intent.stages:
        add(stage_name)
    return ordered


# --------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------
@dataclass
class StageOutput:
    """What happened to one stage. `value` never leaves this object.

    Segmentation values hold numpy mask planes, which are neither JSON nor small.
    as_dict() drops `value` for that reason: the response reports that a stage ran
    and what model ran it, and each answer builder decides what -- if anything --
    of the payload is safe and meaningful to publish.
    """

    stage: str
    status: str  # ran | reused | skipped | unavailable | failed
    model: str | None = None
    reason: str | None = None
    message: str | None = None
    duration_ms: int | None = None
    params: tuple = ()
    value: object = None

    @property
    def usable(self) -> bool:
        return self.status in ("ran", "reused")

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "status": self.status,
            "model": self.model,
            "reason": self.reason,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "describe": STAGES[self.stage].describe,
        }


@dataclass
class RunState:
    digest: str
    analysis_id: str
    image_path: str
    stages: dict[str, StageOutput] = field(default_factory=dict)
    image: object = None
    image_size: tuple[int, int] | None = None
    created_at: float = field(default_factory=time.time)


def _digest(image_path: str) -> str:
    """Identify an image by its bytes, not its filename.

    Every upload lands under a fresh uuid, so a path-keyed cache would never hit
    and "reuse previous results" would be a claim with no mechanism behind it.
    """
    h = hashlib.sha256()
    with open(image_path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Orchestrator:
    """Plans, runs and reports a single analysis.

    One instance per process, sharing one AnalysisPipeline, so model weights load
    once and the run cache actually spans requests.
    """

    def __init__(self, pipeline: AnalysisPipeline | None = None, cache_size: int = 8):
        self._pipeline = pipeline
        self._runs: "OrderedDict[str, RunState]" = OrderedDict()
        self._by_id: dict[str, str] = {}
        self._cache_size = cache_size

    @property
    def pipeline(self) -> AnalysisPipeline:
        if self._pipeline is None:
            self._pipeline = AnalysisPipeline()
        return self._pipeline

    # ---------------------------------------------------------------- caching
    def _state_for(self, image_path: str) -> RunState:
        digest = _digest(image_path)
        state = self._runs.get(digest)
        if state is not None:
            self._runs.move_to_end(digest)
            # A re-upload of identical bytes lands on a new path; point the cached
            # run at the file that actually exists now.
            state.image_path = image_path
            return state

        state = RunState(digest=digest, analysis_id=str(uuid.uuid4()), image_path=image_path)
        self._runs[digest] = state
        self._by_id[state.analysis_id] = digest
        while len(self._runs) > self._cache_size:
            evicted, _ = self._runs.popitem(last=False)
            self._by_id = {k: v for k, v in self._by_id.items() if v != evicted}
        return state

    def state_by_id(self, analysis_id: str) -> RunState | None:
        digest = self._by_id.get(analysis_id)
        return self._runs.get(digest) if digest else None

    def _image(self, state: RunState):
        if state.image is None:
            state.image = self.pipeline.load_and_validate_image(state.image_path)
            state.image_size = tuple(state.image.size)
        return state.image

    # ------------------------------------------------------------- execution
    def _execute(self, state: RunState, name: str, params: tuple) -> StageOutput:
        stage = STAGES[name]

        prior = state.stages.get(name)
        if prior is not None and prior.usable and prior.params == params:
            reused = replace(prior, status="reused", duration_ms=0)
            state.stages[name] = reused
            return reused

        # A stage whose prerequisite never produced anything cannot be attempted,
        # and saying "skipped, detect was unavailable" is more use than a
        # traceback about a missing key.
        for prerequisite in stage.needs:
            upstream = state.stages.get(prerequisite)
            if upstream is None or not upstream.usable:
                out = StageOutput(
                    stage=name,
                    status="skipped",
                    reason="prerequisite_unavailable",
                    message=f"'{prerequisite}' did not produce a result, so '{name}' was not attempted.",
                    params=params,
                )
                state.stages[name] = out
                return out

        # The registry is consulted before any checkpoint is touched. This is
        # where "no model is allowed to serve this" is decided, and deciding it
        # here is what keeps an unserved task from ever reaching a loader.
        model_key = None
        for task in stage.tasks:
            try:
                record = self.pipeline.registry.resolve(task)
            except TaskUnavailableError as e:
                out = StageOutput(
                    stage=name,
                    status="unavailable",
                    reason="no_served_model",
                    message=str(e),
                    params=params,
                )
                state.stages[name] = out
                return out
            model_key = model_key or record.key

        started = time.perf_counter()
        try:
            value = getattr(self, f"_run_{name}")(state, params)
        except StageUnavailable as e:
            out = StageOutput(
                stage=name, status="unavailable", model=model_key,
                reason=e.reason, message=e.message, params=params,
            )
        except ModelNotAvailableError as e:
            out = StageOutput(
                stage=name, status="unavailable", model=model_key,
                reason="no_served_model", message=str(e), params=params,
            )
        except Exception:  # noqa: BLE001
            # Logged in full, reported safely: a stack trace in an API response is
            # a leak, and one failed stage must not lose the stages that worked.
            logger.exception("Stage '%s' failed", name)
            out = StageOutput(
                stage=name, status="failed", model=model_key,
                reason="stage_error",
                message=f"The '{name}' stage failed. The other stages in this analysis are unaffected.",
                params=params,
            )
        else:
            out = StageOutput(
                stage=name, status="ran", model=model_key, params=params, value=value,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        state.stages[name] = out
        return out

    # ------------------------------------------------------------ stage bodies
    def _run_detect(self, state: RunState, params: tuple) -> list[dict]:
        detections = self.pipeline.detect_seeds(state.image_path)
        return [
            {
                "seed_index": i,
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
                # The same short-side measure the resolution floor was measured
                # in, carried per seed because it decides per seed which
                # pixel-level answers exist for it.
                "kernel_px": resolution_gate.kernel_px_from_bbox(det["bbox"]),
            }
            for i, det in enumerate(detections)
        ]

    def _run_classify(self, state: RunState, params: tuple) -> dict:
        image = self._image(state)
        out = {}
        for seed in state.stages["detect"].value:
            crop = self.pipeline.crop_seed(image, seed["bbox"])
            variety, quality = self.pipeline.classify_unified(crop)
            out[seed["seed_index"]] = {
                "variety_prediction": variety,
                "quality_prediction": quality,
            }
        return out

    def _run_resolution(self, state: RunState, params: tuple) -> dict:
        seeds = state.stages["detect"].value
        return self.pipeline.segmentation_status([s["kernel_px"] for s in seeds])

    def _run_segment(self, state: RunState, params: tuple) -> dict:
        status = state.stages["resolution"].value
        if status.get("reason") == "insufficient_resolution" and status.get("status") == "unavailable":
            # Every kernel in this frame is under the floor. Refuse once, here,
            # rather than per seed: below the floor a mask is not a worse
            # measurement, it is an unmeasured one.
            raise StageUnavailable("insufficient_resolution", resolution_gate.SEGMENTATION_UNAVAILABLE)

        image = self._image(state)
        floor = (status.get("resolution") or {}).get("min_kernel_px")
        results = {}
        for seed in state.stages["detect"].value:
            if floor is not None and seed["kernel_px"] < floor:
                results[seed["seed_index"]] = {
                    "available": False,
                    "reason": "insufficient_resolution",
                    "message": resolution_gate.SEGMENTATION_UNAVAILABLE,
                }
                continue
            crop = self.pipeline.crop_seed(image, seed["bbox"])
            results[seed["seed_index"]] = self.pipeline.segment_defects(crop, bbox=seed["bbox"])
        return results

    def _run_similarity(self, state: RunState, params: tuple) -> dict:
        top_k = params[0] if params else 5
        image = self._image(state)
        out = {}
        for seed in state.stages["detect"].value:
            crop = self.pipeline.crop_seed(image, seed["bbox"])
            try:
                out[seed["seed_index"]] = self.pipeline.embed_and_search(crop, "unified", top_k)
            except (IndexEncoderMismatch, KeyError) as e:
                raise StageUnavailable("index_unavailable", f"Similarity search is unavailable: {e}") from e
        return out

    # ------------------------------------------------------------------- seeds
    def _seeds(self, state: RunState) -> list[dict]:
        """The per-seed record, assembled from whichever stages produced one.

        A seed carries a variety prediction only if classify ran. There is no
        placeholder and no default class: a field that is absent means the stage
        that fills it did not run, which the stage report says explicitly.
        """
        detect = state.stages.get("detect")
        if detect is None or not detect.usable:
            return []

        classify = state.stages.get("classify")
        similarity = state.stages.get("similarity")
        segment = state.stages.get("segment")

        seeds = []
        for seed in detect.value:
            entry = dict(seed)
            index = seed["seed_index"]
            if classify is not None and classify.usable:
                entry.update(classify.value.get(index, {}))
            if similarity is not None and similarity.usable:
                entry["similarity_results"] = similarity.value.get(index, [])
            if segment is not None and segment.usable:
                mask = segment.value.get(index, {})
                # The planes themselves stay in the cache. What travels is
                # whether a mask exists for this seed and what it is made of --
                # the pixels are Phase 10's to render and Phase 2's to measure.
                entry["segmentation"] = {
                    "available": bool(mask.get("available")),
                    "reason": mask.get("reason"),
                    "message": mask.get("message"),
                    "model": mask.get("model"),
                    "channels": mask.get("channels"),
                    "is_synthetic_model": mask.get("is_synthetic_model"),
                }
            seeds.append(entry)
        return seeds

    # ---------------------------------------------------------------- answers
    def _quality_breakdown(self, seeds: list[dict]) -> dict:
        """Counts that respect the distribution gate instead of overruling it.

        A grade the gate flagged is an extrapolation, so it is reported in its own
        bucket rather than added to the defective count. Folding those kernels in
        would let imagery the quality model was never validated on inflate a
        number people act on.
        """
        buckets: dict[str, list[int]] = {
            "sound": [],
            "defective": [],
            "defective_low_confidence": [],
            "unverified_out_of_distribution": [],
            "ungraded": [],
        }
        for seed in seeds:
            grade = seed.get("quality_prediction")
            if not grade:
                buckets["ungraded"].append(seed["seed_index"])
                continue
            if grade.get("out_of_distribution"):
                buckets["unverified_out_of_distribution"].append(seed["seed_index"])
                continue
            label = str(grade.get("predicted_class", "")).strip().lower()
            if label == "bad":
                target = "defective" if grade.get("confidence", 0) >= LOW_CONFIDENCE else "defective_low_confidence"
            else:
                target = "sound"
            buckets[target].append(seed["seed_index"])

        return {
            "counts": {k: len(v) for k, v in buckets.items()},
            "seed_indices": buckets,
            "low_confidence_threshold": LOW_CONFIDENCE,
        }

    def _unbuilt_answer(self, capability: str) -> dict:
        """The refusal for a capability no phase has built yet.

        `blocked_by` is read from the live registry rather than written down, so
        the day a real segmenter is declared this sentence changes on its own
        instead of quietly contradicting the model list.
        """
        cap = UNBUILT[capability]
        blocked_by = []
        if cap.requires_task:
            try:
                record = self.pipeline.registry.resolve(cap.requires_task)
            except TaskUnavailableError as e:
                blocked_by.append({"task": cap.requires_task, "reason": "no_served_model", "detail": str(e)})
            else:
                blocked_by.append({"task": cap.requires_task, "reason": None, "detail": f"served by {record.key}"})

        return {
            "capability": cap.name,
            "label": cap.label,
            "available": False,
            "reason": "not_implemented",
            "message": f"{cap.label} is not available. It is owned by {cap.owner}.",
            "owner": cap.owner,
            "blocked_by": blocked_by,
            "note": cap.note,
        }

    def _answer(self, state: RunState, intent: Intent, seeds: list[dict]) -> dict:
        builder = getattr(self, f"_answer_{intent.name}")
        return builder(state, seeds)

    def _answer_analyze(self, state: RunState, seeds: list[dict]) -> dict:
        varieties: dict[str, int] = {}
        confidences = []
        for seed in seeds:
            prediction = seed.get("variety_prediction")
            if prediction:
                name = prediction["predicted_class"]
                varieties[name] = varieties.get(name, 0) + 1
                confidences.append(prediction["confidence"])

        return {
            "capability": "seed_analysis",
            "available": bool(seeds) or state.stages.get("detect", StageOutput("detect", "skipped")).usable,
            "reason": None,
            "seed_count": len(seeds),
            "variety_distribution": varieties,
            "average_variety_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "quality": self._quality_breakdown(seeds),
            "caveats": [
                "Variety and grade are whole-kernel predictions. Neither says which part of "
                "a kernel is affected, and neither is a defect area.",
            ],
        }

    def _answer_find_defects(self, state: RunState, seeds: list[dict]) -> dict:
        classify = state.stages.get("classify")
        if classify is None or not classify.usable:
            return {
                "capability": "kernel_quality_grade",
                "available": False,
                "reason": classify.reason if classify else "not_run",
                "message": classify.message if classify else "The quality stage did not run.",
            }

        breakdown = self._quality_breakdown(seeds)
        return {
            "capability": "kernel_quality_grade",
            "label": "Good / Bad kernel grade",
            "available": True,
            "reason": None,
            **breakdown,
            "basis": (
                "Good/Bad head of the unified model, trained on the Mendeley EfficientMaize "
                "expert-assigned kernel labels."
            ),
            "caveats": [
                "A Bad grade is a judgement about the whole kernel. It does not localise the "
                "defect and it is not a defect area.",
                "Kernels the distribution gate flagged are counted separately as unverified, "
                "not as defective: their grade is an extrapolation rather than a measurement.",
            ],
        }

    def _answer_localize_defect(self, state: RunState, seeds: list[dict]) -> dict:
        segment = state.stages.get("segment")
        answer = {
            "capability": "defect_localisation",
            "label": "Pixel-level defect localisation",
            "available": bool(segment and segment.usable),
            "reason": segment.reason if segment else "not_run",
            "message": segment.message if segment else "The segmentation stage did not run.",
            # Rule 9, stated where someone asking exactly this question will read
            # it: the heatmap that looks like an answer is not one.
            "not_a_substitute": {
                "grad_cam": (
                    "Grad-CAM is available at POST /api/explain/gradcam and shows which regions "
                    "influenced the classification. It is model attention over a feature map, not "
                    "a defect boundary: it must not be thresholded into a mask or measured as an area."
                ),
                "detection_box": (
                    "The detection box bounds the whole kernel, not the defect on it."
                ),
            },
        }
        if segment and segment.usable:
            answer["seeds"] = {
                str(index): {"available": bool(mask.get("available")), "channels": mask.get("channels")}
                for index, mask in segment.value.items()
            }
        return answer

    def _answer_measure_defect(self, state: RunState, seeds: list[dict]) -> dict:
        answer = self._unbuilt_answer("defect_area")
        resolution = state.stages.get("resolution")
        if resolution and resolution.usable:
            # Useful even in a refusal: if the kernels are also under the floor,
            # a better photograph is a prerequisite for the capability once it
            # exists, and the reader may as well learn that now.
            answer["resolution"] = resolution.value.get("resolution")
        return answer

    def _answer_severity(self, state: RunState, seeds: list[dict]) -> dict:
        answer = self._unbuilt_answer("defect_severity")
        classify = state.stages.get("classify")
        if classify and classify.usable:
            answer["nearest_supported"] = {
                "capability": "kernel_quality_grade",
                **self._quality_breakdown(seeds),
                "caveat": (
                    "A binary Good/Bad grade is not a severity score. It has two states and no "
                    "ordering between degrees of damage, so it cannot be read as LOW/MODERATE/HIGH."
                ),
            }
        return answer

    def _answer_foreign_objects(self, state: RunState, seeds: list[dict]) -> dict:
        answer = self._unbuilt_answer("foreign_object")
        classify = state.stages.get("classify")
        if classify and classify.usable:
            flagged = [
                s["seed_index"]
                for s in seeds
                if (s.get("quality_prediction") or {}).get("out_of_distribution")
            ]
            answer["related_signal"] = {
                "signal": "distribution_gate",
                "objects_flagged_as_unlike_training_data": len(flagged),
                "seed_indices": flagged,
                "what_it_means": (
                    "These kernels sit further from the quality model's training representation "
                    "than its calibrated threshold allows, so its grade for them is unverified."
                ),
                "what_it_does_not_mean": (
                    "It is not a foreign-object finding. The gate was calibrated on maize imagery "
                    "and has never been evaluated against labelled non-seed objects, so it cannot "
                    "say that an object is a stone, husk or debris -- or that it is not maize."
                ),
            }
        return answer

    def _answer_explain(self, state: RunState, seeds: list[dict]) -> dict:
        """An account of the run, assembled only from what the run recorded.

        No stage runs for this intent and no language model is consulted. Every
        line below is read back out of a StageOutput, which is what makes it an
        explanation of this analysis rather than a plausible story about one.
        """
        registry = self.pipeline.registry
        models = []
        seen = set()
        for output in state.stages.values():
            if not (output.usable and output.model) or output.model in seen:
                continue
            seen.add(output.model)
            try:
                record = registry.get(output.model)
            except Exception:  # noqa: BLE001
                continue
            models.append({
                "stage": output.stage,
                "key": record.key,
                "name": record.name,
                "version": record.version,
                "architecture": record.architecture,
                "dataset": record.dataset,
                "metrics_binding": record.metrics_binding,
            })

        findings = []
        detect = state.stages.get("detect")
        if detect and detect.usable:
            findings.append(f"{len(detect.value)} kernel(s) were detected above the confidence threshold.")
        classify = state.stages.get("classify")
        if classify and classify.usable:
            breakdown = self._quality_breakdown(seeds)["counts"]
            findings.append(
                f"{breakdown['sound']} graded sound, {breakdown['defective']} graded defective, "
                f"{breakdown['unverified_out_of_distribution']} withheld as unverified by the "
                f"distribution gate."
            )
        resolution = state.stages.get("resolution")
        if resolution and resolution.usable:
            summary = resolution.value.get("resolution") or {}
            if summary.get("kernel_px_median") is not None:
                findings.append(
                    f"Kernels measure {summary['kernel_px_median']} px across (median) against a "
                    f"{summary['min_kernel_px']} px measured floor for pixel-level analysis."
                )

        withheld = [
            {"stage": o.stage, "reason": o.reason, "message": o.message}
            for o in state.stages.values()
            if o.status in ("unavailable", "failed", "skipped")
        ]

        return {
            "capability": "analysis_provenance",
            "available": bool(state.stages),
            "reason": None if state.stages else "nothing_to_explain",
            "message": None if state.stages else "No analysis has been run on this image yet.",
            "models_used": models,
            "findings": findings,
            "withheld": withheld,
            "note": (
                "Assembled from this run's own recorded outputs. No inference ran to produce it "
                "and no language model was involved; POST /api/copilot/explain-analysis adds a "
                "natural-language layer over exactly these values."
            ),
        }

    # --------------------------------------------------------------------- run
    def run(
        self,
        image_path: str | None = None,
        question: str | None = None,
        intent: str | None = None,
        analysis_id: str | None = None,
        top_k: int = 5,
    ) -> dict:
        """Answer one question about one image, and say what that took.

        Either `image_path` or an `analysis_id` from an earlier run is required.
        Passing the id alone is the zero-inference path: every stage comes back
        reused, which is what makes a follow-up question cheap rather than a
        second full analysis.
        """
        routing = resolve_intent(question, intent)
        chosen = INTENTS_BY_NAME[routing["resolved"]]

        if image_path is not None:
            state = self._state_for(image_path)
        elif analysis_id is not None:
            state = self.state_by_id(analysis_id)
            if state is None:
                raise OrchestratorError(
                    f"No cached analysis '{analysis_id}'. Re-submit the image; the orchestrator "
                    f"keeps the most recent {self._cache_size} analyses in memory."
                )
        else:
            raise OrchestratorError("Provide an image or the analysis_id of an earlier run.")

        planned = plan_for(chosen.name)
        for name in planned:
            params = (top_k,) if name == "similarity" else ()
            self._execute(state, name, params)

        seeds = self._seeds(state)
        # Reported in plan order first, then anything an earlier question left in
        # the cache: a reader should see what THIS question needed without losing
        # what is already known about the image. The two are never conflated --
        # an out-of-plan entry is restated as "cached" so a stage an earlier
        # question ran can never be read as work this one caused.
        in_plan = set(planned)
        stages = []
        for name in planned + [n for n in state.stages if n not in in_plan]:
            output = state.stages.get(name)
            if output is None:
                continue
            entry = output.as_dict()
            if name not in in_plan:
                entry["status"] = "cached"
                entry["duration_ms"] = None
            stages.append(entry)

        # Scoped to this question's plan for the same reason: a segmentation
        # refusal from a previous question is not something THIS question asked
        # for and was denied.
        unavailable = [
            {
                "capability": STAGES[name].describe,
                "stage": name,
                "reason": state.stages[name].reason,
                "message": state.stages[name].message,
            }
            for name in planned
            if name in state.stages and state.stages[name].status in ("unavailable", "failed")
        ]
        if chosen.capability:
            cap = UNBUILT[chosen.capability]
            unavailable.append({
                "capability": cap.label,
                "stage": None,
                "reason": "not_implemented",
                "message": f"{cap.label} is not available. It is owned by {cap.owner}.",
            })

        warnings = []
        if detect := state.stages.get("detect"):
            if detect.usable and not detect.value:
                warnings.append("No seeds detected above the confidence threshold.")

        resolution = state.stages.get("resolution")
        return {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "analysis_id": state.analysis_id,
            "image_digest": state.digest,
            "image_path": state.image_path,
            "intent": routing,
            "plan": [
                {
                    "stage": name,
                    "tasks": list(STAGES[name].tasks),
                    "expensive": STAGES[name].expensive,
                    "describe": STAGES[name].describe,
                }
                for name in planned
            ],
            "stages": stages,
            "executed": [n for n in planned if n in state.stages and state.stages[n].status == "ran"],
            "reused": [n for n in planned if n in state.stages and state.stages[n].status == "reused"],
            "seed_count": len(seeds),
            "seeds": seeds,
            "segmentation": resolution.value if resolution and resolution.usable else None,
            "answer": self._answer(state, chosen, seeds),
            "unavailable": unavailable,
            "warnings": warnings,
        }

    # ------------------------------------------------------------ capabilities
    def capabilities(self) -> dict:
        """What each question can honestly be answered with, right now.

        Derived from the live registry every call, for the same reason Phase 14
        reads system-info from disk: a written-down capability list is a claim
        that goes stale the moment a model is declared or retired.
        """
        registry = self.pipeline.registry
        served = {}
        for task in registry.tasks:
            try:
                record = registry.resolve(task)
            except TaskUnavailableError as e:
                served[task] = {"served": False, "model": None, "detail": str(e)}
            else:
                served[task] = {"served": True, "model": record.key, "detail": None}

        entries = []
        for intent in INTENTS:
            stages = plan_for(intent.name)
            blocking = sorted({
                task
                for name in stages
                for task in STAGES[name].tasks
                if not served[task]["served"]
            })
            unbuilt = UNBUILT[intent.capability] if intent.capability else None
            entries.append({
                "intent": intent.name,
                "question": intent.canonical,
                "stages": stages,
                "answerable": not blocking and unbuilt is None,
                "blocked_by_tasks": blocking,
                "not_implemented": None if unbuilt is None else {
                    "capability": unbuilt.label,
                    "owner": unbuilt.owner,
                    "note": unbuilt.note,
                },
            })

        return {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "tasks": served,
            "intents": entries,
            "note": (
                "Model selection is the orchestrator's, not the user's: a question names a "
                "capability and the registry names the model. An intent listed as not "
                "answerable is reported unavailable rather than approximated with a model "
                "trained for something else."
            ),
        }
