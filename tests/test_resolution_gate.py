"""Phase 8: the platform refuses to answer pixel-level questions it cannot answer.

Two refusals live here and the tests keep them apart on purpose. One says "no model
is allowed to serve this task"; the other says "a model could, but not about this
image". Only the second is fixable by re-photographing the sample, so a test suite
that let them blur into a single "unavailable" would be certifying the exact mistake
this phase exists to prevent.

The floor itself is not an opinion. outputs/metrics/segmentation_resolution_floor.json
round-trips TRUE masks through a downscale/upscale at each resolution, which is an
information-loss ceiling: no segmenter can beat it at a given kernel size. Below it a
mask is not a worse measurement, it is an unmeasured one -- so these tests check that
no mask comes back at all, not that a poor one is flagged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.pipeline import resolution_gate
from src.pipeline.unified_pipeline import AnalysisPipeline, ModelNotAvailableError
from src.registry import get_registry

registry = get_registry()
SEG_KEY = "defect_segmenter_synthetic"
FLOOR_FILE = Path("outputs/metrics/segmentation_resolution_floor.json")


# ------------------------------------------------------- measuring the kernel
def test_kernel_size_is_the_short_side_of_the_box():
    """A kernel photographed at 200x30 px carries the detail of a 30 px object.
    The floor was measured with the short side (validate_resolution_floor.py:
    kernel_short_side), so the gate must compare the same quantity -- taking the
    long side would pass images that carry none of the detail the floor requires."""
    assert resolution_gate.kernel_px_from_bbox([10, 20, 210, 50]) == 30
    assert resolution_gate.kernel_px_from_bbox([10, 20, 40, 220]) == 30


def test_kernel_size_survives_a_box_given_in_either_corner_order():
    assert resolution_gate.kernel_px_from_bbox([210, 50, 10, 20]) == 30


def test_a_whole_frame_kernel_is_measured_by_its_short_side_too():
    assert resolution_gate.kernel_px_from_size(400, 260) == 260


# ------------------------------------------------------------------ the gate
def test_the_gate_reads_its_floor_from_the_registry_not_from_a_constant():
    """Retiring the synthetic segmenter for a real one must change the gate by
    editing the YAML the model was declared in, not by editing this module."""
    record = registry.get(SEG_KEY)
    declared = record.requires_resolution["min_kernel_px"]
    verdict = resolution_gate.check(record, declared)
    assert verdict.min_kernel_px == declared


def test_a_kernel_exactly_at_the_floor_is_sufficient():
    """The floor is the smallest size the round trip still met this project's
    tolerances at, so it is inclusive. An off-by-one here would silently discard
    the measurement the whole phase rests on."""
    record = registry.get(SEG_KEY)
    floor = record.requires_resolution["min_kernel_px"]
    assert resolution_gate.check(record, floor).sufficient is True
    assert resolution_gate.check(record, floor - 1).sufficient is False


def test_a_sufficient_verdict_carries_no_refusal_message():
    record = registry.get(SEG_KEY)
    verdict = resolution_gate.check(record, 400)
    assert verdict.sufficient is True
    assert verdict.message is None


def test_the_refusal_uses_the_exact_sentence_phase_8_requires():
    record = registry.get(SEG_KEY)
    verdict = resolution_gate.check(record, 4)
    assert verdict.message == "Segmentation unavailable — insufficient image resolution."
    assert verdict.message == resolution_gate.SEGMENTATION_UNAVAILABLE


def test_a_refusal_shows_the_threshold_and_where_it_came_from():
    """"Too small" is only actionable if the reader can see the number they missed
    and check the file that set it."""
    record = registry.get(SEG_KEY)
    verdict = resolution_gate.check(record, 4).as_dict()
    assert verdict["min_kernel_px"] == record.requires_resolution["min_kernel_px"]
    assert verdict["kernel_px"] == 4
    assert FLOOR_FILE.as_posix() in verdict["source"]


def test_a_model_that_declares_no_requirement_is_not_silently_gated_at_zero():
    """An absent declaration means "this model has no measured floor", which must
    not be read as "this model refuses everything"."""
    record = registry.get("unified_seed_model")
    assert record.requires_resolution is None
    verdict = resolution_gate.check(record, 1)
    assert verdict.sufficient is True
    assert verdict.min_kernel_px is None


# ------------------------------------------------------------ declared floors
def test_the_floor_is_declared_even_though_no_model_serves_the_task():
    """The whole point of reading requires_resolution off the task's records rather
    than through resolve(): an unavailable capability still has a measured
    requirement, and the response says both things at once."""
    # for_task() lists SERVERS and is correctly empty here; the record still exists.
    assert registry.for_task("defect_segmentation") == []
    assert any(r.task == "defect_segmentation" for r in registry.all())
    with pytest.raises(Exception):
        registry.resolve("defect_segmentation")
    floor = resolution_gate.declared_floor(registry, "defect_segmentation")
    assert floor["min_kernel_px"] == 24


def test_a_task_with_no_declared_floor_reports_none_rather_than_a_default():
    assert resolution_gate.declared_floor(registry, "variety") is None


def test_the_declared_floor_matches_the_file_that_measured_it():
    """A number typed into the YAML that drifted from the experiment would be a
    fabricated threshold wearing a measurement's clothes."""
    measured = json.loads(FLOOR_FILE.read_text(encoding="utf-8"))
    floor = resolution_gate.declared_floor(registry, "defect_segmentation")
    assert floor["min_kernel_px"] == measured["resolution_floor_kernel_px"]


# --------------------------------------------------- the two refusals, apart
def test_an_unserved_task_refuses_on_capability_not_on_resolution():
    """Resolved by task, with a crop far above any floor. If this ever came back
    as a resolution problem it would send an operator to buy a better camera for a
    capability that does not exist."""
    pipeline = AnalysisPipeline()
    with pytest.raises(ModelNotAvailableError) as excinfo:
        pipeline.segment_defects(Image.new("RGB", (512, 512)))
    assert "resolution" not in str(excinfo.value).lower()


def test_status_reports_the_capability_refusal_and_the_measurement_together():
    """Both facts are true and a reader needs both: nothing serves this task, and
    one of your three seeds would have been too small even if something did."""
    status = AnalysisPipeline().segmentation_status([12, 30, 40])
    assert status["status"] == "unavailable"
    assert status["reason"] == "no_served_model"
    assert status["resolution"]["min_kernel_px"] == 24
    assert status["resolution"]["seeds_measured"] == 3
    assert status["resolution"]["seeds_below_floor"] == 1
    assert status["resolution"]["kernel_px_median"] == 30


def test_status_with_no_seeds_still_reports_the_floor():
    status = AnalysisPipeline().segmentation_status([])
    assert status["resolution"]["min_kernel_px"] == 24
    assert status["resolution"]["seeds_measured"] == 0
    assert status["resolution"]["kernel_px_median"] is None


def test_status_never_carries_a_mask_or_an_area():
    """This method measures the submitted image; it must not look like it measured
    a defect. Phase 2 owns coverage, and only once a real segmenter serves."""
    status = AnalysisPipeline().segmentation_status([30, 40])
    assert "masks" not in status
    assert not any("area" in k or "coverage" in k for k in status["resolution"])


# ------------------------------------------------- no mask below the floor
@pytest.mark.skipif(
    not registry.get(SEG_KEY).present, reason=f"{SEG_KEY} checkpoint not on disk"
)
def test_below_the_floor_no_mask_is_produced_at_all():
    """Not a low-confidence mask, not a smaller one -- no mask. The explicit
    model_key opts past the capability refusal so that the RESOLUTION branch is the
    only thing under test here."""
    pipeline = AnalysisPipeline()
    out = pipeline.segment_defects(
        Image.new("RGB", (512, 512)), bbox=[0, 0, 20, 12], model_key=SEG_KEY
    )
    assert out["available"] is False
    assert out["reason"] == "insufficient_resolution"
    assert out["message"] == resolution_gate.SEGMENTATION_UNAVAILABLE
    assert "masks" not in out


@pytest.mark.skipif(
    not registry.get(SEG_KEY).present, reason=f"{SEG_KEY} checkpoint not on disk"
)
def test_above_the_floor_a_mask_comes_back_stamped_with_its_provenance():
    """The one path that produces pixels is reachable only by naming a synthetic
    model explicitly, and what comes back says so -- otherwise a mask painted from
    this project's own transformations could be read as a real-defect finding."""
    pipeline = AnalysisPipeline()
    out = pipeline.segment_defects(Image.new("RGB", (256, 256)), model_key=SEG_KEY)
    assert out["available"] is True
    assert out["is_synthetic_model"] is True
    assert "synthetic" in (out["label_note"] or "").lower()
    for channel, mask in out["masks"].items():
        assert mask.shape == (256, 256), f"{channel} mask is not the crop's own size"
        assert set(mask.flatten().tolist()) <= {0, 1}, f"{channel} mask is not binary"


@pytest.mark.skipif(
    not registry.get(SEG_KEY).present, reason=f"{SEG_KEY} checkpoint not on disk"
)
def test_thresholds_are_the_ones_tuned_on_val_and_are_json_safe():
    """Read from the checkpoint, not assumed: a segmenter retrained with different
    channels would otherwise be read with the previous run's meanings. float()
    because these travel out through an API and numpy scalars are not JSON."""
    out = AnalysisPipeline().segment_defects(
        Image.new("RGB", (256, 256)), model_key=SEG_KEY
    )
    assert set(out["thresholds"]) == set(out["channels"])
    for value in out["thresholds"].values():
        assert type(value) is float
    json.dumps(out["thresholds"])
