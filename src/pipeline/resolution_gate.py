"""Is this image large enough for the answer to mean anything?

Phase 8. The floor enforced here is not a preference: it comes from
outputs/metrics/segmentation_resolution_floor.json, which round-trips TRUE masks
through a downscale/upscale at each resolution. That is an information-loss
ceiling -- no segmenter, however good, can beat it at a given kernel size. Below
24 px of kernel short side even a perfect mask falls under this project's own
tolerances (IoU median >= 0.85, area error p90 <= 0.05), so a mask produced there
would be pixels without a measurement behind it.

The gate therefore refuses rather than degrades. That refusal is deliberately a
different fact from the registry's: Phase 7 answers "no model is allowed to serve
this task", this module answers "a model may serve it, but not on this image".
Collapsing the two would tell a user to find a better camera when the real answer
is that the capability does not exist, or the reverse.

Nothing here loads a model or touches an image buffer -- it only decides, so the
decision can be tested without a GPU and reported without running inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# The wording Phase 8 requires, kept in one place so the API, the tests and the
# frontend all quote the same sentence rather than three paraphrases of it.
SEGMENTATION_UNAVAILABLE = "Segmentation unavailable — insufficient image resolution."

_MESSAGES = {
    "defect_segmentation": SEGMENTATION_UNAVAILABLE,
}


def _message_for(task: str) -> str:
    return _MESSAGES.get(task, f"{task} unavailable — insufficient image resolution.")


def kernel_px_from_bbox(bbox) -> int:
    """Kernel size in pixels, measured the way the floor was measured.

    src/segmentation/validate_resolution_floor.py:kernel_short_side takes the
    SHORT side of the region's pixel extent, because a kernel photographed at
    200x30 px carries the detail of a 30 px object, not a 200 px one. A detector
    bbox is that same extent, so the number the gate compares is the number the
    instrument produced -- not a rescaled stand-in for it.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return int(min(abs(x2 - x1), abs(y2 - y1)))


def kernel_px_from_size(width: int, height: int) -> int:
    """Same measure for a single-kernel image with no detection to bound it.

    This assumes the kernel fills the frame, which is what a cropped upload is.
    It over-states the kernel for a wide shot -- but only in the permissive
    direction, so it is paired with detection wherever detection exists.
    """
    return int(min(width, height))


@dataclass(frozen=True)
class ResolutionVerdict:
    """A decision, plus everything needed to justify it to the person affected.

    `min_kernel_px` and `source` travel with the verdict on purpose: "too small"
    is only actionable if the reader can see the threshold and where it came
    from. `sufficient` is True when the model declares no requirement -- an
    absent declaration is not a silent zero-tolerance gate.
    """

    task: str
    model: str
    sufficient: bool
    kernel_px: int
    min_kernel_px: int | None
    source: str | None
    message: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def check(record, kernel_px: int, task: str | None = None) -> ResolutionVerdict:
    """Decide whether `record` may run on a kernel of `kernel_px` pixels.

    `record` is a registry ModelRecord; the requirement is read from it rather
    than from a constant here so that retiring the synthetic segmenter for a real
    one changes the gate by editing the YAML the model was declared in.
    """
    task = task or record.task
    requirement = record.requires_resolution or {}
    minimum = requirement.get("min_kernel_px")

    if minimum is None:
        return ResolutionVerdict(
            task=task,
            model=record.key,
            sufficient=True,
            kernel_px=kernel_px,
            min_kernel_px=None,
            source=None,
            message=None,
        )

    sufficient = kernel_px >= int(minimum)
    return ResolutionVerdict(
        task=task,
        model=record.key,
        sufficient=sufficient,
        kernel_px=kernel_px,
        min_kernel_px=int(minimum),
        source=requirement.get("source"),
        message=None if sufficient else _message_for(task),
    )


def declared_floor(registry, task: str) -> dict | None:
    """The floor declared for `task`, whether or not a model currently serves it.

    A capability that is unavailable still has a measured requirement, and the
    analysis response says so: "no model serves this" and "your kernels are 12 px
    across" are both true and a reader needs both. Reading it off the records for
    the task -- rather than resolve() -- is what makes that possible. When more
    than one model declares the task the strictest floor wins, because reporting
    the laxest would promise a resolution the served model may not accept.
    """
    declared = [
        r.requires_resolution
        for r in registry.all()
        if r.task == task and r.requires_resolution
    ]
    if not declared:
        return None
    strictest = max(declared, key=lambda d: int(d["min_kernel_px"]))
    return {
        "min_kernel_px": int(strictest["min_kernel_px"]),
        "source": strictest.get("source"),
    }
