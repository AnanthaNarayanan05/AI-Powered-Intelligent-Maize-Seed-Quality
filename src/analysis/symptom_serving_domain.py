"""Measures what the visible-symptom gate actually does on the images users submit.

WHY THIS EXISTS
---------------
Phase 4 shipped a gate whose threshold was calibrated on one image domain and
then applied to another, and the two behaved nothing alike: a budget of 5%
became a flag rate of 80% on real uploads. That was found by measuring the
serving domain rather than trusting the calibration domain, and the same check
is owed to every gate this project ships.

The visible-symptom classifier is trained and calibrated entirely on GrainSpace
M600 kernel crops. The images the platform is actually given are YOLO crops out
of user uploads, which come predominantly from the Mendeley quality set. Nothing
guarantees the gate behaves the same way on those, so this script measures it.

WHAT IT CAN AND CANNOT SAY
--------------------------
It can measure COVERAGE -- how often the gate commits to a category and how
often it withholds, and which categories it names. That is a real, checkable
property of the serving domain.

It cannot measure ACCURACY there. The uploads carry no visible-condition labels;
nobody graded them. Producing an accuracy number for this domain would mean
inventing the labels to score against, so none is produced and none may be
quoted. The held-out accuracy in symptom_gate.json describes the GrainSpace test
split and nothing else.

The distribution of names is reported for the same reason the coverage is: if
the gate commits almost exclusively to one class on real uploads, that is a fact
a reader needs when interpreting a report, whether it reflects genuinely clean
kernels or a representation that does not transfer. This script does not decide
between those two readings, because the data to decide it does not exist.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.unified_pipeline import AnalysisPipeline, ModelNotAvailableError  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("symptom_serving_domain")

UPLOADS = "data_processed/uploads/*.jpg"
OUT_PATH = "outputs/metrics/symptom_serving_domain.json"


def measure(pipeline: AnalysisPipeline, paths: list[str]) -> dict:
    status: Counter = Counter()
    named: Counter = Counter()
    argmax: Counter = Counter()
    images, unreadable, kernels = 0, 0, 0

    for path in paths:
        try:
            result = pipeline.analyze_image(path, run_similarity=False)
        except Exception as e:  # truncated JPEGs are a known property of this folder
            unreadable += 1
            logger.info(f"skipped {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue
        images += 1
        for seed in result["seeds"]:
            sp = seed.get("symptom_prediction")
            if not sp:
                continue
            kernels += 1
            key = sp["status"] if sp["status"] != "withheld" else f"withheld:{sp['reason']}"
            status[key] += 1
            argmax[sp["argmax_class_before_gate"]] += 1
            if sp["status"] == "reported":
                named[sp["predicted_class"]] += 1

    return {
        "images_analysed": images,
        "images_unreadable": unreadable,
        "kernels_scored": kernels,
        "status_counts": dict(status),
        "coverage": (status["reported"] / kernels) if kernels else None,
        "reported_class_counts": dict(named),
        "argmax_class_counts_before_gate": dict(argmax),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=UPLOADS)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))[: args.limit]
    if not paths:
        raise SystemExit(f"no images matched {args.glob!r}")

    pipeline = AnalysisPipeline()
    try:
        gate = pipeline.symptom_status()
    except ModelNotAvailableError as e:
        raise SystemExit(str(e))
    if gate["status"] != "available":
        raise SystemExit(f"visible-symptom capability is not served: {gate['message']}")

    stats = measure(pipeline, paths)
    payload = {
        "source": args.glob,
        "domain": ("YOLO kernel crops from real user uploads (data_processed/uploads), "
                   "predominantly Mendeley-domain imagery"),
        "calibration_domain": "GrainSpace M600 kernel crops",
        "confidence_threshold": gate["confidence_threshold"],
        "validated_classes": gate["validated_classes"],
        "held_out_coverage": gate["held_out_coverage"],
        "held_out_accuracy": gate["held_out_accuracy"],
        **stats,
        "accuracy": None,
        "accuracy_note": (
            "Deliberately null. These uploads carry no visible-condition labels, so "
            "no accuracy can be computed here without inventing the ground truth to "
            "score against. Only the GrainSpace held-out accuracy is a measured "
            "figure, and it describes that split alone."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    n = stats["kernels_scored"]
    logger.info(f"{stats['images_analysed']} images, {n} kernels "
                f"({stats['images_unreadable']} unreadable)")
    for k, v in sorted(stats["status_counts"].items(), key=lambda kv: -kv[1]):
        logger.info(f"  {k:<30} {v:5d}  {v / n:.3f}" if n else f"  {k}: {v}")
    logger.info(f"  serving-domain coverage {stats['coverage']:.3f} vs "
                f"held-out {gate['held_out_coverage']:.3f}")
    logger.info(f"  named: {stats['reported_class_counts']}")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
