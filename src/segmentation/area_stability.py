"""Is defect AREA a stable quantity for each defect channel?

This asks a question that comes BEFORE model quality, and that no amount of training
can change. Take the ground-truth mask -- a perfect prediction -- and move its
boundary by one pixel, the smallest error any segmenter can make. Then read the
VISIBLE DEFECT AREA % off it again. If a one-pixel boundary error changes the
reported percentage by 100%, then the percentage is not a measurement; it is a
number that happens to be printed, and the honest thing is to not print it.

Why one pixel: a segmenter's decision boundary is a thresholded probability field.
Sub-pixel boundary placement is not available to it, so +/-1px on the boundary is the
floor of achievable error, not a pessimistic assumption.

Why this separates cracks from blobs: for a compact region, perimeter grows as the
square root of area, so a 1px dilation adds a small FRACTION of the area. For a
1px-wide line, the region IS its own boundary, so a 1px dilation roughly triples it.
That is geometry, not a training deficiency, and it is the reason this project
reports crack DETECTION but not crack AREA.

Reads the same manifest the segmenter trains on, at the same 224px working
resolution, so the numbers apply to the deployed pipeline rather than to the
originals on disk.

Usage:
    python -m src.segmentation.area_stability
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from src.segmentation.build_synthetic_manifest import DEFECT_CHANNELS

# A single cut-off would decide a knife-edge case by rounding: on this data
# insect_damaged lands at 25.7%, so a 25% line would call it unmeasurable and a 26%
# line would call it exact, which is a property of the line and not of the seed.
# Three tiers instead, because what changes across them is what may honestly be SAID:
#
#   QUANTITATIVE  <10%  the printed percentage is worth its digits
#   BANDED     10-50%   the number cannot be quoted, but it still separates "a little"
#                       from "a lot", which is all severity scoring needs
#   DETECTION_ONLY >50% the error bar spans the value; no area figure at all
#
# The error here is a SYSTEMATIC boundary shift, which is exactly what choosing a
# probability threshold does to a mask -- raise the threshold and every boundary moves
# inward at once. It is the worst case for a 1px error, not the average one; random
# per-pixel errors partly cancel and this does not.
TIER_QUANTITATIVE_MAX_REL_ERR_PCT = 10.0
TIER_BANDED_MAX_REL_ERR_PCT = 50.0


def area_tier(worst_rel_err_pct: float) -> str:
    if worst_rel_err_pct <= TIER_QUANTITATIVE_MAX_REL_ERR_PCT:
        return "QUANTITATIVE"
    if worst_rel_err_pct <= TIER_BANDED_MAX_REL_ERR_PCT:
        return "BANDED"
    return "DETECTION_ONLY"


def _coverage_pct(mask: np.ndarray, body: np.ndarray) -> float:
    b = int(body.sum())
    return 100.0 * int((mask & body).sum()) / b if b else 0.0


def measure(manifest: str, image_size: int = 224, per_class: int = 300,
            seed: int = 0) -> dict:
    rows = [r for r in csv.DictReader(open(manifest)) if r["label"] in DEFECT_CHANNELS]
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)

    k = np.ones((3, 3), np.uint8)   # 8-connected: exactly one pixel of boundary motion
    rng = np.random.default_rng(seed)
    out = {}

    for label, group in sorted(by.items()):
        pick = [group[i] for i in rng.permutation(len(group))[:per_class]]
        rel_dil, rel_ero, true_cov, widths = [], [], [], []
        for r in pick:
            m = cv2.imread(r["defect_mask_path"], cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(r["body_mask_path"], cv2.IMREAD_GRAYSCALE)
            if m is None or b is None:
                continue
            s = image_size
            mm = cv2.resize((m > 127).astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
            bb = cv2.resize((b > 127).astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
            base = _coverage_pct(mm, bb)
            if base <= 0:
                continue
            true_cov.append(base)
            rel_dil.append(100.0 * abs(_coverage_pct(cv2.dilate(mm, k, 1), bb) - base) / base)
            rel_ero.append(100.0 * abs(_coverage_pct(cv2.erode(mm, k, 1), bb) - base) / base)
            widths.append(float(2 * cv2.distanceTransform(mm, cv2.DIST_L2, 3).max()))

        q = lambda a, p: round(float(np.percentile(a, p)), 3) if len(a) else None
        worst = max(q(rel_dil, 50) or 0.0, q(rel_ero, 50) or 0.0)
        out[label] = {
            "n": len(true_cov),
            "true_coverage_pct_median": q(true_cov, 50),
            "max_stroke_width_px_median": q(widths, 50),
            "rel_err_pct_median_dilate1px": q(rel_dil, 50),
            "rel_err_pct_median_erode1px": q(rel_ero, 50),
            "rel_err_pct_p90_dilate1px": q(rel_dil, 90),
            "worst_median_rel_err_pct": round(worst, 3),
            "area_tier": area_tier(worst),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data_processed/manifest_segmentation_synthetic.csv")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--out", default="outputs/metrics/segmentation_area_stability.json")
    args = ap.parse_args()

    res = measure(args.manifest, args.image_size, args.per_class)
    payload = {
        "instrument": "1px dilate/erode of the GROUND-TRUTH mask = best case achievable "
                      "boundary error; the resulting change in VISIBLE DEFECT AREA % is a "
                      "lower bound on any segmenter's area error",
        "working_resolution_px": args.image_size,
        "tier_thresholds_rel_err_pct": {
            "QUANTITATIVE": TIER_QUANTITATIVE_MAX_REL_ERR_PCT,
            "BANDED": TIER_BANDED_MAX_REL_ERR_PCT,
        },
        "per_channel": res,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)

    print(f"1px boundary perturbation of the TRUE mask, at {args.image_size}px:\n")
    print(f"{'channel':18s} {'width px':>9s} {'true cov%':>10s} {'+1px err':>10s} "
          f"{'-1px err':>10s}   verdict")
    for ch, d in res.items():
        print(f"{ch:18s} {d['max_stroke_width_px_median']:9.2f} "
              f"{d['true_coverage_pct_median']:9.3f}% "
              f"{d['rel_err_pct_median_dilate1px']:9.1f}% "
              f"{d['rel_err_pct_median_erode1px']:9.1f}%   "
              f"{d['area_tier']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
