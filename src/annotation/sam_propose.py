"""SAM-assisted defect-mask PROPOSALS for real maize kernels.

This script produces candidates, never labels. Every mask it writes is stamped
status=proposed and lives under data_processed/annotation/proposals/. Nothing
here is training data until a human has looked at it in the review tool and
recorded a decision; src/annotation/build_real_manifest.py refuses to admit any
row that has not been through that step.

WHY A HUMAN IS NOT OPTIONAL. The saliency cue driving these proposals is colour
distance from healthy maize. Specular highlights, cast shadows, the pedicel scar
and the pale crown of a perfectly sound kernel all score high on that cue. If
these proposals were accepted wholesale the resulting "defect segmentation"
model would be an expensive reimplementation of a colour threshold, and its IoU
would be measuring agreement with that threshold rather than with a defect.

WHICH DEFECTS ARE SEGMENTED. Only the GrainSpace classes whose defect is
genuinely a REGION of the kernel:
    AP (attacked by pests) -> insect_damaged   -- bore holes and galleries
    BN (broken)            -> cracked          -- fracture faces
    FM (fusarium/mildew)   -> discolored_mold  -- discoloured/mouldy patches
    MY (mouldy)            -> discolored_mold
    NOR (normal)           -> no defect, body only (the clean negatives)
HD (heat damaged) and SD (sprouted) are deliberately excluded. Both are
whole-kernel conditions: the honest mask for a uniformly heat-damaged kernel is
its entire body, which is a degenerate segmentation target that would inflate
IoU while teaching the model nothing about localisation. They remain available
to the visible-symptom CLASSIFIER, which is the right tool for a whole-kernel
condition.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import warnings

import cv2
import numpy as np

from src.annotation.healthy_reference import REFERENCE_PATH, load, saliency
from src.annotation.kernel_body import propose_body
from src.annotation.splits import assign_split, plate_id

# GrainSpace condition code -> this project's existing defect channel. Keeping
# the synthetic model's channel names means the real segmenter is a drop-in
# replacement for it and the two sets of metrics are directly comparable.
CLASS_TO_CHANNEL = {
    "AP": "insect_damaged",
    "BN": "cracked",
    "FM": "discolored_mold",
    "MY": "discolored_mold",
    "NOR": None,
}
EXCLUDED_CLASSES = {
    "HD": "whole-kernel discolouration, not a localisable region",
    "SD": "whole-kernel structural change, not a localisable region",
}

PROPOSAL_ROOT = "data_processed/annotation/proposals"
PROPOSAL_CSV = "data_processed/annotation/proposals.csv"

# Smallest region this project is willing to call measurable, carried over from
# outputs/metrics/segmentation_resolution_floor.json so the annotation stage and
# the evaluation stage agree on what counts as a region.
MIN_REGION_PX = 8

# Saliency thresholds swept to generate candidates. Low catches faint mould,
# high catches only the strongest bore holes; the reviewer picks.
SEED_THRESHOLDS = (2.5, 3.5, 4.5)

# The body boundary is where SAM is least certain, and one row of misassigned
# background pixels is maize-vs-white -- the largest colour distance in the
# frame. Seeds are therefore only sought inside an eroded body, though a
# SAM-refined candidate may still grow back out to the true edge, because a real
# fracture face genuinely does reach the kernel outline.
RIM_ERODE_PX = 3

MAX_CANDIDATES = 5


def _seed_regions(sal: np.ndarray, inner: np.ndarray) -> list[dict]:
    """Connected components of thresholded saliency, ranked by salience mass."""
    regions = []
    for thr in SEED_THRESHOLDS:
        hot = ((sal > thr) & inner).astype(np.uint8)
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_lab, lab, stats, cent = cv2.connectedComponentsWithStats(hot)
        for i in range(1, n_lab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_REGION_PX:
                continue
            comp = lab == i
            regions.append({
                "threshold": thr,
                "area": area,
                "mass": float(area * sal[comp].mean()),
                "centroid": (float(cent[i][0]), float(cent[i][1])),
                "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                         int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])),
                "mask": comp,
            })
    regions.sort(key=lambda r: r["mass"], reverse=True)
    return regions


def _refine_with_sam(region: dict, sal: np.ndarray, body: np.ndarray, predictor) -> np.ndarray | None:
    """Snap a colour-threshold region onto real image boundaries with SAM.

    The colour blob says roughly where; SAM says exactly where. Prompting with
    both a box and interior points matters: the box alone tends to return the
    whole kernel when the defect is large, and points alone drift onto
    neighbouring texture when the defect is small.
    """
    comp = region["mask"]
    ys, xs = np.nonzero(comp)
    if len(xs) == 0:
        return None
    # Up to three interior prompts, taken at the most salient pixels of the
    # component rather than its centroid -- for a crescent-shaped fracture face
    # the centroid can land outside the region entirely.
    order = np.argsort(sal[ys, xs])[::-1]
    take = [order[0]]
    for idx in order[1:]:
        if len(take) >= 3:
            break
        if all((xs[idx] - xs[t]) ** 2 + (ys[idx] - ys[t]) ** 2 > 64 for t in take):
            take.append(idx)
    points = np.array([[xs[t], ys[t]] for t in take])
    labels = np.ones(len(points), dtype=int)

    x0, y0, x1, y1 = region["bbox"]
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=np.array([x0, y0, x1, y1]),
        multimask_output=True,
    )
    best, best_score = None, -1.0
    for mask, score in zip(masks, scores):
        cand = mask & body
        frac = cand.sum() / max(body.sum(), 1)
        # A "defect" covering essentially the whole kernel is SAM returning the
        # kernel, not the defect. The genuinely-whole-kernel case is offered
        # separately as its own candidate, so it is not lost by this guard.
        if cand.sum() < MIN_REGION_PX or frac > 0.97:
            continue
        if score > best_score:
            best, best_score = cand, float(score)
    return best


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def propose_for_image(image: np.ndarray, predictor, mu, icov) -> dict:
    """Body mask plus a short, deduplicated list of defect candidates."""
    predictor.set_image(image)
    body, plausible = propose_body(image, predictor)
    sal = saliency(image, body, mu, icov)
    inner = cv2.erode(body.astype(np.uint8), np.ones((RIM_ERODE_PX * 2 + 1,) * 2, np.uint8)).astype(bool)
    if inner.sum() < MIN_REGION_PX:
        inner = body

    candidates: list[dict] = []

    def add(mask, source):
        if mask is None or mask.sum() < MIN_REGION_PX:
            return
        # Reject crescents that hug the body outline. SAM's body edge is loose by
        # a pixel or two, and just outside it sits the white plate -- the largest
        # colour distance anywhere in the frame -- so a thin shell of background
        # scores as the most "defective" region on an otherwise sound kernel. A
        # real defect has substance inside the kernel; a rim artifact does not.
        if (mask & inner).sum() < 0.30 * mask.sum():
            return
        for existing in candidates:
            if _iou(existing["mask"], mask) > 0.85:
                return
        candidates.append({"mask": mask, "source": source,
                           "coverage_pct": float(mask.sum() / max(body.sum(), 1) * 100)})

    regions = _seed_regions(sal, inner)
    for region in regions[:MAX_CANDIDATES]:
        add(_refine_with_sam(region, sal, body, predictor), f"sam@{region['threshold']}")
        if len(candidates) >= MAX_CANDIDATES:
            break

    # The raw colour-threshold union, as a fallback for diffuse mould that has no
    # boundary for SAM to snap to.
    for thr in SEED_THRESHOLDS:
        union = ((sal > thr) & inner)
        union = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        add(union, f"colour@{thr}")

    # Whole body, for the uniformly-affected kernel where that IS the answer.
    add(body.copy(), "whole_body")

    return {"body": body, "body_plausible": plausible, "candidates": candidates,
            "saliency_max": float(sal.max()), "n_seed_regions": len(regions)}


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-root", default="datasets/grainspace_maize_m600/val")
    ap.add_argument("--sam-checkpoint", default="weights/sam_b.pt")
    ap.add_argument("--reference", default=REFERENCE_PATH)
    ap.add_argument("--out-root", default=PROPOSAL_ROOT)
    ap.add_argument("--out-csv", default=PROPOSAL_CSV)
    ap.add_argument("--per-class", type=int, default=0,
                    help="Cap images per class (0 = all). Proposal generation is "
                         "cheap; human review is not, so this exists to size a "
                         "review session rather than to save GPU time.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint).to(device).eval()
    predictor = SamPredictor(sam)
    mu, icov = load(args.reference)

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows, n_implausible = [], 0
    for cls in sorted(CLASS_TO_CHANNEL):
        files = sorted(glob.glob(os.path.join(args.image_root, cls, "*.png")))
        if args.per_class:
            files = files[: args.per_class]
        for path in files:
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = propose_for_image(image, predictor, mu, icov)
            if not out["body_plausible"]:
                n_implausible += 1

            stem = os.path.splitext(os.path.basename(path))[0]
            dest = os.path.join(args.out_root, cls, stem)
            os.makedirs(dest, exist_ok=True)
            cv2.imwrite(os.path.join(dest, "body.png"), out["body"].astype(np.uint8) * 255)

            cands = []
            # NOR carries no defect candidates by construction: a normal kernel's
            # annotation is an empty defect mask, and offering the reviewer a
            # candidate would invite them to confirm one that is not there.
            if CLASS_TO_CHANNEL[cls] is not None:
                for i, cand in enumerate(out["candidates"]):
                    name = f"cand_{i:02d}.png"
                    cv2.imwrite(os.path.join(dest, name), cand["mask"].astype(np.uint8) * 255)
                    cands.append({"file": name, "source": cand["source"],
                                  "coverage_pct": round(cand["coverage_pct"], 3)})

            rows.append({
                "image_path": path.replace("\\", "/"),
                "grainspace_class": cls,
                "defect_channel": CLASS_TO_CHANNEL[cls] or "",
                "proposal_dir": dest.replace("\\", "/"),
                "body_mask": "body.png",
                "body_plausible": int(out["body_plausible"]),
                "body_px": int(out["body"].sum()),
                "n_candidates": len(cands),
                "candidates_json": json.dumps(cands),
                "saliency_max": round(out["saliency_max"], 3),
                "group": plate_id(path),
                "split": assign_split(plate_id(path)),
                "status": "proposed",
                "reviewer": "",
                "reviewed_at": "",
                "decision": "",
                "final_mask": "",
                "review_note": "",
            })
        print(f"  {cls}: {len(files)} images proposed")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} proposals -> {args.out_csv}")
    print(f"  implausible body masks flagged for review: {n_implausible}")
    print(f"  excluded classes: {', '.join(f'{k} ({v})' for k, v in EXCLUDED_CLASSES.items())}")
    print("  status=proposed on every row. None of this is training data yet.")


if __name__ == "__main__":
    main()
