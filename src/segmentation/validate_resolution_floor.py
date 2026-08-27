"""Determine the resolution floor below which a mask-derived area measurement
stops being meaningful.

Why the first version of this script was thrown away
---------------------------------------------------
The original approach segmented dataset_4 crops with SAM and compared native vs
downscaled masks. Two things invalidated it:

  1. dataset_4's LARGE images are not high-resolution kernels -- they are group
     photos holding 3+ kernels. Median kernel short side is 59px across the whole
     set, and only 26-35px inside the >=128px images. There is no high-resolution
     single-kernel subset to measure against.
  2. Using SAM as its own reference measured SAM's prompt semantics, not
     resolution. Recovered area tracked the PROMPT BOX area (52-74% of it)
     regardless of the true region size, and IoU fell as the target grew, which
     is geometrically impossible for a sound instrument.

Both are recorded in docs/ so the mistake is not repeated.

The instrument used here instead
--------------------------------
dataset_5 (Zenodo 7577017, CC-BY-4.0) supplies 195 strips x 3 views of single
maize kernels at ~100-120px per kernel with REAL HUMAN pixel masks of the embryo.
The embryo is a sub-kernel region -- structurally the same measurement problem as
a defect patch -- so it is a fair proxy for how a region-area measurement behaves.

Primary measurement is model-free and therefore not contaminated by any
segmenter's behaviour:

    downscale the TRUE mask to resolution S, upscale it back, compare to itself.

Whatever error survives that round trip is pure information loss. No segmenter,
however good, can do better at that resolution. It is a hard CEILING on
achievable accuracy, which is exactly what a floor should be derived from.

The secondary measurement runs SAM at each resolution to show what the annotation
assistant realistically achieves, which sits below the ceiling.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import PROJECT_ROOT
from src.utils.seed import set_seed

# Kernel short-side resolutions to simulate. 59 and 35 are marked because they
# are what dataset_4 and dataset_3 actually deliver today.
TARGET_KERNEL_PX = [24, 35, 48, 59, 72, 96, 120]

VIEWS_PER_STRIP = 3

# A coverage percentage shown to a user should not move by more than a few
# percent for reasons that are purely artefacts of image size.
AREA_ERR_TOLERANCE = 0.05
IOU_TOLERANCE = 0.85


def split_views(img: np.ndarray) -> list[np.ndarray]:
    """Each file stacks VIEWS_PER_STRIP views of one kernel vertically."""
    h = img.shape[0] // VIEWS_PER_STRIP
    return [img[i * h:(i + 1) * h] for i in range(VIEWS_PER_STRIP)]


def group_key(path: Path) -> str:
    """One physical kernel may appear under several capture timestamps. Group on
    plate + kernel id so repeated captures of one kernel never straddle a split."""
    stem = path.stem
    kern = re.search(r"Kernel_(\d+)", stem)
    plate = re.search(r"(plate_[\w\.]+?)_Speed", stem)
    return f"{plate.group(1) if plate else 'na'}#{kern.group(1) if kern else stem}"


def kernel_short_side(mask_region: np.ndarray) -> int:
    ys, xs = np.where(mask_region)
    if len(xs) == 0:
        return 0
    return int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def kernel_body(view: np.ndarray) -> np.ndarray:
    """Kernel body vs the dark conveyor background. Used only to measure the
    kernel's pixel size so resolutions can be expressed per kernel, not per file."""
    import cv2

    g = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lb, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return th.astype(bool)
    big = 1 + int(np.argmax(stats[1:, 4]))
    return lb == big


def roundtrip_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    """Downscale a binary mask to `scale` and back. Area is thresholded at 0.5
    on the way down, which is what any honest resampling does."""
    h, w = mask.shape
    sh, sw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((sw, sh), Image.BILINEAR)
    ) > 127
    back = np.array(
        Image.fromarray(small.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    ) > 127
    return back


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def load_pairs(root: Path):
    ins = sorted((root / "InputImages").glob("*.png"))
    outs = {p.name: p for p in (root / "OutputImages").glob("*.png")}
    pairs = []
    for i in ins:
        o = outs.get(i.name)
        if o is not None:
            pairs.append((i, o))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/dataset_5_embryo_seg")
    ap.add_argument("--weights", default="weights/sam_b.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--with-sam", action="store_true",
                    help="also measure what SAM recovers at each resolution")
    ap.add_argument("--out", default="outputs/metrics/segmentation_resolution_floor.json")
    args = ap.parse_args()

    set_seed(42)
    root = PROJECT_ROOT / args.root
    pairs = load_pairs(root)
    print(f"{len(pairs)} strips -> up to {len(pairs) * VIEWS_PER_STRIP} views")

    sam = None
    if args.with_sam:
        from ultralytics import SAM

        sam = SAM(str(PROJECT_ROOT / args.weights))
        sam.to(args.device)

    ceiling = {s: {"iou": [], "area_err": []} for s in TARGET_KERNEL_PX}
    sam_res = {s: {"iou": [], "area_err": []} for s in TARGET_KERNEL_PX}
    # The quantity that actually governs area error is the size of the REGION
    # being measured, not the size of the kernel containing it. Kernel resolution
    # only matters through its effect on region size. Recording (region_px,
    # area_err) pairs gives a curve that transfers to defects of any size,
    # whereas a kernel-px curve only transfers to regions as large as an embryo.
    by_region_px: list[tuple[float, float, float]] = []
    native_kernel_px = []
    native_region_px = []
    used_views = 0

    for idx, (ip, op) in enumerate(pairs, 1):
        img = np.array(Image.open(ip).convert("RGB"))
        msk = np.array(Image.open(op)).astype(bool)
        for view_img, view_msk in zip(split_views(img), split_views(msk)):
            if view_msk.sum() < 20:
                continue  # embryo not visible from this angle -- a real annotation, not a gap
            body = kernel_body(view_img)
            k_px = kernel_short_side(body)
            if k_px < 40:
                continue
            used_views += 1
            native_kernel_px.append(k_px)
            native_region_px.append(kernel_short_side(view_msk))
            true_area = float(view_msk.sum())

            for s in TARGET_KERNEL_PX:
                if s >= k_px:
                    continue  # would be an upscale
                scale = s / k_px
                rt = roundtrip_mask(view_msk, scale)
                err = abs(float(rt.sum()) - true_area) / true_area
                ceiling[s]["iou"].append(iou(rt, view_msk))
                ceiling[s]["area_err"].append(err)
                # effective size of the measured region at this resolution
                by_region_px.append(
                    (kernel_short_side(view_msk) * scale, err, iou(rt, view_msk))
                )

                if sam is not None:
                    h, w = view_img.shape[:2]
                    small = np.array(
                        Image.fromarray(view_img).resize(
                            (max(1, round(w * scale)), max(1, round(h * scale))),
                            Image.BILINEAR,
                        )
                    )
                    ys, xs = np.where(view_msk)
                    box = [
                        xs.min() * scale, ys.min() * scale,
                        xs.max() * scale, ys.max() * scale,
                    ]
                    r = sam(small, bboxes=[box], verbose=False)
                    if r and r[0].masks is not None and len(r[0].masks.data):
                        pm = r[0].masks.data[0].cpu().numpy().astype(bool)
                        up = np.array(
                            Image.fromarray(pm.astype(np.uint8) * 255).resize(
                                (w, h), Image.NEAREST
                            )
                        ) > 127
                        sam_res[s]["iou"].append(iou(up, view_msk))
                        sam_res[s]["area_err"].append(
                            abs(float(up.sum()) - true_area) / true_area
                        )
        if idx % 50 == 0:
            print(f"  {idx}/{len(pairs)} strips")

    def summarise(store):
        out = {}
        for s in TARGET_KERNEL_PX:
            if not store[s]["iou"]:
                continue
            out[str(s)] = {
                "n": len(store[s]["iou"]),
                "iou_median": round(float(np.median(store[s]["iou"])), 4),
                "area_err_median": round(float(np.median(store[s]["area_err"])), 4),
                "area_err_p90": round(float(np.percentile(store[s]["area_err"], 90)), 4),
            }
        return out

    ceil_s = summarise(ceiling)
    sam_s = summarise(sam_res) if sam is not None else {}

    # Area error as a function of the measured region's own pixel size. This is
    # the curve that generalises to defects, which are far smaller than embryos.
    region_bins = [(4, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 48), (48, 1e9)]
    region_summary = {}
    for lo, hi in region_bins:
        sel = [(e, i) for px, e, i in by_region_px if lo <= px < hi]
        if len(sel) < 15:
            continue
        errs = np.array([e for e, _ in sel])
        ious = np.array([i for _, i in sel])
        label = f"{lo}-{int(hi)}" if hi < 1e9 else f"{lo}+"
        region_summary[label] = {
            "n": len(sel),
            "iou_median": round(float(np.median(ious)), 4),
            "area_err_median": round(float(np.median(errs)), 4),
            "area_err_p90": round(float(np.percentile(errs, 90)), 4),
        }
    passing_region = [
        k for k, v in region_summary.items()
        if v["area_err_p90"] <= AREA_ERR_TOLERANCE and v["iou_median"] >= IOU_TOLERANCE
    ]
    min_region_px = (
        int(passing_region[0].split("-")[0].rstrip("+")) if passing_region else None
    )

    passing = [
        int(s) for s, v in ceil_s.items()
        if v["iou_median"] >= IOU_TOLERANCE and v["area_err_p90"] <= AREA_ERR_TOLERANCE
    ]
    floor = min(passing) if passing else None

    result = {
        "dataset": "dataset_5_embryo_seg (Zenodo 7577017, CC-BY-4.0)",
        "instrument": "true-mask downscale/upscale round trip = information-loss "
                      "ceiling; no segmenter can beat this at a given resolution",
        "proxy_note": "embryo region used as a stand-in for a defect patch: both are "
                      "sub-kernel regions measured as a fraction of kernel area",
        "views_used": used_views,
        "native_kernel_px_median": int(np.median(native_kernel_px)) if native_kernel_px else None,
        "native_region_px_median": int(np.median(native_region_px)) if native_region_px else None,
        "tolerances": {"iou_median_min": IOU_TOLERANCE, "area_err_p90_max": AREA_ERR_TOLERANCE},
        "ceiling_by_kernel_px": ceil_s,
        "sam_by_kernel_px": sam_s,
        "ceiling_by_region_px": region_summary,
        "min_measurable_region_px": min_region_px,
        "resolution_floor_kernel_px": floor,
        "project_datasets_for_reference": {
            "dataset_4_quality_median_kernel_px": 59,
            "dataset_3_detection_median_kernel_px": 35,
        },
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nviews used: {used_views}   native kernel px median: "
          f"{result['native_kernel_px_median']}   region px median: "
          f"{result['native_region_px_median']}")
    print("\nCEILING (best any model could do at this kernel resolution)")
    print("kernel_px |    n | IoU med | dA med | dA p90 | verdict")
    for s in TARGET_KERNEL_PX:
        v = ceil_s.get(str(s))
        if not v:
            continue
        ok = v["iou_median"] >= IOU_TOLERANCE and v["area_err_p90"] <= AREA_ERR_TOLERANCE
        print(f"{s:>9} | {v['n']:>4} | {v['iou_median']:>7.3f} | {v['area_err_median']:>6.3f} | "
              f"{v['area_err_p90']:>6.3f} | {'PASS' if ok else 'fail'}")
    if sam_s:
        print("\nSAM (what the annotation assistant actually recovers)")
        print("kernel_px |    n | IoU med | dA med | dA p90")
        for s in TARGET_KERNEL_PX:
            v = sam_s.get(str(s))
            if not v:
                continue
            print(f"{s:>9} | {v['n']:>4} | {v['iou_median']:>7.3f} | {v['area_err_median']:>6.3f} | "
                  f"{v['area_err_p90']:>6.3f}")
    print("\nCEILING BY MEASURED-REGION SIZE (the curve that transfers to defects)")
    print("region_px |    n | IoU med | dA med | dA p90 | verdict")
    for k, v in region_summary.items():
        ok = v["area_err_p90"] <= AREA_ERR_TOLERANCE and v["iou_median"] >= IOU_TOLERANCE
        print(f"{k:>9} | {v['n']:>4} | {v['iou_median']:>7.3f} | {v['area_err_median']:>6.3f} | "
              f"{v['area_err_p90']:>6.3f} | {'PASS' if ok else 'fail'}")
    print(f"\nsmallest reliably measurable region: {min_region_px} px across")
    print(f"resolution floor (kernel short side): {floor} px")
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
