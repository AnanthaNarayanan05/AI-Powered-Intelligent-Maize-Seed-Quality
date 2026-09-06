"""Global healthy-kernel colour model, used to propose WHERE a defect is.

WHY GLOBAL AND NOT PER-IMAGE. The obvious approach is to take each kernel's own
dominant colour as its healthy reference and flag pixels that deviate from it.
That works on a kernel with a local blemish and fails completely on a uniformly
mouldy or heat-damaged one: there the reference is itself the defect colour, the
deviation is zero everywhere, and the proposal comes back clean. Referencing the
NOR class instead means a fully-affected kernel scores high across its whole
area, which is the correct proposal for a fully-affected kernel.

WHAT THIS IS NOT. Mahalanobis distance from healthy maize colour is a SALIENCY
CUE for proposing candidate regions to a human reviewer. It is not a defect
detector, it is not a label, and no number it produces ever reaches a reported
metric. Shadow, specular highlight and the pedicel scar all score high and are
all not defects; that is precisely what the reviewer is there to throw out.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

from src.annotation.kernel_body import propose_body
from src.annotation.splits import assign_split, plate_id

REFERENCE_PATH = "data_processed/annotation/healthy_reference.json"

# Sampled from the kernel INTERIOR: the boundary ring is a blend of kernel and
# background and would drag the mean toward the white plate.
_INNER_ERODE = 11
_MIN_INNER_PX = 300
_PX_PER_KERNEL = 400


def build(image_dir: str, predictor, limit: int = 150, seed: int = 42) -> dict:
    """Fit a Lab-space Gaussian to healthy kernel interiors.

    Only TRAIN-split plates contribute. The reference steers proposals on
    defective kernels, so fitting it on plates that later appear in val or test
    would leak evaluation data into the annotation process. Weakly, since a human
    makes the final call -- but excluding them costs nothing.
    """
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    files = [f for f in files if assign_split(plate_id(f)) == "train"]
    rng = np.random.default_rng(seed)

    samples, used, skipped = [], [], 0
    for path in files:
        if len(used) >= limit:
            break
        bgr = cv2.imread(path)
        if bgr is None:
            skipped += 1
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        body, plausible = propose_body(image, predictor)
        if not plausible:
            skipped += 1
            continue
        inner = cv2.erode(body.astype(np.uint8), np.ones((_INNER_ERODE,) * 2, np.uint8)).astype(bool)
        if inner.sum() < _MIN_INNER_PX:
            skipped += 1
            continue
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)[inner]
        take = rng.permutation(len(lab))[:_PX_PER_KERNEL]
        samples.append(lab[take])
        used.append(os.path.basename(path))

    if not samples:
        raise SystemExit(
            f"No usable healthy kernels in {image_dir}. The reference cannot be "
            "fitted, and without it the proposal step would be measuring colour "
            "deviation from nothing."
        )

    X = np.concatenate(samples, axis=0)
    mu = X.mean(axis=0)
    # Ridge on the covariance: L, a and b are strongly correlated across a single
    # crop type, so the raw covariance can sit near-singular and its inverse
    # becomes a noise amplifier rather than a distance.
    cov = np.cov(X.T) + np.eye(3) * 1e-3
    return {
        "space": "CIELAB (OpenCV 8-bit convention, L/a/b each 0-255)",
        "mu": mu.tolist(),
        "cov": cov.tolist(),
        "n_kernels": len(used),
        "n_pixels": int(len(X)),
        "n_skipped": skipped,
        "source_dir": image_dir,
        "split_restriction": "train plates only",
        "purpose": (
            "Saliency cue for PROPOSING candidate defect regions to a human "
            "reviewer. Not a detector, not a label, never a reported metric."
        ),
    }


def load(path: str = REFERENCE_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, inverse covariance)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run src/annotation/healthy_reference.py first.")
    with open(path, "r", encoding="utf-8") as fh:
        ref = json.load(fh)
    return np.asarray(ref["mu"], np.float32), np.linalg.inv(np.asarray(ref["cov"], np.float32))


def saliency(image: np.ndarray, body: np.ndarray, mu: np.ndarray, icov: np.ndarray) -> np.ndarray:
    """Per-pixel Mahalanobis distance from healthy maize colour, zero outside `body`."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = lab - mu
    dist = np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, icov, d), 0.0))
    dist[~body] = 0.0
    return dist


def main() -> None:
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", default="datasets/grainspace_maize_m600/val/NOR")
    ap.add_argument("--sam-checkpoint", default="weights/sam_b.pt")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", default=REFERENCE_PATH)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint).to(device).eval()
    ref = build(args.image_dir, SamPredictor(sam), limit=args.limit)
    ref["device"] = device

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=2)
    print(f"healthy reference: {ref['n_kernels']} kernels, {ref['n_pixels']} px, "
          f"{ref['n_skipped']} skipped -> {args.out}")
    print("  mu(Lab) =", [round(v, 1) for v in ref["mu"]])


if __name__ == "__main__":
    main()
