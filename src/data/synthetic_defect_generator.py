"""Synthetic defect/quality dataset generator.

See docs/06_SYNTHETIC_DEFECT_POLICY.md for the full rationale and constraints. In
short: no real defect-labeled maize dataset was provided, so this module builds one
deterministically from the REAL corn images already in Datasets A/B by applying
controlled, logged OpenCV transformations. Because we perform the transformation, we
know the ground-truth label exactly — nothing is guessed or invented. Every image
this produces is written under a `synthetic/` subtree and the exact parameters used
are recorded in a manifest CSV so the whole dataset is reproducible and auditable.

Classes (chosen because they are the defect types explicitly documented in the
literature survey, docs/02_LITERATURE_SURVEY_ANALYSIS.md rows 16/23/37/41/43):
    healthy            - the source image, unmodified
    cracked            - procedural fracture-line + local texture disruption
    discolored_mold    - procedural blotchy dark/discolored patches
    insect_damaged     - procedural small punched dark holes

This is NOT a substitute for real pathology data. Every consumer of this model
(backend schemas, frontend labels, Gemini prompts) must keep the word "synthetic" in
the model name and output field names.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np

SYNTHETIC_CLASSES = ["healthy", "cracked", "discolored_mold", "insect_damaged"]


def _seed_foreground_mask(img: np.ndarray, erode_iters: int = 2) -> np.ndarray:
    """Approximate the seed's foreground region so defects are placed ON the seed,
    not on the (near-black) background — otherwise a dark hole/crack on a dark
    background is invisible and the "defect" label would not match what's visible.
    Simple, deterministic, and logged: grayscale threshold + largest contour.

    erode_iters=2 (default) shrinks the region so sampled defect CENTRES land well
    inside the seed. erode_iters=0 returns the seed's true extent, which is what a
    defect may be painted onto and what its mask may claim: a defect that stops 4px
    short of the kernel edge would under-report its own area.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(gray.shape, dtype=np.uint8) * 255
    largest = max(contours, key=cv2.contourArea)
    fg = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(fg, [largest], -1, 255, -1)
    if erode_iters > 0:
        # erode slightly so blob/hole centers land well inside the seed, not its edge
        fg = cv2.erode(fg, np.ones((5, 5), np.uint8), iterations=erode_iters)
    return fg if fg.sum() > 0 else np.ones(gray.shape, dtype=np.uint8) * 255


def _sample_point_in_mask(mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w // 2, h // 2
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def _apply_crack(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    out = img.copy()
    fg = _seed_foreground_mask(img)
    body = _seed_foreground_mask(img, erode_iters=0)
    scratch = out.copy()
    n_lines = rng.randint(1, 2)
    params = {"n_lines": n_lines, "lines": []}
    for _ in range(n_lines):
        x1, y1 = _sample_point_in_mask(fg, rng)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.randint(int(0.15 * w), int(0.35 * w))
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        thickness = rng.randint(1, 2)
        dark = rng.randint(20, 60)
        cv2.line(scratch, (x1, y1), (x2, y2), (dark, dark, dark), thickness, cv2.LINE_AA)
        params["lines"].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "thickness": thickness})
    # a fracture line drawn across the background is not a crack in the seed
    out = np.where(body[:, :, None] > 0, scratch, out)
    return out, params


def _apply_discoloration(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    fg = _seed_foreground_mask(img)
    n_blobs = rng.randint(1, 3)
    params = {"n_blobs": n_blobs, "blobs": []}
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        cx, cy = _sample_point_in_mask(fg, rng)
        radius = rng.randint(int(0.10 * w), int(0.22 * w))
        blob_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(blob_mask, (cx, cy), radius, 1.0, -1)
        blob_mask = cv2.GaussianBlur(blob_mask, (0, 0), sigmaX=radius / 4)
        mask = np.maximum(mask, blob_mask)
        params["blobs"].append({"cx": cx, "cy": cy, "radius": radius})
    body = _seed_foreground_mask(img, erode_iters=0)
    mask = mask * (body.astype(np.float32) / 255.0)  # keep the effect on the seed only

    darken = rng.uniform(0.55, 0.8)
    # shift toward a brown/grey moldy tone and darken
    tone = np.array([35, 50, 55], dtype=np.float32)  # BGR brownish-grey
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - mask * darken) + tone[c] * mask * darken
    out = np.clip(out, 0, 255).astype(np.uint8)
    params["darken_factor"] = round(darken, 3)
    return out, params


def _apply_insect_damage(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    out = img.copy()
    fg = _seed_foreground_mask(img)
    body = _seed_foreground_mask(img, erode_iters=0)
    scratch = out.copy()
    n_holes = rng.randint(3, 7)
    params = {"n_holes": n_holes, "holes": []}
    for _ in range(n_holes):
        cx, cy = _sample_point_in_mask(fg, rng)
        radius = rng.randint(max(2, int(0.025 * w)), max(3, int(0.055 * w)))
        cv2.circle(scratch, (cx, cy), radius, (8, 8, 8), -1, cv2.LINE_AA)
        cv2.circle(scratch, (cx, cy), radius, (25, 30, 35), 1, cv2.LINE_AA)  # rim
        params["holes"].append({"cx": cx, "cy": cy, "radius": radius})
    # a bore hole hanging off the edge of the kernel is background, not damage
    out = np.where(body[:, :, None] > 0, scratch, out)
    return out, params


# --- pixel-level ground-truth masks -------------------------------------------
# The generators above KNOW where they painted, so a defect mask here is not an
# estimate or a guess: it is the same geometry re-rasterised onto an empty canvas.
# Every parameter needed is already recorded in `transform_params`, which is what
# keeps the manifest auditable -- a reviewer can regenerate any mask from the CSV
# alone and check it against the image.
#
# One judgement call is unavoidable. Discoloration paints a BLURRED disc, so its
# alpha fades continuously to zero and "the defect region" has to be cut somewhere.
# We cut at alpha >= 0.5, i.e. the area where the mould tone contributes more than
# the original pixel does. That is a stated convention, not a measurement, so it is
# written into the manifest next to the mask it produced.
MASK_ALPHA_THRESHOLD = 0.5


def reconstruct_defect_mask(source_img: np.ndarray, defect_cls: str, params: dict) -> np.ndarray:
    """Exact ground-truth defect mask, re-derived from the logged parameters.

    `source_img` must be the ORIGINAL pre-defect image, because the foreground mask
    that constrained the painting is computed from it. Returns uint8 HxW in {0,255}.
    `healthy` is defect-free by construction, hence an all-zero mask.
    """
    h, w = source_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if defect_cls == "healthy":
        return mask

    body = _seed_foreground_mask(source_img, erode_iters=0)

    if defect_cls == "cracked":
        for ln in params["lines"]:
            # LINE_8, not LINE_AA: a mask records membership, and an antialiased
            # edge pixel is partial coverage, not membership.
            cv2.line(mask, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), 255, ln["thickness"], cv2.LINE_8)
        return cv2.bitwise_and(mask, body)

    if defect_cls == "insect_damaged":
        for ho in params["holes"]:
            cv2.circle(mask, (ho["cx"], ho["cy"]), ho["radius"], 255, -1, cv2.LINE_8)
        return cv2.bitwise_and(mask, body)

    if defect_cls == "discolored_mold":
        alpha = np.zeros((h, w), dtype=np.float32)
        for bl in params["blobs"]:
            blob = np.zeros((h, w), dtype=np.float32)
            cv2.circle(blob, (bl["cx"], bl["cy"]), bl["radius"], 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=bl["radius"] / 4)
            alpha = np.maximum(alpha, blob)
        alpha = alpha * (body.astype(np.float32) / 255.0)   # same clamp the painter used
        mask[alpha >= MASK_ALPHA_THRESHOLD] = 255
        return mask

    raise ValueError(f"no mask rule for defect class {defect_cls!r}")


GENERATORS = {
    "cracked": _apply_crack,
    "discolored_mold": _apply_discoloration,
    "insect_damaged": _apply_insect_damage,
}


def generate_synthetic_dataset(
    source_root: str,
    out_root: str,
    manifest_path: str,
    per_class_synthetic_ratio: float = 1.0,
    seed: int = 42,
    masks_root: str | None = None,
) -> list[dict]:
    """For every real image, keep it as `healthy`, and generate one synthetic
    variant per defect class (subject to per_class_synthetic_ratio, so the dataset
    doesn't get 4x larger than necessary for a quick demo run). Returns the manifest
    rows and also writes them to manifest_path.

    If masks_root is given, each generated defect also gets a pixel-level mask PNG
    written under masks_root/<class>/, and the manifest gains a defect_mask_path
    column. Healthy rows keep an empty mask path: their mask is all-zero by
    definition and writing thousands of black PNGs would only waste disk.
    """
    rng = random.Random(seed)
    source_path = Path(source_root)
    out_path = Path(out_root)
    manifest_rows = []

    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    for cls_dir in class_dirs:
        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # healthy = original, unmodified, just referenced (path recorded, not copied,
            # to avoid duplicating the whole real dataset on disk)
            manifest_rows.append({
                "synthetic_image_path": str(img_path),
                "synthetic_label": "healthy",
                "source_image": str(img_path),
                "source_variety_class": cls_dir.name,
                "transform_params": "{}",
                "defect_mask_path": "",
            })

            for defect_cls, gen_fn in GENERATORS.items():
                if rng.random() > per_class_synthetic_ratio:
                    continue
                out_img, params = gen_fn(img, rng)
                out_dir = out_path / defect_cls
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{img_path.stem}__{defect_cls}.jpg"
                cv2.imwrite(str(out_file), out_img)

                mask_file = ""
                if masks_root:
                    defect_mask = reconstruct_defect_mask(img, defect_cls, params)
                    mask_dir = Path(masks_root) / defect_cls
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = mask_dir / f"{img_path.stem}__{defect_cls}.png"
                    cv2.imwrite(str(mask_path), defect_mask)   # PNG: lossless, a JPEG mask is not a mask
                    mask_file = str(mask_path)
                    if defect_cls == "discolored_mold":
                        params = {**params, "mask_alpha_threshold": MASK_ALPHA_THRESHOLD}

                manifest_rows.append({
                    "synthetic_image_path": str(out_file),
                    "synthetic_label": defect_cls,
                    "source_image": str(img_path),
                    "source_variety_class": cls_dir.name,
                    "transform_params": str(params),
                    "defect_mask_path": mask_file,
                })

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "synthetic_image_path", "synthetic_label", "source_image",
            "source_variety_class", "transform_params", "defect_mask_path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic defect dataset")
    parser.add_argument("--source-root", required=True, help="Real variety dataset root (folder-per-class)")
    parser.add_argument("--out-root", required=True, help="Where to write generated synthetic-defect images")
    parser.add_argument("--manifest", required=True, help="Output manifest CSV path")
    parser.add_argument("--ratio", type=float, default=1.0, help="Fraction of images to synthesize per defect class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--masks-root", default=None,
                        help="If set, also write pixel-level defect mask PNGs here")
    args = parser.parse_args()

    rows = generate_synthetic_dataset(args.source_root, args.out_root, args.manifest,
                                      args.ratio, args.seed, args.masks_root)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["synthetic_label"]] = counts.get(r["synthetic_label"], 0) + 1
    print(f"Generated {len(rows)} manifest rows. Per-class counts: {counts}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
