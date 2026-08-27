"""Turn the synthetic-defect manifest into a segmentation manifest.

What this does NOT do is invent labels. Every defect mask referenced here was
re-derived from the transformation parameters this project logged when it painted the
defect (src/data/synthetic_defect_generator.reconstruct_defect_mask), so the mask is
exact by construction rather than estimated.

The seed-body channel is a different kind of label and is marked as such. It is
produced by the same deterministic threshold-plus-largest-contour rule that decided
where a defect was allowed to be painted -- so on this studio-background imagery it is
reliable, but it is an ALGORITHMIC label, not a human annotation, and no claim about
segmentation of kernels on cluttered backgrounds follows from it. It is included
because defect coverage % needs a denominator (VISIBLE DEFECT AREA % = defect pixels /
valid visible seed pixels), and a predicted denominator beats a hard-coded one.

Splitting is on SOURCE SEEDS, matching src/training/train_synthetic_defect.py, so a
kernel that the segmenter trains on never reappears in its own test set wearing a
different defect.

Usage:
    python -m src.segmentation.build_synthetic_manifest
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path

import cv2

from src.data.group_split import split_groups
from src.data.synthetic_defect_generator import (
    SYNTHETIC_CLASSES,
    _seed_foreground_mask,
    reconstruct_defect_mask,
)

# The order here defines the checkpoint's channel order. seed_body last so that
# slicing [:len(DEFECT_CHANNELS)] always gives exactly the defect channels.
DEFECT_CHANNELS = [c for c in SYNTHETIC_CLASSES if c != "healthy"]
CHANNELS = DEFECT_CHANNELS + ["seed_body"]

FIELDS = ["image_path", "label", "defect_mask_path", "body_mask_path", "split", "group"]


def build(
    manifest_path: str,
    out_manifest: str,
    body_root: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    rebuild_masks: bool = False,
) -> list[dict]:
    with open(manifest_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{manifest_path} is empty; run src.data.synthetic_defect_generator first")
    if "defect_mask_path" not in rows[0]:
        raise SystemExit(
            f"{manifest_path} has no defect_mask_path column. Re-run "
            "src.data.synthetic_defect_generator with --masks-root."
        )

    sources = sorted({r["source_image"] for r in rows})
    variety_of = {r["source_image"]: r["source_variety_class"] for r in rows}
    assignment = split_groups(
        sources,
        group_of=lambda s: s,
        label_of=lambda s: variety_of[s],
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    body_dir = Path(body_root)
    body_dir.mkdir(parents=True, exist_ok=True)
    body_for_source: dict[str, str] = {}
    out_rows: list[dict] = []
    missing_defect_masks = 0

    for src_path in sources:
        img = cv2.imread(src_path)
        if img is None:
            raise SystemExit(f"cannot read source image {src_path}")
        # one body mask per physical seed, shared by its healthy row and every variant
        stem = Path(src_path).stem.replace(" ", "_")
        parent = Path(src_path).parent.name
        body_path = body_dir / f"{parent}__{stem}.png"
        if rebuild_masks or not body_path.exists():
            cv2.imwrite(str(body_path), _seed_foreground_mask(img, erode_iters=0))
        body_for_source[src_path] = str(body_path)

    for r in rows:
        mask_path = r["defect_mask_path"]
        label = r["synthetic_label"]
        if label != "healthy":
            if not mask_path or not os.path.exists(mask_path):
                missing_defect_masks += 1
                continue
            if rebuild_masks:
                srcim = cv2.imread(r["source_image"])
                params = ast.literal_eval(r["transform_params"])
                cv2.imwrite(mask_path, reconstruct_defect_mask(srcim, label, params))
        out_rows.append({
            "image_path": r["synthetic_image_path"],
            "label": label,
            "defect_mask_path": mask_path if label != "healthy" else "",
            "body_mask_path": body_for_source[r["source_image"]],
            "split": assignment[r["source_image"]],
            "group": r["source_image"],
        })

    if missing_defect_masks:
        raise SystemExit(
            f"{missing_defect_masks} defect rows reference a mask file that is not on "
            "disk. Re-run src.data.synthetic_defect_generator with --masks-root rather "
            "than training on a manifest that overstates what it has."
        )

    os.makedirs(os.path.dirname(out_manifest) or ".", exist_ok=True)
    with open(out_manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data_processed/manifest_synthetic_defects.csv")
    ap.add_argument("--out", default="data_processed/manifest_segmentation_synthetic.csv")
    ap.add_argument("--body-root", default="data_processed/synthetic_defect_masks/seed_body")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-masks", action="store_true",
                    help="Re-derive every mask from transform_params instead of trusting the PNGs on disk")
    args = ap.parse_args()

    rows = build(args.manifest, args.out, args.body_root, seed=args.seed,
                 rebuild_masks=args.rebuild_masks)

    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        counts[(r["split"], r["label"])] = counts.get((r["split"], r["label"]), 0) + 1
    print(f"channels: {CHANNELS}")
    print(f"{len(rows)} rows -> {args.out}")
    for split in ("train", "val", "test"):
        per = {lbl: n for (s, lbl), n in sorted(counts.items()) if s == split}
        print(f"  {split:5s} {sum(per.values()):5d}  {per}")
    groups = {s: {r['group'] for r in rows if r['split'] == s} for s in ('train', 'val', 'test')}
    print("  seed overlap train/test:", len(groups['train'] & groups['test']),
          " train/val:", len(groups['train'] & groups['val']))


if __name__ == "__main__":
    main()
