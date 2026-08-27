"""YOLO-format detection dataset validator. Verifies data.yaml consistency, every
image<->label pairing, coordinate/box validity, and computes instance statistics.
Read-only — never modifies the original dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def validate_yolo_dataset(root: str) -> dict:
    root_path = Path(root)
    yaml_path = root_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found under {root}")

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    report = {
        "root": str(root_path),
        "data_yaml": data_cfg,
        "splits": {},
        "total_instances": 0,
        "invalid_boxes": [],
        "missing_labels": [],
        "empty_labels": [],
        "image_label_mismatch": [],
    }

    for split in ["train", "valid", "test"]:
        img_dir = root_path / split / "images"
        lbl_dir = root_path / split / "labels"
        if not img_dir.exists():
            continue

        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        split_stats = {
            "num_images": len(images),
            "num_labels": 0,
            "num_instances": 0,
            "objects_per_image": [],
        }

        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                report["missing_labels"].append(str(img))
                continue
            split_stats["num_labels"] += 1
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            if not lines:
                report["empty_labels"].append(str(lbl))
            split_stats["objects_per_image"].append(len(lines))
            split_stats["num_instances"] += len(lines)
            report["total_instances"] += len(lines)

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    report["invalid_boxes"].append({"file": str(lbl), "line": line, "reason": "wrong field count"})
                    continue
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                valid_class = 0 <= cid < len(data_cfg.get("names", []))
                valid_coords = 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                if not (valid_class and valid_coords):
                    report["invalid_boxes"].append({"file": str(lbl), "line": line, "reason": "out of range"})

        if split_stats["objects_per_image"]:
            opi = split_stats["objects_per_image"]
            split_stats["avg_objects_per_image"] = round(sum(opi) / len(opi), 2)
            split_stats["min_objects_per_image"] = min(opi)
            split_stats["max_objects_per_image"] = max(opi)
        del split_stats["objects_per_image"]  # keep report compact
        report["splits"][split] = split_stats

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = validate_yolo_dataset(args.root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Classes: {report['data_yaml'].get('names')}")
    print(f"Splits: {report['splits']}")
    print(f"Total instances: {report['total_instances']}")
    print(f"Invalid boxes: {len(report['invalid_boxes'])}  Missing labels: {len(report['missing_labels'])}  Empty labels: {len(report['empty_labels'])}")


if __name__ == "__main__":
    main()
