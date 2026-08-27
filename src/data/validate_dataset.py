"""Generic folder-per-class image dataset validator/auditor.

Reusable for Dataset A (corn_3_classes) and Dataset B (maize_data). Computes exactly
the numbers required by docs/03_DATASET_AUDIT.md: counts, corruption, duplicates,
resolutions, class balance. Never modifies the original dataset — writes a JSON
report to data_processed/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image


def audit_classification_dataset(root: str) -> dict:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    report = {
        "root": str(root_path),
        "classes": classes,
        "per_class_counts": {},
        "total_images": 0,
        "corrupted_files": [],
        "duplicate_pairs": [],
        "resolutions": defaultdict(int),
        "extensions": defaultdict(int),
    }

    hash_to_path: dict[str, str] = {}

    for cls in classes:
        cls_dir = root_path / cls
        files = [f for f in cls_dir.iterdir() if f.is_file()]
        report["per_class_counts"][cls] = len(files)
        report["total_images"] += len(files)

        for f in files:
            report["extensions"][f.suffix.lower()] += 1
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    report["resolutions"][f"{im.size[0]}x{im.size[1]}"] += 1
            except Exception as e:  # noqa: BLE001
                report["corrupted_files"].append({"path": str(f), "error": str(e)})
                continue

            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest in hash_to_path:
                report["duplicate_pairs"].append([hash_to_path[digest], str(f)])
            else:
                hash_to_path[digest] = str(f)

    report["resolutions"] = dict(report["resolutions"])
    report["extensions"] = dict(report["extensions"])

    counts = list(report["per_class_counts"].values())
    if counts:
        report["class_imbalance_ratio"] = round(max(counts) / max(min(counts), 1), 3)

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit a folder-per-class image dataset")
    parser.add_argument("--root", required=True, help="Path to dataset root (contains one folder per class)")
    parser.add_argument("--out", required=True, help="Path to write the JSON report")
    args = parser.parse_args()

    report = audit_classification_dataset(args.root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Total images: {report['total_images']}")
    print(f"Classes: {report['per_class_counts']}")
    print(f"Corrupted: {len(report['corrupted_files'])}")
    print(f"Duplicate pairs: {len(report['duplicate_pairs'])}")
    print(f"Class imbalance ratio (max/min): {report.get('class_imbalance_ratio')}")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
