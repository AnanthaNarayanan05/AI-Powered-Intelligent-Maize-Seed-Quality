"""Measures the train/serve gap for variety classification.

`train_variety.py` trains and evaluates the classifier on **whole images** from the
dataset manifest. The deployed system (`unified_pipeline.analyze_image`) instead runs
YOLO detection first and classifies each **cropped** seed. Those are different input
distributions, so the reported test accuracy does not automatically describe the
accuracy of the served system.

This script measures both on the same held-out test split, so the report can quote the
number that matches how the system is actually used.

    python -m src.analysis.eval_serving_gap --dataset a
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from PIL import Image

from src.pipeline.unified_pipeline import AnalysisPipeline
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="a", choices=["a", "b"])
    parser.add_argument("--limit", type=int, default=0, help="0 = whole test split")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config()
    manifest = os.path.join(cfg["paths"]["processed"], f"manifest_dataset_{args.dataset}.csv")
    rows = [r for r in csv.DictReader(open(manifest)) if r["split"] == "test"]
    if args.limit:
        rows = rows[: args.limit]

    pipe = AnalysisPipeline()
    whole_ok = crop_ok = crop_total = no_detect = 0
    crop_pred_counts: collections.Counter = collections.Counter()
    disagreements = []

    for i, r in enumerate(rows):
        path, truth = r["filepath"], r["label"]
        img = Image.open(path).convert("RGB")

        w = pipe.classify_variety(img, dataset=args.dataset)
        whole_ok += w["predicted_class"] == truth

        dets = pipe.detect_seeds(path)
        if not dets:
            no_detect += 1
            continue

        best = max(dets, key=lambda d: d.get("confidence", 0))
        c = pipe.classify_variety(pipe.crop_seed(img, best["bbox"]), dataset=args.dataset)
        crop_total += 1
        crop_ok += c["predicted_class"] == truth
        crop_pred_counts[c["predicted_class"]] += 1

        if w["predicted_class"] != c["predicted_class"]:
            disagreements.append({
                "file": os.path.basename(path),
                "truth": truth,
                "whole_pred": w["predicted_class"],
                "whole_conf": round(float(w["confidence"]), 4),
                "crop_pred": c["predicted_class"],
                "crop_conf": round(float(c["confidence"]), 4),
            })

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(rows)}", flush=True)

    n = len(rows)
    result = {
        "dataset": args.dataset,
        "test_images": n,
        "whole_image_accuracy": round(whole_ok / n, 4),
        "served_crop_accuracy": round(crop_ok / crop_total, 4) if crop_total else None,
        "images_with_no_detection": no_detect,
        "disagreements": len(disagreements),
        "crop_prediction_distribution": dict(crop_pred_counts),
        "examples": disagreements[:20],
    }

    print()
    print("=" * 64)
    print(f"Dataset {args.dataset.upper()} — held-out test split, {n} images")
    print("=" * 64)
    print(f"whole image (as trained/evaluated): {whole_ok}/{n} = {whole_ok/n*100:.2f}%")
    if crop_total:
        print(f"YOLO crop   (as actually served) : {crop_ok}/{crop_total} = {crop_ok/crop_total*100:.2f}%")
    print(f"detection found nothing on        : {no_detect} images")
    print(f"whole-vs-crop disagreements       : {len(disagreements)}")
    print(f"crop prediction distribution      : {dict(crop_pred_counts)}")

    out = args.out or os.path.join(cfg["paths"]["metrics"], f"serving_gap_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
