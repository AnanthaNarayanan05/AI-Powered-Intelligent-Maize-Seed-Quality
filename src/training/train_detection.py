"""Phase 13: YOLO seed detection training, using Ultralytics on Dataset C
(single class "Corn", verified in docs/03_DATASET_AUDIT.md). Wraps Ultralytics
rather than reimplementing YOLO, per the master prompt's Phase 13 instruction to
"use a suitable Ultralytics YOLO implementation."

Run locally on the RTX 4060:
    python -m src.training.train_detection
"""
from __future__ import annotations

import argparse
import json
import os

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("train_detection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", default=None, help="e.g. yolov8n.pt (nano, recommended for RTX 4060)")
    args = parser.parse_args()

    cfg = load_config()
    det_cfg = cfg["detection"]

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics --break-system-packages"
        ) from e

    data_yaml = os.path.join(cfg["paths"]["dataset_c"], "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"{data_yaml} not found. Extract 'Grain and Objects Detection.v1i.yolov11.zip' "
            f"into {cfg['paths']['dataset_c']} first."
        )

    model_name = args.model or det_cfg["model"]
    epochs = args.epochs or det_cfg["epochs"]

    # Ultralytics resolves a relative `project` against its own runs/ directory, which
    # would put the weights somewhere the inference pipeline never looks. Pin it.
    project_dir = os.path.abspath(cfg["paths"]["checkpoints"])

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=det_cfg["image_size"],
        batch=det_cfg["batch_size"],
        project=project_dir,
        name="detection_corn",
        seed=cfg["seed"],
        exist_ok=True,
    )

    metrics = model.val(data=data_yaml, split="test")
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    with open(os.path.join(cfg["paths"]["metrics"], "detection.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Detection test metrics: {summary}")
    logger.info(f"Best weights: {os.path.join(project_dir, 'detection_corn', 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
