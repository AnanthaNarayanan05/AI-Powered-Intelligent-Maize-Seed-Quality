"""Trains the synthetic defect/quality classifier described in
docs/06_SYNTHETIC_DEFECT_POLICY.md. Same architecture as the variety classifier, but
trained on procedurally generated defect patterns whose ground truth is exact by
construction (we performed the transformation). MUST be run only after
src/data/synthetic_defect_generator.py has produced the manifest.

Run locally:
    python -m src.data.synthetic_defect_generator \
        --source-root datasets/dataset_variety_a_corntype/Corn_3_Classes_Image_Dataset \
        --out-root data_processed/synthetic_defects \
        --manifest data_processed/manifest_synthetic_defects.csv
    python -m src.training.train_synthetic_defect
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticDefectDataset
from src.data.group_split import split_groups
from src.data.synthetic_defect_generator import SYNTHETIC_CLASSES
from src.contrastive.simclr import build_eval_transform, build_simclr_augmentation
from src.models.variety_classifier import CognitiveAttentionClassifier
from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger
from src.training.train_variety import evaluate  # reuse the same evaluation routine

logger = get_logger("train_synthetic_defect")


def make_split_indices(manifest_path: str, val_split: float, test_split: float, seed: int):
    """Split on SOURCE SEEDS, never on rows.

    Every source image produces four rows here: the untouched original as `healthy`
    plus one painted variant per defect class. A row-level stratified split therefore
    puts `chulpi_cancha 1` in train as healthy and the very same kernel in test as
    cracked, so the test set measures how well the model recognises kernels it has
    already memorised rather than how well it recognises defects. This is the same
    leakage that inflated the Dataset B variety accuracies to 99.9% (docs/09 section
    6.9); the fix is the same one -- group by provenance and deal out whole groups.

    Grouping by source image also keeps the four defect classes perfectly balanced
    across splits for free, since each group contributes exactly one row per class.
    Groups are stratified by the source variety class so no variety is concentrated
    in one split.
    """
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    sources = sorted({r["source_image"] for r in rows})
    variety_of = {r["source_image"]: r["source_variety_class"] for r in rows}
    assignment = split_groups(
        sources,
        group_of=lambda src: src,
        label_of=lambda src: variety_of[src],
        val_frac=val_split,
        test_frac=test_split,
        seed=seed,
    )

    buckets = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        buckets[assignment[r["source_image"]]].append(i)
    return buckets["train"], buckets["val"], buckets["test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_processed/manifest_synthetic_defects.csv")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    logger.info(f"Device: {device}")

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"{args.manifest} not found. Run src/data/synthetic_defect_generator.py first "
            f"(see docs/06_SYNTHETIC_DEFECT_POLICY.md)."
        )

    train_idx, val_idx, test_idx = make_split_indices(
        args.manifest, cfg["training"]["val_split"], cfg["training"]["test_split"], cfg["seed"]
    )

    image_size = cfg["training"]["image_size"]
    train_tf = build_simclr_augmentation(image_size)
    eval_tf = build_eval_transform(image_size)

    train_ds = SyntheticDefectDataset(args.manifest, SYNTHETIC_CLASSES, train_idx, train_tf)
    val_ds = SyntheticDefectDataset(args.manifest, SYNTHETIC_CLASSES, val_idx, eval_tf)
    test_ds = SyntheticDefectDataset(args.manifest, SYNTHETIC_CLASSES, test_idx, eval_tf)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = CognitiveAttentionClassifier(
        num_classes=len(SYNTHETIC_CLASSES),
        backbone_name=cfg["backbone"],
        pretrained=args.pretrained,
        use_attention=True,
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs or cfg["training"]["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["device"]["mixed_precision"] and device.type == "cuda")

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["logs"], "synthetic_defect.jsonl")
    best_ckpt_path = os.path.join(ckpt_dir, "synthetic_defect_classifier_best.pt")

    best_val_f1 = -1.0
    patience = cfg["training"]["early_stopping_patience"]
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["device"]["mixed_precision"] and device.type == "cuda"):
                logits = model(imgs, mode="classify")
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, SYNTHETIC_CLASSES)
        elapsed = time.time() - t0
        logger.info(
            f"[synthetic_defect] epoch {epoch+1}/{epochs} "
            f"train_loss={total_loss/max(len(train_loader),1):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} ({elapsed:.1f}s)"
        )
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "val_accuracy": val_metrics["accuracy"], "val_f1_macro": val_metrics["f1_macro"]}) + "\n")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": SYNTHETIC_CLASSES,
                "backbone": cfg["backbone"],
                "use_attention": True,
                "is_synthetic": True,
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, SYNTHETIC_CLASSES)
    with open(os.path.join(cfg["paths"]["metrics"], "synthetic_defect.json"), "w") as f:
        json.dump({"best_val_f1_macro": best_val_f1, "test_metrics": test_metrics, "note": "SYNTHETIC labels — see docs/06_SYNTHETIC_DEFECT_POLICY.md"}, f, indent=2)

    logger.info(f"[SYNTHETIC] Test accuracy: {test_metrics['accuracy']:.4f}  F1 macro: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
