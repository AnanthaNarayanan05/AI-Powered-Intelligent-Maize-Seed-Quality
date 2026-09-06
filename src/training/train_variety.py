"""Phase 7/11 training script for a variety classifier (Dataset A or B), supporting
the 4 ablation experiments defined in docs/05_ARCHITECTURE.md:
    1. baseline                          --no-attention --no-contrastive
    2. +attention                        --no-contrastive
    3. +contrastive                      --no-attention (loads pretrained encoder)
    4. +contrastive +attention (proposed) (default)

Run locally on the RTX 4060, e.g.:
    python -m src.training.train_variety --dataset a --experiment full
    python -m src.training.train_variety --dataset b --experiment baseline

Saves: best checkpoint, training log (jsonl), metrics (json incl. confusion matrix).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)

from src.data.datasets import VarietyImageDataset
from src.contrastive.simclr import build_eval_transform, build_simclr_augmentation
from src.models.variety_classifier import CognitiveAttentionClassifier
from src.registry.model_registry import checkpoint_sha256
from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger

logger = get_logger("train_variety")

EXPERIMENTS = {
    "baseline": {"use_attention": False, "use_contrastive": False},
    "attention_only": {"use_attention": True, "use_contrastive": False},
    "contrastive_only": {"use_attention": False, "use_contrastive": True},
    "full": {"use_attention": True, "use_contrastive": True},
}


def evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs, mode="classify")
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
        "precision_weighted": precision_w, "recall_weighted": recall_w, "f1_weighted": f1_w,
        "confusion_matrix": cm,
        "per_class_report": report,
        "_preds": all_preds,
        "_labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["a", "b", "quality"], required=True,
                         help="'quality' is the Mendeley EfficientMaize Good/Bad set, used "
                              "for the quality-head ablation (docs/09 sec 6.7 item 3) with "
                              "the same 4-experiment machinery as a/b.")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                         help="Skip ImageNet-pretrained weights (use when the weights host is unreachable)")
    parser.add_argument("--manifest", default=None,
                         help="Override the manifest CSV. Used to run against the group-aware "
                              "split (manifest_dataset_b_grouped.csv) without disturbing the "
                              "original random-split results.")
    parser.add_argument("--encoder", default=None,
                         help="Path to the contrastive encoder checkpoint. Point this at the "
                              "train-split-only encoder when running a group-aware experiment.")
    parser.add_argument("--tag", default=None,
                         help="Suffix for checkpoint/log/metric filenames. Defaults to the "
                              "experiment name; set it to keep a re-run from overwriting an "
                              "earlier run's artefacts.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    logger.info(f"Device: {device} | dataset={args.dataset} | experiment={args.experiment}")

    exp_cfg = EXPERIMENTS[args.experiment]
    ds_cfg = (cfg["dataset_a"] if args.dataset == "a"
              else cfg["dataset_b"] if args.dataset == "b" else cfg["dataset_quality"])
    classes = ds_cfg["classes"]
    data_root = (cfg["paths"]["dataset_a"] if args.dataset == "a"
                 else cfg["paths"]["dataset_b"] if args.dataset == "b"
                 else cfg["paths"]["dataset_quality"])

    tag = args.tag or args.experiment
    manifest_path = args.manifest or os.path.join(
        cfg["paths"]["processed"], f"manifest_dataset_{args.dataset}.csv"
    )
    if args.manifest and not os.path.exists(manifest_path):
        raise SystemExit(f"--manifest {manifest_path} does not exist")
    if not os.path.exists(manifest_path):
        from src.data.split_dataset import build_manifest
        import csv as _csv
        classes_found, rows = build_manifest(
            data_root, cfg["training"]["val_split"], cfg["training"]["test_split"], cfg["seed"]
        )
        os.makedirs(cfg["paths"]["processed"], exist_ok=True)
        with open(manifest_path, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["filepath", "label", "split"])
            writer.writerows(rows)
        logger.info(f"Manifest generated at {manifest_path}")

    image_size = cfg["training"]["image_size"]
    train_tf = build_simclr_augmentation(image_size)  # mild augmentation reused for supervised training too
    eval_tf = build_eval_transform(image_size)

    train_ds = VarietyImageDataset(manifest_path, "train", classes, train_tf)
    val_ds = VarietyImageDataset(manifest_path, "val", classes, eval_tf)
    test_ds = VarietyImageDataset(manifest_path, "test", classes, eval_tf)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    # persistent_workers matters on Windows: workers are spawned, not forked, so
    # rebuilding them every epoch costs more than the epoch itself on small datasets.
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = CognitiveAttentionClassifier(
        num_classes=len(classes),
        backbone_name=cfg["backbone"],
        pretrained=args.pretrained,
        use_attention=exp_cfg["use_attention"],
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
    ).to(device)

    if exp_cfg["use_contrastive"]:
        ckpt_path = args.encoder or os.path.join(
            cfg["paths"]["checkpoints"], f"contrastive_encoder_{args.dataset}.pt"
        )
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            model.backbone.load_state_dict(state["backbone_state_dict"])
            logger.info(f"Loaded contrastive-pretrained encoder from {ckpt_path}")
        else:
            logger.warning(
                f"--experiment {args.experiment} requested contrastive pretraining but "
                f"{ckpt_path} does not exist yet. Run pretrain_contrastive.py first. "
                f"Continuing with ImageNet-only initialization."
            )

    class_counts = [0] * len(classes)
    for row in train_ds.rows:
        class_counts[train_ds.class_to_idx[row["label"]]] += 1
    weights = torch.tensor([1.0 / max(c, 1) for c in class_counts], dtype=torch.float32, device=device)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    epochs = args.epochs or cfg["training"]["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["device"]["mixed_precision"] and device.type == "cuda")

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["logs"], f"variety_{args.dataset}_{tag}.jsonl")
    best_ckpt_path = os.path.join(ckpt_dir, f"variety_{args.dataset}_{tag}_best.pt")

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

        val_metrics = evaluate(model, val_loader, device, classes)
        elapsed = time.time() - t0
        train_loss = total_loss / max(len(train_loader), 1)
        logger.info(
            f"[{args.dataset}/{args.experiment}] epoch {epoch+1}/{epochs} "
            f"train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} ({elapsed:.1f}s)"
        )
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1, "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"], "val_f1_macro": val_metrics["f1_macro"],
                "seconds": elapsed,
            }) + "\n")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "backbone": cfg["backbone"],
                "use_attention": exp_cfg["use_attention"],
                "experiment": args.experiment,
                "dataset": args.dataset,
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    # Final test-set evaluation using the best checkpoint
    state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, classes)
    preds = test_metrics.pop("_preds")
    labels = test_metrics.pop("_labels")

    # Image-level accuracy overstates the sample size whenever a manifest carries
    # groups: Dataset B's 2,669 test images come from only 19 physical seeds, so
    # 2,669 is not an independent-sample count. Score each group once by majority
    # vote and report that alongside, with the group count as the honest n.
    group_metrics = None
    if test_ds.rows and "group" in test_ds.rows[0]:
        from collections import Counter, defaultdict
        votes, truth = defaultdict(Counter), {}
        for row, pred, lab in zip(test_ds.rows, preds, labels):
            votes[row["group"]][pred] += 1
            truth[row["group"]] = lab
        correct = sum(1 for g, c in votes.items() if c.most_common(1)[0][0] == truth[g])
        group_metrics = {
            "n_groups": len(votes),
            "group_majority_vote_accuracy": correct / len(votes),
            "note": "Each group is one physical source seed, scored once. n_groups is "
                    "the independent-sample count; the image-level accuracy above "
                    "counts augmented copies of the same seed repeatedly.",
        }

    split_sizes = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        entry = {"images": len(ds.rows)}
        if ds.rows and "group" in ds.rows[0]:
            entry["groups"] = len({r["group"] for r in ds.rows})
        split_sizes[name] = entry

    metrics_path = os.path.join(cfg["paths"]["metrics"], f"variety_{args.dataset}_{tag}.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "manifest": manifest_path,
            "experiment": args.experiment,
            "tag": tag,
            "split_sizes": split_sizes,
            "best_val_f1_macro": best_val_f1,
            "test_metrics": test_metrics,
            "group_level": group_metrics,
            "checkpoint_sha256": checkpoint_sha256(best_ckpt_path),
        }, f, indent=2)

    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    if group_metrics:
        logger.info(f"Group-level (n={group_metrics['n_groups']} source seeds): "
                    f"{group_metrics['group_majority_vote_accuracy']:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt_path}")
    logger.info(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
