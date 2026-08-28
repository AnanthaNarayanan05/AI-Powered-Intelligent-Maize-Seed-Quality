"""Trains the unified two-head seed model on every dataset at once.

    python -m src.training.train_unified --encoder outputs/checkpoints/contrastive_encoder_unified_unified.pt

Each batch mixes variety-labelled images (Datasets A and B) with quality-labelled
images (the Mendeley set). Loss is masked per head via ignore_index=-1, so the
variety head never receives gradient from an image whose variety nobody recorded,
and vice versa. Both heads share the trunk, so all 23,605 images shape the
representation.

Heads are weighted by their share of labelled data rather than equally: quality
rows are ~20% of the corpus, and an unweighted sum lets the variety task dominate
the trunk. Reported metrics are per-head and per-split; there is no combined
"accuracy" number, because averaging across two different label spaces would not
mean anything.
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

from src.contrastive.simclr import build_eval_transform, build_simclr_augmentation
from src.data.datasets import UnifiedSeedDataset
from src.models.unified_model import UnifiedSeedModel
from src.utils.config import load_config, get_device
from src.registry.model_registry import checkpoint_sha256
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger

logger = get_logger("train_unified")

MANIFEST = "data_processed/manifest_unified.csv"
VARIETY_CLASSES = ["Zea_mays_Chulpi_Cancha", "Zea_mays_Indurata", "Zea_mays_Rugosa",
                   "Bhihilifa", "SanzalSima", "WangDataa"]
QUALITY_CLASSES = ["Good", "Bad"]
IGNORE = -1


def _head_metrics(preds, labels, classes):
    if not labels:
        return None
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    present = sorted(set(labels) | set(preds))
    return {
        "n": len(labels),
        "accuracy": acc,
        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(classes)))).tolist(),
        "per_class_report": classification_report(
            labels, preds, labels=present,
            target_names=[classes[i] for i in present],
            output_dict=True, zero_division=0),
    }


def evaluate(model, loader, device):
    model.eval()
    vp, vl, qp, ql = [], [], [], []
    with torch.no_grad():
        for imgs, v, q in loader:
            vlog, qlog = model(imgs.to(device), mode="classify")
            vm, qm = v != IGNORE, q != IGNORE
            if vm.any():
                vp.extend(vlog[vm].argmax(1).cpu().tolist()); vl.extend(v[vm].tolist())
            if qm.any():
                qp.extend(qlog[qm].argmax(1).cpu().tolist()); ql.extend(q[qm].tolist())
    return {
        "variety": _head_metrics(vp, vl, VARIETY_CLASSES),
        "quality": _head_metrics(qp, ql, QUALITY_CLASSES),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="Contrastively pretrained trunk to start from.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-attention", dest="attention", action="store_false", default=True)
    ap.add_argument("--tag", default="unified")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])

    img_size = cfg["training"]["image_size"]
    train_ds = UnifiedSeedDataset(MANIFEST, "train", VARIETY_CLASSES, QUALITY_CLASSES,
                                  build_simclr_augmentation(img_size))
    val_ds = UnifiedSeedDataset(MANIFEST, "val", VARIETY_CLASSES, QUALITY_CLASSES,
                                build_eval_transform(img_size))
    test_ds = UnifiedSeedDataset(MANIFEST, "test", VARIETY_CLASSES, QUALITY_CLASSES,
                                 build_eval_transform(img_size))

    n_var = sum(1 for r in train_ds.rows if r.get("variety_label"))
    n_qual = sum(1 for r in train_ds.rows if r.get("quality_label"))
    logger.info(f"train: {len(train_ds)} images ({n_var} variety-labelled, {n_qual} quality-labelled)")

    nw = cfg["training"]["num_workers"]
    lk = {"num_workers": nw, "pin_memory": device.type == "cuda", "persistent_workers": nw > 0}
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **lk)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **lk)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **lk)

    model = UnifiedSeedModel(
        VARIETY_CLASSES, QUALITY_CLASSES,
        backbone_name=cfg["backbone"], pretrained=True,
        use_attention=args.attention,
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
    ).to(device)

    if args.encoder:
        if not os.path.exists(args.encoder):
            raise SystemExit(f"--encoder {args.encoder} not found")
        model.load_pretrained_trunk(args.encoder, map_location=device)
        logger.info(f"Loaded contrastive trunk from {args.encoder}")

    # Weight each head by its share of labelled rows, so the smaller quality task is
    # not drowned out by the 13,329 variety rows.
    total = n_var + n_qual
    w_var, w_qual = n_var / total, n_qual / total
    logger.info(f"head loss weights -- variety {w_var:.3f}, quality {w_qual:.3f}")

    crit_v = nn.CrossEntropyLoss(ignore_index=IGNORE)
    crit_q = nn.CrossEntropyLoss(ignore_index=IGNORE)

    epochs = args.epochs or cfg["training"]["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = cfg["device"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"{args.tag}_seed_model_best.pt")
    log_path = os.path.join(cfg["paths"]["logs"], f"{args.tag}.jsonl")

    best_score, patience, stale = -1.0, cfg["training"]["early_stopping_patience"], 0

    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for imgs, v, q in train_loader:
            imgs, v, q = imgs.to(device), v.to(device), q.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                vlog, qlog = model(imgs, mode="classify")
                lv = crit_v(vlog, v) if (v != IGNORE).any() else vlog.sum() * 0.0
                lq = crit_q(qlog, q) if (q != IGNORE).any() else qlog.sum() * 0.0
                loss = w_var * lv + w_qual * lq
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()

        val = evaluate(model, val_loader, device)
        # Selection criterion is the mean of the two head F1s: optimising one head's
        # score alone would let the other quietly regress.
        parts = [m["f1_macro"] for m in val.values() if m]
        score = sum(parts) / len(parts)
        logger.info(
            f"epoch {ep+1}/{epochs} loss={tot/max(len(train_loader),1):.4f} "
            f"variety_f1={val['variety']['f1_macro']:.4f} "
            f"quality_f1={val['quality']['f1_macro']:.4f} "
            f"mean={score:.4f} ({time.time()-t0:.1f}s)"
        )
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep + 1, "loss": tot / max(len(train_loader), 1),
                                "val": {k: (m["f1_macro"] if m else None) for k, m in val.items()}}) + "\n")

        if score > best_score:
            best_score, stale = score, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "variety_classes": VARIETY_CLASSES,
                        "quality_classes": QUALITY_CLASSES,
                        "use_attention": args.attention}, best_path)
        else:
            stale += 1
            if stale >= patience:
                logger.info(f"Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    test = evaluate(model, test_loader, device)

    dst = os.path.join(cfg["paths"]["metrics"], f"{args.tag}_seed_model.json")
    with open(dst, "w") as f:
        json.dump({
            "manifest": MANIFEST,
            "encoder": args.encoder,
            "use_attention": args.attention,
            "variety_classes": VARIETY_CLASSES,
            "quality_classes": QUALITY_CLASSES,
            "split_sizes": {k: len(d) for k, d in
                            (("train", train_ds), ("val", val_ds), ("test", test_ds))},
            "best_val_mean_f1": best_score,
            "test_metrics": test,
            # Binds these numbers to the artefact that produced them. The registry
            # recomputes it; retraining without re-evaluating shows up as a stale
            # binding instead of the old score being reported for the new model.
            "checkpoint_sha256": checkpoint_sha256(best_path),
        }, f, indent=2)

    for head, m in test.items():
        if m:
            logger.info(f"TEST {head}: n={m['n']} acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")
    logger.info(f"checkpoint: {best_path}")
    logger.info(f"metrics: {dst}")


if __name__ == "__main__":
    main()
