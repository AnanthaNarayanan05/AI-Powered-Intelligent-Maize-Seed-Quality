"""Phase 8: SimCLR contrastive pretraining on a variety dataset (A or B), unlabeled.
Run locally on the RTX 4060:
    python -m src.training.pretrain_contrastive --dataset a --epochs 60
Produces outputs/checkpoints/contrastive_encoder_<dataset>.pt (backbone weights only,
loaded by train_variety.py when --use-contrastive-pretrain is passed).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data.datasets import ContrastiveImageDataset, list_all_filepaths
from src.contrastive.simclr import build_simclr_augmentation, nt_xent_loss
from src.models.variety_classifier import CognitiveAttentionClassifier
from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger

logger = get_logger("pretrain_contrastive")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["a", "b", "unified", "quality"], required=True,
                         help="'unified' pools every dataset's TRAIN rows via --manifest; "
                              "it has no data_root of its own and requires --manifest. "
                              "'quality' is the Mendeley EfficientMaize set (docs/09 sec "
                              "6.7 item 3's quality-head ablation) and also requires "
                              "--manifest, for the same reason.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--manifest", default=None,
                         help="Restrict pretraining to the train split of this manifest. "
                              "Without it, SimCLR sees every image in the dataset root "
                              "INCLUDING val and test, which lets the encoder learn "
                              "representations of the evaluation images before they are "
                              "ever evaluated. Required for a clean group-aware experiment.")
    parser.add_argument("--tag", default=None,
                         help="Suffix for the saved encoder, so a clean re-run does not "
                              "overwrite an earlier encoder.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    logger.info(f"Device: {device}")

    if args.dataset in ("unified", "quality") and not args.manifest:
        raise SystemExit(f"--dataset {args.dataset} requires --manifest")
    data_root = (cfg["paths"]["dataset_a"] if args.dataset == "a"
                 else cfg["paths"]["dataset_b"] if args.dataset == "b"
                 else cfg["paths"].get("dataset_quality") if args.dataset == "quality" else None)
    image_size = cfg["training"]["image_size"]
    epochs = args.epochs or cfg["contrastive"]["pretrain_epochs"]
    batch_size = args.batch_size or cfg["contrastive"]["pretrain_batch_size"]
    temperature = cfg["contrastive"]["temperature"]

    if args.manifest:
        import csv as _csv
        with open(args.manifest) as fh:
            rows = [r for r in _csv.DictReader(fh) if r["split"] == "train"]
        filepaths = [r["filepath"] for r in rows]
        n_groups = len({r["group"] for r in rows}) if rows and "group" in rows[0] else None
        logger.info(
            f"Contrastive pretraining on the TRAIN SPLIT ONLY of {args.manifest}: "
            f"{len(filepaths)} images"
            + (f" from {n_groups} source groups" if n_groups else "")
        )
    else:
        filepaths = list_all_filepaths(data_root)
        logger.warning(
            f"Pretraining on ALL {len(filepaths)} images in {data_root}, val and test "
            f"included. The resulting encoder must not be used for a held-out evaluation."
        )

    aug = build_simclr_augmentation(image_size)
    ds = ContrastiveImageDataset(filepaths, aug)
    num_workers = cfg["training"]["num_workers"]
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    num_classes_placeholder = 3  # unused during contrastive mode, head not trained here
    model = CognitiveAttentionClassifier(
        num_classes=num_classes_placeholder,
        backbone_name=cfg["backbone"],
        pretrained=args.pretrained,
        use_attention=cfg["attention"]["use_channel_attention"] or cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["device"]["mixed_precision"] and device.type == "cuda")

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg["paths"]["logs"], f"contrastive_{args.dataset}.jsonl")
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["device"]["mixed_precision"] and device.type == "cuda"):
                z1 = model(view1, mode="contrastive")
                z2 = model(view2, mode="contrastive")
                loss = nt_xent_loss(z1, z2, temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        elapsed = time.time() - t0
        logger.info(f"[{args.dataset}] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} ({elapsed:.1f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed}) + "\n")

    _tag = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(ckpt_dir, f"contrastive_encoder_{args.dataset}{_tag}.pt")
    torch.save({"backbone_state_dict": model.backbone.state_dict(), "config": cfg["contrastive"]}, ckpt_path)
    logger.info(f"Saved pretrained encoder to {ckpt_path}")


if __name__ == "__main__":
    main()
