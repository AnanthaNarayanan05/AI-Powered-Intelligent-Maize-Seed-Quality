"""Builds the FAISS similarity index for a trained variety model (Phase 12). Run
locally after train_variety.py has produced a best checkpoint:
    python -m src.similarity.build_index --dataset a
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import VarietyImageDataset
from src.contrastive.simclr import build_eval_transform
from src.models.variety_classifier import CognitiveAttentionClassifier
from src.similarity.embedding_index import EmbeddingIndex
from src.utils.config import load_config, get_device
from src.utils.logging_utils import get_logger

logger = get_logger("build_index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["a", "b"], required=True)
    parser.add_argument("--experiment", default="full")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device(cfg["device"]["prefer"])
    ds_cfg = cfg["dataset_a"] if args.dataset == "a" else cfg["dataset_b"]
    classes = ds_cfg["classes"]

    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], f"variety_{args.dataset}_{args.experiment}_best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model = CognitiveAttentionClassifier(
        num_classes=len(classes), backbone_name=state["backbone"],
        pretrained=False, use_attention=state["use_attention"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    manifest_path = os.path.join(cfg["paths"]["processed"], f"manifest_dataset_{args.dataset}.csv")
    eval_tf = build_eval_transform(cfg["training"]["image_size"])
    ds = VarietyImageDataset(manifest_path, "train", classes, eval_tf)  # index built from train split (gallery)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    all_embeddings, all_meta = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            emb = model(imgs, mode="embedding").cpu().numpy()
            all_embeddings.append(emb)
            batch_rows = ds.rows[i * loader.batch_size: i * loader.batch_size + imgs.size(0)]
            for row, label in zip(batch_rows, labels.tolist()):
                all_meta.append({"path": row["filepath"], "label": classes[label]})

    embeddings = np.concatenate(all_embeddings, axis=0)
    index = EmbeddingIndex(dim=embeddings.shape[1])
    index.add(embeddings, all_meta)

    out_path = cfg["paths"][f"faiss_index_{args.dataset}"].rsplit(".index", 1)[0]
    index.save(out_path)
    logger.info(f"Saved FAISS index with {len(all_meta)} embeddings to {out_path}.faiss")


if __name__ == "__main__":
    main()
