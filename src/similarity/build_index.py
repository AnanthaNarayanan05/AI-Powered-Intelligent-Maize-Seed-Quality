"""Builds a FAISS gallery index for a trained seed model (Phase 12/13).

    python -m src.similarity.build_index --dataset unified
    python -m src.similarity.build_index --dataset a          # ablation checkpoints
    python -m src.similarity.build_index --dataset b

The gallery is built from the TRAIN split only. Test and validation images are held
out so that similarity results are never a lookup of the evaluation set, and so that
a neighbour shown in the UI is an image the model was fitted on rather than one of
the images its reported metrics were measured against.

Each index is bound to the encoder that built it (see EmbeddingIndex): the query
model and the gallery model must be the same model, because an L2 distance between
two different feature spaces is not a distance between anything.

Metadata is copied from the manifest verbatim. Labels a dataset never carried are
stored as null; nothing is inferred from a neighbouring column, from folder names,
or from the model's own predictions.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.contrastive.simclr import build_eval_transform
from src.data.datasets import _read_manifest
from src.similarity.embedding_index import EmbeddingIndex, encoder_id, index_path
from src.utils.config import load_config, get_device
from src.utils.logging_utils import get_logger

logger = get_logger("build_index")


class _GalleryDataset(Dataset):
    """Images only. The index needs no labels to be built, so none are looked up
    here — that keeps the 3,401 quality-only rows of the unified manifest usable
    without inventing a variety for them."""

    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img = Image.open(self.rows[idx]["filepath"]).convert("RGB")
        return self.transform(img), idx


def _clean(value):
    """Manifest blanks mean 'this dataset never labelled this'. Keep that as null."""
    value = (value or "").strip()
    return value or None


def _load_encoder(cfg, dataset: str, experiment: str, device):
    """Returns (model, checkpoint_path, manifest_path, gallery_source)."""
    ckpt_dir = cfg["paths"]["checkpoints"]

    if dataset == "unified":
        from src.models.unified_model import UnifiedSeedModel
        from src.registry import get_registry

        # The gallery must be built by the same checkpoint the serving path
        # loads, or the index is bound to an encoder that never answers a query.
        # Asking the registry is what guarantees they are the same file.
        ckpt_path = get_registry().resolve("embedding").checkpoint
        state = torch.load(ckpt_path, map_location=device)
        model = UnifiedSeedModel(
            state["variety_classes"], state["quality_classes"],
            backbone_name=cfg["backbone"], pretrained=False,
            use_attention=state.get("use_attention", True),
        ).to(device)
        model.load_state_dict(state["model_state_dict"])
        manifest = os.path.join(cfg["paths"]["processed"], "manifest_unified.csv")
        return model, ckpt_path, manifest, "unified"

    from src.models.variety_classifier import CognitiveAttentionClassifier

    # Ablation encoders are addressed by path, not by the registry: they are the
    # variants the registry deliberately does not serve, and their indices exist
    # only to reproduce the comparison.
    ds_cfg = cfg["dataset_a"] if dataset == "a" else cfg["dataset_b"]
    ckpt_path = os.path.join(ckpt_dir, f"variety_{dataset}_{experiment}_best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model = CognitiveAttentionClassifier(
        num_classes=len(ds_cfg["classes"]), backbone_name=state["backbone"],
        pretrained=False, use_attention=state["use_attention"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    manifest = os.path.join(cfg["paths"]["processed"], f"manifest_dataset_{dataset}.csv")
    return model, ckpt_path, manifest, f"dataset_{dataset}"


def _row_metadata(row: dict, dataset: str, gallery_source: str) -> dict:
    """One metadata entry per gallery vector, in a schema shared by all indices."""
    if dataset == "unified":
        return {
            "path": row["filepath"],
            "variety_label": _clean(row.get("variety_label")),
            "quality_label": _clean(row.get("quality_label")),
            "source": _clean(row.get("source")) or gallery_source,
        }
    return {
        "path": row["filepath"],
        "variety_label": _clean(row.get("label")),
        "quality_label": None,  # a/b were variety-only datasets
        "source": gallery_source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["a", "b", "unified"], required=True)
    parser.add_argument("--experiment", default="full", help="ignored for --dataset unified")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device(cfg["device"]["prefer"])

    model, ckpt_path, manifest_path, gallery_source = _load_encoder(
        cfg, args.dataset, args.experiment, device
    )
    model.eval()

    rows = _read_manifest(manifest_path, args.split)
    if not rows:
        raise SystemExit(f"No '{args.split}' rows in {manifest_path}")

    eval_tf = build_eval_transform(cfg["training"]["image_size"])
    ds = _GalleryDataset(rows, eval_tf)
    loader = DataLoader(
        ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    all_embeddings, all_meta = [], []
    with torch.no_grad():
        for imgs, indices in loader:
            emb = model(imgs.to(device), mode="embedding").cpu().numpy()
            all_embeddings.append(emb)
            # Metadata is keyed by the index the loader returned alongside the image,
            # so the vector/metadata pairing cannot drift from batch arithmetic.
            for i in indices.tolist():
                all_meta.append(_row_metadata(rows[i], args.dataset, gallery_source))

    embeddings = np.concatenate(all_embeddings, axis=0)
    index = EmbeddingIndex(dim=embeddings.shape[1], info={
        "encoder": encoder_id(args.dataset, args.experiment),
        "checkpoint": ckpt_path.replace("\\", "/"),
        "gallery_manifest": manifest_path.replace("\\", "/"),
        "gallery_split": args.split,
        "note": "VISUAL/FEATURE SIMILARITY only. Neighbour labels describe the "
                "neighbour images, not the query.",
    })
    index.add(embeddings, all_meta)

    out_path = index_path(cfg, args.dataset)
    index.save(out_path)
    labelled = sum(1 for m in all_meta if m["variety_label"])
    logger.info(
        f"Saved {len(all_meta)} {embeddings.shape[1]}-d embeddings to {out_path}.faiss "
        f"(encoder={index.info['encoder']}, split={args.split}, "
        f"{labelled} with a variety label, {len(all_meta) - labelled} without)"
    )


if __name__ == "__main__":
    main()
