"""FAISS-backed similarity search over seed-model embeddings.

Explicitly labeled throughout as VISUAL/FEATURE SIMILARITY (docs/05_ARCHITECTURE.md
Phase 12) — never presented as agricultural certification or purity verification.
A neighbour is an image that lands near the query in one particular model's feature
space. It is not evidence about the query's variety, and the labels attached to the
neighbours belong to *those* images, not to the query.

Encoder binding
---------------
An embedding only means something relative to the encoder that produced it. Two
different models both emit 1280-d vectors from an EfficientNet-B0 trunk, so a query
embedded by model X searched against an index built by model Y raises no error
anywhere in FAISS — it just returns confident nonsense. Every index therefore
carries a provenance sidecar naming the encoder that built it, and `load()` refuses
an index whose encoder does not match the one the caller is about to query with.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


class IndexEncoderMismatch(RuntimeError):
    """Raised when an index would be queried with a different encoder than built it."""


# Metadata schema written for every gallery image. Labels are copied from the
# manifest and are None when that dataset never carried the label — never inferred.
_META_KEYS = ("path", "variety_label", "quality_label", "source")


class EmbeddingIndex:
    def __init__(self, dim: int, info: dict | None = None):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list[dict] = []  # parallel list, one entry per vector
        self.info: dict = dict(info or {})

    # ---------- build ----------
    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"{embeddings.shape[0]} embeddings but {len(metadata)} metadata rows"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings are {embeddings.shape[1]}-d, index is {self.dim}-d"
            )
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)

    # ---------- query ----------
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise IndexEncoderMismatch(
                f"query embedding is {query.shape[1]}-d but the index is {self.index.d}-d"
            )
        top_k = max(1, min(int(top_k), self.index.ntotal))
        distances, indices = self.index.search(query, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append({
                **entry,
                # Kept for callers that predate the split variety/quality schema.
                "label": entry.get("variety_label"),
                "distance": float(dist),
                # Bounded (0,1], higher = nearer. A monotone rescaling of L2 distance
                # for display only — it is not a probability and not a confidence.
                "similarity_score": float(1.0 / (1.0 + dist)),
            })
        return results

    # ---------- persistence ----------
    def save(self, path: str):
        import faiss

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta.json", "w") as f:
            json.dump(self.metadata, f)
        info = {
            **self.info,
            "dim": int(self.index.d),
            "ntotal": int(self.index.ntotal),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(path + ".info.json", "w") as f:
            json.dump(info, f, indent=2)
        self.info = info

    @classmethod
    def load(cls, path: str, dim: int | None = None, encoder: str | None = None):
        """Load an index, refusing one that cannot honestly answer the caller's query.

        `dim` and `encoder` describe the model the caller intends to embed with. Both
        are checked, because a silent mismatch produces plausible-looking neighbours
        rather than an error.
        """
        import faiss

        faiss_index = faiss.read_index(path + ".faiss")
        obj = cls(dim=int(faiss_index.d))
        obj.index = faiss_index
        with open(path + ".meta.json") as f:
            obj.metadata = json.load(f)

        info_path = path + ".info.json"
        obj.info = {}
        if os.path.exists(info_path):
            with open(info_path) as f:
                obj.info = json.load(f)

        if dim is not None and int(dim) != obj.index.d:
            raise IndexEncoderMismatch(
                f"index at {path} is {obj.index.d}-d but the query encoder emits {dim}-d "
                f"embeddings. Rebuild it with src.similarity.build_index."
            )
        built_by = obj.info.get("encoder")
        if encoder is not None and built_by is not None and built_by != encoder:
            raise IndexEncoderMismatch(
                f"index at {path} was built by encoder '{built_by}' but the query would "
                f"be embedded by '{encoder}'. Distances between two different feature "
                f"spaces are meaningless. Rebuild it with src.similarity.build_index."
            )
        if len(obj.metadata) != obj.index.ntotal:
            raise IndexEncoderMismatch(
                f"index at {path} holds {obj.index.ntotal} vectors but "
                f"{len(obj.metadata)} metadata rows; the mapping is not trustworthy."
            )
        return obj


def encoder_id(dataset: str, experiment: str = "full") -> str:
    """Canonical name of the model that embeds queries for a given gallery.

    Written into the index sidecar at build time and passed to `load()` at query
    time, so the two can be compared rather than assumed equal.
    """
    if dataset == "unified":
        return "unified_seed_model"
    return f"variety_{dataset}_{experiment}"


def index_path(cfg: dict, dataset: str) -> str:
    """Path stem for a dataset's index; FAISS/metadata/info suffixes are appended."""
    key = f"faiss_index_{dataset}"
    if key not in cfg["paths"]:
        raise KeyError(f"configs/config.yaml has no paths.{key}")
    return cfg["paths"][key].rsplit(".index", 1)[0]
