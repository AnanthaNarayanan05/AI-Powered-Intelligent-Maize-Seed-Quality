"""FAISS-backed similarity search over variety-model embeddings. Explicitly labeled
throughout as VISUAL/FEATURE SIMILARITY (docs/05_ARCHITECTURE.md Phase 12) — never
presented as agricultural certification or purity verification."""
from __future__ import annotations

import json
import os

import numpy as np


class EmbeddingIndex:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list[dict] = []  # parallel list: {"path":..., "label":...}

    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        assert embeddings.shape[0] == len(metadata)
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({
                **self.metadata[idx],
                "distance": float(dist),
                "similarity_score": float(1.0 / (1.0 + dist)),  # bounded (0,1], higher = more similar
            })
        return results

    def save(self, path: str):
        import faiss

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta.json", "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str, dim: int):
        import faiss

        obj = cls(dim)
        obj.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta.json") as f:
            obj.metadata = json.load(f)
        return obj
