"""Calibrates a distribution gate for the quality head.

The quality head scores 97.25% on the Mendeley test split, and calls 71% of
Dataset A defective at 89% mean confidence. Both facts are true at once: it is
accurate on kernel close-ups resembling its training data, and it extrapolates
confidently on imagery it has never seen. Softmax confidence does not separate the
two -- it is *higher* on the out-of-distribution set for Dataset B (96.2%) than on
some in-distribution data.

So the gate uses representation distance instead of confidence. Embeddings of the
Mendeley training split define the region where the head has evidence. At serve
time an input's distance to its nearest training neighbours says whether the
quality prediction is interpolation or extrapolation.

The threshold is calibrated on the Mendeley *validation* split, never on test.

The operating point was chosen by measurement, not by convention. An initial 95th
percentile flagged only 5% of in-distribution kernels but caught just 45% of
serve-time crops from Dataset C -- on which the head graded 51 of 51 kernels `Bad`
at median confidence 1.000, which cannot be right. Sweeping the percentile
(src/analysis/eval_quality_gate.py):

    val pct   thresh   in-dist flagged   Dataset A caught   Dataset C caught
         95   0.2056             6.1%             100.0%              45.2%
         90   0.1654            10.5%             100.0%              58.0%
         80   0.1225            20.7%             100.0%              76.8%
    --> 70   0.0929            28.9%             100.0%              88.4%
         60   0.0788            36.4%             100.0%              96.4%
         50   0.0648            47.1%             100.0%              98.4%

70 is chosen because the two error types are not equally costly. A false flag
withholds a valid grade and says "unverified"; a miss asserts a defect that may be
wrong. Confidently mislabelling a sound kernel as defective is the worse failure,
so the gate is deliberately biased toward withholding.

Three alternative scores were tested and rejected -- class-conditional kNN (AUROC
0.879 vs 0.880, no better), class-conditional Mahalanobis in PCA space (0.647), and
distance projected onto the quality head's own weights (0.496, chance). The pooled
kNN score is the best available discriminator; the original gate's weakness was its
threshold, not its scoring function.

Writes outputs/checkpoints/quality_reference.npz
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from PIL import Image

from src.pipeline.unified_pipeline import AnalysisPipeline

MANIFEST = "data_processed/manifest_dataset_4_quality.csv"
K = 5  # neighbours averaged, so one outlying training image cannot admit a whole region
VAL_PERCENTILE = 70  # see the sweep in the module docstring


def embed_split(pipeline, model, tf, split, limit=None):
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["split"] == split]
    if limit:
        rows = rows[:limit]
    out = np.zeros((len(rows), model.feature_dim), dtype=np.float32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            img = Image.open(r["filepath"]).convert("RGB")
            e = model(tf(img).unsqueeze(0).to(pipeline.device), mode="embedding")
            out[i] = e.cpu().numpy()[0]
    # Cosine geometry: normalise so distance reflects direction, not activation scale.
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def knn_distance(queries: np.ndarray, reference: np.ndarray, k: int = K,
                 exclude_self: bool = False) -> np.ndarray:
    sims = queries @ reference.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    top = np.partition(sims, -k, axis=1)[:, -k:]
    return 1.0 - top.mean(axis=1)  # mean cosine distance to k nearest


def main():
    p = AnalysisPipeline()
    model, _, _ = p._get_unified_model()
    tf = p._get_eval_transform()

    print("embedding Mendeley train split (the region where the head has evidence)...")
    train = embed_split(p, model, tf, "train")
    print(f"  reference set: {train.shape[0]} embeddings, dim {train.shape[1]}")

    print("calibrating threshold on the validation split...")
    val = embed_split(p, model, tf, "val")
    d_val = knn_distance(val, train)
    threshold = float(np.percentile(d_val, VAL_PERCENTILE))

    print(f"  validation kNN distance: median={np.median(d_val):.4f} "
          f"p{VAL_PERCENTILE}={threshold:.4f} max={d_val.max():.4f}")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    dst = "outputs/checkpoints/quality_reference.npz"
    np.savez_compressed(dst, reference=train, threshold=threshold, k=K,
                        val_percentile=VAL_PERCENTILE)
    print(f"written: {dst}")

    # Report the gate's behaviour on data it was not calibrated against.
    print("\ngate behaviour (share flagged as outside the validated range):")
    for name, path, split in [
        ("Dataset 4 test  (in-distribution)", MANIFEST, "test"),
        ("Dataset A test  (never quality-labelled)", "data_processed/manifest_dataset_a.csv", "test"),
        ("Dataset B test  (never quality-labelled)", "data_processed/manifest_dataset_b_grouped.csv", "test"),
    ]:
        rows = [r for r in csv.DictReader(open(path)) if r["split"] == split][:400]
        embs = np.zeros((len(rows), model.feature_dim), dtype=np.float32)
        with torch.no_grad():
            for i, r in enumerate(rows):
                img = Image.open(r["filepath"]).convert("RGB")
                embs[i] = model(tf(img).unsqueeze(0).to(p.device), mode="embedding").cpu().numpy()[0]
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        d = knn_distance(embs, train)
        print(f"  {name:<42} n={len(rows):4d}  flagged={np.mean(d > threshold)*100:5.1f}%  "
              f"median dist={np.median(d):.4f}")


if __name__ == "__main__":
    main()
