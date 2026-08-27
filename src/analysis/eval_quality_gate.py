"""Compares candidate out-of-distribution scores for the quality head.

The first gate used cosine distance to the k nearest training embeddings, pooled
across both classes. It caught 100% of Dataset A but let 38 of 51 Dataset C crops
through while the head graded every one of them `Bad` at median confidence 1.000 --
a tray of corn is not 100% defective, so those grades needed flagging and were not.

The likely reason is that a pooled distance measures "does this look like a maize
kernel photograph", which Dataset C crops emphatically do. What the gate actually
needs to measure is "does this look like the kernels that taught the head what
`Bad` means" -- a class-conditional question.

This script scores four candidates on the same data and reports how well each
separates in-distribution kernels (Mendeley test, which must NOT be flagged) from
imagery the head was never validated on (Datasets A and C crops, which must be):

  pooled_knn        cosine distance to k nearest training embeddings, any class
  classwise_knn     cosine distance to k nearest embeddings OF THE PREDICTED CLASS
  mahalanobis       class-conditional Mahalanobis in a PCA-reduced space
  quality_proj      cosine distance after projecting onto the quality head's own
                    weight directions, so only quality-relevant variation counts

Reports AUROC plus, at each score's own 95th-percentile-of-validation threshold,
the false-flag rate on in-distribution data and the catch rate on each OOD set.

    python -m src.analysis.eval_quality_gate
"""
from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from src.pipeline.unified_pipeline import AnalysisPipeline

QUALITY_MANIFEST = "data_processed/manifest_dataset_4_quality.csv"
DATASET_A_MANIFEST = "data_processed/manifest_dataset_a.csv"
K = 5
N_OOD = 250
PCA_DIM = 64


def _rows(manifest, split):
    with open(manifest) as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split]


def _embed(pipe, model, tf, images):
    """Returns (normalised embeddings, predicted quality class indices)."""
    embs = np.zeros((len(images), model.feature_dim), dtype=np.float32)
    preds = np.zeros(len(images), dtype=np.int64)
    with torch.no_grad():
        for i, im in enumerate(images):
            t = tf(im).unsqueeze(0).to(pipe.device)
            e = model(t, mode="embedding")
            q = model.quality_head(e)
            embs[i] = e.cpu().numpy()[0]
            preds[i] = int(q.argmax(1).item())
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs, preds


def _load(paths):
    return [Image.open(p).convert("RGB") for p in paths]


def _serve_crops(pipe, image_paths, limit):
    """Crops produced the way serving produces them: detect, then crop with the
    configured context padding. This is what the gate actually sees in production,
    and it differs from a raw dataset image."""
    crops = []
    for p in image_paths:
        if len(crops) >= limit:
            break
        try:
            img = pipe.load_and_validate_image(p)
            for det in pipe.detect_seeds(p):
                crops.append(pipe.crop_seed(img, det["bbox"]))
                if len(crops) >= limit:
                    break
        except Exception:
            continue
    return crops


# ---------- candidate scores ----------

def pooled_knn(q, ref, ref_y, qpred, k=K, **kw):
    sims = q @ ref.T
    return 1.0 - np.partition(sims, -k, axis=1)[:, -k:].mean(1)


def classwise_knn(q, ref, ref_y, qpred, k=K, **kw):
    out = np.empty(len(q), dtype=np.float32)
    for c in np.unique(ref_y):
        idx = np.where(qpred == c)[0]
        if not len(idx):
            continue
        sub = ref[ref_y == c]
        sims = q[idx] @ sub.T
        kk = min(k, sub.shape[0])
        out[idx] = 1.0 - np.partition(sims, -kk, axis=1)[:, -kk:].mean(1)
    return out


def _fit_pca(ref, dim):
    mu = ref.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(ref - mu, full_matrices=False)
    return mu, vt[:dim].T


def mahalanobis(q, ref, ref_y, qpred, pca=None, **kw):
    mu, w = pca
    R = (ref - mu) @ w
    Q = (q - mu) @ w
    # shared covariance with shrinkage keeps the estimate conditioned
    cent = np.concatenate([R[ref_y == c] - R[ref_y == c].mean(0) for c in np.unique(ref_y)])
    cov = np.cov(cent.T) + np.eye(w.shape[1]) * 1e-3
    inv = np.linalg.inv(cov)
    means = {c: R[ref_y == c].mean(0) for c in np.unique(ref_y)}
    out = np.empty(len(q), dtype=np.float32)
    for i, (v, c) in enumerate(zip(Q, qpred)):
        d = v - means.get(int(c), R.mean(0))
        out[i] = float(np.sqrt(max(d @ inv @ d, 0.0)))
    return out


def quality_proj(q, ref, ref_y, qpred, proj=None, k=K, **kw):
    """Distance measured only along directions the quality head actually reads."""
    R = ref @ proj
    Q = q @ proj
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
    out = np.empty(len(q), dtype=np.float32)
    for c in np.unique(ref_y):
        idx = np.where(qpred == c)[0]
        if not len(idx):
            continue
        sub = R[ref_y == c]
        sims = Q[idx] @ sub.T
        kk = min(k, sub.shape[0])
        out[idx] = 1.0 - np.partition(sims, -kk, axis=1)[:, -kk:].mean(1)
    return out


def auroc(pos, neg):
    """P(score(ood) > score(in-dist)). 1.0 = perfect separation, 0.5 = useless."""
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    sc = np.concatenate([pos, neg])
    order = np.argsort(sc)
    ranks = np.empty(len(sc), dtype=np.float64)
    ranks[order] = np.arange(1, len(sc) + 1)
    n1, n0 = lab.sum(), len(lab) - lab.sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    random.seed(0)
    pipe = AnalysisPipeline()
    model, _, qclasses = pipe._get_unified_model()
    tf = pipe._get_eval_transform()

    print("embedding Mendeley train (reference)...")
    tr = _rows(QUALITY_MANIFEST, "train")
    ref, _ = _embed(pipe, model, tf, _load([r["filepath"] for r in tr]))
    ref_y = np.array([qclasses.index(r["label"]) for r in tr])

    print("embedding Mendeley val (calibration) and test (must NOT be flagged)...")
    va = _rows(QUALITY_MANIFEST, "val")
    val, val_pred = _embed(pipe, model, tf, _load([r["filepath"] for r in va]))
    te = _rows(QUALITY_MANIFEST, "test")
    tst, tst_pred = _embed(pipe, model, tf, _load([r["filepath"] for r in te]))

    ood_sets = {}

    a_rows = _rows(DATASET_A_MANIFEST, "test")
    print(f"embedding Dataset A ({len(a_rows)} images, must be flagged)...")
    ood_sets["dataset_A"] = _embed(pipe, model, tf, _load([r["filepath"] for r in a_rows]))

    c_dir = pipe.cfg["paths"]["dataset_c"]
    c_imgs = []
    for dp, _, fs in os.walk(c_dir):
        for f in fs:
            if f.lower().endswith((".jpg", ".png")):
                c_imgs.append(os.path.join(dp, f))
    random.shuffle(c_imgs)
    print("producing Dataset C SERVE-TIME crops (detect -> crop, must be flagged)...")
    c_crops = _serve_crops(pipe, c_imgs[:40], N_OOD)
    print(f"  {len(c_crops)} crops")
    ood_sets["dataset_C_crops"] = _embed(pipe, model, tf, c_crops)

    pca = _fit_pca(ref, PCA_DIM)
    proj = model.quality_head[1].weight.detach().cpu().numpy().T  # (feature_dim, 2)

    scorers = {
        "pooled_knn": (pooled_knn, {}),
        "classwise_knn": (classwise_knn, {}),
        "mahalanobis": (mahalanobis, {"pca": pca}),
        "quality_proj": (quality_proj, {"proj": proj}),
    }

    results = {}
    print(f"\n{'score':<16}{'thresh':>9}{'in-dist flagged':>17}", end="")
    for name in ood_sets:
        print(f"{name+' caught':>22}", end="")
    print(f"{'AUROC(A)':>10}{'AUROC(C)':>10}")
    print("-" * (16 + 9 + 17 + 22 * len(ood_sets) + 20))

    for name, (fn, kw) in scorers.items():
        s_val = fn(val, ref, ref_y, val_pred, **kw)
        thr = float(np.percentile(s_val, 95))
        s_test = fn(tst, ref, ref_y, tst_pred, **kw)
        false_flag = float((s_test > thr).mean())

        row = {"threshold": thr, "in_distribution_flag_rate": false_flag}
        print(f"{name:<16}{thr:9.4f}{false_flag*100:16.1f}%", end="")
        aurocs = {}
        for oname, (oe, op) in ood_sets.items():
            s_ood = fn(oe, ref, ref_y, op, **kw)
            caught = float((s_ood > thr).mean())
            row[f"{oname}_catch_rate"] = caught
            aurocs[oname] = auroc(s_ood, s_test)
            print(f"{caught*100:21.1f}%", end="")
        print(f"{aurocs['dataset_A']:10.3f}{aurocs['dataset_C_crops']:10.3f}")
        row["auroc"] = aurocs
        results[name] = row

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/quality_gate_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten: outputs/metrics/quality_gate_comparison.json")


if __name__ == "__main__":
    main()
