"""Shared pieces of the maize-identity gate (Phase 4).

Why a second gate exists alongside the quality gate
---------------------------------------------------
``build_quality_reference.py`` already calibrates a k-NN distance gate, and Phase 4
is required to use the existing distribution gating first. It was tried, and it
does not answer this question. Its reference set is the Mendeley quality *training
split alone*, so the question it asks is "does the quality head have evidence for
this kernel?" -- not "is this a maize kernel at all?". Its own calibration sweep
records the consequence: at the chosen operating point it flags 100% of Dataset A
and 88.4% of Dataset C, both of which are maize, plus 28.9% of in-distribution
kernels. Routed to a foreign-object flag it would call most real maize a possible
foreign object, which is worse than having no flag at all.

The scoring function is kept -- pooled k-NN cosine distance was measured to be the
best of four candidates (``eval_quality_gate.py``: AUROC 0.880 against 0.879,
0.647 and 0.496). What changes is the reference set. Here it spans every maize
kernel corpus the project holds, so "far from every known maize kernel" can
honestly mean "not a maize kernel".

Both gates use the same embedding from the same unified model. No new model is
trained for this: the backbone was contrastively pretrained on maize and then
trained on six varieties and two quality classes, so its representation is already
shaped by what maize looks like. Phase 4 is a derived measurement over it.

FLAGGING, NOT CLASSIFICATION
---------------------------
A high distance means the object does not resemble any maize kernel the system
knows. It does not say what the object is. There is no stone/husk/cob/debris
label anywhere in this project's data, so no such claim is ever made.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import os
import random

import numpy as np
import torch
from PIL import Image

K = 5  # neighbours averaged, matching the quality gate

REFERENCE_PATH = "outputs/checkpoints/maize_reference.npz"
CACHE_DIR = "outputs/cache/maize_identity"

UNIFIED_MANIFEST = "data_processed/manifest_unified.csv"
GRAINSPACE_GLOB = "datasets/grainspace_maize_m600/val/*/*.png"
GRAINSET_ROOT = "datasets/grainset_maize_impurities"

# GrainSpace is the project's only local source of visibly damaged maize -- mould,
# insect damage, breakage. A reference without it would put a mouldy kernel far
# from everything it knows and flag it as a possible foreign object, which is the
# single most likely way this feature could embarrass itself. It is split three
# ways by a fixed seed so the reference, the calibration set and the evaluation
# set never share an image.
GRAINSPACE_SPLIT = {"reference": 0.50, "calibration": 0.25, "evaluation": 0.25}
GRAINSPACE_SEED = 0


def grainspace_split(seed: int = GRAINSPACE_SEED) -> dict[str, list[str]]:
    """Partitions the GrainSpace val images, stratified by condition class."""
    by_class: dict[str, list[str]] = {}
    for path in sorted(glob.glob(GRAINSPACE_GLOB)):
        by_class.setdefault(os.path.basename(os.path.dirname(path)), []).append(path)

    out: dict[str, list[str]] = {k: [] for k in GRAINSPACE_SPLIT}
    for cls, paths in sorted(by_class.items()):
        rng = random.Random(f"{seed}:{cls}")
        paths = list(paths)
        rng.shuffle(paths)
        start = 0
        for name, frac in GRAINSPACE_SPLIT.items():
            n = int(round(len(paths) * frac))
            out[name].extend(paths[start:start + n])
            start += n
        out[list(GRAINSPACE_SPLIT)[-1]].extend(paths[start:])
    return out


def unified_rows(split: str) -> list[dict]:
    with open(UNIFIED_MANIFEST) as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split]


def grainset_paths(subdir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(GRAINSET_ROOT, subdir, "*.png")))


def load_grainset_view(path: str) -> Image.Image:
    """One object view from a GrainSet image.

    Every GrainSet image is the same physical object photographed twice, side by
    side, with a dark seam at the exact midpoint -- the mean column-brightness
    profile over 60 impurity images is a symmetric double hump troughing at 50%.
    Scoring both halves would double every object and report 6,000 independent
    impurities where there are 3,000, so only the left view is used and the counts
    stay honest.
    """
    img = Image.open(path).convert("RGB")
    return img.crop((0, 0, img.width // 2, img.height))


def detect_crops(pipeline, image) -> list:
    """The kernel crops the serving path would produce for one image.

    This is not a detail. The gate is calibrated so that a stated fraction of real
    maize is falsely flagged, and that promise is only meaningful if the
    calibration images look like what the gate is actually handed. At serve time
    it is never handed a dataset photograph -- it is handed a YOLO bounding box
    cut out of one. Measured on identical kernels, the two differ by a factor of
    eighteen in median distance (0.0058 as dataset images, 0.1026 as crops), which
    is far more than the margin any threshold here is chosen with.
    """
    crops = []
    for det in pipeline.detect_seeds(image):
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        if x2 > x1 and y2 > y1:
            crops.append(image.crop((x1, y1, x2, y2)))
    return crops


def embed_crops(pipeline, model, transform, paths, loader=None, chunk: int = 128,
                cache_key: str | None = None) -> np.ndarray:
    """Detect-then-embed over a list of images, in the serving domain.

    One image yields zero or more rows, so the result is not aligned with
    ``paths`` -- an object YOLO never finds is never scored, which is a real
    limitation of the flag and not something to paper over by falling back to the
    whole image.
    """
    loader = loader or (lambda p: Image.open(p).convert("RGB"))
    cache = None
    if cache_key:
        os.makedirs(CACHE_DIR, exist_ok=True)
        digest = hashlib.sha256('\n'.join(paths).encode()).hexdigest()[:16]
        cache = os.path.join(CACHE_DIR, f"crop_{cache_key}_{len(paths)}_{digest}.npy")
        if os.path.exists(cache):
            return np.load(cache)

    parts, buf, counts, sizes = [], [], [], []
    for path in paths:
        try:
            crops = detect_crops(pipeline, loader(path))
        except Exception:  # noqa: BLE001 -- a corrupt upload must not stop a sweep
            crops = []
        counts.append(len(crops))
        sizes.extend(min(c.size) for c in crops)
        buf.extend(crops)
        if len(buf) >= chunk:
            parts.append(embed(pipeline, model, transform, buf))
            buf = []
    if buf:
        parts.append(embed(pipeline, model, transform, buf))
    result = (np.concatenate(parts) if parts
              else np.zeros((0, model.feature_dim), np.float32))
    if cache:
        np.save(cache, result)
        # Per-image crop counts, saved beside the embeddings so the detection rate
        # -- the ceiling on end-to-end recall -- costs no second detector pass.
        np.save(cache[:-4] + ".counts.npy", np.asarray(counts, np.int32))
        # Per-crop short side, row-aligned with the embeddings. Not bookkeeping:
        # the distance this reference produces varies strongly with crop size, and
        # a score that is not corrected for it measures resolution as much as it
        # measures maize. Cached here so the correction is fitted from the same
        # pass that produced the embeddings rather than replayed afterwards.
        np.save(cache[:-4] + ".sizes.npy", np.asarray(sizes, np.int32))
    return result


def cached_counts(paths, cache_key: str) -> np.ndarray | None:
    """Per-image crop counts from a previous ``embed_crops`` call, or None."""
    digest = hashlib.sha256('\n'.join(paths).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR,
                        f"crop_{cache_key}_{len(paths)}_{digest}.counts.npy")
    return np.load(path) if os.path.exists(path) else None


def cached_sizes(paths, cache_key: str) -> np.ndarray | None:
    """Per-crop short side from a previous ``embed_crops`` call, or None.

    Row-aligned with the cached embeddings, so a size correction can be fitted
    and applied without re-running the detector.
    """
    digest = hashlib.sha256('\n'.join(paths).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR,
                        f"crop_{cache_key}_{len(paths)}_{digest}.sizes.npy")
    return np.load(path) if os.path.exists(path) else None


def detection_rate(pipeline, paths, loader=None) -> float:
    """Fraction of images YOLO finds at least one object in.

    A hard ceiling on end-to-end recall: an undetected foreign object is never
    scored by any threshold, so the flag's real recall is this times the recall
    measured on detected objects.
    """
    loader = loader or (lambda p: Image.open(p).convert("RGB"))
    found = 0
    for path in paths:
        try:
            found += bool(detect_crops(pipeline, loader(path)))
        except Exception:  # noqa: BLE001
            continue
    return found / len(paths) if paths else 0.0


def embed(pipeline, model, transform, images, batch: int = 32) -> np.ndarray:
    """L2-normalised embeddings, so distance reflects direction not activation scale."""
    out = np.zeros((len(images), model.feature_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            t = torch.stack([transform(im) for im in chunk]).to(pipeline.device)
            out[i:i + len(chunk)] = model(t, mode="embedding").cpu().numpy()
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def embed_paths(pipeline, model, transform, paths, loader=None, chunk: int = 256,
                cache_key: str | None = None) -> np.ndarray:
    """Embeds images by path, in chunks so 17,000 files never sit in memory at once.

    Results are cached by a digest of the path list: the evaluation sweep re-reads
    the same tens of thousands of embeddings and a GPU pass over all of them takes
    minutes.
    """
    loader = loader or (lambda p: Image.open(p).convert("RGB"))
    cache = None
    if cache_key:
        os.makedirs(CACHE_DIR, exist_ok=True)
        digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:16]
        cache = os.path.join(CACHE_DIR, f"{cache_key}_{len(paths)}_{digest}.npy")
        if os.path.exists(cache):
            return np.load(cache)

    parts = []
    for i in range(0, len(paths), chunk):
        parts.append(embed(pipeline, model, transform,
                           [loader(p) for p in paths[i:i + chunk]]))
    result = np.concatenate(parts) if parts else np.zeros((0, model.feature_dim), np.float32)
    if cache:
        np.save(cache, result)
    return result


def knn_distance(queries: np.ndarray, reference: np.ndarray, k: int = K,
                 block: int = 512) -> np.ndarray:
    """Mean cosine distance to the k nearest reference embeddings.

    Blocked because the full similarity matrix against a 17,000-image reference is
    otherwise several gigabytes for a large query set.
    """
    out = np.empty(len(queries), dtype=np.float32)
    kk = min(k, reference.shape[0])
    for i in range(0, len(queries), block):
        sims = queries[i:i + block] @ reference.T
        out[i:i + block] = 1.0 - np.partition(sims, -kk, axis=1)[:, -kk:].mean(1)
    return out


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(score(foreign) > score(maize)). 1.0 = perfect separation, 0.5 = useless."""
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = labels.sum(), len(labels) - labels.sum()
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_reference(pipeline, model, transform):
    """The maize-identity reference, in the domain the gate is actually served.

    Membership is deliberately wide -- three variety/quality datasets plus half of
    GrainSpace's damaged kernels -- because the gate's claim is about maize in
    general, and any mode of maize left out becomes a false alarm at serve time.

    Every image goes through YOLO first and contributes its crops, not itself. An
    earlier build embedded the dataset photographs directly and was measured to be
    wrong for the job: the same kernels presented as detector crops sat 18x further
    from that reference, so a threshold guaranteeing a 2% false-flag rate in
    calibration produced an 80% flag rate on real uploads. Matching the domain is
    what makes the guarantee mean anything.

    Returns (embeddings, provenance) -- provenance counts crops, which is what the
    reference holds, and images, which is what was fed in.
    """
    parts, provenance = [], []

    rows = unified_rows("train")
    for source in sorted({r["source"] for r in rows}):
        paths = [r["filepath"] for r in rows if r["source"] == source]
        emb = embed_crops(pipeline, model, transform, paths,
                          cache_key=f"ref_unified_{source}")
        parts.append(emb)
        provenance.append({"source": f"manifest_unified train (source={source})",
                           "images": len(paths), "count": int(emb.shape[0])})

    gs = grainspace_split()["reference"]
    emb = embed_crops(pipeline, model, transform, gs, cache_key="ref_grainspace")
    parts.append(emb)
    provenance.append({"source": "grainspace_maize_m600 val (50% reference split)",
                       "images": len(gs), "count": int(emb.shape[0])})

    return np.concatenate(parts), provenance


def calibration_spec():
    """(cache_key, paths, loader) for the held-out maize the gate is tuned on.

    Never test, never impurities. Declared once and consumed by both the embedding
    and the size readers below, because the two must describe the same crops in
    the same order. Keeping two copies of a path list is not a hypothetical risk:
    a divergent copy of the damaged-class list once silently produced an empty set.
    """
    return [
        ("cal_unified", [r["filepath"] for r in unified_rows("val")], None),
        ("cal_grainspace", grainspace_split()["calibration"], None),
    ]


def evaluation_spec():
    """(group, name, cache_key, paths, loader) for every set the gate is scored on.

    The GrainSet maize controls matter more than they look. GrainSpace kernels fill
    the frame; GrainSet objects sit on a black background. Without same-rig maize a
    high catch rate on impurities could just mean the gate learned to flag black
    backgrounds. Sound and damaged GrainSet maize hold the imaging constant so the
    only thing left varying is whether the object is a kernel.
    """
    damaged = []
    for cls in ("test/1_F&S", "test/2_SD", "test/3_MY",
                "test/4_AP", "test/5_BN", "test/6_HD"):
        damaged.extend(grainset_paths(cls))
    return [
        ("maize", "unified_test", "ev_unified_test",
         [r["filepath"] for r in unified_rows("test")], None),
        ("maize", "grainspace_damaged_held_out", "ev_grainspace",
         grainspace_split()["evaluation"], None),
        ("maize", "grainset_sound_same_rig", "ev_grainset_nor",
         grainset_paths("test/0_NOR"), load_grainset_view),
        ("maize", "grainset_damaged_same_rig", "ev_grainset_damaged",
         damaged, load_grainset_view),
        ("foreign", "grainset_impurities", "ev_grainset_im",
         grainset_paths("train/7_IM") + grainset_paths("test/7_IM"),
         load_grainset_view),
    ]


def calibration_embeddings(pipeline, model, transform):
    """Held-out maize crops the thresholds are set on.

    Returns (embeddings, n_crops). The second value is crops rather than images
    because a threshold is a statement about scored objects, and an image YOLO
    found nothing in contributes none.
    """
    parts = [embed_crops(pipeline, model, transform, paths, loader=loader,
                         cache_key=key)
             for key, paths, loader in calibration_spec()]
    out = np.concatenate(parts)
    return out, int(out.shape[0])


def calibration_sizes() -> np.ndarray | None:
    """Short side of every calibration crop, in the order ``calibration_embeddings``
    returns them. None if any set has not been embedded yet.

    This is what the size correction is fitted on, and fitting it on calibration
    alone is the point: a correction fitted on the evaluation sets would make the
    gate look well-behaved there by construction.
    """
    parts = [cached_sizes(paths, key) for key, paths, _ in calibration_spec()]
    return np.concatenate(parts) if all(p is not None for p in parts) else None


def evaluation_sets(pipeline, model, transform):
    """The sets the gate is scored on, none of which it was built or tuned against.

    Like the reference, these are detector crops. An object YOLO never finds is
    absent here -- see ``detection_rate``, which measures how often that happens
    and is reported alongside the catch rate as a ceiling on it.
    """
    maize, foreign = {}, {}
    for group, name, key, paths, loader in evaluation_spec():
        target = maize if group == "maize" else foreign
        target[name] = embed_crops(pipeline, model, transform, paths,
                                   loader=loader, cache_key=key)
    return maize, foreign


def evaluation_crop_sizes() -> dict[str, np.ndarray]:
    """Short side of every evaluation crop, keyed by set name and row-aligned with
    ``evaluation_sets``. Sets that have not been embedded yet are omitted."""
    out = {}
    for _group, name, key, paths, _loader in evaluation_spec():
        sizes = cached_sizes(paths, key)
        if sizes is not None:
            out[name] = sizes
    return out
