"""Live system report backing GET /api/system-info.

Nothing here is a stored description of the platform. Every model status, class
list, metric and index size is read from disk at request time, because a
hard-coded copy goes stale the moment a model is retrained or a checkpoint is
moved -- which is exactly how this page came to advertise "Dataset A: 3 classes"
long after a single 6-variety model had replaced it.

What IS declared in this file is the capability wording: which claims the project
is willing to make and which it explicitly refuses to make. That is a statement
about the data, not about the artefacts, so it cannot be derived from them.

Never returns a secret. The Gemini section reports whether a key is configured
and which model name is selected, never the key itself.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone

from backend.config import GEMINI_MODEL, gemini_is_configured
from src.utils.config import load_config

# ---------------------------------------------------------------- model specs
# Only the identity of an artefact is written here (where it lives, what it is,
# whether the serving path uses it). Everything measurable is read from disk.
_MODEL_SPECS = [
    {
        "key": "detection",
        "name": "Seed detection",
        "architecture": "YOLOv8n",
        "role": "Locates and counts individual kernels (single class: Corn).",
        "checkpoint": "detection_corn/weights/best.pt",
        "metrics": "detection",
        "served": True,
    },
    {
        "key": "unified_seed_model",
        "name": "Variety + kernel quality",
        "architecture": "EfficientNet-B0 + cognitive attention, two heads",
        "role": "One pass produces the variety prediction and the Good/Bad grade.",
        "checkpoint": "unified_seed_model_best.pt",
        "metrics": "unified_seed_model",
        "served": True,
    },
    {
        "key": "contrastive_encoder",
        "name": "Contrastive pretraining",
        "architecture": "SimCLR / NT-Xent",
        "role": "Initialises the unified model's backbone; not served on its own.",
        "checkpoint": "contrastive_encoder_unified_unified.pt",
        "metrics": None,
        "served": False,
    },
    {
        "key": "quality_gate",
        "name": "Quality distribution gate",
        "architecture": "kNN cosine distance to the training feature bank",
        "role": (
            "Marks grades on imagery unlike the training data as unverified "
            "instead of reporting them as defects."
        ),
        "checkpoint": "quality_reference.npz",
        "metrics": "quality_gate_comparison",
        "served": True,
    },
    {
        "key": "variety_a_full",
        "name": "Variety model A (ablation)",
        "architecture": "EfficientNet-B0 + cognitive attention, 3 classes",
        "role": (
            "Superseded by the unified model. Kept reachable so the "
            "contrastive/attention ablation stays reproducible."
        ),
        "checkpoint": "variety_a_full_best.pt",
        "metrics": "variety_a_full",
        "served": False,
    },
    {
        "key": "variety_b_full_grouped",
        "name": "Variety model B (ablation)",
        "architecture": "EfficientNet-B0 + cognitive attention, 3 classes",
        "role": (
            "Superseded by the unified model. Group-aware splits; kept for the "
            "same ablation."
        ),
        "checkpoint": "variety_b_full_grouped_best.pt",
        "metrics": "variety_b_full_grouped",
        "served": False,
    },
    {
        "key": "synthetic_defect_classifier",
        "name": "Synthetic defect classifier",
        "architecture": "EfficientNet-B0 + cognitive attention, 4 classes",
        "role": (
            "Trained on defects this project painted. Demonstration only, "
            "disabled in the serving path, never a real-world result."
        ),
        "checkpoint": "synthetic_defect_classifier_best.pt",
        "metrics": "synthetic_defect",
        "served": False,
    },
    {
        "key": "defect_segmenter_synthetic",
        "name": "Defect segmenter (synthetic)",
        "architecture": "EfficientNet-B0 encoder + U-Net decoder",
        "role": (
            "Trained on painted defects. Establishes the resolution floor only; "
            "it is not evidence about real defects and is not served."
        ),
        "checkpoint": "defect_segmenter_synthetic_best.pt",
        "metrics": "defect_segmenter_synthetic",
        "served": False,
    },
]


def _pct(value) -> str:
    return f"{float(value) * 100:.1f}%"


def _headline(key: str, m: dict) -> list[dict]:
    """Real numbers from the model's own evaluation file, in display order."""
    if key == "detection":
        return [
            {"label": "mAP@50", "value": _pct(m["map50"])},
            {"label": "Precision", "value": _pct(m["precision"])},
            {"label": "Recall", "value": _pct(m["recall"])},
        ]
    if key == "unified_seed_model":
        v = m["test_metrics"]["variety"]
        q = m["test_metrics"]["quality"]
        return [
            {"label": "Variety accuracy", "value": _pct(v["accuracy"]), "n": v["n"]},
            {"label": "Variety macro F1", "value": _pct(v["f1_macro"]), "n": v["n"]},
            {"label": "Quality accuracy", "value": _pct(q["accuracy"]), "n": q["n"]},
            {"label": "Quality macro F1", "value": _pct(q["f1_macro"]), "n": q["n"]},
        ]
    if key == "quality_gate":
        g = m["pooled_knn"]
        return [
            {"label": "Distance threshold", "value": f"{g['threshold']:.4f}"},
            {"label": "AUROC (held-out dataset)", "value": f"{g['auroc']['dataset_A']:.3f}"},
            {"label": "Flag rate in-distribution", "value": _pct(g["in_distribution_flag_rate"])},
        ]
    if key == "defect_segmenter_synthetic":
        # test_metrics is keyed per defect channel; report the range rather than a
        # single mean, because the channels differ by more than a factor of two and
        # an average would hide that. Every number here is on painted defects.
        per_channel = m.get("test_metrics") or {}
        dice = {c: v["dice_micro"] for c, v in per_channel.items() if "dice_micro" in v}
        if not dice:
            return []
        worst = min(dice, key=dice.get)
        best = max(dice, key=dice.get)
        return [
            {"label": f"Dice, best channel ({best})", "value": f"{dice[best]:.3f}"},
            {"label": f"Dice, worst channel ({worst})", "value": f"{dice[worst]:.3f}"},
        ]
    test = m.get("test_metrics") or m.get("test") or {}
    if "accuracy" in test:
        out = [{"label": "Test accuracy", "value": _pct(test["accuracy"])}]
        if "f1_macro" in test:
            out.append({"label": "Macro F1", "value": _pct(test["f1_macro"])})
        return out
    return []


def _classes(m: dict) -> dict | None:
    out = {}
    for field in ("variety_classes", "quality_classes", "classes", "channels"):
        if isinstance(m.get(field), list):
            out[field] = m[field]
    return out or None


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _mtime(path: str) -> str | None:
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    except OSError:
        return None
    return stamp.isoformat(timespec="seconds")


def _models(cfg: dict) -> list[dict]:
    ckpt_dir = cfg["paths"]["checkpoints"]
    metrics_dir = cfg["paths"]["metrics"]
    out = []
    for spec in _MODEL_SPECS:
        ckpt = os.path.join(ckpt_dir, spec["checkpoint"])
        present = os.path.exists(ckpt)
        metrics_file = (
            os.path.join(metrics_dir, spec["metrics"] + ".json") if spec["metrics"] else None
        )
        m = _read_json(metrics_file) if metrics_file else None
        # A model is only "ready" if its weights are on this machine. Reporting a
        # trained-elsewhere model as available is how a page starts describing
        # something the server cannot actually do.
        if not present:
            status = "missing"
        elif spec["served"]:
            status = "ready"
        else:
            status = "available"
        out.append(
            {
                "key": spec["key"],
                "name": spec["name"],
                "architecture": spec["architecture"],
                "role": spec["role"],
                "checkpoint": ckpt.replace("\\", "/"),
                "status": status,
                "served": spec["served"],
                "trained_at": _mtime(ckpt) if present else None,
                "metrics": _headline(spec["key"], m) if m else [],
                "metrics_file": metrics_file.replace("\\", "/") if m else None,
                "classes": _classes(m) if m else None,
                "label_provenance": (m or {}).get("label_provenance"),
                # The training script's own caveat, carried through verbatim rather
                # than being re-worded in the UI.
                "note": (m or {}).get("label_note") or (m or {}).get("note"),
            }
        )
    return out


def _similarity_indices(cfg: dict) -> list[dict]:
    """Provenance sidecars written by src.similarity.build_index (Phase 13)."""
    out = []
    for dataset in ("unified", "a", "b"):
        key = f"faiss_index_{dataset}"
        if key not in cfg["paths"]:
            continue
        stem = cfg["paths"][key].rsplit(".index", 1)[0]
        if not os.path.exists(stem + ".faiss"):
            out.append({"dataset": dataset, "status": "not built"})
            continue
        info = _read_json(stem + ".info.json") or {}
        out.append(
            {
                "dataset": dataset,
                "status": "built",
                "encoder": info.get("encoder"),
                "dim": info.get("dim"),
                "gallery_size": info.get("ntotal"),
                "gallery_split": info.get("gallery_split"),
                "built_at": info.get("built_at"),
            }
        )
    return out


def _runtime() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "torch": None,
        "cuda_available": False,
        "device": "cpu",
        "gpu": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        # torch absent or the CUDA driver is unusable -> report CPU, not a 500.
        pass
    return info


def _database(cfg: dict) -> dict:
    path = cfg["paths"]["database"]
    out = {
        "path": path.replace("\\", "/"),
        "engine": "SQLite",
        "status": "not created",
        "tables": {},
    }
    if not os.path.exists(path):
        return out
    out["size_bytes"] = os.path.getsize(path)
    try:
        from sqlalchemy import func, select

        from database.db import session_scope
        from database.models import Analysis, BatchAnalysis, Detection, SimilarityResult

        with session_scope() as db:
            for label, model in (
                ("analyses", Analysis),
                ("detections", Detection),
                ("similarity_results", SimilarityResult),
                ("batches", BatchAnalysis),
            ):
                out["tables"][label] = db.execute(
                    select(func.count()).select_from(model)
                ).scalar_one()
        out["status"] = "connected"
    except Exception:
        out["status"] = "unavailable"
    return out


# ------------------------------------------------------- capability statements
# These are claims about what the DATA can support, so they are written, not
# measured. Each supported line carries its own limit; each unsupported line says
# why the claim is impossible here rather than merely absent.
SUPPORTED = [
    "maize seed detection & counting (single class: Corn)",
    "individual seed cropping",
    "variety classification across 6 varieties in one unified model, via "
    "contrastive-pretrained, cognitive-attention CNN",
    "kernel quality grading (Good/Bad) from real expert-assigned labels",
    "out-of-distribution detection on quality grades — extrapolated grades are "
    "marked unverified rather than reported as defects",
    "seed-lot composition reporting: variety composition, off-type rate, "
    "soundness (explicitly NOT a certification)",
    "visual similarity search over a train-split gallery, in the feature space "
    "of the model that made the prediction — neighbour labels describe the "
    "neighbours, never the query",
    "Grad-CAM explainability, per head and per seed, over the post-attention "
    "feature map (an attention map, not a segmentation mask)",
    "batch analysis and persistent history",
    "Gemini-powered explanation of verified model outputs",
]

NOT_SUPPORTED = [
    "real-world fungal/insect/disease diagnosis — published kernel-level "
    "detection relies on NIR/hyperspectral bands an RGB camera cannot see",
    "defect type classification — the quality label is binary Good/Bad",
    "defect severity scoring — the model outputs confidence, not severity",
    "pixel-level segmentation or exact defect area on real defects — the only "
    "trained segmenter learned defects this project painted",
    "foreign-object detection — the detector has a single class, Corn",
    "certified seed-lot analysis — certification requires an accredited "
    "laboratory and a prescribed sampling protocol",
]

API_VERSION = "0.2.0"


def system_report() -> dict:
    cfg = load_config()
    configured = gemini_is_configured()
    return {
        "backend": {
            "api_version": API_VERSION,
            "python_executable": os.path.basename(sys.executable),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "runtime": _runtime(),
        "models": _models(cfg),
        "similarity_indices": _similarity_indices(cfg),
        "database": _database(cfg),
        # Name of the selected model only. The key itself never leaves the server.
        "gemini": {"configured": configured, "model": GEMINI_MODEL},
        "supported_capabilities": SUPPORTED,
        "explicitly_not_supported": NOT_SUPPORTED,
        # Retained at the top level: existing clients read this flat key.
        "gemini_configured": configured,
    }
