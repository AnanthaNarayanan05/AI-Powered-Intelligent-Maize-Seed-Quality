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
from pathlib import Path

from backend.config import GEMINI_MODEL, gemini_is_configured
from src.registry import get_registry
from src.utils.config import PROJECT_ROOT, load_config

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


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# How trustworthy the reported numbers are, in the words a reader needs rather
# than the registry's one-word state. "unrecorded" deliberately does not say the
# metrics match -- it says nobody wrote down enough to tell.
_BINDING_NOTE = {
    "verified": None,
    "stale": (
        "These metrics were produced by a different build of this checkpoint. "
        "The model was retrained without being re-evaluated, so they do not "
        "describe the weights currently on disk."
    ),
    "unrecorded": (
        "This evaluation predates checkpoint fingerprinting, so it cannot be "
        "confirmed to describe the checkpoint currently on disk."
    ),
    "missing": None,
}


def _display_path(path: str | None) -> str | None:
    """Project-relative, forward-slashed. The registry works in absolute paths so
    it can stat them; the page has no use for this machine's directory layout."""
    if not path:
        return path
    try:
        path = str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        pass
    return path.replace("\\", "/")


def _models() -> list[dict]:
    """One row per registry entry, with metrics formatted for display.

    This function used to carry its own table of model names, architectures,
    checkpoint paths and served flags. That was a second answer to "which model
    does the platform serve", kept in step with configs/model_registry.yaml by
    hand -- which is to say, not kept in step. Identity is now read from the
    registry and only the presentation happens here.

    refresh=True re-probes the filesystem on every request, which is the whole
    point of this endpoint; hashes and class lists are cached against file
    identity, so an unchanged checkpoint costs a stat() rather than a re-read.
    """
    out = []
    for record in get_registry(refresh=True).all():
        row = record.as_dict()
        row["checkpoint"] = _display_path(row["checkpoint"])
        row["metrics_file"] = _display_path(row["metrics_file"])
        # the raw metrics blob is not shipped; the UI gets the headline figures
        row["metrics"] = _headline(record.key, record.metrics) if record.metrics else []
        row["metrics_note"] = _BINDING_NOTE.get(record.metrics_binding)
        out.append(row)
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
        "models": _models(),
        "similarity_indices": _similarity_indices(cfg),
        "database": _database(cfg),
        # Name of the selected model only. The key itself never leaves the server.
        "gemini": {"configured": configured, "model": GEMINI_MODEL},
        "supported_capabilities": SUPPORTED,
        "explicitly_not_supported": NOT_SUPPORTED,
        # Retained at the top level: existing clients read this flat key.
        "gemini_configured": configured,
    }
