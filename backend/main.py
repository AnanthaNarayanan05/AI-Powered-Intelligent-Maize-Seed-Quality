"""FastAPI application entrypoint (Phase 18). Run locally:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, gemini_is_configured
from backend.routes import analyze, detect, classify, similarity, history, copilot, media, stats, lot
from src.utils.logging_utils import get_logger

logger = get_logger("backend_main")

app = FastAPI(
    title="Maize Seed Variety Recognition & Detection API",
    description=(
        "See docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md for exactly what this API does "
        "and does not support. Quality/defect predictions are from a SYNTHETIC "
        "demonstration model, not real-world pathology data."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(detect.router)
app.include_router(classify.router)
app.include_router(similarity.router)
app.include_router(history.router)
app.include_router(copilot.router)
app.include_router(media.router)
app.include_router(stats.router)
app.include_router(lot.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": gemini_is_configured(),
    }


@app.get("/api/system-info")
async def system_info():
    """Phase 23 'Model/System Information' page data — never exposes secrets, only
    which capabilities exist, matching docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md."""
    return {
        "supported_capabilities": [
            "maize seed detection & counting (single class: Corn)",
            "individual seed cropping",
            "variety classification across 6 varieties in one unified model, via "
            "contrastive-pretrained cognitive-attention CNN",
            "kernel quality grading (Good/Bad) from real expert-assigned labels, "
            "97.25% test accuracy",
            "out-of-distribution detection on quality grades — extrapolated grades "
            "are marked unverified rather than reported as defects",
            "seed-lot composition reporting: variety composition, off-type rate, "
            "soundness (explicitly NOT a certification)",
            "feature embeddings + FAISS visual similarity search",
            "Grad-CAM explainability (not segmentation)",
            "batch analysis, persistent history",
            "Gemini-powered explanation/copilot over verified results",
        ],
        "explicitly_not_supported": [
            "real-world fungal/insect/disease diagnosis — published kernel-level "
            "detection relies on NIR/hyperspectral bands an RGB camera cannot see",
            "defect type classification — the quality label is binary Good/Bad",
            "defect severity scoring — the model outputs confidence, not severity",
            "pixel-level segmentation or exact defect area — no dataset has masks",
            "foreign-object detection — the detector has one class, Corn",
            "certified seed-lot analysis — certification requires an accredited "
            "laboratory and a prescribed sampling protocol",
        ],
        "gemini_configured": gemini_is_configured(),
    }


@app.get("/api/system-info")
async def system_info():
    """Phase 23 'Model/System Information' page data — never exposes secrets, only
    which capabilities exist, matching docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md."""
    return {
        "supported_capabilities": [
            "maize seed detection & counting (single class: Corn)",
            "individual seed cropping",
            "variety classification (Dataset A: 3 classes; Dataset B: 3 classes) via "
            "contrastive-pretrained, cognitive-attention CNN",
            "feature embeddings + FAISS visual similarity search",
            "Grad-CAM explainability (not segmentation)",
            "synthetic defect-pattern classification (demonstration only, not real pathology)",
            "batch analysis, persistent history",
            "Gemini-powered explanation/copilot over verified results",
        ],
        "explicitly_not_supported": [
            "real-world fungal/insect/disease diagnosis",
            "defect severity scoring",
            "pixel-level segmentation or exact defect area",
            "foreign-object detection",
            "purity / certified seed-lot analysis",
        ],
        "gemini_configured": gemini_is_configured(),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak internal stack traces or secrets to the client (Phase 18 rule).
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
