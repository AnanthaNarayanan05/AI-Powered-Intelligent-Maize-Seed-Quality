"""FastAPI application entrypoint (Phase 18). Run locally:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, gemini_is_configured
from backend.routes import analyze, detect, classify, similarity, history, copilot, media, stats, lot
from backend.services.system_service import API_VERSION, system_report
from src.utils.logging_utils import get_logger

logger = get_logger("backend_main")

app = FastAPI(
    title="Maize Seed Variety Recognition & Detection API",
    description=(
        "See docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md, or GET /api/system-info for a "
        "live report of the models actually loaded on this server, for exactly what "
        "this API does and does not support. Quality grades come from real "
        "expert-assigned Good/Bad labels; defect segmentation and severity are NOT "
        "served, because the only trained segmenter learned painted defects."
    ),
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Response headers a browser is allowed to READ cross-origin. Without this the
    # Grad-CAM endpoint's "this is not a segmentation mask" note and its which-model
    # /which-head fields are set but invisible to the frontend, which then has to
    # guess what the image it is displaying actually explains.
    expose_headers=[
        "X-Explainability-Note", "X-Gradcam-Model", "X-Gradcam-Head",
        "X-Gradcam-Target-Layer", "X-Gradcam-Predicted-Class", "X-Gradcam-Region",
    ],
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
    """Live report of what this server can actually do right now.

    Read from disk on every request -- checkpoints, evaluation JSON, FAISS index
    sidecars, the database, the CUDA runtime -- so a retrained or missing model
    changes the page instead of the page describing a model that is gone. Never
    exposes secrets: the Gemini section reports configured/not and the model name.
    """
    return system_report()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak internal stack traces or secrets to the client (Phase 18 rule).
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
