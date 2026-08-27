"""Serves dataset/upload images to the frontend as thumbnails.

The similarity index and the history records both reference images by their
server-side filesystem path. The browser cannot read those, so without this route
the UI can only display path strings — which is why similarity results and history
rows previously showed raw Windows paths instead of pictures.

SECURITY: this endpoint takes a caller-supplied path, so it is a path-traversal
target. Every request is resolved to an absolute real path and checked to be inside
one of a small set of allow-listed roots; anything else is rejected with 403 before
the file is ever opened. Symlinks are resolved before the check, so a link pointing
out of an allowed root cannot escape it either.
"""
from __future__ import annotations

import io
import os

from fastapi import APIRouter, HTTPException, Query, Response
from PIL import Image

from src.utils.config import load_config

router = APIRouter(prefix="/api", tags=["media"])

_cfg = load_config()

# Only these trees may ever be served. Everything else is refused.
_ALLOWED_ROOTS = [
    os.path.realpath(p)
    for p in (
        "datasets",
        _cfg["paths"]["processed"],
        os.path.join(_cfg["paths"]["processed"], "synthetic_defects"),
    )
    if os.path.isdir(p)
]

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Bounded so a caller cannot request a 20000px render and exhaust memory.
_MAX_EDGE = 1024


def _resolve_within_allowed(raw_path: str) -> str:
    """Return an absolute path, or raise if it escapes the allow-listed roots."""
    if not raw_path or "\x00" in raw_path:
        raise HTTPException(status_code=400, detail="Invalid path")

    candidate = os.path.realpath(os.path.normpath(raw_path))

    for root in _ALLOWED_ROOTS:
        # commonpath avoids the classic '/data-evil' matching '/data' prefix bug
        try:
            if os.path.commonpath([candidate, root]) == root:
                break
        except ValueError:  # different drives on Windows
            continue
    else:
        raise HTTPException(status_code=403, detail="Path is outside the served roots")

    if os.path.splitext(candidate)[1].lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=403, detail="Unsupported file type")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


@router.get("/media/image")
async def media_image(
    path: str = Query(..., description="Server-side image path from a history or similarity result"),
    size: int = Query(0, ge=0, le=_MAX_EDGE, description="Longest edge in px; 0 = original"),
):
    """Serve an allow-listed dataset/upload image, optionally downscaled."""
    resolved = _resolve_within_allowed(path)

    try:
        with Image.open(resolved) as img:
            img = img.convert("RGB")
            if size:
                img.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
    except HTTPException:
        raise
    except Exception:
        # Never surface a stack trace or the resolved server path to the client.
        raise HTTPException(status_code=422, detail="Image could not be read")

    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )
