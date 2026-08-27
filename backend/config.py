"""Backend settings. GEMINI_API_KEY is read ONLY from the environment (via .env,
loaded here with python-dotenv) and is never logged, printed, returned in any API
response, or referenced from the frontend. See docs/06_SYNTHETIC_DEFECT_POLICY.md and
PROJECT_MASTER_PROMPT.md's Gemini security rules.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; real key must never be committed (see .gitignore)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data_processed/uploads")
MAX_UPLOAD_MB = 15
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")


def gemini_is_configured() -> bool:
    """Never logs or returns the key itself — only whether one is present."""
    return len(GEMINI_API_KEY) > 0
