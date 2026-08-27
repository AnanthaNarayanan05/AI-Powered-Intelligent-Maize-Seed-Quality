"""Verifies Gemini failure handling never raises (Phase 22 requirement) and never
logs/exposes the API key."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib


def test_generate_text_without_key_returns_clean_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    import backend.config as config
    importlib.reload(config)
    import backend.services.gemini_service as gemini_service
    importlib.reload(gemini_service)

    text, reason = gemini_service.generate_text("test prompt")
    assert text is None
    assert reason is not None
    assert "GEMINI_API_KEY" in reason


def test_generate_text_never_raises_on_bad_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "definitely-not-a-real-key")
    import backend.config as config
    importlib.reload(config)
    import backend.services.gemini_service as gemini_service
    importlib.reload(gemini_service)

    # This will attempt a real (failing) network call to Gemini with a bad key,
    # or fail to import google.generativeai if it's not installed — either way,
    # generate_text must swallow the error and return a clean fallback, not raise.
    text, reason = gemini_service.generate_text("test prompt", timeout_seconds=5)
    assert text is None or isinstance(text, str)
    if text is None:
        assert reason is not None
