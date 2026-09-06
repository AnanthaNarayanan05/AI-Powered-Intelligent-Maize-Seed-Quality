"""Verifies Gemini failure handling never raises (Phase 22 requirement), never
logs/exposes the API key, and that the Phase 18 google-genai migration kept the
contract the rest of the backend depends on.

The migration is the reason most of these tests exist. Swapping SDKs changed the
call shape underneath `generate_text` while its signature stayed identical, so
nothing above it would have failed loudly if a detail were mistranslated -- the
worst outcomes (a 20-second budget silently becoming 20 milliseconds, or the
limitation ground rules quietly not being sent) both look exactly like "the AI is
having a bad day" from the outside. These pin the details down instead.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib

import pytest


def _reload_with_key(monkeypatch, key: str):
    """Reload config + service so module-level GEMINI_API_KEY picks up `key`."""
    monkeypatch.setenv("GEMINI_API_KEY", key)
    import backend.config as config
    importlib.reload(config)
    import backend.services.gemini_service as gemini_service
    importlib.reload(gemini_service)
    return gemini_service


class _StubResponse:
    def __init__(self, text=None, prompt_feedback=None, candidates=None):
        self.text = text
        self.prompt_feedback = prompt_feedback
        self.candidates = candidates or []


class _RecordingModels:
    """Stands in for client.models, capturing exactly what the SDK was handed."""

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._raises is not None:
            raise self._raises
        return self._response


class _StubClient:
    def __init__(self, models):
        self.models = models


def _install_stub_client(monkeypatch, gemini_service, models):
    monkeypatch.setattr(gemini_service, "_client", _StubClient(models))


def test_generate_text_without_key_returns_clean_fallback(monkeypatch):
    gemini_service = _reload_with_key(monkeypatch, "")

    text, reason = gemini_service.generate_text("test prompt")
    assert text is None
    assert reason is not None
    assert "GEMINI_API_KEY" in reason


def test_generate_text_never_raises_on_bad_key(monkeypatch):
    gemini_service = _reload_with_key(monkeypatch, "definitely-not-a-real-key")

    # Attempts a real (failing) network call to Gemini with a bad key, or fails to
    # import google.genai if it is not installed -- either way, generate_text must
    # swallow the error and return a clean fallback, not raise.
    text, reason = gemini_service.generate_text("test prompt", timeout_seconds=5)
    assert text is None or isinstance(text, str)
    if text is None:
        assert reason is not None


def test_sdk_exception_becomes_a_fallback_not_a_crash(monkeypatch):
    gemini_service = _reload_with_key(monkeypatch, "some-key")
    _install_stub_client(monkeypatch, gemini_service,
                         _RecordingModels(raises=RuntimeError("upstream exploded")))

    text, reason = gemini_service.generate_text("test prompt")
    assert text is None
    assert "temporarily unavailable" in reason


def test_failure_reason_never_carries_the_key_or_the_prompt(monkeypatch):
    """A fallback string is rendered in the UI, so it must leak neither secret."""
    gemini_service = _reload_with_key(monkeypatch, "sk-super-secret-value")
    _install_stub_client(
        monkeypatch, gemini_service,
        _RecordingModels(raises=RuntimeError("auth failed for sk-super-secret-value")),
    )

    text, reason = gemini_service.generate_text("a user question about seed 3")
    assert text is None
    assert "sk-super-secret-value" not in reason
    assert "seed 3" not in reason


def test_timeout_is_converted_to_milliseconds(monkeypatch):
    """The old SDK took seconds; HttpOptions.timeout is milliseconds.

    Passing the number straight through would have made every call time out
    ~1000x early, which surfaces as an outage rather than as a bug.
    """
    gemini_service = _reload_with_key(monkeypatch, "some-key")
    models = _RecordingModels(response=_StubResponse(text="hello"))
    _install_stub_client(monkeypatch, gemini_service, models)

    gemini_service.generate_text("test prompt", timeout_seconds=20)

    assert models.calls[0]["config"].http_options.timeout == 20_000


def test_limitation_context_is_sent_with_every_call(monkeypatch):
    """The ground rules moved from a model object to the per-request config.

    They are what stops Gemini claiming segmentation, defect area, severity or a
    disease diagnosis, so a call that forgot to attach them would be a fabrication
    risk that still returned confident-looking text.
    """
    gemini_service = _reload_with_key(monkeypatch, "some-key")
    models = _RecordingModels(response=_StubResponse(text="hello"))
    _install_stub_client(monkeypatch, gemini_service, models)

    gemini_service.generate_text("first")
    gemini_service.generate_text("second")

    assert len(models.calls) == 2
    for call in models.calls:
        assert call["config"].system_instruction == gemini_service.LIMITATION_CONTEXT


def test_automatic_function_calling_is_disabled(monkeypatch):
    """No tools are ever passed, so the SDK's AFC loop is dead weight."""
    gemini_service = _reload_with_key(monkeypatch, "some-key")
    models = _RecordingModels(response=_StubResponse(text="hello"))
    _install_stub_client(monkeypatch, gemini_service, models)

    gemini_service.generate_text("test prompt")

    assert models.calls[0]["config"].automatic_function_calling.disable is True


def test_configured_model_is_the_one_requested(monkeypatch):
    gemini_service = _reload_with_key(monkeypatch, "some-key")
    models = _RecordingModels(response=_StubResponse(text="hello"))
    _install_stub_client(monkeypatch, gemini_service, models)

    gemini_service.generate_text("test prompt")

    from backend.config import GEMINI_MODEL
    assert models.calls[0]["model"] == GEMINI_MODEL


def test_empty_model_name_falls_back_to_the_default(monkeypatch):
    """Copying .env.example sets GEMINI_MODEL to "", which is not a model name."""
    monkeypatch.setenv("GEMINI_MODEL", "")
    import backend.config as config
    importlib.reload(config)
    assert config.GEMINI_MODEL == "gemini-3.6-flash"
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    importlib.reload(config)


@pytest.mark.parametrize(
    "response, expected",
    [
        (_StubResponse(text=None), "empty response"),
        (_StubResponse(text=""), "empty response"),
    ],
)
def test_textless_response_is_reported_not_faked(monkeypatch, response, expected):
    """No text means no answer. Never a placeholder the UI would show as Gemini's."""
    gemini_service = _reload_with_key(monkeypatch, "some-key")
    _install_stub_client(monkeypatch, gemini_service, _RecordingModels(response=response))

    text, reason = gemini_service.generate_text("test prompt")
    assert text is None
    assert expected in reason


def test_blocked_prompt_is_distinguished_from_an_empty_one(monkeypatch):
    class _Feedback:
        block_reason = "SAFETY"

    gemini_service = _reload_with_key(monkeypatch, "some-key")
    _install_stub_client(
        monkeypatch, gemini_service,
        _RecordingModels(response=_StubResponse(text=None, prompt_feedback=_Feedback())),
    )

    text, reason = gemini_service.generate_text("test prompt")
    assert text is None
    assert "declined" in reason


def test_truncated_response_is_distinguished_from_an_empty_one(monkeypatch):
    class _Candidate:
        finish_reason = "MAX_TOKENS"

    gemini_service = _reload_with_key(monkeypatch, "some-key")
    _install_stub_client(
        monkeypatch, gemini_service,
        _RecordingModels(response=_StubResponse(text=None, candidates=[_Candidate()])),
    )

    text, reason = gemini_service.generate_text("test prompt")
    assert text is None
    assert "cut off" in reason


def test_client_is_built_once_and_reused(monkeypatch):
    """A client owns a connection pool; rebuilding it per request throws it away."""
    gemini_service = _reload_with_key(monkeypatch, "some-key")

    built = []

    class _FakeGenaiClient:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.models = _RecordingModels(response=_StubResponse(text="hello"))

    # _get_client does `from google import genai` then `genai.Client(...)`, so
    # patching the attribute on the real module intercepts the construction.
    import google.genai as genai_mod
    monkeypatch.setattr(gemini_service, "_client", None)
    monkeypatch.setattr(genai_mod, "Client", _FakeGenaiClient)

    first = gemini_service._get_client()
    second = gemini_service._get_client()

    assert first is second
    assert len(built) == 1


def test_the_retired_sdk_is_gone(monkeypatch):
    """google-generativeai is retired. If it were still installed, an accidental
    import of it would keep working and hide an incomplete migration."""
    import importlib.util
    assert importlib.util.find_spec("google.generativeai") is None
