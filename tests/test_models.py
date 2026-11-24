import pytest

from playground import models


def test_missing_keys_raises_friendly_error():
    with pytest.raises(models.MissingAPIKeyError):
        models.resolve_provider()


def test_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    assert models.resolve_provider() == "anthropic"


def test_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    assert models.resolve_provider() == "openai"


def test_explicit_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "OpenAI")
    assert models.resolve_provider() == "openai"
    monkeypatch.setenv("MODEL_PROVIDER", "nope")
    with pytest.raises(ValueError):
        models.resolve_provider()


def test_get_chat_model_builds_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    model = models.get_chat_model()
    assert type(model).__name__ == "ChatAnthropic"
