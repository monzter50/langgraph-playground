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


def test_ollama_is_never_auto_detected(monkeypatch):
    # No keys and no MODEL_PROVIDER: still the friendly error, not a silent local fallback.
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    with pytest.raises(models.MissingAPIKeyError):
        models.resolve_provider()


def test_explicit_provider_wins_over_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    assert models.resolve_provider() == "ollama"


def test_get_chat_model_builds_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    assert type(models.get_chat_model()).__name__ == "ChatOpenAI"


def test_get_chat_model_passes_temperature(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert models.get_chat_model(temperature=0).temperature == 0


def test_get_chat_model_builds_ollama_defaults(monkeypatch):
    # Constructing ChatOllama doesn't contact the server, so no Ollama is needed.
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_NAME", "claude-sonnet-5-5")  # cloud name must not leak in
    model = models.get_chat_model()
    assert type(model).__name__ == "ChatOllama"
    assert model.model == "qwen3:8b"
    assert model.base_url == "http://localhost:11434"
    assert model.reasoning is False
    assert model.temperature is None


def test_get_chat_model_ollama_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    monkeypatch.setenv("OLLAMA_REASONING", "true")
    model = models.get_chat_model(temperature=0.2)
    assert model.model == "llama3.1:8b"
    assert model.base_url == "http://gpu-box:11434"
    assert model.reasoning is True
    assert model.temperature == 0.2


def test_ollama_reasoning_kwarg_beats_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_REASONING", "true")
    assert models.get_chat_model(reasoning=False).reasoning is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("off", False),
        ("0", False),
        ("TRUE", True),
        ("1", True),
        ("high", "high"),
        ("low", "low"),
    ],
)
def test_resolve_ollama_reasoning(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("OLLAMA_REASONING", raw)
    assert models.resolve_ollama_reasoning() == expected


def test_resolve_ollama_reasoning_rejects_garbage(monkeypatch):
    monkeypatch.setenv("OLLAMA_REASONING", "maybe")
    with pytest.raises(ValueError):
        models.resolve_ollama_reasoning()
