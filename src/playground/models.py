"""Shared helper that picks a chat model based on environment variables.

Every exercise calls ``get_chat_model()`` instead of hard-coding a provider, so
you can switch between Anthropic, OpenAI and a local Ollama model without
touching exercise code.

Resolution order:
1. ``MODEL_PROVIDER`` (``anthropic`` | ``openai`` | ``ollama``) if set explicitly.
2. Otherwise Anthropic if ``ANTHROPIC_API_KEY`` is set.
3. Otherwise OpenAI if ``OPENAI_API_KEY`` is set.
4. Otherwise raise a friendly error.

Ollama is never auto-detected: it needs no key, so opt in with
``MODEL_PROVIDER=ollama``.

``MODEL_NAME`` overrides the default model for Anthropic / OpenAI. Ollama uses
its own variables so a cloud ``MODEL_NAME`` in ``.env`` doesn't leak into it:

- ``OLLAMA_MODEL``     (default ``qwen3:8b``)
- ``OLLAMA_BASE_URL``  (default ``http://localhost:11434``)
- ``OLLAMA_REASONING`` (default off): ``true`` keeps the model's thinking but moves
  it to ``message.additional_kwargs["reasoning_content"]``; ``low`` / ``medium`` /
  ``high`` set an effort level on models that support it. Either way
  ``message.content`` stays a clean final answer.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Load variables from a local .env file (if present). Safe to call many times.
load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5-mini",
    "ollama": "qwen3:8b",
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}
_EFFORTS = {"low", "medium", "high"}


class MissingAPIKeyError(RuntimeError):
    """Raised when no provider API key is configured."""


def resolve_provider() -> str:
    """Return the provider name to use, based on env vars."""
    explicit = os.getenv("MODEL_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown MODEL_PROVIDER={explicit!r}; expected one of {sorted(DEFAULT_MODELS)}"
            )
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise MissingAPIKeyError(
        "No model API key found. Copy .env.example to .env and set "
        "ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY, "
        "or set MODEL_PROVIDER=ollama to use a local model."
    )


def resolve_ollama_reasoning() -> bool | str:
    """Parse ``OLLAMA_REASONING`` into ChatOllama's ``reasoning`` option.

    Default is ``False`` (thinking disabled), not ``None``: with ``None``, thinking
    models such as qwen3 put ``<think>...</think>`` straight into the answer text.
    """
    raw = os.getenv("OLLAMA_REASONING", "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    if raw in _EFFORTS:
        return raw
    raise ValueError(
        f"Unknown OLLAMA_REASONING={raw!r}; expected true/false or one of {sorted(_EFFORTS)}"
    )


def get_chat_model(**kwargs) -> BaseChatModel:
    """Build a chat model for the configured provider.

    For Anthropic / OpenAI, ``init_chat_model`` (LangChain's provider-agnostic
    factory) returns a ``ChatAnthropic`` or ``ChatOpenAI``. For Ollama we build a
    ``ChatOllama`` directly. All of them share the same interface (``invoke``,
    ``stream``, ``bind_tools``, ``with_structured_output``...).

    Extra ``kwargs`` (e.g. ``temperature=0``, ``max_tokens=512``) are passed
    straight through to the underlying chat model class, for every provider.
    """
    provider = resolve_provider()
    if provider == "ollama":
        from langchain_ollama import ChatOllama  # imported lazily: only needed locally

        kwargs.setdefault("reasoning", resolve_ollama_reasoning())
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL") or DEFAULT_MODELS["ollama"],
            base_url=os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
            **kwargs,
        )
    model_name = os.getenv("MODEL_NAME") or DEFAULT_MODELS[provider]
    return init_chat_model(model_name, model_provider=provider, **kwargs)
