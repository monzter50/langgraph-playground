"""Shared helper that picks a chat model based on environment variables.

Every exercise calls ``get_chat_model()`` instead of hard-coding a provider, so
you can switch between Anthropic and OpenAI without touching exercise code.

Resolution order:
1. ``MODEL_PROVIDER`` (``anthropic`` | ``openai``) if set explicitly.
2. Otherwise Anthropic if ``ANTHROPIC_API_KEY`` is set.
3. Otherwise OpenAI if ``OPENAI_API_KEY`` is set.
4. Otherwise raise a friendly error.

``MODEL_NAME`` overrides the default model for the chosen provider.
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
}


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
        "ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY."
    )


def get_chat_model(**kwargs) -> BaseChatModel:
    """Build a chat model for the configured provider.

    ``init_chat_model`` is LangChain's provider-agnostic factory: it returns a
    ``ChatAnthropic`` or ``ChatOpenAI`` instance that share the same interface
    (``invoke``, ``stream``, ``bind_tools``, ``with_structured_output``...).

    Extra ``kwargs`` (e.g. ``temperature=0``, ``max_tokens=512``) are passed
    straight through to the underlying chat model class.
    """
    provider = resolve_provider()
    model_name = os.getenv("MODEL_NAME") or DEFAULT_MODELS[provider]
    return init_chat_model(model_name, model_provider=provider, **kwargs)
