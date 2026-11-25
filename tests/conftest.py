"""Shared test fixtures: fake chat models so tests never need API keys."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolCallingModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts ``bind_tools`` (it just ignores the tools).

    Feed it pre-scripted AIMessages — including ones with ``tool_calls`` — to
    drive agent graphs deterministically.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002 - signature mirrors real models
        return self


def fake_model(*responses: AIMessage | str) -> FakeToolCallingModel:
    return FakeToolCallingModel(messages=iter(responses))


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """Make sure no test accidentally talks to a real provider."""
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "MODEL_PROVIDER",
        "MODEL_NAME",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_REASONING",
        "LANGSMITH_TRACING",
    ):
        monkeypatch.delenv(var, raising=False)
