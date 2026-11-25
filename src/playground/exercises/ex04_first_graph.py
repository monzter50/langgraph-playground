"""Exercise 04 — Your first LangGraph ``StateGraph``.

Run:  uv run python -m playground.exercises.ex04_first_graph

Core ideas:
- **State**: a TypedDict shared by every node. Each node receives the current
  state and returns a *partial* update (a dict with only the keys it changes).
- **Nodes**: plain functions ``state -> dict``.
- **Edges**: fixed transitions between nodes. ``START`` and ``END`` are
  special virtual nodes.

    START -> clean_text -> count_words -> summarize -> END
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from playground.models import get_chat_model


class State(TypedDict, total=False):
    text: str  # input
    word_count: int  # filled by count_words
    summary: str  # filled by summarize


def build_graph(model: BaseChatModel | None = None):
    """Build and compile the graph. ``model`` is injectable for tests."""

    # Pure-Python nodes: no LLM needed. Great for learning how state flows.
    def clean_text(state: State) -> dict:
        return {"text": " ".join(state["text"].split())}

    def count_words(state: State) -> dict:
        return {"word_count": len(state["text"].split())}

    # An LLM node. The model is created lazily so importing this module never
    # requires an API key (important for `langgraph dev` and tests).
    def summarize(state: State) -> dict:
        llm = model or get_chat_model()
        msg = llm.invoke(
            f"Summarize in one short sentence ({state['word_count']} words given):\n\n"
            f"{state['text']}"
        )
        return {"summary": msg.text}

    builder = StateGraph(State)
    builder.add_node("clean_text", clean_text)
    builder.add_node("count_words", count_words)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "clean_text")
    builder.add_edge("clean_text", "count_words")
    builder.add_edge("count_words", "summarize")
    builder.add_edge("summarize", END)

    # compile() validates the graph and returns a Runnable (invoke/stream...).
    return builder.compile()


# Exported for langgraph.json / LangGraph Studio.
graph = build_graph()


def main() -> None:
    print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live
    result = graph.invoke(
        {
            "text": "LangGraph   models agent workflows as graphs.\n"
            "Nodes do work,   edges decide what runs next, and a shared state "
            "object carries data between them."
        }
    )
    print("Final state:", result)


# ---------------------------------------------------------------------------
# TRY THIS
# - Add a ``keywords: list[str]`` key and a node that extracts them.
# - Make ``count_words`` and a new ``detect_language`` node run in PARALLEL:
#   add two edges out of ``clean_text`` and join them before ``summarize``.
# - Use ``Annotated[list[str], operator.add]`` as a key type: that's a
#   *reducer* — updates get appended instead of overwritten.
# - Run ``graph.stream(inputs, stream_mode="updates")`` to watch each node.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
