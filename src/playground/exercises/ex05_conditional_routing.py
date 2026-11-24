"""Exercise 05 — Conditional edges / routing.

Run:  uv run python -m playground.exercises.ex05_conditional_routing

``add_conditional_edges(source, router_fn, path_map)`` lets the graph decide
at runtime which node runs next. The router reads the state and returns a
key; ``path_map`` maps that key to a node name.

                 ┌─> math_expert ────┐
    START -> classify                ├─> END
                 ├─> code_expert ────┤
                 └─> general_expert ─┘
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from playground.models import get_chat_model

Category = Literal["math", "code", "general"]


class State(TypedDict, total=False):
    question: str
    category: Category
    answer: str


def build_graph(model: BaseChatModel | None = None):
    def llm() -> BaseChatModel:
        return model or get_chat_model()

    def classify(state: State) -> dict:
        msg = llm().invoke(
            "Classify the question as exactly one word: math, code or general.\n"
            f"Question: {state['question']}\nCategory:"
        )
        raw = msg.text.strip().lower()
        category = next((c for c in ("math", "code") if c in raw), "general")
        return {"category": category}

    def make_expert(persona: str):
        def expert(state: State) -> dict:
            msg = llm().invoke(f"You are {persona}. Answer briefly: {state['question']}")
            return {"answer": msg.text}

        return expert

    # The router: a plain function of the state that returns a branch key.
    def route(state: State) -> Category:
        return state["category"]

    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("math_expert", make_expert("a patient math tutor"))
    builder.add_node("code_expert", make_expert("a senior Python engineer"))
    builder.add_node("general_expert", make_expert("a friendly generalist"))

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {"math": "math_expert", "code": "code_expert", "general": "general_expert"},
    )
    for node in ("math_expert", "code_expert", "general_expert"):
        builder.add_edge(node, END)
    return builder.compile()


graph = build_graph()


def main() -> None:
    print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live
    for q in [
        "What is the derivative of x^3?",
        "How do I reverse a list in Python?",
        "Why is the sky blue?",
    ]:
        result = graph.invoke({"question": q})
        print(f"\n[{result['category']}] {q}\n -> {result['answer']}")


# ---------------------------------------------------------------------------
# TRY THIS
# - Replace the string-matching classifier with ``with_structured_output``
#   (exercise 02) returning ``category: Category``.
# - Add a "refuse" branch for off-topic questions that goes straight to END.
# - Instead of a router function, return ``Command(goto="math_expert",
#   update={...})`` from ``classify`` (``from langgraph.types import Command``).
# - Add a loop: a ``review`` node that sends the answer back to the expert if
#   it is too long (conditional edge pointing backwards).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
