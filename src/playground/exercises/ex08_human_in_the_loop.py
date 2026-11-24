"""Exercise 08 — Human-in-the-loop with ``interrupt``.

Run:  uv run python -m playground.exercises.ex08_human_in_the_loop

``interrupt(value)`` pauses the graph *inside* a node and surfaces ``value``
to the caller. The state is saved by the checkpointer (required!). Later you
resume with ``Command(resume=answer)``: the node re-runs from the start and
this time ``interrupt(...)`` returns ``answer``.

    START -> write_draft -> human_review --approve--> publish -> END
                 ^               |
                 └───revise──────┘
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from playground.models import get_chat_model


class State(TypedDict, total=False):
    topic: str
    draft: str
    feedback: str
    published: bool


def build_graph(model: BaseChatModel | None = None, checkpointer=None):
    def write_draft(state: State) -> dict:
        llm = model or get_chat_model()
        prompt = f"Write a one-sentence tweet about {state['topic']}."
        if state.get("feedback"):
            prompt += (
                f"\nPrevious draft: {state['draft']}"
                f"\nRevise it using this feedback: {state['feedback']}"
            )
        return {"draft": llm.invoke(prompt).text}

    def human_review(state: State) -> Command:
        # Execution stops here until someone resumes with Command(resume=...).
        # NOTE: code before interrupt() runs again on resume — keep it idempotent.
        decision = interrupt(
            {
                "question": "Approve this draft? Reply 'yes' or give feedback.",
                "draft": state["draft"],
            }
        )
        if str(decision).strip().lower() in {"yes", "y", "ok", "approve"}:
            return Command(goto="publish")
        # Command can update state AND choose the next node at the same time.
        return Command(goto="write_draft", update={"feedback": str(decision)})

    def publish(state: State) -> dict:
        print(f"📣 Published: {state['draft']}")
        return {"published": True}

    builder = StateGraph(State)
    builder.add_node("write_draft", write_draft)
    # ``destinations`` only helps graph drawing / Studio show the Command edges.
    builder.add_node("human_review", human_review, destinations=("publish", "write_draft"))
    builder.add_node("publish", publish)
    builder.add_edge(START, "write_draft")
    builder.add_edge("write_draft", "human_review")
    builder.add_edge("publish", END)
    return builder.compile(checkpointer=checkpointer)


# For `langgraph dev` (the server supplies the checkpointer).
graph = build_graph()


def main() -> None:
    app = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "tweet-1"}}

    result = app.invoke({"topic": "learning LangGraph"}, config)
    # When the graph is paused, the result contains "__interrupt__".
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\nDraft: {payload['draft']}")
        answer = input(f"{payload['question']}\n> ")
        result = app.invoke(Command(resume=answer), config)

    print("\nFinal state:", result)


# ---------------------------------------------------------------------------
# TRY THIS
# - While paused, inspect ``app.get_state(config).next`` — which node is waiting?
# - Edit the state before resuming: ``app.update_state(config, {"draft": "..."})``.
# - Add an approval step to the ReAct agent (exercise 06): interrupt inside a
#   tool before it does something "dangerous" (e.g. ``send_email``).
# - Use ``interrupt_before=["publish"]`` in ``compile()`` instead of calling
#   interrupt() — a static breakpoint, handy for debugging.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
