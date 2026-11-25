"""Exercise 07 — Memory / persistence with a checkpointer + ``thread_id``.

Run:  uv run python -m playground.exercises.ex07_memory

A graph compiled with a *checkpointer* saves its state after every step.
Invoke it again with the same ``thread_id`` and it resumes from that state —
that's how a chatbot "remembers" a conversation. A different ``thread_id``
is a brand-new conversation.

``InMemorySaver`` keeps checkpoints in RAM (lost on exit). For real apps, use
``langgraph-checkpoint-sqlite`` / ``-postgres``.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from playground.models import get_chat_model


def build_graph(model: BaseChatModel | None = None, checkpointer=None):
    def chatbot(state: MessagesState) -> dict:
        llm = model or get_chat_model()
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


# For `langgraph dev`: no checkpointer here — the LangGraph server provides
# its own persistence and manages threads for you in Studio.
graph = build_graph()


def chat(app, thread_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [("user", text)]}, config)
    return result["messages"][-1].text


def main() -> None:
    app = build_graph(checkpointer=InMemorySaver())

    print("alice:", chat(app, "alice", "Hi! My name is Alice and I love climbing."))
    print("alice:", chat(app, "alice", "What's my name and hobby?"))
    print("bob:  ", chat(app, "bob", "What's my name?"))  # different thread: no memory

    # Inspect the saved state for a thread.
    snapshot = app.get_state({"configurable": {"thread_id": "alice"}})
    print(f"\nalice thread has {len(snapshot.values['messages'])} messages")

    # Every step is a checkpoint — you can list them (time travel!).
    history = list(app.get_state_history({"configurable": {"thread_id": "alice"}}))
    print(f"alice thread has {len(history)} checkpoints")


# ---------------------------------------------------------------------------
# TRY THIS
# - Write a small REPL: ``while True: text = input("> ")`` with a fixed thread_id.
# - Long chats grow forever. Trim history before calling the model with
#   ``langchain_core.messages.trim_messages``.
# - Time travel: pick a checkpoint from ``get_state_history`` and invoke with
#   its ``config`` to fork the conversation from that point.
# - Swap InMemorySaver for SqliteSaver (``uv add langgraph-checkpoint-sqlite``)
#   and confirm memory survives a restart.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
