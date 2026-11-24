"""Exercise 06 — A ReAct-style agent with tools, built in LangGraph.

Run:  uv run python -m playground.exercises.ex06_react_agent

ReAct = Reason + Act. The model reasons, optionally calls tools, reads the
results, and repeats until it can answer. In exercise 03 we wrote that loop
by hand; here the graph IS the loop:

    START -> agent --(tool_calls?)--> tools
               ^                        |
               └────────────────────────┘
             (no tool calls) -> END

- ``MessagesState`` is a prebuilt state with a ``messages`` list whose reducer
  *appends* new messages instead of replacing the list.
- ``ToolNode`` executes every tool call in the last AIMessage.
- ``tools_condition`` routes to "tools" if there are tool calls, else END.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from playground.exercises.ex03_tools import TOOLS
from playground.models import get_chat_model

SYSTEM = SystemMessage(
    "You are a helpful assistant. Use the tools for arithmetic and weather; "
    "never do math in your head."
)


def build_graph(model: BaseChatModel | None = None, checkpointer=None):
    def agent(state: MessagesState) -> dict:
        llm = (model or get_chat_model()).bind_tools(TOOLS)
        response = llm.invoke([SYSTEM, *state["messages"]])
        return {"messages": [response]}  # appended thanks to the reducer

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" | END
    builder.add_edge("tools", "agent")  # after tools, think again
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def main() -> None:
    question = "Multiply 12 by 7, then add 5. Then tell me the weather in CDMX."
    # stream_mode="updates" yields {node_name: update} after each step,
    # so you can watch the Reason -> Act -> Observe loop happen.
    for step in graph.stream({"messages": [("user", question)]}, stream_mode="updates"):
        for node, update in step.items():
            for msg in update["messages"]:
                print(f"--- {node} ---")
                msg.pretty_print()


# ---------------------------------------------------------------------------
# TRY THIS
# - Compare with the one-liner prebuilt agent:
#       from langchain.agents import create_agent
#       agent = create_agent(get_chat_model(), TOOLS, system_prompt="...")
# - Set ``recursion_limit`` in the config (``graph.invoke(..., {"recursion_limit": 4})``)
#   and ask something that needs many tool calls. What error do you get?
# - Add a tool that raises an exception. ``ToolNode`` turns errors into
#   ToolMessages by default — how does the model react?
# - Open this graph in LangGraph Studio (``uv run langgraph dev``) and step through it.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
