"""Exercise 03 — Tools / tool calling.

Run:  uv run python -m playground.exercises.ex03_tools

A *tool* is a Python function plus a schema the model can read. The model does
NOT run the tool: it returns ``tool_calls`` (name + args) and YOU execute them,
then send the results back as ``ToolMessage``s so the model can answer.

This is the manual loop. Exercise 06 lets LangGraph run the loop for you.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from playground.models import get_chat_model


# The @tool decorator builds the schema from the type hints + docstring.
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (fake data for practice)."""
    fake = {"madrid": "sunny, 24°C", "london": "rainy, 12°C", "cdmx": "cloudy, 19°C"}
    return fake.get(city.lower(), f"no data for {city}")


TOOLS = [add, multiply, get_weather]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_with_tools(question: str, model: BaseChatModel | None = None) -> list[BaseMessage]:
    """One round of tool calling: ask -> execute tool calls -> final answer."""
    model = model or get_chat_model()
    # bind_tools attaches the tool schemas to every request.
    llm_with_tools = model.bind_tools(TOOLS)

    messages: list[BaseMessage] = [HumanMessage(question)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Each tool call has an id that the ToolMessage must reference.
    for call in ai_msg.tool_calls:
        print(f"  -> model wants {call['name']}({call['args']})")
        result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    if ai_msg.tool_calls:
        # Send the tool results back so the model can write the final answer.
        messages.append(llm_with_tools.invoke(messages))
    return messages


def main() -> None:
    # Inspect what the model "sees" for a tool:
    print("Schema for multiply:", multiply.args_schema.model_json_schema(), "\n")

    messages = run_with_tools("What's (3.5 + 4) * 2? Also, how's the weather in Madrid?")
    print("\nFinal answer:", messages[-1].text)


# ---------------------------------------------------------------------------
# TRY THIS
# - Call a tool directly: ``add.invoke({"a": 1, "b": 2})``.
# - Ask a question that needs two *sequential* calls (multiply, then add the
#   result). One round is not enough — turn ``run_with_tools`` into a loop
#   that continues while ``ai_msg.tool_calls`` is non-empty.
# - Add your own tool (e.g. ``search_notes(query: str)``) and see if the model
#   picks it.
# - Try ``model.bind_tools(TOOLS, tool_choice="get_weather")`` to force a tool.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
