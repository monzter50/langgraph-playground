"""Run every exercise against fake models — no API keys, no network."""

from __future__ import annotations

import importlib

import pytest
from conftest import fake_model
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

EXERCISES = [
    "ex01_lcel_chain",
    "ex02_structured_output",
    "ex03_tools",
    "ex04_first_graph",
    "ex05_conditional_routing",
    "ex06_react_agent",
    "ex07_memory",
    "ex08_human_in_the_loop",
]


@pytest.mark.parametrize("name", EXERCISES)
def test_exercise_imports_without_api_keys(name):
    importlib.import_module(f"playground.exercises.{name}")


def test_ex01_chain():
    from playground.exercises.ex01_lcel_chain import build_chain

    chain = build_chain(fake_model("Embeddings are number lists."))
    assert chain.invoke({"topic": "x", "audience": "y"}) == "Embeddings are number lists."


def test_ex02_schema():
    from playground.exercises.ex02_structured_output import Recipe

    r = Recipe(name="Soup", servings=4, ingredients=["tomato"], vegetarian=True)
    assert r.servings == 4
    with pytest.raises(ValueError):
        Recipe(name="Soup", servings=0, ingredients=[], vegetarian=True)


def test_ex03_tools_loop():
    from playground.exercises.ex03_tools import add, run_with_tools

    assert add.invoke({"a": 2, "b": 3}) == 5
    model = fake_model(
        AIMessage("", tool_calls=[{"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c1"}]),
        AIMessage("It is 12."),
    )
    messages = run_with_tools("3*4?", model)
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.content == "12.0"
    assert messages[-1].text == "It is 12."


def test_ex04_first_graph():
    from playground.exercises.ex04_first_graph import build_graph

    graph = build_graph(fake_model("A summary."))
    result = graph.invoke({"text": "  hello    graph   world "})
    assert result == {"text": "hello graph world", "word_count": 3, "summary": "A summary."}


@pytest.mark.parametrize(
    ("label", "category"), [("math", "math"), ("Code.", "code"), ("history", "general")]
)
def test_ex05_routing(label, category):
    from playground.exercises.ex05_conditional_routing import build_graph

    graph = build_graph(fake_model(label, "expert answer"))
    updates = list(graph.stream({"question": "q"}, stream_mode="updates"))
    assert updates[0] == {"classify": {"category": category}}
    assert updates[1] == {f"{category}_expert": {"answer": "expert answer"}}


def test_ex06_react_agent_runs_tools():
    from playground.exercises.ex06_react_agent import build_graph

    model = fake_model(
        AIMessage("", tool_calls=[{"name": "add", "args": {"a": 2, "b": 2}, "id": "c1"}]),
        AIMessage("", tool_calls=[{"name": "get_weather", "args": {"city": "London"}, "id": "c2"}]),
        AIMessage("4, and London is rainy."),
    )
    result = build_graph(model).invoke({"messages": [("user", "2+2 and London weather?")]})
    tool_outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_outputs == ["4.0", "rainy, 12°C"]
    assert result["messages"][-1].text == "4, and London is rainy."


def test_ex07_memory_is_per_thread():
    from playground.exercises.ex07_memory import build_graph, chat

    app = build_graph(fake_model("hi alice", "you are alice", "who?"), checkpointer=InMemorySaver())
    chat(app, "alice", "I'm Alice")
    chat(app, "alice", "who am I?")
    chat(app, "bob", "who am I?")

    def count(thread):
        return len(app.get_state({"configurable": {"thread_id": thread}}).values["messages"])

    assert count("alice") == 4
    assert count("bob") == 2


def test_ex08_interrupt_and_resume():
    from playground.exercises.ex08_human_in_the_loop import build_graph

    app = build_graph(fake_model("draft v1", "draft v2"), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    result = app.invoke({"topic": "tests"}, config)
    assert result["__interrupt__"][0].value["draft"] == "draft v1"
    assert app.get_state(config).next == ("human_review",)

    result = app.invoke(Command(resume="make it shorter"), config)  # revise
    assert result["__interrupt__"][0].value["draft"] == "draft v2"

    result = app.invoke(Command(resume="yes"), config)  # approve
    assert result["published"] is True
    assert "__interrupt__" not in result


def test_debugging_demo_runs_with_fake_model(capsys):
    from playground.debugging import main

    for mode in ("updates", "values", "debug", "messages"):
        main(["--fake", "--mode", mode])
    assert "math_expert" in capsys.readouterr().out
