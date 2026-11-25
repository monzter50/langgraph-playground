"""Debugging helpers & demo: see what your chains and graphs are doing.

Run:
    uv run python -m playground.debugging                      # stream_mode="updates"
    uv run python -m playground.debugging --mode debug         # every task/checkpoint event
    uv run python -m playground.debugging --mode values        # full state after each step
    uv run python -m playground.debugging --mode messages      # LLM tokens as they stream
    uv run python -m playground.debugging --set-debug          # LangChain global debug logs
    uv run python -m playground.debugging --verbose            # lighter global logs
    uv run python -m playground.debugging --fake               # no API key: uses a fake model

Uses the routing graph from exercise 05. See docs/DEBUGGING.md.
"""

from __future__ import annotations

import argparse
import json

from langchain_core.globals import set_debug, set_verbose
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from playground.exercises.ex05_conditional_routing import build_graph


def _show(obj) -> str:
    """Pretty-print anything (messages, dicts, pydantic models)."""
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", default="updates", choices=["updates", "values", "debug", "messages"]
    )
    parser.add_argument(
        "--set-debug", action="store_true", help="langchain_core.globals.set_debug(True)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="langchain_core.globals.set_verbose(True)"
    )
    parser.add_argument("--fake", action="store_true", help="use a fake model (no API key)")
    parser.add_argument("--question", default="What is 17 * 23?")
    args = parser.parse_args(argv)

    # set_debug(True): logs EVERY input/output of every Runnable (prompts,
    # raw model responses, parser results...). Very noisy but very complete.
    set_debug(args.set_debug)
    # set_verbose(True): a lighter version that only some components honour.
    set_verbose(args.verbose)

    model = FakeListChatModel(responses=["math", "17 * 23 = 391"]) if args.fake else None
    graph = build_graph(model=model)

    print(f"=== stream_mode={args.mode!r} ===")
    for chunk in graph.stream({"question": args.question}, stream_mode=args.mode):
        if args.mode == "messages":
            # (message_chunk, metadata) tuples: token-level streaming from LLM nodes
            token, meta = chunk
            print(f"[{meta.get('langgraph_node')}] {token.text!r}")
            continue
        elif args.mode == "debug":
            # {"type": "task" | "task_result" | "checkpoint", "step": n, "payload": {...}}
            print(f"step={chunk['step']} type={chunk['type']}")
            print(_show(chunk["payload"]))
        else:
            # updates: {node_name: partial_update}; values: full state dict
            print(_show(chunk))
        print("-" * 60)


if __name__ == "__main__":
    main()
