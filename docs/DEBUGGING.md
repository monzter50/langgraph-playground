# Debugging LangChain & LangGraph locally

This guide walks through every debugging tool set up in this repo, from the
lightest (print what's happening) to the heaviest (breakpoints inside a running
LangGraph server).

| Tool | Best for | Needs API key? |
|------|----------|----------------|
| [1. Stream intermediate steps](#1-stream-intermediate-steps) | Seeing which node ran and what it returned | No (with `--fake`) |
| [2. `set_debug` / `set_verbose`](#2-set_debug--set_verbose) | Seeing exact prompts and raw model responses | No (with `--fake`) |
| [3. VS Code breakpoints](#3-vs-code-breakpoints-on-an-exercise) | Stepping through node code line by line | Yes for real runs |
| [4. LangGraph Studio](#4-langgraph-studio-langgraph-dev) | Visualising the graph, threads, interrupts, state edits | Yes to run graphs |
| [5. Attach debugger to `langgraph dev`](#5-breakpoints-inside-langgraph-dev) | Breakpoints while driving runs from Studio | Yes to run graphs |
| [6. LangSmith tracing](#6-langsmith-tracing-optional) | Persisted traces: tokens, latency, every call | Yes + LangSmith key |

Before starting: `uv sync` and `cp .env.example .env` (see the README).

---

## 1. Stream intermediate steps

`graph.invoke()` only returns the final state. `graph.stream()` yields events
while the graph runs; `stream_mode` picks what you get:

| `stream_mode` | Yields | Use it to |
|---------------|--------|-----------|
| `"values"` | full state after every step | watch state evolve |
| `"updates"` | `{node_name: partial_update}` per node | see *who* changed *what* |
| `"debug"` | `{"type": "task" \| "task_result" \| "checkpoint", "step", "payload"}` | the most detail: node inputs, results, errors, interrupts, checkpoints |
| `"messages"` | `(token_chunk, metadata)` | token streaming from LLM calls inside nodes |

You can also pass a list (`stream_mode=["updates", "messages"]`) and get
`(mode, chunk)` tuples.

Try it with the demo script (`src/playground/debugging.py`, built on the
exercise 05 routing graph). `--fake` uses a scripted fake model, so no key is needed:

```sh
uv run python -m playground.debugging --fake                   # updates
uv run python -m playground.debugging --fake --mode values
uv run python -m playground.debugging --fake --mode debug
uv run python -m playground.debugging --fake --mode messages
uv run python -m playground.debugging --question "Why is the sky blue?"   # real model
```

In your own code:

```python
for step in graph.stream(inputs, stream_mode="updates"):
    print(step)
```

Exercise 06 (`ex06_react_agent.py`) already uses `stream_mode="updates"` to print
each Reason → Act → Observe step.

## 2. `set_debug` / `set_verbose`

LangChain has two global switches in `langchain_core.globals`:

```python
from langchain_core.globals import set_debug, set_verbose

set_debug(True)    # log EVERY runnable's input/output: prompts, raw LLM responses, parsers, graph nodes
set_verbose(True)  # lighter, human-readable logging (only some components honour it)
```

`set_debug(True)` is noisy but shows exactly what is sent to the model, which is
the fastest way to debug prompt templates. Try:

```sh
uv run python -m playground.debugging --fake --set-debug
uv run python -m playground.debugging --fake --verbose
```

Turn them off again with `set_debug(False)`. Don't leave them on in tests.

Other handy one-liners:

```python
print(graph.get_graph().draw_mermaid())   # paste at https://mermaid.live
graph.get_state(config)                   # current state of a thread (needs a checkpointer)
list(graph.get_state_history(config))     # every checkpoint (time travel)
```

## 3. VS Code breakpoints on an exercise

`.vscode/launch.json` ships these configurations (install the **Python** and
**Python Debugger** extensions first):

| Configuration | What it does |
|---------------|--------------|
| **Debug current exercise file** | Runs the open `src/playground/exercises/exNN_*.py` as a module (`python -m playground.exercises.exNN_...`) with `.env` loaded. |
| **Debug current exercise file (step into LangChain/LangGraph)** | Same with `justMyCode: false`, so you can step into library code (e.g. `ToolNode`, `StateGraph.invoke`). |
| **Debug debugging.py demo** | Runs `playground.debugging --fake --mode debug`. |
| **Debug pytest (all tests)** | Runs the test suite under the debugger — no API keys needed. |
| **Attach to langgraph dev (port 5678)** | See section 5. |

Steps:

1. `uv sync`, then in VS Code: *Python: Select Interpreter* → `./.venv/bin/python`
   (`.vscode/settings.json` already points there).
2. Open e.g. `src/playground/exercises/ex06_react_agent.py`.
3. Click in the gutter to set a breakpoint inside the `agent` node function.
4. *Run and Debug* (⇧⌘D) → **Debug current exercise file** → F5.
5. When it stops, inspect `state["messages"]` in the Variables pane, step with
   F10/F11, or evaluate `response.tool_calls` in the Debug Console.

> Tip: the "current exercise file" config only works for files inside
> `src/playground/exercises/` (it builds the module name from the folder name).
> For `debugging.py` use its own config.

Want breakpoints without real API calls? Put the breakpoint in the exercise
and run **Debug pytest** — the tests drive each graph with fake models.

## 4. LangGraph Studio (`langgraph dev`)

`langgraph.json` registers the graphs from exercises 04–08:

| Graph id | File |
|----------|------|
| `ex04_first_graph` | `src/playground/exercises/ex04_first_graph.py:graph` |
| `ex05_routing` | `src/playground/exercises/ex05_conditional_routing.py:graph` |
| `ex06_react_agent` | `src/playground/exercises/ex06_react_agent.py:graph` |
| `ex07_memory` | `src/playground/exercises/ex07_memory.py:graph` |
| `ex08_human_in_the_loop` | `src/playground/exercises/ex08_human_in_the_loop.py:graph` |

Start the local, in-memory LangGraph server:

```sh
uv run langgraph dev
```

You'll see:

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

The browser opens Studio (use `--no-browser` to skip). Studio is a web UI that
talks to *your local server*; your code and data stay on your machine. In it you can:

- pick a graph from the dropdown and see it drawn,
- run it with an input (e.g. for `ex06_react_agent`: a message
  `{"role": "user", "content": "What's 12*7?"}`),
- watch each node's output, open **threads** and inspect/edit state,
- for `ex08_human_in_the_loop`: the run pauses at `human_review`; answer the
  interrupt with `yes` or feedback text to resume,
- set *interrupts* before/after any node to step through like breakpoints,
- fork from any earlier checkpoint (time travel).

Notes:

- The server reads `.env` (`"env": ".env"` in `langgraph.json`) and hot-reloads
  when you save a file.
- The exported graphs are compiled **without** a checkpointer — the server adds
  its own persistence and thread management. The `main()` functions in ex07/ex08
  compile with `InMemorySaver` for standalone runs.
- If Safari or a strict browser blocks the `https → http://127.0.0.1` connection,
  use Chrome, or `uv run langgraph dev --tunnel`.
- Local server state goes into `.langgraph_api/` (gitignored).

## 5. Breakpoints inside `langgraph dev`

Combine Studio with VS Code breakpoints:

1. Start the server with a debug port:

   ```sh
   uv run langgraph dev --debug-port 5678
   # add --wait-for-client to block startup until the debugger attaches
   ```

   It prints `🐛 Debugger listening on port 5678`. (A warning about "frozen
   modules" is harmless; set `PYDEVD_DISABLE_FILE_VALIDATION=1` to silence it.)
2. In VS Code: *Run and Debug* → **Attach to langgraph dev (port 5678)** → F5.
3. Set a breakpoint in a node, e.g. `human_review` in `ex08_human_in_the_loop.py`.
4. Run that graph from Studio. VS Code stops on your breakpoint; inspect `state`,
   step, then continue (F5) to let Studio receive the result.

`debugpy` is already a dev dependency, which `--debug-port` requires.

## 6. LangSmith tracing (optional)

[LangSmith](https://smith.langchain.com) records every chain/graph run as a
trace tree: inputs, outputs, prompts, tool calls, token usage, latency and
errors. It needs no code changes — just env vars in `.env`:

```sh
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...           # Settings → API Keys in LangSmith
LANGSMITH_PROJECT=langgraph-playground
```

Then run any exercise and open the project in LangSmith. Each node, model and
tool call is a span you can click into. Set `LANGSMITH_TRACING=false` (or remove
it) to turn tracing off. Tests force it off.

Useful extras:

```python
# Tag / name runs so they're easy to find in LangSmith
graph.invoke(inputs, {"run_name": "ex06-try-1", "tags": ["practice"], "metadata": {"lesson": 6}})
```

## Local models (Ollama)

Everything above works the same with `MODEL_PROVIDER=ollama` (see the README's
"Running locally with Ollama"): `stream_mode`, `set_debug`, VS Code breakpoints,
LangGraph Studio and LangSmith tracing don't care which provider is behind
`get_chat_model()`. Local runs show up in LangSmith as `ChatOllama` spans (traces
are still sent to LangSmith if tracing is on, so turn it off if you want runs to
stay fully offline). The "Needs API key?" column above becomes "needs `ollama serve`
running" instead.

Tracing is especially handy with small local models: when exercise 03 or 06
misbehaves, the trace shows whether the model emitted a tool call at all and
with what arguments.

## Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| `MissingAPIKeyError: No model API key found` | `cp .env.example .env` and fill `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, or set `MODEL_PROVIDER=ollama`. |
| `ConnectError` / `model "qwen3:8b" not found` with Ollama | Start Ollama (`ollama serve` or the app) and `ollama pull qwen3:8b`; check `OLLAMA_BASE_URL`. |
| `<think>` text in answers | Leave `OLLAMA_REASONING` unset/`false`, or set `true` to move thinking out of `content`. |
| `ModuleNotFoundError: playground` | Run via `uv run python -m playground....` from the repo root (and `uv sync`). |
| Breakpoints are ignored | Check the interpreter is `.venv/bin/python`; use the `justMyCode: false` config for library code. |
| `GraphRecursionError` | The agent looped too long; inspect with `stream_mode="updates"` or raise `recursion_limit`. |
| Interrupt "doesn't pause" | `interrupt()` needs a checkpointer and a `thread_id` in the config. |
