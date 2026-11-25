# langgraph-playground

Practice playground for [LangChain](https://python.langchain.com/) and
[LangGraph](https://langchain-ai.github.io/langgraph/) (Python): eight small,
commented exercises plus a local debugging setup (LangGraph Studio, VS Code
breakpoints, LangSmith tracing, streaming).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/)
(`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```sh
uv sync                    # creates .venv and installs everything (incl. dev tools)
cp .env.example .env       # then add ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

The model is chosen in [`src/playground/models.py`](src/playground/models.py):
Anthropic Claude by default (`claude-sonnet-5-5`), OpenAI as a fallback if only
`OPENAI_API_KEY` is set. Override with `MODEL_PROVIDER` / `MODEL_NAME` in `.env`,
or run everything on a local model with Ollama (next section).

## Running locally with Ollama

No API key needed: the exercises can run against a local model served by
[Ollama](https://ollama.com/).

```sh
ollama pull qwen3:8b        # ~5 GB download, once
echo "MODEL_PROVIDER=ollama" >> .env
uv run python -m playground.exercises.ex01_lcel_chain
uv run langgraph dev        # Studio works the same with local models
```

Settings (all optional, see [`.env.example`](.env.example)):

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_MODEL` | `qwen3:8b` | Any model you have pulled (`ollama list`). `MODEL_NAME` is ignored for Ollama. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where `ollama serve` is listening. |
| `OLLAMA_REASONING` | off | qwen3 "thinks" before answering. Off disables thinking so `message.content` is a clean answer (and it's faster). `true` keeps the thinking but moves it to `message.additional_kwargs["reasoning_content"]`; `low` / `medium` / `high` set an effort level where supported. |

**Pick a model that fits your RAM.** On a 16 GB Mac, 8B models such as
`qwen3:8b` (~5 GB) run comfortably; the larger `gemma4:26b` (~17 GB) does not
fit in memory and will swap or fail to load, so don't use it here.

**Expect rougher edges than with Claude or GPT.** Small local models are less
reliable at tool calling and multi-step agent loops, so exercises **03** (tools),
**06** (ReAct agent) and **08** (human-in-the-loop) are the most likely to
misbehave: skipped or malformed tool calls, doing math "in their head", or
stopping early. That's a useful thing to observe and debug, not a bug in the
exercise; switch back to a cloud model to compare.

## Exercises

Each file is small, runnable on its own, and ends with a **TRY THIS** section of
practice ideas.

| # | Topic | Run |
|---|-------|-----|
| 01 | Prompt + model + output parser (LCEL chain) | `uv run python -m playground.exercises.ex01_lcel_chain` |
| 02 | Structured output with Pydantic | `uv run python -m playground.exercises.ex02_structured_output` |
| 03 | Tools / tool calling (manual loop) | `uv run python -m playground.exercises.ex03_tools` |
| 04 | First `StateGraph` (state, nodes, edges) | `uv run python -m playground.exercises.ex04_first_graph` |
| 05 | Conditional edges / routing | `uv run python -m playground.exercises.ex05_conditional_routing` |
| 06 | ReAct agent with tools in LangGraph | `uv run python -m playground.exercises.ex06_react_agent` |
| 07 | Memory: checkpointer + `thread_id` | `uv run python -m playground.exercises.ex07_memory` |
| 08 | Human-in-the-loop with `interrupt` | `uv run python -m playground.exercises.ex08_human_in_the_loop` |

Exercises live in [`src/playground/exercises/`](src/playground/exercises/).

## Debugging

```sh
uv run langgraph dev                         # LangGraph Studio on the graphs from ex04–ex08
uv run langgraph dev --debug-port 5678       # ...and attach VS Code's debugger
uv run python -m playground.debugging --fake --mode debug   # stream intermediate steps
```

👉 Full step-by-step guide: **[docs/DEBUGGING.md](docs/DEBUGGING.md)** (Studio,
VS Code breakpoints, LangSmith tracing, `stream_mode`, `set_debug`).

## Tests & lint

Tests run every graph against fake chat models, so **no API keys are needed**.

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

CI (GitHub Actions) runs the same on every push / PR.

## Layout

```
src/playground/
  models.py            # get_chat_model(): picks Anthropic / OpenAI / Ollama from env
  debugging.py         # stream_mode + set_debug/set_verbose demo
  exercises/ex01..08   # the exercises
tests/                 # pytest, fake models only
langgraph.json         # graphs registered for `langgraph dev` / Studio
.vscode/launch.json    # debug configs (current file, pytest, attach to langgraph dev)
docs/DEBUGGING.md
```

## License

MIT
