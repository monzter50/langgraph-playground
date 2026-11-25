"""Exercise 01 — Prompt + model + output parser (an LCEL chain).

Run:  uv run python -m playground.exercises.ex01_lcel_chain

LCEL (LangChain Expression Language) lets you compose "Runnables" with the
``|`` operator. Every piece — prompt, model, parser — is a Runnable with the
same interface: ``invoke``, ``batch``, ``stream`` (and async ``a*`` variants).

    prompt  ->  model  ->  parser
    dict        list of    AIMessage   ->  str
                messages
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from playground.models import get_chat_model

# 1) A prompt template. ``{topic}`` and ``{audience}`` are input variables
#    that must be supplied when the chain is invoked.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise teacher. Answer in at most 3 sentences."),
        ("human", "Explain {topic} to {audience}."),
    ]
)


def build_chain(model: BaseChatModel | None = None) -> Runnable:
    """Compose prompt | model | parser.

    ``model`` is injectable so tests can pass a fake model (no API key needed).
    """
    model = model or get_chat_model()
    # 2) StrOutputParser turns the AIMessage returned by the model into a str.
    return prompt | model | StrOutputParser()


def main() -> None:
    chain = build_chain()

    # invoke: one input -> one output
    answer = chain.invoke({"topic": "vector embeddings", "audience": "a 10 year old"})
    print("invoke ->", answer, "\n")

    # stream: tokens arrive as they are generated (the parser streams too!)
    print("stream -> ", end="")
    for chunk in chain.stream({"topic": "recursion", "audience": "a chef"}):
        print(chunk, end="", flush=True)
    print("\n")

    # batch: many inputs, run concurrently
    results = chain.batch(
        [
            {"topic": "HTTP", "audience": "a cat"},
            {"topic": "git rebase", "audience": "a pirate"},
        ]
    )
    for r in results:
        print("batch ->", r)


# ---------------------------------------------------------------------------
# TRY THIS
# - Print ``prompt.invoke({...})`` on its own to see the messages sent to the model.
# - Remove ``StrOutputParser()`` and inspect the raw AIMessage (``.content``,
#   ``.response_metadata``, ``.usage_metadata``).
# - Add a second step: pipe the answer into another prompt that translates it
#   to Spanish. Hint: ``{"text": chain} | translate_prompt | model | parser``.
# - Use ``print(chain.get_graph().draw_mermaid())`` to visualise the chain.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
