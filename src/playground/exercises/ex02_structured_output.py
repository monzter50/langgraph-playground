"""Exercise 02 — Structured output with Pydantic.

Run:  uv run python -m playground.exercises.ex02_structured_output

Instead of parsing free text, ask the model to return data that matches a
schema. ``model.with_structured_output(Schema)`` uses the provider's native
tool-calling / JSON features under the hood and returns a validated Pydantic
object.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from playground.models import get_chat_model


# Field descriptions are sent to the model — they ARE part of the prompt.
class Recipe(BaseModel):
    """A cooking recipe extracted from free-form text."""

    name: str = Field(description="Short name of the dish")
    servings: int = Field(description="How many people it serves", ge=1)
    ingredients: list[str] = Field(description="Ingredients with quantities")
    vegetarian: bool = Field(description="True if the dish has no meat or fish")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract a structured recipe from the user's text."),
        ("human", "{text}"),
    ]
)


def build_chain(model: BaseChatModel | None = None) -> Runnable:
    model = model or get_chat_model()
    # The result of this chain is a ``Recipe`` instance, not a message.
    return prompt | model.with_structured_output(Recipe)


def main() -> None:
    chain = build_chain()
    text = (
        "My grandma's tomato soup feeds four: 1kg ripe tomatoes, 1 onion, "
        "2 garlic cloves, 500ml vegetable stock, olive oil and basil."
    )
    recipe: Recipe = chain.invoke({"text": text})
    print(type(recipe).__name__, "->")
    print(recipe.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# TRY THIS
# - Add a ``steps: list[str]`` field and a ``difficulty: Literal["easy", "medium", "hard"]``.
# - Pass ``include_raw=True`` to ``with_structured_output`` and inspect the raw
#   AIMessage + parsing errors.
# - Feed it text that is NOT a recipe. What happens? Make fields Optional to
#   let the model say "unknown".
# - Extract a ``list[Recipe]`` by wrapping it in another model (``class Menu``).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
