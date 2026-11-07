"""Answer a question from the top-k retrieved chunks, numbered in a prompt read from a file."""

import re
from collections.abc import Callable
from importlib import resources
from typing import Any

from dotenv import find_dotenv, load_dotenv

from rageval.cache import Retriever
from rageval.embed import Embedder
from rageval.retrieve import retrieve

REFUSAL_PHRASE = "I cannot answer this from the provided documents."
"""The fixed sentence the prompt tells the model to reply with when contexts lack the answer.

Refusal detection (later tickets) matches on this constant; `prompts/answer.txt` must contain it.
"""

DEFAULT_MODEL = "gpt-4.1-mini"

Completion = Callable[..., Any]
"""Called like `litellm.completion(model=..., messages=..., temperature=...)`; inject a fake."""


def _default_completion(**kwargs: Any) -> Any:
    """litellm.completion, imported and keyed on first use so imports need no key."""
    load_dotenv(find_dotenv(usecwd=True))
    import litellm

    return litellm.completion(**kwargs)


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    """Fill the prompt file with the question and contexts numbered [1], [2], ..."""
    template = resources.files("rageval").joinpath("prompts", "answer.txt").read_text("utf-8")
    numbered = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(contexts, start=1))
    values = {"contexts": numbered, "question": question}
    return re.sub(r"\{(contexts|question)\}", lambda match: values[match.group(1)], template)


def generate(
    question: str,
    k: int,
    collection: str = "default",
    embed: Embedder | None = None,
    completion: Completion | None = None,
    model: str = DEFAULT_MODEL,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    """Retrieve the top-k chunks and answer from them at temperature 0.

    Returns {"answer": str, "contexts": list of retrieve() dicts, best first}. Raises
    ValueError if the collection was never ingested. `retriever(question, k)` replaces the
    direct `retrieve` call (e.g. the on-disk cache); the completion is never cached.
    """
    if retriever is not None:
        contexts = retriever(question, k)
    else:
        contexts = retrieve(question, k, collection=collection, embed=embed)
    call = completion or _default_completion
    response = call(
        model=model,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        temperature=0,
    )
    answer = str(response["choices"][0]["message"]["content"]).strip()
    return {"answer": answer, "contexts": contexts}
