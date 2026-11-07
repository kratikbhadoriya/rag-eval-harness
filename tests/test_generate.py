"""Behavior of rageval.generate through its public function, with fake completion and embedder."""

from pathlib import Path
from typing import Any

import pytest

from rageval import generate, ingest
from rageval.embed import Embedder
from rageval.generate import REFUSAL_PHRASE


class FakeCompletion:
    """Records calls and replies like litellm.completion would."""

    def __init__(self, reply: str = "Cats purr [1].") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.reply}}]}


def _prompt(fake: FakeCompletion) -> str:
    return "\n".join(m["content"] for m in fake.calls[0]["messages"])


def test_returns_answer_and_contexts_as_plain_dicts(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([multi_pdf], embed=fake_embed)
    fake = FakeCompletion("Rivers flow [1].")
    result = generate("Rivers", k=2, embed=fake_embed, completion=fake)
    assert set(result) == {"answer", "contexts"}
    assert result["answer"] == "Rivers flow [1]."
    assert len(result["contexts"]) == 2
    assert all(isinstance(c, dict) and "text" in c for c in result["contexts"])
    assert "Rivers" in result["contexts"][0]["text"]


def test_contexts_are_numbered_in_prompt_in_retrieval_order(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([multi_pdf], embed=fake_embed)
    fake = FakeCompletion()
    result = generate("Satellites", k=3, embed=fake_embed, completion=fake)
    prompt = _prompt(fake)
    for i, ctx in enumerate(result["contexts"], start=1):
        assert f"[{i}] {ctx['text']}" in prompt
    assert "Satellites" in prompt  # the question is in the prompt too


def test_k_is_respected(multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder) -> None:
    ingest([multi_pdf], embed=fake_embed)
    fake = FakeCompletion()
    result = generate("Rivers", k=1, embed=fake_embed, completion=fake)
    assert len(result["contexts"]) == 1
    prompt = _prompt(fake)
    assert "[1]" in prompt
    assert "[2]" not in prompt


def test_prompt_comes_from_file_and_carries_refusal_phrase(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([multi_pdf], embed=fake_embed)
    fake = FakeCompletion()
    generate("Rivers", k=1, embed=fake_embed, completion=fake)
    assert REFUSAL_PHRASE in _prompt(fake)
    prompt_file = Path(__file__).parents[1] / "src" / "rageval" / "prompts" / "answer.txt"
    assert REFUSAL_PHRASE in prompt_file.read_text()


def test_calls_completion_at_temperature_zero_with_a_groq_model(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([multi_pdf], embed=fake_embed)
    fake = FakeCompletion()
    generate("Rivers", k=1, embed=fake_embed, completion=fake)
    assert fake.calls[0]["temperature"] == 0
    assert fake.calls[0]["model"].startswith("groq/")


def test_unknown_collection_raises_value_error(chroma_dir: Path, fake_embed: Embedder) -> None:
    with pytest.raises(ValueError):
        generate("x", k=1, collection="nothing", embed=fake_embed, completion=FakeCompletion())
