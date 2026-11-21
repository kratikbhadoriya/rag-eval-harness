"""Behavior of the default rerank scorer, stubbed at the CrossEncoder class (offline, no torch)."""

import sys
import types
from typing import ClassVar

import pytest

from rageval import rerank_scorer


class StubCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder (no download, no torch)."""

    inits: ClassVar[list[str]] = []
    predicted: ClassVar[list[list[tuple[str, str]]]] = []

    def __init__(self, model_name: str) -> None:
        StubCrossEncoder.inits.append(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        StubCrossEncoder.predicted.append(list(pairs))
        return [float(len(text)) for _, text in pairs]


@pytest.fixture
def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> type[StubCrossEncoder]:
    StubCrossEncoder.inits = []
    StubCrossEncoder.predicted = []
    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    # `_model` is a module-level cache; reset it so every test starts with the model unloaded
    monkeypatch.setattr(rerank_scorer, "_model", None)
    return StubCrossEncoder


def test_import_does_not_load_the_model(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    assert stub_cross_encoder.inits == []


def test_default_score_returns_one_score_per_text_in_order(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    scores = rerank_scorer.default_score("q", ["a", "abc", "ab"])
    assert scores == [1.0, 3.0, 2.0]
    assert stub_cross_encoder.predicted == [[("q", "a"), ("q", "abc"), ("q", "ab")]]


def test_model_is_loaded_once_on_first_call(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    rerank_scorer.default_score("q", ["a"])
    rerank_scorer.default_score("q", ["b"])
    assert stub_cross_encoder.inits == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]


def test_missing_rerank_group_says_to_sync_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import raises ImportError
    monkeypatch.setattr(rerank_scorer, "_model", None)
    with pytest.raises(ImportError, match="rerank"):
        rerank_scorer.default_score("q", ["a"])


def test_empty_texts_return_empty_without_loading(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    assert rerank_scorer.default_score("q", []) == []
    assert stub_cross_encoder.inits == []


def test_score_pairs_single_query_is_one_model_call(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    assert rerank_scorer.score_pairs([("q", "a"), ("q", "abc")]) == [1.0, 3.0]
    assert stub_cross_encoder.predicted == [[("q", "a"), ("q", "abc")]]


def test_score_pairs_mixed_queries_keep_input_order(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    pairs = [("q1", "a"), ("q2", "abcd"), ("q1", "ab")]
    assert rerank_scorer.score_pairs(pairs) == [1.0, 4.0, 2.0]


def test_score_pairs_empty_returns_empty(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    assert rerank_scorer.score_pairs([]) == []
    assert stub_cross_encoder.inits == []
