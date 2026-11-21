"""Reranker: ordering, top-k, field preservation, ties, empty input, latency wrapper."""

from typing import Any

from rageval.ingest import CHUNK_SIZE
from rageval.rerank import percentile, rerank, reranked_retriever


def chunk(cid: str, text: str, score: float = 1.0) -> dict[str, Any]:
    return {"chunk_id": cid, "text": text, "source": "a.pdf", "page": 3, "score": score}


class FakeScorer:
    """Scores a pair by looking up the chunk text in a table; records every call."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self.table.get(text, 0.0) for _, text in pairs]


def test_orders_by_scorer_keeps_k_and_preserves_fields() -> None:
    scorer = FakeScorer({"low": 0.1, "high": 9.0, "mid": 5.0})
    out = rerank("q", [chunk("1", "low"), chunk("2", "high"), chunk("3", "mid")], 2, scorer)
    assert [c["chunk_id"] for c in out] == ["2", "3"]
    assert out[0] == {"chunk_id": "2", "text": "high", "source": "a.pdf", "page": 3, "score": 9.0}
    assert scorer.calls[0][0] == ("q", "low")


def test_ties_keep_retriever_order() -> None:
    scorer = FakeScorer({})
    out = rerank("q", [chunk("1", "a"), chunk("2", "b"), chunk("3", "c")], 3, scorer)
    assert [c["chunk_id"] for c in out] == ["1", "2", "3"]


def test_empty_candidates_never_call_scorer() -> None:
    scorer = FakeScorer({})
    assert rerank("q", [], 5, scorer) == []
    assert scorer.calls == []


def test_input_chunks_not_mutated() -> None:
    cands = [chunk("1", "a", 7.0)]
    rerank("q", cands, 1, FakeScorer({"a": 2.0}))
    assert cands[0]["score"] == 7.0


def test_wrapper_fetches_20_and_returns_k() -> None:
    asked: list[tuple[str, int]] = []

    def fetch(question: str, k: int) -> list[dict[str, Any]]:
        asked.append((question, k))
        return [chunk(str(i), f"t{i}") for i in range(k)]

    scorer = FakeScorer({"t7": 3.0, "t2": 2.0, "t11": 1.0})
    timings: list[float] = []
    retrieve_fn = reranked_retriever(fetch, scorer, timings)
    out = retrieve_fn("q", 2)
    assert asked == [("q", 20)]
    assert [c["chunk_id"] for c in out] == ["7", "2"]


def test_warmup_untimed_and_one_timing_per_query() -> None:
    scorer = FakeScorer({})
    timings: list[float] = []
    retrieve_fn = reranked_retriever(lambda q, k: [chunk("1", "a")], scorer, timings)
    assert timings == []
    assert len(scorer.calls) == 1  # warm-up happened at construction
    retrieve_fn("q1", 5)
    retrieve_fn("q2", 5)
    assert len(timings) == 2
    assert all(t >= 0 for t in timings)


def test_warmup_scores_full_size_batch_untimed() -> None:
    scorer = FakeScorer({})
    timings: list[float] = []
    # The fetch function is never called at construction (only the warm-up runs), so it is a
    # placeholder; what matters is the batch the warm-up scorer sees.
    reranked_retriever(lambda q, k: [], scorer, timings, candidates=7)
    assert timings == []
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == 7
    # Cross-encoder cost grows with text length, so the warm-up must use chunk-sized text,
    # the size real candidates have (ingest.CHUNK_SIZE), or the first real query is still cold.
    assert all(len(text) == CHUNK_SIZE for _q, text in scorer.calls[0])


def test_empty_fetch_still_records_one_timing_without_scoring() -> None:
    scorer = FakeScorer({})
    timings: list[float] = []
    retrieve_fn = reranked_retriever(lambda q, k: [], scorer, timings)
    calls_after_warmup = len(scorer.calls)
    assert retrieve_fn("q", 5) == []
    assert len(scorer.calls) == calls_after_warmup
    assert len(timings) == 1


def test_percentile_linear_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0
    assert abs(percentile(values, 95) - 38.5) < 1e-9
    assert percentile([40.0, 10.0, 30.0, 20.0], 50) == 25.0  # unsorted input


def test_percentile_single_value() -> None:
    assert percentile([7.5], 50) == 7.5
    assert percentile([7.5], 95) == 7.5
