"""Reranker: rescore retrieved chunks with a cross-encoder and keep the best k.

Reranker scores are a separate scale from a Retriever's Score; never compare them.
The scoring step is injectable: a function over (query, chunk text) pairs.
"""

import time
from collections.abc import Callable
from typing import Any

from rageval.ingest import CHUNK_SIZE

Scorer = Callable[[list[tuple[str, str]]], list[float]]
RetrieveFn = Callable[[str, int], list[dict[str, Any]]]

CANDIDATES = 20  # chunks fetched from the Retriever before reranking


def rerank(
    query: str, chunks: list[dict[str, Any]], k: int, scorer: Scorer
) -> list[dict[str, Any]]:
    """Return the k chunks by scorer score, best first, with `score` replaced.

    Ties keep the incoming (Retriever) order. An empty list never calls the scorer.
    """
    if not chunks:
        return []
    scores = scorer([(query, c["text"]) for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])  # stable: ties keep order
    return [{**chunks[i], "score": float(scores[i])} for i in order[:k]]


def reranked_retriever(
    fetch: RetrieveFn,
    scorer: Scorer,
    timings_ms: list[float],
    candidates: int = CANDIDATES,
) -> RetrieveFn:
    """Wrap `fetch(question, k)`: fetch `candidates` chunks, rerank, return k.

    Runs one untimed warm-up scorer call now, on `candidates` dummy pairs of CHUNK_SIZE
    characters, so a full-size batch is scored once before any timing.
    Each call then appends the milliseconds spent in
    the rerank step alone (not the fetch) to `timings_ms`, which the caller owns.
    """
    scorer([("warm-up", "w" * CHUNK_SIZE)] * candidates)

    def retrieve_reranked(question: str, k: int) -> list[dict[str, Any]]:
        found = fetch(question, candidates)
        start = time.perf_counter()
        out = rerank(question, found, k, scorer)
        timings_ms.append((time.perf_counter() - start) * 1000)
        return out

    return retrieve_reranked


def percentile(values: list[float], q: float) -> float:
    """q-th percentile (0-100) by linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
