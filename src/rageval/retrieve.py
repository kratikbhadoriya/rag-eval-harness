"""Retrieve the most similar chunks for a query."""

import re
from typing import Any

from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from rageval.embed import Embedder, default_embed
from rageval.store import chroma_client, chunk_list_stamp, load_chunks

RETRIEVERS: tuple[str, ...] = ("dense", "bm25", "hybrid")
DEFAULT_RETRIEVER = "bm25"  # won #38 on Context recall; see metrics.md

# Reciprocal rank fusion constant; 60 is the standard value from the original RRF paper.
RRF_K = 60
# Each Retriever contributes this many top candidates to fusion.
CANDIDATE_DEPTH = 50


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; shared by Chunks and queries. No stemming, no stopwords."""
    return re.findall(r"\w+", text.lower())


_bm25_indexes: dict[str, tuple[tuple[str, int, int], BM25Okapi | None, list[dict[str, Any]]]] = {}


def _retrieve_bm25(query: str, k: int, collection: str) -> list[dict[str, Any]]:
    chunks = load_chunks(collection)
    if chunks is None:
        raise ValueError(f"collection {collection!r} was never ingested")
    stamp = chunk_list_stamp(collection)
    cached = _bm25_indexes.get(stamp[0])
    if cached is None or cached[0] != stamp:
        index = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
        cached = (stamp, index, chunks)
        _bm25_indexes[stamp[0]] = cached
    _, index, chunks = cached
    terms = tokenize(query)
    if index is None or not terms:
        return []
    scores = index.get_scores(terms)
    term_set = set(terms)
    results = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "source": c["source"],
            "page": int(c["page"]),
            "score": float(score),
        }
        for c, score in zip(chunks, scores, strict=True)
        if term_set & set(tokenize(c["text"]))
    ]
    return sorted(results, key=lambda r: r["score"], reverse=True)[:k]


def _retrieve_dense(
    query: str, k: int, collection: str, embed: Embedder | None
) -> list[dict[str, Any]]:
    if embed is None:
        embed = default_embed
    try:
        chunks = chroma_client().get_collection(collection)
    except NotFoundError as exc:
        raise ValueError(f"collection {collection!r} was never ingested") from exc
    res = chunks.query(
        query_embeddings=embed([query]),  # type: ignore[arg-type]
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = (res["documents"] or [[]])[0]
    metas = (res["metadatas"] or [[]])[0]
    dists = (res["distances"] or [[]])[0]
    results = [
        {
            "chunk_id": cid,
            "text": doc,
            "source": str(meta["source"]),
            "page": int(meta["page"]),  # type: ignore[arg-type]
            "score": 1.0 - float(dist),
        }
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True)
    ]
    return sorted(results, key=lambda r: r["score"], reverse=True)


def _retrieve_hybrid(
    query: str, k: int, collection: str, embed: Embedder | None
) -> list[dict[str, Any]]:
    rankings = [
        _retrieve_dense(query, CANDIDATE_DEPTH, collection, embed),
        _retrieve_bm25(query, CANDIDATE_DEPTH, collection),
    ]
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            cid = result["chunk_id"]
            by_id[cid] = result
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
    return [{**by_id[cid], "score": scores[cid]} for cid in ordered]


def retrieve(
    query: str,
    k: int,
    collection: str = "default",
    embed: Embedder | None = None,
    retriever: str = DEFAULT_RETRIEVER,
) -> list[dict[str, Any]]:
    """Return up to k chunks for the query, best first.

    retriever is "bm25" (default), "dense" or "hybrid"; anything else raises ValueError.
    Each result has chunk_id, text, source, page and score: 1 - cosine distance for dense,
    the raw BM25 score for bm25, the summed reciprocal-rank-fusion score for hybrid.
    bm25 never calls the embedder.

    Raises ValueError if the collection was never ingested (a visible mistake, not an
    empty list). A collection that exists but has no matches returns [].
    """
    if retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}; choose from {', '.join(RETRIEVERS)}")
    if retriever == "bm25":
        return _retrieve_bm25(query, k, collection)
    if retriever == "hybrid":
        return _retrieve_hybrid(query, k, collection, embed)
    return _retrieve_dense(query, k, collection, embed)
