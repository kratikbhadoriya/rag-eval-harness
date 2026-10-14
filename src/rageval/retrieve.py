"""Retrieve the most similar chunks for a query."""

import os
from typing import Any

import chromadb

from rageval.embed import default_embed
from rageval.ingest import Embedder


def retrieve(
    query: str,
    k: int,
    collection: str = "default",
    embed: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Return up to k chunks most similar to the query, best first.

    Each result has chunk_id, text, source, page and score (1 - cosine distance).

    Raises ValueError if the collection was never ingested (a visible mistake, not an
    empty list). A collection that exists but has no matches returns [].
    """
    if embed is None:
        embed = default_embed
    client = chromadb.PersistentClient(path=os.environ.get("RAGEVAL_CHROMA_DIR", "./.chroma"))
    try:
        col = client.get_collection(collection)
    except Exception as exc:  # chromadb raises NotFoundError / InvalidArgumentError by version
        existing = {c.name if hasattr(c, "name") else str(c) for c in client.list_collections()}
        if collection in existing:
            raise
        raise ValueError(f"collection {collection!r} was never ingested") from exc
    res = col.query(
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
