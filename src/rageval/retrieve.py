"""Retrieve the most similar chunks for a query."""

import os
from typing import Any

import chromadb

from rageval.ingest import Embedder


def retrieve(
    query: str,
    k: int,
    collection: str = "default",
    embed: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Return up to k chunks most similar to the query, best first.

    Each result has chunk_id, text, source, page and score (1 - cosine distance).
    """
    if embed is None:
        raise NotImplementedError("default embedder is not implemented yet")
    client = chromadb.PersistentClient(path=os.environ.get("RAGEVAL_CHROMA_DIR", "./.chroma"))
    col = client.get_collection(collection)
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
