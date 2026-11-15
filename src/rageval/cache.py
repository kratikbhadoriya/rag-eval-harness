"""On-disk cache for retrieval output (never generation or Judge calls).

Key: question, k, collection, chunk size, chunk overlap, corpus checksum, embedder name, Retriever.
Directory: `RAGEVAL_CACHE_DIR` (default `./.eval_cache`, gitignored).
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rageval.embed import Embedder, default_embed
from rageval.ingest import CHUNK_OVERLAP, CHUNK_SIZE
from rageval.retrieve import DEFAULT_RETRIEVER, RetrieverName, retrieve

RetrieveFn = Callable[[str, int], list[dict[str, Any]]]
"""(question, k) -> retrieve() result dicts."""

DEFAULT_CORPUS_DIR = Path("data/corpus")


def cache_dir() -> Path:
    return Path(os.environ.get("RAGEVAL_CACHE_DIR", ".eval_cache"))


def corpus_checksum(corpus_dir: Path) -> str:
    """SHA-256 over the relative names and bytes of every file under corpus_dir, sorted."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    digest = hashlib.sha256()
    for path in sorted(p for p in corpus_dir.rglob("*") if p.is_file()):
        digest.update(path.relative_to(corpus_dir).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def embedder_name(embed: Embedder) -> str:
    """The embedder's `name` attribute if it has one, else its module-qualified function name."""
    name = getattr(embed, "name", None)
    if isinstance(name, str):
        return name
    return f"{getattr(embed, '__module__', '?')}.{getattr(embed, '__qualname__', repr(embed))}"


def cached_retriever(
    collection: str = "default",
    embed: Embedder | None = None,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    retriever: RetrieverName = DEFAULT_RETRIEVER,
) -> RetrieveFn:
    """Return a retriever that reads/writes the cache around `retrieve`.

    The corpus checksum is computed once here, not per question. A miss calls `retrieve`
    (and so the embedder) and stores the result.
    """
    embed = embed or default_embed
    key_parts = {
        "collection": collection,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "corpus": corpus_checksum(corpus_dir),
        "embedder": embedder_name(embed),
        "retriever": retriever,
    }

    def fetch(question: str, k: int) -> list[dict[str, Any]]:
        key = json.dumps({**key_parts, "question": question, "k": k}, sort_keys=True)
        path = cache_dir() / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        if path.exists():
            return list(json.loads(path.read_text(encoding="utf-8")))
        results = retrieve(question, k, collection=collection, embed=embed, retriever=retriever)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results), encoding="utf-8")
        return results

    return fetch
