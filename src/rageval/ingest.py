"""Ingest PDFs into a collection."""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rageval.embed import default_embed

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

Embedder = Callable[[list[str]], list[list[float]]]


def ingest(
    paths: Sequence[str | Path],
    collection: str = "default",
    embed: Embedder | None = None,
) -> int:
    """Chunk PDFs page by page, embed and store them. Returns the number of chunks stored."""
    embed = embed or default_embed
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict[str, str | int]] = []
    for path in paths:
        name = Path(path).name
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            for page_no, page in enumerate(doc, start=1):
                for index, chunk in enumerate(splitter.split_text(page.get_text())):
                    ids.append(f"{name}:{page_no}:{index}")
                    texts.append(chunk)
                    metas.append({"source": name, "page": page_no})
    if not ids:
        return 0
    client = chromadb.PersistentClient(path=os.environ.get("RAGEVAL_CHROMA_DIR", "./.chroma"))
    col = client.get_or_create_collection(collection, configuration={"hnsw": {"space": "cosine"}})
    col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embed(texts))  # type: ignore[arg-type]
    return len(ids)
