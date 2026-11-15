"""Ingest PDFs into a collection."""

from collections.abc import Sequence
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rageval.embed import Embedder, default_embed
from rageval.store import chroma_client, save_chunks

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400


def ingest(
    paths: Sequence[str | Path],
    collection: str = "default",
    embed: Embedder | None = None,
) -> int:
    """Chunk PDFs page by page, embed and store them. Returns the number of chunks stored."""
    if embed is None:
        embed = default_embed
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
    collection_ = chroma_client().get_or_create_collection(
        collection, configuration={"hnsw": {"space": "cosine"}}
    )
    if ids:
        collection_.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,  # type: ignore[arg-type]
            embeddings=embed(texts),  # type: ignore[arg-type]
        )
    save_chunks(
        collection,
        [
            {"chunk_id": i, "text": t, "source": m["source"], "page": m["page"]}
            for i, t, m in zip(ids, texts, metas, strict=True)
        ],
    )
    return len(ids)
