"""Behavior of rageval.ingest through its public function."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb

from rageval import ingest

Embed = Callable[[list[str]], list[list[float]]]


def _stored(chroma_dir: Path, collection: str = "default") -> Any:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(collection).get(include=["metadatas", "documents"])


def test_single_page_returns_count_and_records_metadata(
    single_pdf: Path, chroma_dir: Path, fake_embed: Embed
) -> None:
    n = ingest([single_pdf], embed=fake_embed)
    assert n == 1
    got = _stored(chroma_dir)
    assert got["ids"] == ["single.pdf:1:0"]
    assert got["metadatas"][0]["source"] == "single.pdf"
    assert got["metadatas"][0]["page"] == 1
    assert "Cats purr" in got["documents"][0]


def test_multi_page_chunks_never_span_pages(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embed
) -> None:
    n = ingest([multi_pdf], embed=fake_embed)
    got = _stored(chroma_dir)
    assert n == len(got["ids"]) > 3
    for cid, meta, doc in zip(got["ids"], got["metadatas"], got["documents"], strict=True):
        base, page, index = cid.split(":")
        assert base == "multi.pdf"
        assert int(page) == meta["page"]
        assert index.isdigit()
        if meta["page"] == 1:
            assert "Apples" in doc
            assert "Rivers" not in doc
        if meta["page"] == 3:
            assert "Satellites" in doc
            assert "Rivers" not in doc
    assert {m["page"] for m in got["metadatas"]} == {1, 2, 3}
    assert "multi.pdf:2:1" in got["ids"]


def test_path_accepts_string(single_pdf: Path, chroma_dir: Path, fake_embed: Embed) -> None:
    assert ingest([str(single_pdf)], collection="docs", embed=fake_embed) == 1
    assert _stored(chroma_dir, "docs")["ids"] == ["single.pdf:1:0"]
