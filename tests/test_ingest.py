"""Behavior of rageval.ingest, observed through rageval.retrieve."""

from pathlib import Path
from typing import Any

import pymupdf
import pytest

from rageval import ingest, retrieve
from rageval.embed import Embedder


def _all_chunks(embed: Embedder, collection: str = "default") -> list[dict[str, Any]]:
    return retrieve("the", k=1000, collection=collection, embed=embed, retriever="dense")


def test_single_page_returns_count_and_records_metadata(
    single_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    n = ingest([single_pdf], embed=fake_embed)
    assert n == 1
    got = _all_chunks(fake_embed)
    assert [r["chunk_id"] for r in got] == ["single.pdf:1:0"]
    assert got[0]["source"] == "single.pdf"
    assert got[0]["page"] == 1
    assert "Cats purr" in got[0]["text"]


def test_multi_page_chunks_never_span_pages(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    n = ingest([multi_pdf], embed=fake_embed)
    got = _all_chunks(fake_embed)
    assert n == len(got) > 3
    for r in got:
        base, page, index = r["chunk_id"].split(":")
        assert base == "multi.pdf"
        assert int(page) == r["page"]
        assert index.isdigit()
        if r["page"] == 1:
            assert "Apples" in r["text"]
            assert "Rivers" not in r["text"]
        if r["page"] == 3:
            assert "Satellites" in r["text"]
            assert "Rivers" not in r["text"]
    assert {r["page"] for r in got} == {1, 2, 3}
    assert "multi.pdf:2:1" in {r["chunk_id"] for r in got}


def test_path_accepts_string(single_pdf: Path, chroma_dir: Path, fake_embed: Embedder) -> None:
    assert ingest([str(single_pdf)], collection="docs", embed=fake_embed) == 1
    assert [r["chunk_id"] for r in _all_chunks(fake_embed, "docs")] == ["single.pdf:1:0"]


def test_reingest_leaves_chunk_count_unchanged(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    first = ingest([multi_pdf], embed=fake_embed)
    before = sorted(r["chunk_id"] for r in _all_chunks(fake_embed))
    assert ingest([multi_pdf], embed=fake_embed) == first
    assert sorted(r["chunk_id"] for r in _all_chunks(fake_embed)) == before


def test_multiple_paths_in_one_call_are_all_counted(
    single_pdf: Path, multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    n = ingest([single_pdf, multi_pdf], embed=fake_embed)
    got = _all_chunks(fake_embed)
    assert n == len(got)
    assert {r["source"] for r in got} == {"single.pdf", "multi.pdf"}


def test_collections_are_isolated(
    single_pdf: Path, multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([single_pdf], collection="alpha", embed=fake_embed)
    ingest([multi_pdf], collection="beta", embed=fake_embed)
    assert {r["source"] for r in _all_chunks(fake_embed, "alpha")} == {"single.pdf"}
    assert {r["source"] for r in _all_chunks(fake_embed, "beta")} == {"multi.pdf"}


def test_retrieve_from_never_ingested_collection_raises_clear_error(
    single_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([single_pdf], collection="alpha", embed=fake_embed)
    with pytest.raises(ValueError, match="never ingested"):
        retrieve("cats", k=3, collection="missing", embed=fake_embed)


def test_ingest_with_zero_chunks_still_creates_collection(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    blank = tmp_path / "blank.pdf"
    with pymupdf.open() as doc:  # type: ignore[no-untyped-call]
        doc.new_page()
        doc.save(blank)
    assert ingest([blank], collection="empty", embed=fake_embed) == 0
    assert retrieve("anything", k=3, collection="empty", embed=fake_embed) == []
