"""Behavior of rageval.ingest, observed through rageval.retrieve."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rageval import ingest, retrieve

Embed = Callable[[list[str]], list[list[float]]]


def _all_chunks(embed: Embed, collection: str = "default") -> list[dict[str, Any]]:
    return retrieve("the", k=1000, collection=collection, embed=embed)


def test_single_page_returns_count_and_records_metadata(
    single_pdf: Path, chroma_dir: Path, fake_embed: Embed
) -> None:
    n = ingest([single_pdf], embed=fake_embed)
    assert n == 1
    got = _all_chunks(fake_embed)
    assert [r["chunk_id"] for r in got] == ["single.pdf:1:0"]
    assert got[0]["source"] == "single.pdf"
    assert got[0]["page"] == 1
    assert "Cats purr" in got[0]["text"]


def test_multi_page_chunks_never_span_pages(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embed
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


def test_path_accepts_string(single_pdf: Path, chroma_dir: Path, fake_embed: Embed) -> None:
    assert ingest([str(single_pdf)], collection="docs", embed=fake_embed) == 1
    assert [r["chunk_id"] for r in _all_chunks(fake_embed, "docs")] == ["single.pdf:1:0"]
