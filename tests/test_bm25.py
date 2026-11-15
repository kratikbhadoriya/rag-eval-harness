"""Behavior of the bm25 Retriever through ingest() then retrieve()."""

from pathlib import Path
from typing import Any

import pymupdf
import pytest

from rageval import ingest, retrieve
from rageval.embed import Embedder


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    for text in pages:
        doc.new_page().insert_text((72, 72), text)  # type: ignore[no-untyped-call]
    doc.save(path)  # type: ignore[no-untyped-call]
    doc.close()  # type: ignore[no-untyped-call]
    return path


RARE_PAGES = [
    "xylophone alpha beta gamma delta epsilon zeta eta",
    "alpha alpha alpha alpha",
    "river sea water",
    "cats purr softly",
]
QUERY = "xylophone alpha"


def test_default_retrieve_is_bm25_and_never_calls_the_embedder(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    default = retrieve(QUERY, k=4, embed=_never_called)
    assert default == retrieve(QUERY, k=4, embed=fake_embed, retriever="bm25")
    assert default[0]["page"] == 1


def test_bm25_ranks_rare_word_chunk_first_where_dense_ranks_another_higher(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    dense = retrieve(QUERY, k=4, embed=fake_embed, retriever="dense")
    assert dense[0]["page"] == 2
    bm25 = retrieve(QUERY, k=4, embed=fake_embed, retriever="bm25")
    assert bm25[0]["page"] == 1
    assert "xylophone" in bm25[0]["text"]


def _never_called(texts: list[str]) -> list[list[float]]:
    raise AssertionError("bm25 must not call the embedder")


def _text_by_id(results: list[dict[str, Any]]) -> dict[str, str]:
    return {r["chunk_id"]: r["text"] for r in results}


def _both_index_same_text(embed: Embedder) -> None:
    dense = retrieve("anything", k=1000, embed=embed, retriever="dense")
    all_words = " ".join(r["text"] for r in dense)
    bm25 = retrieve(all_words, k=1000, embed=embed, retriever="bm25")
    assert _text_by_id(bm25) == _text_by_id(dense)
    assert len(dense) > 0


def test_bm25_needs_no_embedder_at_query_time(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    results = retrieve(QUERY, k=2, embed=_never_called, retriever="bm25")
    assert results[0]["page"] == 1


def test_bm25_and_vector_store_index_identical_text_across_reingest_and_second_file(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    first = make_pdf(tmp_path / "one.pdf", RARE_PAGES)
    ingest([first], embed=fake_embed)
    _both_index_same_text(fake_embed)
    ingest([first], embed=fake_embed)
    _both_index_same_text(fake_embed)
    second = make_pdf(tmp_path / "two.pdf", ["comet orbit sun", "moon tide ocean"])
    ingest([second], embed=fake_embed)
    _both_index_same_text(fake_embed)
    assert len(retrieve("comet", k=100, embed=fake_embed, retriever="bm25")) == 1
    assert len(retrieve("anything", k=100, embed=fake_embed, retriever="dense")) == 6


def test_collections_are_isolated(tmp_path: Path, chroma_dir: Path, fake_embed: Embedder) -> None:
    ingest([make_pdf(tmp_path / "a.pdf", ["comet orbit sun"])], collection="aaa", embed=fake_embed)
    ingest([make_pdf(tmp_path / "b.pdf", ["river sea water"])], collection="bbb", embed=fake_embed)
    assert retrieve("comet", k=5, collection="bbb", retriever="bm25") == []
    got = retrieve("comet", k=5, collection="aaa", retriever="bm25")
    assert [r["source"] for r in got] == ["a.pdf"]


def test_never_ingested_collection_raises_value_error(chroma_dir: Path) -> None:
    with pytest.raises(ValueError, match="never ingested"):
        retrieve("comet", k=5, collection="nothing", retriever="bm25")


def test_ingested_collection_with_no_matches_returns_empty_list(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "a.pdf", RARE_PAGES)], embed=fake_embed)
    assert retrieve("qqqq", k=5, retriever="bm25") == []


def test_zero_chunk_ingest_still_creates_collection(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    assert ingest([make_pdf(tmp_path / "blank.pdf", [""])], embed=fake_embed) == 0
    assert retrieve("comet", k=5, retriever="bm25") == []


def test_results_have_exact_keys_sorted_best_first_and_capped_at_k(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    results = retrieve("alpha xylophone beta", k=2, retriever="bm25")
    assert len(results) == 2
    for r in results:
        assert set(r) == {"chunk_id", "text", "source", "page", "score"}
    assert results[0]["score"] >= results[1]["score"]


def test_tokenisation_ignores_case_and_punctuation(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "a.pdf", ["Comet, ORBIT! sun.", "river sea"])], embed=fake_embed)
    assert [r["page"] for r in retrieve("comet orbit", k=5, retriever="bm25")] == [1]


def test_unknown_retriever_raises_value_error(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "a.pdf", RARE_PAGES)], embed=fake_embed)
    with pytest.raises(ValueError, match="unknown retriever"):
        retrieve("alpha", k=5, embed=fake_embed, retriever="fuzzy")
