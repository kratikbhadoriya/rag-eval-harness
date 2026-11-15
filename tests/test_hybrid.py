"""Behavior of the hybrid Retriever through ingest() then retrieve()."""

from pathlib import Path

import pymupdf
import pytest

from rageval import ingest, retrieve
from rageval.embed import Embedder

RARE_PAGES = [
    "xylophone alpha beta gamma delta epsilon zeta eta",
    "alpha alpha alpha alpha",
    "river sea water",
    "cats purr softly",
]
QUERY = "xylophone alpha"


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    for text in pages:
        doc.new_page().insert_text((72, 72), text)  # type: ignore[no-untyped-call]
    doc.save(path)  # type: ignore[no-untyped-call]
    doc.close()  # type: ignore[no-untyped-call]
    return path


def _ranks(results: list[dict[str, object]]) -> dict[str, int]:
    return {str(r["chunk_id"]): i for i, r in enumerate(results, start=1)}


def test_hybrid_scores_are_summed_reciprocal_ranks_of_both_rankings(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    dense = _ranks(retrieve(QUERY, k=50, embed=fake_embed))
    bm25 = _ranks(retrieve(QUERY, k=50, embed=fake_embed, retriever="bm25"))
    hybrid = retrieve(QUERY, k=10, embed=fake_embed, retriever="hybrid")
    assert hybrid
    for r in hybrid:
        cid = r["chunk_id"]
        expected = sum(1 / (60 + ranks[cid]) for ranks in (dense, bm25) if cid in ranks)
        assert r["score"] == pytest.approx(expected)


def test_chunk_found_by_one_retriever_still_appears(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    bm25_ids = {r["chunk_id"] for r in retrieve(QUERY, k=50, embed=fake_embed, retriever="bm25")}
    hybrid = retrieve(QUERY, k=50, embed=fake_embed, retriever="hybrid")
    hybrid_ids = {r["chunk_id"] for r in hybrid}
    assert len(hybrid_ids) == 4  # dense returns every chunk, bm25 only two
    assert len(bm25_ids) == 2
    assert bm25_ids < hybrid_ids


def test_chunk_ranked_well_by_both_outranks_chunk_ranked_well_by_one(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    # dense ranks page 2 first, bm25 ranks page 1 first; both find both
    hybrid = retrieve(QUERY, k=4, embed=fake_embed, retriever="hybrid")
    assert {hybrid[0]["page"], hybrid[1]["page"]} == {1, 2}
    assert all(r["page"] in (3, 4) for r in hybrid[2:])
    assert hybrid[1]["score"] > hybrid[2]["score"]


def test_results_have_exact_keys_sorted_and_capped_at_k(
    tmp_path: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([make_pdf(tmp_path / "rare.pdf", RARE_PAGES)], embed=fake_embed)
    results = retrieve(QUERY, k=3, embed=fake_embed, retriever="hybrid")
    assert len(results) == 3
    for r in results:
        assert set(r) == {"chunk_id", "text", "source", "page", "score"}
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_never_ingested_collection_raises_value_error(
    chroma_dir: Path, fake_embed: Embedder
) -> None:
    with pytest.raises(ValueError):
        retrieve("x", k=3, collection="nothing", embed=fake_embed, retriever="hybrid")


def test_ingested_collection_with_no_matches_returns_empty_list(
    chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([], embed=fake_embed, collection="emptyc")
    assert retrieve("x", k=3, collection="emptyc", embed=fake_embed, retriever="hybrid") == []
