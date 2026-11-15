"""Behavior of rageval.retrieve through its public functions."""

from pathlib import Path

from rageval import ingest, retrieve
from rageval.embed import Embedder


def test_query_sharing_words_returns_that_chunk_first_with_page(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([multi_pdf], embed=fake_embed)
    results = retrieve("Satellites", k=3, embed=fake_embed)
    assert results[0]["page"] == 3
    assert results[0]["source"] == "multi.pdf"
    assert "Satellites" in results[0]["text"]


def test_results_have_exact_keys_at_most_k_and_sorted_descending(
    multi_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([multi_pdf], embed=fake_embed)
    results = retrieve("Rivers", k=2, embed=fake_embed)
    assert len(results) == 2
    for r in results:
        assert set(r) == {"chunk_id", "text", "source", "page", "score"}
        assert r["chunk_id"].startswith("multi.pdf:")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_score_is_one_minus_cosine_distance(
    single_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([single_pdf], embed=fake_embed)
    text = retrieve("anything", k=1, embed=fake_embed, retriever="dense")[0]["text"]
    # Querying with the chunk's own text: cosine distance 0, so score 1.
    assert abs(retrieve(text, k=1, embed=fake_embed, retriever="dense")[0]["score"] - 1.0) < 1e-6
    # A query sharing no words with the chunk: distance near 1, score near 0.
    other = retrieve("zzqx", k=1, embed=fake_embed, retriever="dense")[0]["score"]
    assert -0.01 < other < 0.5


def test_k_larger_than_corpus_returns_all(
    single_pdf: Path, chroma_dir: Path, fake_embed: Embedder
) -> None:
    ingest([single_pdf], embed=fake_embed)
    assert len(retrieve("cats", k=10, embed=fake_embed)) == 1
