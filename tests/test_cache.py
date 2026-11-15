"""Retrieval cache: repeat runs skip the embedder, any key change misses, generation is live."""

from pathlib import Path
from typing import Any

import pytest

from rageval.cache import cached_retriever, corpus_checksum, embedder_name
from rageval.embed import Embedder, default_embed
from rageval.generate import generate
from rageval.ingest import ingest


class CountingEmbedder:
    def __init__(self, inner: Embedder, name: str = "counting") -> None:
        self.inner = inner
        self.name = name
        self.calls = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return self.inner(texts)


@pytest.fixture
def env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chroma_dir: Path,
    single_pdf: Path,
    fake_embed: Embedder,
) -> tuple[Path, CountingEmbedder]:
    monkeypatch.setenv("RAGEVAL_CACHE_DIR", str(tmp_path / "cache"))
    ingest([single_pdf], collection="docs", embed=fake_embed)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha")
    return corpus, CountingEmbedder(fake_embed)


def _hits(corpus: Path, emb: CountingEmbedder, k: int = 2, **kw: Any) -> list[dict[str, Any]]:
    retrieve = cached_retriever("docs", embed=emb, corpus_dir=corpus, **kw)
    return retrieve("what is this?", k)


def test_repeat_run_hits_cache_without_embedding(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    first = _hits(corpus, emb)
    assert emb.calls == 1
    second = _hits(corpus, emb)
    assert emb.calls == 1
    assert second == first


def test_changed_k_misses(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    _hits(corpus, emb)
    _hits(corpus, emb, k=1)
    assert emb.calls == 2


@pytest.mark.parametrize("change", [{"chunk_size": 999}, {"chunk_overlap": 7}])
def test_changed_chunk_settings_miss(
    env: tuple[Path, CountingEmbedder], change: dict[str, Any]
) -> None:
    corpus, emb = env
    _hits(corpus, emb)
    _hits(corpus, emb, **change)
    assert emb.calls == 2


def test_changed_corpus_misses(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    _hits(corpus, emb)
    (corpus / "a.txt").write_text("beta")
    _hits(corpus, emb)
    assert emb.calls == 2


def test_changed_embedder_name_misses(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    _hits(corpus, emb)
    emb.name = "other"
    _hits(corpus, emb)
    assert emb.calls == 2


def test_generation_is_never_cached(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    retrieve = cached_retriever("docs", embed=emb, corpus_dir=corpus)
    calls: list[Any] = []

    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": f"answer {len(calls)}"}}]}

    a = generate("what is this?", 2, collection="docs", completion=completion, retriever=retrieve)
    b = generate("what is this?", 2, collection="docs", completion=completion, retriever=retrieve)
    assert emb.calls == 1
    assert len(calls) == 2
    assert (a["answer"], b["answer"]) == ("answer 1", "answer 2")


def test_missing_corpus_dir_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus"):
        corpus_checksum(tmp_path / "nope")


def test_default_embedder_name_includes_model_identity() -> None:
    assert "all-MiniLM-L6-v2" in embedder_name(default_embed)


def test_changed_retriever_misses_and_repeat_hits(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    _hits(corpus, emb)
    assert emb.calls == 1
    _hits(corpus, emb, retriever="hybrid")
    assert emb.calls == 2
    _hits(corpus, emb, retriever="hybrid")
    _hits(corpus, emb)
    assert emb.calls == 2


def test_bm25_run_is_not_served_dense_results(env: tuple[Path, CountingEmbedder]) -> None:
    corpus, emb = env
    dense = _hits(corpus, emb)
    bm25 = _hits(corpus, emb, retriever="bm25")
    assert emb.calls == 1
    assert [r["score"] for r in bm25] != [r["score"] for r in dense]
