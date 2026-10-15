"""Smoke tests: the pipeline works offline with no secrets."""

import os
import socket
from pathlib import Path

import pytest

from rageval import ingest, retrieve
from rageval.embed import Embedder


def test_ingest_and_retrieve_need_no_network_or_keys(
    single_pdf: Path,
    chroma_dir: Path,
    fake_embed: Embedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert "GROQ_API_KEY" not in os.environ
    assert ingest([single_pdf], embed=fake_embed) == 1
    assert "Cats purr" in retrieve("cats", k=1, embed=fake_embed)[0]["text"]
