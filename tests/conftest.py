"""Shared fixtures: deterministic offline fake embedder, fixture PDFs, temp Chroma dir."""

import hashlib
import math
import re
from pathlib import Path

import pytest

from rageval.embed import Embedder

DIM = 64


def _embed_one(text: str) -> list[float]:
    vec = [0.0] * DIM
    for token in re.findall(r"\w+", text.lower()):
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def fake_embed() -> Embedder:
    """Hashed bag-of-words embedder: shared words mean nearby vectors. No model, no network."""

    def embed(texts: list[str]) -> list[list[float]]:
        return [_embed_one(t) for t in texts]

    return embed


@pytest.fixture
def chroma_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "chroma"
    monkeypatch.setenv("RAGEVAL_CHROMA_DIR", str(d))
    return d


@pytest.fixture
def single_pdf() -> Path:
    return Path(__file__).parent / "fixtures" / "single.pdf"


@pytest.fixture
def multi_pdf() -> Path:
    return Path(__file__).parent / "fixtures" / "multi.pdf"


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see an API key, whatever the developer's environment holds."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
