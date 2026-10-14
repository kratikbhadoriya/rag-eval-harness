"""Behavior of the default embedder, stubbed at the ONNX model boundary (offline)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from chromadb.utils.embedding_functions import onnx_mini_lm_l6_v2

from rageval import embed as embed_module
from rageval import ingest, retrieve


class StubModel:
    """Stands in for Chroma's bundled all-MiniLM-L6-v2 (no download, no onnxruntime)."""

    calls: ClassVar[list[list[str]]] = []

    def __call__(self, input: list[str]) -> list[list[float]]:
        StubModel.calls.append(list(input))
        return [[0.6, 0.8, 0.0] for _ in input]


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> type[StubModel]:
    StubModel.calls = []
    monkeypatch.setattr(embed_module, "_model", None)
    monkeypatch.setattr(onnx_mini_lm_l6_v2, "ONNXMiniLM_L6_V2", StubModel)
    return StubModel


def test_ingest_without_embed_uses_bundled_model(
    single_pdf: Path, chroma_dir: Path, stub_model: type[StubModel]
) -> None:
    assert ingest([single_pdf]) == 1
    assert len(stub_model.calls) == 1
    assert "Cats purr" in stub_model.calls[0][0]


def test_retrieve_without_embed_uses_bundled_model(
    single_pdf: Path, chroma_dir: Path, stub_model: type[StubModel]
) -> None:
    ingest([single_pdf])
    stub_model.calls = []
    results = retrieve("do cats purr?", k=1)
    assert stub_model.calls == [["do cats purr?"]]
    assert len(results) == 1
    assert "Cats purr" in results[0]["text"]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_dotenv_is_loaded_lazily_by_default_embedder(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("RAGEVAL_TEST_DOTENV=loaded\n")
    code = (
        "import os, rageval, rageval.embed as e;"
        "assert 'RAGEVAL_TEST_DOTENV' not in os.environ, 'loaded at import';"
        "import chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 as m;"
        "m.ONNXMiniLM_L6_V2 = lambda: (lambda texts: [[0.0] for _ in texts]);"
        "e.default_embed(['x']);"
        "assert os.environ['RAGEVAL_TEST_DOTENV'] == 'loaded'"
    )
    env = {k: v for k, v in os.environ.items() if k != "RAGEVAL_TEST_DOTENV"}
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
