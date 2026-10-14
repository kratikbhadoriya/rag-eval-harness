"""Default local embedder: Chroma's bundled all-MiniLM-L6-v2 (ONNX), no API key needed."""

from collections.abc import Callable
from typing import Any

from dotenv import find_dotenv, load_dotenv

Embedder = Callable[[list[str]], list[list[float]]]
"""Maps texts to one vector each; inject a fake in tests."""

_model: Any = None


def default_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with the bundled model. Loads `.env` and the model on first use."""
    global _model
    if _model is None:
        load_dotenv(find_dotenv(usecwd=True))
        from chromadb.utils.embedding_functions import onnx_mini_lm_l6_v2

        _model = onnx_mini_lm_l6_v2.ONNXMiniLM_L6_V2()
    return [[float(x) for x in vec] for vec in _model(texts)]
