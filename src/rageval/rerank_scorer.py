"""Default local cross-encoder scorer for reranking (optional `rerank` dependency group)."""

from typing import Any

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: Any = None


def default_score(query: str, texts: list[str]) -> list[float]:
    """Score each text against the query; returns one float per text, in input order.

    Higher means more relevant. These are raw cross-encoder scores, a separate scale from any
    Retriever's Score. Loads the model on first non-empty call only (never at import);
    raises ImportError telling you to sync the `rerank` group if it is not installed.
    """
    global _model
    if not texts:
        return []
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Reranking needs the optional `rerank` dependency group: "
                "run `uv sync --group rerank`."
            ) from e
        _model = CrossEncoder(MODEL_NAME)
    return [float(s) for s in _model.predict([(query, t) for t in texts])]
