"""Chroma persistent client shared by ingest and retrieve."""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI


def chroma_client() -> ClientAPI:
    """Open the persistent Chroma store at RAGEVAL_CHROMA_DIR (default ./.chroma)."""
    return chromadb.PersistentClient(path=os.environ.get("RAGEVAL_CHROMA_DIR", "./.chroma"))


def _chunk_list_path(collection: str) -> Path:
    base = Path(os.environ.get("RAGEVAL_CHROMA_DIR", "./.chroma"))
    return base / "rageval_chunks" / f"{collection}.jsonl"


def save_chunks(collection: str, chunks: list[dict[str, Any]]) -> None:
    """Merge chunks into the collection's saved Chunk list by chunk_id (upsert semantics).

    Creates the (possibly empty) list file if missing, mirroring get_or_create_collection.
    """
    merged = {c["chunk_id"]: c for c in (load_chunks(collection) or [])}
    for chunk in chunks:
        merged[chunk["chunk_id"]] = {
            key: chunk[key] for key in ("chunk_id", "text", "source", "page")
        }
    path = _chunk_list_path(collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in merged.values()), encoding="utf-8")


def load_chunks(collection: str) -> list[dict[str, Any]] | None:
    """Return the saved Chunk list, or None if the collection was never ingested."""
    path = _chunk_list_path(collection)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def chunk_list_stamp(collection: str) -> tuple[str, int, int]:
    """Identify the current saved Chunk list file, to know when a cached index is stale."""
    path = _chunk_list_path(collection)
    st = path.stat()
    return (str(path.resolve()), st.st_mtime_ns, st.st_size)
