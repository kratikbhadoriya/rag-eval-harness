"""Chroma persistent client shared by ingest and retrieve."""

import os

import chromadb
from chromadb.api import ClientAPI


def chroma_client() -> ClientAPI:
    """Open the persistent Chroma store at RAGEVAL_CHROMA_DIR (default ./.chroma)."""
    return chromadb.PersistentClient(path=os.environ.get("RAGEVAL_CHROMA_DIR", "./.chroma"))
