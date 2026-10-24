"""Golden set and frozen corpus checks: pure file reads, no network, no embedder."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "data" / "corpus"
GOLDEN = ROOT / "golden" / "golden.jsonl"

TYPES = {"factual", "multi_hop", "comparative", "unanswerable"}
STATUSES = {"verified", "corrected"}
REQUIRED = {
    "id",
    "question",
    "reference_answer",
    "type",
    "source_file",
    "source_page",
    "source_section",
    "evidence",
    "status",
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def corpus_text() -> dict[str, list[str]]:
    """Page texts per PDF, extracted once."""
    out: dict[str, list[str]] = {}
    for pdf in sorted(CORPUS.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            out[pdf.name] = [page.get_text() for page in doc]
    return out


def test_corpus_is_frozen() -> None:
    lines = (CORPUS / "SHA256SUMS").read_text().splitlines()
    expected = {name: digest for digest, name in (line.split(maxsplit=1) for line in lines)}
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CORPUS.glob("*.pdf"))
    }
    assert len(expected) == 8
    assert actual == expected


def test_thirty_rows(rows: list[dict]) -> None:
    assert len(rows) == 30


def test_ids_unique(rows: list[dict]) -> None:
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_schema(rows: list[dict]) -> None:
    for r in rows:
        assert r.keys() >= REQUIRED, r["id"]
        assert r["type"] in TYPES, r["id"]
        assert r["status"] in STATUSES, r["id"]
        for key in ("id", "question", "reference_answer", "evidence"):
            assert isinstance(r[key], str) and r[key].strip(), (r["id"], key)
        for key in ("source_file", "source_page", "source_section"):
            assert isinstance(r[key], list) and r[key], (r["id"], key)
            assert all(isinstance(v, str) for v in r[key]), (r["id"], key)
        for page in r["source_page"]:
            assert page.isdigit() or (page == "n/a" and r["type"] == "unanswerable"), r["id"]


def test_source_files_exist(rows: list[dict]) -> None:
    for r in rows:
        for name in r["source_file"]:
            assert (CORPUS / name).is_file(), (r["id"], name)


def test_source_pages_in_range(rows: list[dict], corpus_text: dict[str, list[str]]) -> None:
    for r in rows:
        if r["type"] == "unanswerable":
            continue
        n_pages = min(len(corpus_text[f]) for f in r["source_file"])
        for page in r["source_page"]:
            assert 1 <= int(page) <= n_pages, (r["id"], page)


def test_at_least_five_unanswerable(rows: list[dict]) -> None:
    assert Counter(r["type"] for r in rows)["unanswerable"] >= 5
