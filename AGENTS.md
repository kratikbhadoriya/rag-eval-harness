## Stack

Python 3.12, uv, chromadb (local persistent), pymupdf, langchain-text-splitters, litellm (OpenAI), ruff, strict mypy, pytest, pre-commit.

## Layout

- `src/rageval/ingest.py`, `retrieve.py`: plain functions over plain dicts
- `src/rageval/embed.py`: `Embedder` type and lazy `default_embed`
- `src/rageval/store.py`: shared Chroma persistent client and the saved Chunk list (JSONL)
- `src/rageval/generate.py`: retrieve top-k, answer via litellm (OpenAI gpt-4.1-mini, a different model from the gpt-4o-mini Judge); prompt in `src/rageval/prompts/answer.txt`
- `src/rageval/eval.py`: runs the Golden set through generate + Judge (`--retriever {dense,bm25,hybrid}`, default bm25), writes `results.json` / `results.md`
- `src/rageval/cache.py`: on-disk retrieval cache (`RAGEVAL_CACHE_DIR`), never caches generation or Judge calls
- `src/rageval/gate.py`: quality and Refusal gate over `results.json` and `eval_thresholds.json`
- `tests/`: offline tests with a fake embedder and small fixture PDFs
- `metrics.md`: metrics log; `GLOSSARY.md`: domain terms

## Commands

- `make install`, `make lint`, `make fmt`, `make typecheck`, `make test`, `make check`
- `make ingest` (ingest `data/corpus`), `make eval [SAMPLE=n]` (needs `OPENAI_API_KEY`), `make gate` (checks `results.json` against `eval_thresholds.json`)

## Rules

1. Functions take and return plain dicts and ints. Reason: keeps the contract trivially testable and framework-free.
2. No framework wrappers (LangChain chains/retrievers) around retrieval. Reason: they hide what is being measured.
3. Every metric change gets a row in `metrics.md`. Reason: quality changes must stay traceable.
4. Tests run with no network and no API keys; inject the fake embedder. Reason: CI is fast, free and deterministic.

## Default embedder

When `embed` is omitted, `rageval.embed.default_embed` uses Chroma's bundled ONNX all-MiniLM-L6-v2. On first real use it downloads the model (~80 MB) once to `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/`, so that first call needs network; later runs are offline. `.env` is loaded lazily inside the embedder, never at import. Tests never trigger the download: they stub the model class or pass the fake embedder.

## Config

`RAGEVAL_CHROMA_DIR` sets the Chroma persistence directory (default `./.chroma`). `ingest` also saves the exact Chunk list per Collection as `rageval_chunks/<collection>.jsonl` inside it (merged by `chunk_id`), which the `bm25` Retriever indexes (`retrieve(..., retriever="bm25"|"dense"|"hybrid")`); tests point it at a temp dir via the `chroma_dir` fixture. `OPENAI_API_KEY` is never needed by tests and is deleted for them by an autouse fixture.

`RAGEVAL_CACHE_DIR` sets the retrieval cache directory (default `./.eval_cache`, gitignored); tests use a temp dir. `rageval.cache.cached_retriever` caches `retrieve` output only (never generation or Judge), keyed on question, k, collection, Retriever, chunk size and overlap (`ingest.CHUNK_SIZE`/`CHUNK_OVERLAP`), a SHA-256 of the files in `data/corpus`, and the embedder name (its `name` attribute, else module-qualified function name). Delete the directory to clear it.

`make eval RETRIEVER=hybrid` (or `python -m rageval.eval --retriever {dense,bm25,hybrid}`, default `bm25`) picks the Retriever; `results.json` records `retriever` and `golden_set_version` (first 8 hex characters of the SHA-256 of the Golden set file).

## Gate thresholds

`make gate` runs `python -m rageval.gate`: reads `results.json` and the committed `eval_thresholds.json`, prints each failing check with metric, value and threshold, exits 1 on failure. Schema: `{"faithfulness": float, "context_recall": float, "baseline_refusal_count": int}`. Refusal gate requires `refusal.count >= min(4, baseline_refusal_count - 1)`. Missing metrics fail; answer relevancy is not gated. Changing the file needs a `metrics.md` row.

## Collections

`retrieve` from a collection that was never ingested raises `ValueError` (a visible mistake); an existing collection with no matches returns `[]`. Collection names must be 3+ characters (Chroma rule).

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (`kratikbhadoriya/rag-eval-harness`) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
