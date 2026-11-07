.PHONY: install lint fmt typecheck test check ingest eval gate

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	env -u OPENAI_API_KEY uv run pytest

check: lint typecheck test

ingest:
	uv run python -c "from pathlib import Path; from rageval.ingest import ingest; print(ingest(sorted(Path('data/corpus').glob('*.pdf'))), 'chunks')"

gate:
	uv run python -m rageval.gate

eval:
	uv run python -m rageval.eval $(if $(SAMPLE),--sample $(SAMPLE))
