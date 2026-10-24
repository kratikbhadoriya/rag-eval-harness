.PHONY: install lint fmt typecheck test check eval

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
	env -u GROQ_API_KEY uv run pytest

check: lint typecheck test

eval:
	uv run python -m rageval.eval $(if $(SAMPLE),--sample $(SAMPLE))
