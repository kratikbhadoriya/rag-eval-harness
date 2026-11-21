"""Run the Golden set through generate + Judge and write results.json / results.md.

`python -m rageval.eval` is the CLI behind `make eval`.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from rageval import rerank_scorer
from rageval.cache import cached_retriever
from rageval.generate import REFUSAL_PHRASE, generate
from rageval.rerank import CANDIDATES, Scorer, percentile, reranked_retriever
from rageval.retrieve import DEFAULT_RETRIEVER, RETRIEVERS, RetrieverName

METRICS = ("faithfulness", "context_recall")

Generator = Callable[[str, int], dict[str, Any]]
"""(question, k) -> {"answer": str, "contexts": [{"text": ...}, ...]}."""

Judge = Callable[[dict[str, Any]], dict[str, float]]
"""Scores one sample {question, answer, contexts: list[str], reference}.

Returns {"faithfulness", "context_recall", "input_tokens", "output_tokens", "cost"}; a metric
the Judge could not score is NaN.
"""

DEFAULT_K = 10

DEFAULT_GOLDEN = Path("golden/golden.jsonl")


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a new config with every default filled in; the one place defaults live.

    Keys: golden (Golden set path), k, sample (None runs everything), collection, retriever,
    rerank. Raises ValueError on an unknown key. `config` is not mutated.
    """
    defaults: dict[str, Any] = {
        "golden": DEFAULT_GOLDEN,
        "k": DEFAULT_K,
        "sample": None,
        "collection": "default",
        "retriever": DEFAULT_RETRIEVER,
        "rerank": False,
    }
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    return {**defaults, **config}


_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


class JudgeError(RuntimeError):
    """The Judge returned NaN for a sample even after one retry.

    `sample_id` names the failing sample; `nan_count`, `input_tokens`, `output_tokens` and `cost`
    hold what the run had seen and spent when it failed.
    """

    def __init__(
        self,
        message: str,
        sample_id: str,
        nan_count: int,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.nan_count = nan_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost


def strip_citations(answer: str) -> str:
    """Drop `[n]` markers; the spike's Faithfulness 0.0 on a supported answer is suspected to be
    caused by them, so the Judge never sees them. Results keep the raw answer."""
    return _CITATION.sub("", answer)


def _load_golden(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def sample_rows(rows: list[dict[str, Any]], size: int | None) -> list[dict[str, Any]]:
    """Deterministic slice of the Golden set: stratified by type, ordered by id, no randomness.

    `size` of None or >= len(rows) returns every row. Otherwise each type gets a share of `size`
    proportional to its size (largest remainder, ties by type name); within a type, rows are
    picked evenly spaced over id order. Unanswerable rows keep at least one slot when size >= 2, so
    the Refusal check stays meaningful (the slot is taken from the largest other share).
    """
    ordered = sorted(rows, key=lambda row: row["id"])
    if size is None or size >= len(ordered):
        return ordered
    size = max(size, 0)
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        strata.setdefault(str(row.get("type")), []).append(row)
    total = len(ordered)
    quota = {kind: size * len(group) // total for kind, group in strata.items()}
    by_remainder = sorted(strata, key=lambda kind: (-(size * len(strata[kind]) % total), kind))
    for kind in by_remainder[: size - sum(quota.values())]:
        quota[kind] += 1
    if "unanswerable" in strata and size >= 2 and quota["unanswerable"] == 0:
        donor = max(
            (kind for kind in strata if kind != "unanswerable"),
            key=lambda kind: (quota[kind], kind),
        )
        quota[donor] -= 1
        quota["unanswerable"] = 1
    picked = [
        group[i * len(group) // quota[kind]]
        for kind, group in strata.items()
        if quota[kind]
        for i in range(quota[kind])
    ]
    return sorted(picked, key=lambda row: row["id"])


def _score(
    sample: dict[str, Any], judge: Judge, sample_id: str, tally: dict[str, Any]
) -> dict[str, float]:
    """Judge one sample, retrying once on NaN; tokens and cost of every attempt are tallied."""
    for _attempt in (1, 2):
        out = judge(sample)
        tally["input"] += int(out.get("input_tokens", 0))
        tally["output"] += int(out.get("output_tokens", 0))
        tally["cost"] += float(out.get("cost", 0.0))
        scores = {metric: float(out[metric]) for metric in METRICS}
        if not any(math.isnan(value) for value in scores.values()):
            return scores
        tally["nan"] += 1
    raise JudgeError(
        f"Judge returned NaN twice for sample {sample_id!r}",
        sample_id,
        tally["nan"],
        tally["input"],
        tally["output"],
        tally["cost"],
    )


def run(
    golden_path: Path,
    k: int,
    sample: int | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
    collection: str = "default",
    retriever: RetrieverName = DEFAULT_RETRIEVER,
    rerank: bool = False,
    scorer: Scorer | None = None,
) -> dict[str, Any]:
    """Evaluate the answerable Golden set rows and return the results dict.

    `sample` limits the Golden set with `sample_rows` (None runs everything).
    Unanswerable rows go through the generator only (never the Judge) and are
    counted in `skipped_unanswerable` and `refusal`.

    Results: k, sample, retriever, golden_set_version (first 8 hex of the Golden set file's
    SHA-256), rows_total (Golden set size), rows_scored, metrics (mean per metric),
    samples (id, type, question, answer (raw),
    contexts, reference, one score per metric), nan_count, judge_tokens {input, output},
    judge_cost_usd, skipped_unanswerable, refusal {count, total} (answers with REFUSAL_PHRASE),
    answerable_refused {count, total} (answerable rows the generator refused: usually a retrieval
    miss, scored 0 Faithfulness).

    `rerank` wraps the default generator's cached retriever in a Reranker (top-20
    candidates, keep k; `scorer` defaults to the local cross-encoder) and adds a `rerank` block:
    candidates, k, latency_ms {n, p50, p95} (None percentiles when n is 0). Off, no block.

    Raises JudgeError if a sample is still NaN after one retry.
    """
    if rerank and generator is not None:
        raise ValueError(
            "rerank=True needs the default generator: an injected generator bypasses the "
            "Reranker, so no reranking would happen"
        )
    timings_ms: list[float] = []
    if generator is None:
        fetch = cached_retriever(collection, retriever=retriever)
        if rerank:
            fetch = reranked_retriever(
                fetch, scorer or rerank_scorer.score_pairs, timings_ms, candidates=CANDIDATES
            )

        def generator(question: str, k: int) -> dict[str, Any]:
            return generate(question, k, collection=collection, retriever=fetch)

    if judge is None:
        judge = ragas_judge()
    tally: dict[str, Any] = {"input": 0, "output": 0, "cost": 0.0, "nan": 0}
    golden = _load_golden(golden_path)
    rows = sample_rows(golden, sample)
    answerable = [r for r in rows if r.get("type") != "unanswerable"]
    samples: list[dict[str, Any]] = []
    for row in answerable:
        gen = generator(row["question"], k)
        contexts = [c["text"] for c in gen["contexts"]]
        scores = _score(
            {
                "question": row["question"],
                "answer": strip_citations(gen["answer"]),
                "contexts": contexts,
                "reference": row["reference_answer"],
            },
            judge,
            row["id"],
            tally,
        )
        samples.append(
            {
                "id": row["id"],
                "type": row.get("type"),
                "question": row["question"],
                "answer": gen["answer"],
                "contexts": contexts,
                "reference": row["reference_answer"],
                "refused": REFUSAL_PHRASE in gen["answer"],
                **scores,
            }
        )
    unanswerable = [r for r in rows if r.get("type") == "unanswerable"]
    refused = sum(REFUSAL_PHRASE in generator(r["question"], k)["answer"] for r in unanswerable)
    means = {
        m: (sum(s[m] for s in samples) / len(samples) if samples else math.nan) for m in METRICS
    }
    results: dict[str, Any] = {
        "k": k,
        "sample": sample,
        "retriever": retriever,
        "golden_set_version": hashlib.sha256(golden_path.read_bytes()).hexdigest()[:8],
        "rows_total": len(golden),
        "rows_scored": len(rows),
        "metrics": means,
        "samples": samples,
        "nan_count": tally["nan"],
        "judge_tokens": {"input": tally["input"], "output": tally["output"]},
        "judge_cost_usd": tally["cost"],
        "skipped_unanswerable": len(rows) - len(answerable),
        "refusal": {"count": refused, "total": len(unanswerable)},
        "answerable_refused": {
            "count": sum(s["refused"] for s in samples),
            "total": len(samples),
        },
    }
    if rerank:
        results["rerank"] = {
            "candidates": CANDIDATES,
            "k": k,
            "latency_ms": {
                "n": len(timings_ms),
                "p50": percentile(timings_ms, 50) if timings_ms else None,
                "p95": percentile(timings_ms, 95) if timings_ms else None,
            },
        }
    return results


def format_markdown(results: dict[str, Any]) -> str:
    """Render a results dict as the results.md text. Pure: no I/O."""
    means = results["metrics"]
    tokens = results["judge_tokens"]
    lines = [
        "# Evaluation results",
        "",
        f"k = {results['k']}, answerable samples = {len(results['samples'])}, "
        f"skipped unanswerable = {results['skipped_unanswerable']}",
        "",
        "| Metric | Mean |",
        "| --- | --- |",
        f"| Faithfulness | {means['faithfulness']:.3f} |",
        f"| Context recall | {means['context_recall']:.3f} |",
        "",
        "| Id | Type | Faithfulness | Context recall | Refused |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {s['id']} | {s['type']} | {s['faithfulness']:.2f} | {s['context_recall']:.2f} "
        f"| {'yes' if s['refused'] else ''} |"
        for s in results["samples"]
    ]
    lines += [
        "",
        f"Refusals on answerable: {results['answerable_refused']['count']} of "
        f"{results['answerable_refused']['total']}",
        f"Refusals on unanswerable: {results['refusal']['count']} of {results['refusal']['total']}",
        f"Judge NaN scores (retried): {results['nan_count']}",
        f"Judge tokens: {tokens['input']} in, {tokens['output']} out; "
        f"cost ${results['judge_cost_usd']:.4f}",
        "",
    ]
    return "\n".join(lines)


JUDGE_MODEL = "gpt-4o-mini"
_COST_PER_INPUT_TOKEN = 0.15e-6
_COST_PER_OUTPUT_TOKEN = 0.60e-6


def ragas_judge() -> Judge:
    """Real Judge: RAGAS Faithfulness + LLMContextRecall on gpt-4o-mini. Needs OPENAI_API_KEY.

    ragas and langchain are imported here, not at module import, so tests need neither.
    """
    load_dotenv(find_dotenv(usecwd=True))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env or the environment)")
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))

    def judge(sample: dict[str, Any]) -> dict[str, float]:
        data = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=sample["question"],
                    response=sample["answer"],
                    retrieved_contexts=sample["contexts"],
                    reference=sample["reference"],
                )
            ]
        )
        res: Any = evaluate(
            data,
            metrics=[Faithfulness(), LLMContextRecall()],
            llm=llm,
            token_usage_parser=get_token_usage_for_openai,
            show_progress=False,
        )
        scores = res.scores[0]
        usage = res.total_tokens()
        return {
            "faithfulness": float(scores["faithfulness"]),
            "context_recall": float(scores["context_recall"]),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost": float(
                res.total_cost(
                    cost_per_input_token=_COST_PER_INPUT_TOKEN,
                    cost_per_output_token=_COST_PER_OUTPUT_TOKEN,
                )
            ),
        }

    return judge


def _print_totals(input_tokens: int, output_tokens: int, cost: float) -> None:
    print(f"Judge tokens: {input_tokens} in, {output_tokens} out; cost ${cost:.4f}")


def _write_failure(out_dir: Path, exc: JudgeError) -> None:
    """Replace any earlier results with a metric-free failure marker, so the gate fails."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").unlink(missing_ok=True)
    marker = {
        "failed_sample": exc.sample_id,
        "nan_count": exc.nan_count,
        "judge_tokens": {"input": exc.input_tokens, "output": exc.output_tokens},
        "judge_cost_usd": exc.cost,
    }
    (out_dir / "results.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def main(
    argv: list[str] | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="rageval.eval", description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("golden/golden.jsonl"))
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"contexts per question, also the kept count with --rerank (default {DEFAULT_K})",
    )
    parser.add_argument("--sample", type=int, default=None, help="score n rows (stratified)")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--retriever", choices=RETRIEVERS, default=DEFAULT_RETRIEVER)
    parser.add_argument(
        "--rerank", action="store_true", help="cross-encoder rerank the retriever top-20, keep k"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    k = args.k if args.k is not None else DEFAULT_K

    load_dotenv(find_dotenv(usecwd=True))
    needs_key = judge is None or generator is None
    missing = [v for v in ("OPENAI_API_KEY",) if needs_key and not os.environ.get(v)]
    if missing:
        print(
            f"error: missing {', '.join(missing)} (set in .env or the environment)", file=sys.stderr
        )
        return 2
    try:
        results = run(
            args.golden,
            k,
            args.sample,
            generator,
            judge,
            args.collection,
            args.retriever,
            rerank=args.rerank,
        )
    except JudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _print_totals(exc.input_tokens, exc.output_tokens, exc.cost)
        _write_failure(args.out_dir, exc)
        return 1
    except json.JSONDecodeError:
        raise
    except ValueError as exc:
        print(
            f"error: {exc}\nrun `make ingest` first (collection {args.collection!r} has no chunks)",
            file=sys.stderr,
        )
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "results.md").write_text(format_markdown(results), encoding="utf-8")
    tokens = results["judge_tokens"]
    _print_totals(tokens["input"], tokens["output"], results["judge_cost_usd"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
