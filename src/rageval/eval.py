"""Run the Golden set through generate + Judge and write results.json / results.md.

`python -m rageval.eval` is the CLI behind `make eval`.
"""

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from rageval.generate import generate

METRICS = ("faithfulness", "context_recall")

Generator = Callable[[str, int], dict[str, Any]]
"""(question, k) -> {"answer": str, "contexts": [{"text": ...}, ...]}."""

Judge = Callable[[dict[str, Any]], dict[str, float]]
"""Scores one sample {question, answer, contexts: list[str], reference}.

Returns {"faithfulness", "context_recall", "input_tokens", "output_tokens", "cost"}; a metric
the Judge could not score is NaN.
"""

_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


class JudgeError(RuntimeError):
    """The Judge returned NaN for a sample even after one retry.

    `input_tokens`, `output_tokens` and `cost` hold what the run had spent when it failed.
    """

    def __init__(self, message: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        super().__init__(message)
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


def _score(
    sample: dict[str, Any], judge: Judge, sample_id: str, tally: dict[str, Any]
) -> dict[str, float]:
    """Judge one sample, retrying once on NaN; tokens and cost of every attempt are tallied."""
    for attempt in (1, 2):
        out = judge(sample)
        tally["input"] += int(out.get("input_tokens", 0))
        tally["output"] += int(out.get("output_tokens", 0))
        tally["cost"] += float(out.get("cost", 0.0))
        scores = {m: float(out[m]) for m in METRICS}
        if not any(math.isnan(v) for v in scores.values()):
            return scores
        tally["nan"] += 1
        if attempt == 2:
            raise JudgeError(
                f"Judge returned NaN twice for sample {sample_id!r}",
                tally["input"],
                tally["output"],
                tally["cost"],
            )
    raise AssertionError("unreachable")


def run(
    golden_path: Path,
    k: int,
    sample: int | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
    collection: str = "default",
) -> dict[str, Any]:
    """Evaluate the answerable Golden set rows and return the results dict.

    `sample` is accepted for the stratified sampling of a later ticket; for now all answerable
    rows are always run. Unanswerable rows are skipped (counted in `skipped_unanswerable`).

    Results: k, sample, metrics (mean per metric), samples (id, type, question, answer (raw),
    contexts, reference, one score per metric), nan_count, judge_tokens {input, output},
    judge_cost_usd, skipped_unanswerable, refusal (None until the refusal ticket fills it).

    Raises JudgeError if a sample is still NaN after one retry.
    """
    if generator is None:

        def generator(question: str, k: int) -> dict[str, Any]:
            return generate(question, k, collection=collection)

    if judge is None:
        judge = ragas_judge()
    tally: dict[str, Any] = {"input": 0, "output": 0, "cost": 0.0, "nan": 0}
    rows = _load_golden(golden_path)
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
                **scores,
            }
        )
    means = {
        m: (sum(s[m] for s in samples) / len(samples) if samples else math.nan) for m in METRICS
    }
    return {
        "k": k,
        "sample": sample,
        "metrics": means,
        "samples": samples,
        "nan_count": tally["nan"],
        "judge_tokens": {"input": tally["input"], "output": tally["output"]},
        "judge_cost_usd": tally["cost"],
        "skipped_unanswerable": len(rows) - len(answerable),
        "refusal": None,
    }


def format_markdown(results: dict[str, Any]) -> str:
    """Render a results dict as the results.md text. Pure: no I/O."""
    m = results["metrics"]
    tokens = results["judge_tokens"]
    lines = [
        "# Evaluation results",
        "",
        f"k = {results['k']}, answerable samples = {len(results['samples'])}, "
        f"skipped unanswerable = {results['skipped_unanswerable']}",
        "",
        "| Metric | Mean |",
        "| --- | --- |",
        f"| Faithfulness | {m['faithfulness']:.3f} |",
        f"| Context recall | {m['context_recall']:.3f} |",
        "",
        "| Id | Type | Faithfulness | Context recall |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {s['id']} | {s['type']} | {s['faithfulness']:.2f} | {s['context_recall']:.2f} |"
        for s in results["samples"]
    ]
    lines += [
        "",
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


def main(
    argv: list[str] | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="rageval.eval", description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("golden/golden.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=None, help="accepted; not yet implemented")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv(usecwd=True))
    needed = (("OPENAI_API_KEY",) if judge is None else ()) + (
        ("GROQ_API_KEY",) if generator is None else ()
    )
    missing = [v for v in needed if not os.environ.get(v)]
    if missing:
        print(
            f"error: missing {', '.join(missing)} (set in .env or the environment)", file=sys.stderr
        )
        return 2
    if args.sample is not None:
        print("note: --sample is not implemented yet; running all answerable rows", file=sys.stderr)
    try:
        results = run(args.golden, args.k, args.sample, generator, judge, args.collection)
    except JudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _print_totals(exc.input_tokens, exc.output_tokens, exc.cost)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "results.md").write_text(format_markdown(results), encoding="utf-8")
    t = results["judge_tokens"]
    _print_totals(t["input"], t["output"], results["judge_cost_usd"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
