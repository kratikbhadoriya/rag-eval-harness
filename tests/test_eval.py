"""Behavior of rageval.eval.run and format_markdown with fake generator and judge."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from rageval.eval import JudgeError, format_markdown, main, run
from rageval.generate import REFUSAL_PHRASE


def _write_golden(tmp_path: Path) -> Path:
    rows = [
        {"id": "f01", "question": "Q1?", "reference_answer": "R1", "type": "factual"},
        {"id": "u01", "question": "Q2?", "reference_answer": "none", "type": "unanswerable"},
        {"id": "f02", "question": "Q3?", "reference_answer": "R3", "type": "factual"},
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, question: str, k: int) -> dict[str, Any]:
        self.calls.append((question, k))
        return {
            "answer": f"Answer to {question} [1].",
            "contexts": [{"text": "ctx one"}, {"text": "ctx two"}],
        }


class FakeJudge:
    """Scores per question; `nan_first` lists questions whose first attempt is NaN."""

    def __init__(self, nan_first: tuple[str, ...] = (), always_nan: tuple[str, ...] = ()) -> None:
        self.nan_first = set(nan_first)
        self.always_nan = set(always_nan)
        self.samples: list[dict[str, Any]] = []

    def __call__(self, sample: dict[str, Any]) -> dict[str, float]:
        self.samples.append(sample)
        q = sample["question"]
        seen = sum(1 for s in self.samples if s["question"] == q)
        nan = q in self.always_nan or (q in self.nan_first and seen == 1)
        faith = math.nan if nan else (1.0 if q == "Q1?" else 0.5)
        return {
            "faithfulness": faith,
            "context_recall": 0.5 if q == "Q1?" else 1.0,
            "input_tokens": 100,
            "output_tokens": 10,
            "cost": 0.001,
        }


def test_scores_answerable_rows_and_skips_unanswerable(tmp_path: Path) -> None:
    gen, judge = FakeGenerator(), FakeJudge()
    res = run(_write_golden(tmp_path), k=3, generator=gen, judge=judge)
    assert [s["id"] for s in res["samples"]] == ["f01", "f02"]
    assert gen.calls == [("Q1?", 3), ("Q3?", 3), ("Q2?", 3)]
    assert res["k"] == 3
    assert res["skipped_unanswerable"] == 1
    assert res["metrics"] == {"faithfulness": 0.75, "context_recall": 0.75}
    assert res["samples"][0]["faithfulness"] == 1.0
    assert res["nan_count"] == 0
    assert res["judge_tokens"] == {"input": 200, "output": 20}
    assert res["judge_cost_usd"] == pytest.approx(0.002)


class RefusingGenerator(FakeGenerator):
    """Refuses (with a citation-free phrase inside a longer reply) on unanswerable Q2."""

    def __call__(self, question: str, k: int) -> dict[str, Any]:
        out = super().__call__(question, k)
        if question == "Q2?":
            out["answer"] = f"Sorry. {REFUSAL_PHRASE}"
        return out


def test_unanswerable_rows_go_to_generator_not_judge_and_refusals_are_counted(
    tmp_path: Path,
) -> None:
    gen, judge = RefusingGenerator(), FakeJudge()
    res = run(_write_golden(tmp_path), k=3, generator=gen, judge=judge)
    assert ("Q2?", 3) in gen.calls
    assert all(s["question"] != "Q2?" for s in judge.samples)
    assert [s["id"] for s in res["samples"]] == ["f01", "f02"]
    assert res["metrics"] == {"faithfulness": 0.75, "context_recall": 0.75}
    assert res["refusal"] == {"count": 1, "total": 1}


def test_non_refusing_answer_to_unanswerable_is_not_counted(tmp_path: Path) -> None:
    res = run(_write_golden(tmp_path), k=3, generator=FakeGenerator(), judge=FakeJudge())
    assert res["refusal"] == {"count": 0, "total": 1}


def test_citation_markers_stripped_for_judge_but_kept_in_results(tmp_path: Path) -> None:
    judge = FakeJudge()
    res = run(_write_golden(tmp_path), k=2, generator=FakeGenerator(), judge=judge)
    assert judge.samples[0]["answer"] == "Answer to Q1?."
    assert judge.samples[0]["contexts"] == ["ctx one", "ctx two"]
    assert judge.samples[0]["reference"] == "R1"
    assert res["samples"][0]["answer"] == "Answer to Q1? [1]."


def test_nan_is_retried_once_and_counted(tmp_path: Path) -> None:
    judge = FakeJudge(nan_first=("Q3?",))
    res = run(_write_golden(tmp_path), k=2, generator=FakeGenerator(), judge=judge)
    assert res["nan_count"] == 1
    assert res["samples"][1]["faithfulness"] == 0.5
    assert res["judge_tokens"] == {"input": 300, "output": 30}  # retry tokens are billed too


def test_persistent_nan_fails_run_naming_sample(tmp_path: Path) -> None:
    judge = FakeJudge(always_nan=("Q3?",))
    with pytest.raises(JudgeError, match="f02"):
        run(_write_golden(tmp_path), k=2, generator=FakeGenerator(), judge=judge)
    assert sum(1 for s in judge.samples if s["question"] == "Q3?") == 2


def _results(tmp_path: Path) -> dict[str, Any]:
    return run(_write_golden(tmp_path), k=3, generator=FakeGenerator(), judge=FakeJudge())


def test_format_markdown_snapshot(tmp_path: Path) -> None:
    assert format_markdown(_results(tmp_path)) == (
        "# Evaluation results\n"
        "\n"
        "k = 3, answerable samples = 2, skipped unanswerable = 1\n"
        "\n"
        "| Metric | Mean |\n"
        "| --- | --- |\n"
        "| Faithfulness | 0.750 |\n"
        "| Context recall | 0.750 |\n"
        "\n"
        "| Id | Type | Faithfulness | Context recall |\n"
        "| --- | --- | --- | --- |\n"
        "| f01 | factual | 1.00 | 0.50 |\n"
        "| f02 | factual | 0.50 | 1.00 |\n"
        "\n"
        "Refusals on unanswerable: 0 of 1\n"
        "Judge NaN scores (retried): 0\n"
        "Judge tokens: 200 in, 20 out; cost $0.0020\n"
    )


def test_cli_writes_files_and_prints_judge_totals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    code = main(
        ["--golden", str(_write_golden(tmp_path)), "--k", "2", "--out-dir", str(out)],
        generator=FakeGenerator(),
        judge=FakeJudge(),
    )
    assert code == 0
    assert json.loads((out / "results.json").read_text())["metrics"]["faithfulness"] == 0.75
    assert "| Faithfulness | 0.750 |" in (out / "results.md").read_text()
    assert capsys.readouterr().out.strip() == "Judge tokens: 200 in, 20 out; cost $0.0020"


def test_cli_fails_naming_sample_and_still_prints_totals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--golden", str(_write_golden(tmp_path)), "--out-dir", str(tmp_path / "out")],
        generator=FakeGenerator(),
        judge=FakeJudge(always_nan=("Q3?",)),
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "f02" in captured.err
    assert "Judge tokens: 300 in, 30 out" in captured.out
    assert "metrics" not in json.loads((tmp_path / "out" / "results.json").read_text())


def test_cli_judge_failure_replaces_stale_results_with_failure_marker(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.json").write_text('{"metrics": {"faithfulness": 1.0}}')
    (out / "results.md").write_text("stale")
    code = main(
        ["--golden", str(_write_golden(tmp_path)), "--out-dir", str(out)],
        generator=FakeGenerator(),
        judge=FakeJudge(always_nan=("Q3?",)),
    )
    assert code == 1
    assert not (out / "results.md").exists()
    written = json.loads((out / "results.json").read_text())
    assert written.pop("judge_cost_usd") == pytest.approx(0.003)
    assert written == {
        "failed_sample": "f02",
        "nan_count": 2,
        "judge_tokens": {"input": 300, "output": 30},
    }


class UningestedGenerator:
    def __call__(self, question: str, k: int) -> dict[str, Any]:
        raise ValueError("collection 'default' has not been ingested")


def test_cli_never_ingested_collection_exits_2_with_ingest_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--golden", str(_write_golden(tmp_path)), "--out-dir", str(tmp_path / "out")],
        generator=UningestedGenerator(),
        judge=FakeJudge(),
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "make ingest" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "out" / "results.json").exists()


def test_cli_reports_missing_keys_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert main(["--golden", str(_write_golden(tmp_path))]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def _rows() -> list[dict[str, Any]]:
    counts = {"factual": 13, "multi_hop": 7, "comparative": 5, "unanswerable": 5}
    rows = [
        {"id": f"{t[0]}{i:02d}", "question": f"{t}{i}?", "reference_answer": "r", "type": t}
        for t, c in counts.items()
        for i in range(1, c + 1)
    ]
    return rows[::-1]  # shuffled input order must not matter


def _types(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["type"]] = out.get(r["type"], 0) + 1
    return out


def test_sample_rows_is_deterministic_and_ordered_by_id() -> None:
    from rageval.eval import sample_rows

    a, b = sample_rows(_rows(), 10), sample_rows(list(reversed(_rows())), 10)
    assert a == b
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)


def test_sample_rows_stratifies_by_type_with_unanswerable() -> None:
    from rageval.eval import sample_rows

    assert _types(sample_rows(_rows(), 10)) == {
        "factual": 4,
        "multi_hop": 2,
        "comparative": 2,
        "unanswerable": 2,
    }
    # n=3 would round unanswerable to 0; one is still kept
    got = _types(sample_rows(_rows(), 3))
    assert sum(got.values()) == 3 and got["unanswerable"] == 1


def test_sample_rows_runs_everything_when_n_covers_total_or_omitted() -> None:
    from rageval.eval import sample_rows

    full = sorted(_rows(), key=lambda r: r["id"])
    assert sample_rows(_rows(), None) == full
    assert sample_rows(_rows(), 30) == full
    assert sample_rows(_rows(), 99) == full


def test_run_samples_golden_rows(tmp_path: Path) -> None:
    gen, judge = FakeGenerator(), FakeJudge()
    path = tmp_path / "g.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _rows()) + "\n")
    res = run(path, k=2, sample=10, generator=gen, judge=judge)
    assert res["sample"] == 10
    assert len(res["samples"]) == 8
    assert res["refusal"]["total"] == 2
    assert res["skipped_unanswerable"] == 2


def test_results_record_rows_total_and_rows_scored(tmp_path: Path) -> None:
    full = run(_write_golden(tmp_path), k=2, generator=FakeGenerator(), judge=FakeJudge())
    assert (full["rows_total"], full["rows_scored"]) == (3, 3)
    part = run(_write_golden(tmp_path), k=2, sample=2, generator=FakeGenerator(), judge=FakeJudge())
    assert (part["rows_total"], part["rows_scored"]) == (3, 2)


def test_judge_model_differs_from_generator_model() -> None:
    from rageval.eval import JUDGE_MODEL
    from rageval.generate import DEFAULT_MODEL

    assert JUDGE_MODEL != DEFAULT_MODEL
    assert JUDGE_MODEL.split("/")[-1] != DEFAULT_MODEL.split("/")[-1]
