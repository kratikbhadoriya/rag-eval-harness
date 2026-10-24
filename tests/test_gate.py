import json
import math
from pathlib import Path
from typing import Any

import pytest

from rageval.gate import check, main

THRESHOLDS = {"faithfulness": 0.80, "context_recall": 0.70, "baseline_refusal_count": 5}


def _results(faith: float = 0.9, recall: float = 0.9, refusals: int = 5) -> dict[str, Any]:
    return {
        "metrics": {"faithfulness": faith, "context_recall": recall, "answer_relevancy": 0.0},
        "refusal": {"count": refusals, "total": 5},
        "rows_total": 30,
        "rows_scored": 30,
    }


def test_passes_when_all_checks_clear() -> None:
    assert check(_results(), THRESHOLDS) == {"passed": True, "reasons": []}


@pytest.mark.parametrize("metric,threshold", [("faith", 0.80), ("recall", 0.70)])
def test_metric_boundary(metric: str, threshold: float) -> None:
    at = check(_results(**{metric: threshold}), THRESHOLDS)
    under = check(_results(**{metric: threshold - 0.001}), THRESHOLDS)
    over = check(_results(**{metric: threshold + 0.001}), THRESHOLDS)
    assert at["passed"] and over["passed"]
    assert not under["passed"]
    assert len(under["reasons"]) == 1


def test_metrics_are_checked_independently() -> None:
    res = check(_results(faith=0.5, recall=0.5), THRESHOLDS)
    assert len(res["reasons"]) == 2
    assert "faithfulness" in res["reasons"][0] and "0.500" in res["reasons"][0]
    assert "0.800" in res["reasons"][0]
    assert "context_recall" in res["reasons"][1]


def test_answer_relevancy_is_not_gated() -> None:
    results = _results()
    results["metrics"]["answer_relevancy"] = 0.0
    assert check(results, THRESHOLDS)["passed"]


@pytest.mark.parametrize("baseline,required", [(5, 4), (4, 3), (3, 2), (2, 1)])
def test_refusal_boundary(baseline: int, required: int) -> None:
    th = {**THRESHOLDS, "baseline_refusal_count": baseline}
    assert check(_results(refusals=required), th)["passed"]
    assert check(_results(refusals=required + 1), th)["passed"]
    under = check(_results(refusals=required - 1), th)
    assert not under["passed"]
    assert "refusal" in under["reasons"][0]


def test_refusal_rule_caps_at_four() -> None:
    th = {**THRESHOLDS, "baseline_refusal_count": 9}
    assert check(_results(refusals=4), th)["passed"]
    assert not check(_results(refusals=3), th)["passed"]


def test_missing_or_nan_metrics_fail() -> None:
    res = _results()
    del res["metrics"]["faithfulness"]
    res["metrics"]["context_recall"] = math.nan
    out = check(res, THRESHOLDS)
    assert not out["passed"]
    assert len(out["reasons"]) == 2
    assert all("missing" in r for r in out["reasons"])


def test_missing_refusal_fails() -> None:
    res = _results()
    del res["refusal"]
    assert not check(res, THRESHOLDS)["passed"]


def test_missing_threshold_fails() -> None:
    out = check(_results(), {"faithfulness": 0.8})
    assert not out["passed"]
    assert len(out["reasons"]) == 2


def test_partial_run_fails() -> None:
    res = _results() | {"sample": 10, "rows_scored": 10}
    out = check(res, THRESHOLDS)
    assert not out["passed"]
    assert "partial" in out["reasons"][0] and "10 of 30" in out["reasons"][0]


def test_results_without_row_counts_fail() -> None:
    res = _results()
    del res["rows_total"], res["rows_scored"]
    out = check(res, THRESHOLDS)
    assert not out["passed"]
    assert "rows_total" in out["reasons"][0]


def _write(tmp_path: Path, results: dict[str, Any]) -> list[str]:
    (tmp_path / "r.json").write_text(json.dumps(results))
    (tmp_path / "t.json").write_text(json.dumps(THRESHOLDS))
    return ["--results", str(tmp_path / "r.json"), "--thresholds", str(tmp_path / "t.json")]


def test_cli_exit_codes_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_write(tmp_path, _results())) == 0
    assert main(_write(tmp_path, _results(faith=0.1))) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "faithfulness" in out and "0.100" in out and "0.800" in out


def test_cli_missing_file_fails(tmp_path: Path) -> None:
    args = ["--results", str(tmp_path / "no.json"), "--thresholds", str(tmp_path / "no2.json")]
    assert main(args) == 1
