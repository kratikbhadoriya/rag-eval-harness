"""Quality gate and Refusal gate over a results dict and a thresholds dict.

Thresholds schema (eval_thresholds.json): faithfulness, context_recall (floats) and
baseline_refusal_count (int). Answer relevancy is deliberately not gated (see ADR 0001).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

GATED_METRICS = ("faithfulness", "context_recall")
MAX_REQUIRED_REFUSALS = 4


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return float(value)


def check(results: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Return {"passed": bool, "reasons": [str]}; a missing metric or threshold fails."""
    reasons: list[str] = []
    metrics = results.get("metrics") or {}
    for name in GATED_METRICS:
        value = _number(metrics.get(name))
        threshold = _number(thresholds.get(name))
        if threshold is None:
            reasons.append(f"{name}: threshold missing from thresholds file")
        elif value is None:
            reasons.append(f"{name}: metric missing from results (threshold {threshold:.3f})")
        elif value < threshold:
            reasons.append(f"{name}: {value:.3f} is below threshold {threshold:.3f}")

    count = _number((results.get("refusal") or {}).get("count"))
    baseline = _number(thresholds.get("baseline_refusal_count"))
    if baseline is None:
        reasons.append("refusal: baseline_refusal_count missing from thresholds file")
    elif count is None:
        reasons.append("refusal: count missing from results")
    else:
        required = min(MAX_REQUIRED_REFUSALS, baseline - 1)
        if count < required:
            reasons.append(f"refusal: count {count:g} is below threshold {required:g}")
    return {"passed": not reasons, "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rageval.gate", description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    parser.add_argument("--thresholds", type=Path, default=Path("eval_thresholds.json"))
    args = parser.parse_args(argv)
    try:
        results = json.loads(args.results.read_text())
        thresholds = json.loads(args.thresholds.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read gate inputs: {exc}")
        return 1
    outcome = check(results, thresholds)
    if outcome["passed"]:
        print("PASS: quality gate and refusal gate")
        return 0
    for reason in outcome["reasons"]:
        print(f"FAIL: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
