#!/usr/bin/env python3
"""Sequential, leakage-free project proposal assessment for NASA93."""
import argparse
import csv
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .nasa93_effort import TARGET, _frame, _usable, build_pipeline, predictor_names
from .nasa93_summary import read_csv


def assess_arrivals(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Assess CSV-order arrivals; the order is a demonstration sequence, not a verified timeline."""
    rows = list(rows)
    names = predictor_names(columns)
    if not names:
        raise ValueError("input has no usable descriptors")
    history: List[Mapping[str, Any]] = []
    valid_seen = 0
    model_errors: List[float] = []
    baseline_errors: List[float] = []
    results: List[Dict[str, Any]] = []
    for arrival, row in enumerate(rows, 1):
        effort = row.get(TARGET)
        finite_effort = isinstance(effort, (int, float)) and not isinstance(effort, bool) and math.isfinite(float(effort))
        result: Dict[str, Any] = {"arrival": arrival, "identifier": row.get("idX", arrival),
                                  "prior_training_count": len(history), "prediction": None, "baseline": None,
                                  "revealed_effort": effort, "absolute_error": None, "baseline_absolute_error": None,
                                  "cumulative_mae": None, "baseline_cumulative_mae": None, "status": "scored"}
        if not finite_effort:
            result["status"] = "invalid_effort_not_revealed"
            results.append(result)
            continue
        if valid_seen < 2:
            result["status"] = "warm-up"
            valid_seen += 1
            history.append(row)
            results.append(result)
            continue
        model = build_pipeline(names)
        model.fit(_frame(history, names), [float(item[TARGET]) for item in history])
        prediction = float(model.predict(_frame([row], names))[0])
        baseline = sum(float(item[TARGET]) for item in history) / len(history)
        actual = float(effort)
        model_error, baseline_error = abs(prediction - actual), abs(baseline - actual)
        model_errors.append(model_error); baseline_errors.append(baseline_error)
        result.update(prediction=prediction, baseline=baseline, absolute_error=model_error,
                      baseline_absolute_error=baseline_error,
                      cumulative_mae=sum(model_errors) / len(model_errors),
                      baseline_cumulative_mae=sum(baseline_errors) / len(baseline_errors))
        history.append(row)
        results.append(result)
    return results


def assess_csv(path: str) -> List[Dict[str, Any]]:
    data = read_csv(path)
    return assess_arrivals(data["rows"], data["columns"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free sequential NASA93 assessments")
    parser.add_argument("csv_path")
    args = parser.parse_args()
    results = assess_csv(args.csv_path)
    fields = ["arrival", "identifier", "prior_training_count", "prediction", "baseline", "revealed_effort",
              "absolute_error", "baseline_absolute_error", "cumulative_mae", "baseline_cumulative_mae", "status"]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader(); writer.writerows(results)


if __name__ == "__main__":
    main()
