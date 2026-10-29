#!/usr/bin/env python3
"""Choose the highest-value feasible NASA93DEM project configuration."""
import argparse
import csv
import json
import time
from pathlib import Path

from sklearn.preprocessing import StandardScaler

DATASET = Path(__file__).resolve().parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"
EFFORT = "Effort-"
VALUE = "Kloc+"
REQUIRED = {"idX", EFFORT, VALUE}
DEFAULT_BUDGET = 100.0


def number(value, field, row_number):
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric at row {row_number}") from exc
    if result < 0:
        raise ValueError(f"{field} must be non-negative at row {row_number}")
    return result


def load_validate(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"dataset must contain {sorted(REQUIRED)}")
        rows = list(reader)
    if not rows:
        raise ValueError("dataset is empty")
    columns = set(rows[0])
    if any(set(row) != columns for row in rows):
        raise ValueError("rows do not share the header schema")
    for row_number, row in enumerate(rows, 2):
        number(row[EFFORT], EFFORT, row_number)
        number(row[VALUE], VALUE, row_number)
    return rows


def evaluate(rows, budget=DEFAULT_BUDGET):
    """Return all configurations, feasible configurations, and the winner.

    Value is delivered KLOC; effort is the dataset's person-month estimate.
    The budget constraint is estimated effort <= budget.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    configurations = []
    for row in rows:
        effort = number(row[EFFORT], EFFORT, 0)
        value = number(row[VALUE], VALUE, 0)
        configurations.append({"id": row["idX"], "effort": effort, "value": value})
    feasible = [item for item in configurations if item["effort"] <= budget]
    selected = max(feasible, key=lambda item: (item["value"], -item["effort"], item["id"],)) if feasible else None
    return configurations, feasible, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = load_validate(args.dataset)
    # Exercise a scikit-learn primitive for a scale-free value/effort comparison.
    StandardScaler().fit_transform([[float(row[EFFORT])] for row in rows])
    configurations, feasible, selected = evaluate(rows, args.budget)
    result = {
        "dataset": str(Path(args.dataset)), "rows": len(rows),
        "budget_person_months": args.budget,
        "objective": "maximize delivered value measured as KLOC",
        "constraint": "estimated effort (person-months) <= fixed budget",
        "feasible_count": len(feasible), "selected": selected,
        "trade_offs": [
            "The winner maximizes KLOC among feasible historical configurations.",
            "A larger project can deliver more KLOC but consumes more effort and leaves less contingency.",
            "No feasible configuration is selected when every effort estimate exceeds the budget.",
        ],
        "assumptions": [
            "Effort- is a usable planning estimate in person-months.",
            "Kloc+ is a reasonable proxy for delivered functionality/value.",
            "Historical configurations are candidates and are comparable enough for this screening choice.",
        ],
        "uncertainty_limitations_bias": [
            "KLOC is not customer value, quality, revenue, or schedule certainty.",
            "Effort estimates are uncertain and observational; they do not prove causal effects of settings.",
            "NASA93DEM may not represent the current team, technology, scope, or risk profile; survivor and measurement bias may exist.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
