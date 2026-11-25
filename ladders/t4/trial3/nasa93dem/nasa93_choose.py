#!/usr/bin/env python3
"""Choose the best observed NASA93 project record under an Effort- budget."""
import argparse
import math
from typing import Any, Dict, Iterable, Mapping

try:
    from .nasa93_summary import read_csv
except ImportError:
    from nasa93_summary import read_csv

TARGET = "Effort-"
VALUE = "Kloc+"
OUTCOMES = {VALUE, TARGET, "Defects-"}


def _finite(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _budget(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("budget must be finite and nonnegative")
    if not math.isfinite(result) or result < 0:
        raise ValueError("budget must be finite and nonnegative")
    return result


def choose(rows: Iterable[Mapping[str, Any]], columns: Iterable[str], budget: Any) -> Dict[str, Any]:
    """Select an observed row; no prediction or causal interpretation is made."""
    limit = _budget(budget)
    columns = list(columns)
    descriptors = [name for name in columns if not name.endswith("X") and name not in OUTCOMES]
    if not descriptors:
        raise ValueError("input has no descriptor settings")
    material = list(rows)
    total = len(material)
    feasible = []
    for position, row in enumerate(material):
        effort, size = row.get(TARGET), row.get(VALUE)
        if _finite(effort) and _finite(size) and float(effort) <= limit:
            feasible.append((float(size), float(effort), position, row))
    if not feasible:
        raise ValueError("no feasible records with finite Effort- and Kloc+")
    size, effort, position, row = max(feasible, key=lambda item: (item[0], -item[1], -item[2]))
    identifier = row.get("idX", position + 1)
    return {"budget": limit, "feasible_count": len(feasible), "total_count": total,
            "record_identifier": identifier, "descriptors": {name: row.get(name) for name in descriptors},
            "metadata": {name: row.get(name) for name in columns if name.endswith("X")},
            "observed_effort": effort, "delivered_size": size,
            "selection_rule": "maximize observed Kloc+; ties lower observed Effort-; then earliest CSV record",
            "uses_predictions": False}


def choose_csv(path: str, budget: Any) -> Dict[str, Any]:
    data = read_csv(path)
    return choose(data["rows"], data["columns"], budget)


def format_report(result: Dict[str, Any]) -> str:
    lines = ["NASA93 observed project choice", f"Budget (Effort-): {result['budget']:.6g}",
             f"Feasible/total records: {result['feasible_count']}/{result['total_count']}",
             f"Original record identifier: {result['record_identifier']}", "Descriptor settings:"]
    lines += [f"  {key}={value}" for key, value in result["descriptors"].items()]
    lines += ["Metadata (not descriptor settings):"]
    lines += [f"  {key}={value}" for key, value in result["metadata"].items()]
    lines += [f"Observed Effort-: {result['observed_effort']:.6g}", f"Observed delivered size Kloc+: {result['delivered_size']:.6g}",
              f"Selection rule: {result['selection_rule']}",
              "This is retrospective selection among observed outcomes, not a prediction for an unseen proposal.",
              "Kloc+ is an explicitly labeled delivered-size proxy, not business value or productivity; settings are not claimed to cause outcomes."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--budget", required=True, type=float)
    args = parser.parse_args()
    print(format_report(choose_csv(args.csv_path, args.budget)), end="")


if __name__ == "__main__":
    main()
