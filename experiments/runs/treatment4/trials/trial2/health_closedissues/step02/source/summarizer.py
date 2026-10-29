"""Summarize configuration and quality fields in a CSV dataset."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import sklearn


CONFIGURATION_FIELDS = [
    "N_estimators",
    "criterion",
    "Min_sample_leaves",
    "Min_impurity_decrease",
    "Max_depth",
]
OUTCOME_FIELDS = ["MRE-", "ACC+", "PRED40+"]
NUMERIC_FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
REQUIRED_FIELDS = OUTCOME_FIELDS + CONFIGURATION_FIELDS


def _numeric(value: str) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number


def _stats(values: list[float | int]) -> dict[str, float | int]:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mean = sum(ordered) / n
    percentile = lambda p: ordered[(n - 1) * p]
    def quantile(p: float) -> float:
        position = (n - 1) * p
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"count": n, "min": ordered[0], "max": ordered[-1], "mean": mean,
            "std": math.sqrt(sum((x - mean) ** 2 for x in ordered) / n),
            "q25": quantile(0.25), "median": quantile(0.5), "q75": quantile(0.75)}


def summarize_csv(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing_columns = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    missing_values = {field: sum(row.get(field, "") == "" for row in rows) for field in columns}
    numeric = {field: [_numeric(row[field]) for row in rows if row[field] != ""] for field in NUMERIC_FIELDS}
    categorical = {"criterion": dict(sorted(Counter(row["criterion"] for row in rows if row["criterion"] != "").items()))}
    configuration_rows = [tuple(row[field] for field in CONFIGURATION_FIELDS) for row in rows]
    config_counts = Counter(configuration_rows)
    duplicate_rows = sum(count - 1 for count in Counter(tuple(row.get(c, "") for c in columns) for row in rows).values() if count > 1)
    return {
        "record_count": len(rows), "columns": columns,
        "data_types": {field: ("categorical" if field == "criterion" else "numeric") for field in columns},
        "missing_values": missing_values, "numeric_statistics": {field: _stats(vals) for field, vals in numeric.items() if vals},
        "categorical_frequencies": categorical, "configuration_fields": CONFIGURATION_FIELDS,
        "outcome_fields": OUTCOME_FIELDS, "optimization_directions": {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"},
        "distinct_configuration_count": len(config_counts),
        "repeated_configurations": {"|".join(key): count for key, count in sorted(config_counts.items()) if count > 1},
        "duplicate_row_count": duplicate_rows, "scikit_learn_version": sklearn.__version__,
        "parsing_assumptions": ["The first row is a comma-separated header.", "Empty strings are missing values.", "Numeric fields are parsed as decimal numbers; criterion is categorical."],
        "limitations": ["Descriptive statistics do not define metric formulas, units, valid ranges, or compute costs.", "No issue timeline or closure events are inferred from the filename."],
    }


def write_report(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    result = summarize_csv(input_path)
    Path(output_path).write_text(__import__("json").dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
