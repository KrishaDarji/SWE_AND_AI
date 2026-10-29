#!/usr/bin/env python3
"""Independent NASA93 dataset summarizer for Step 1."""
import argparse
import csv
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np
import sklearn


OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}


def _parse(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def read_csv(path: str) -> Dict[str, Any]:
    if not path:
        raise ValueError("input path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {"record_count": 0, "columns": [], "rows": []}
        rows = [{key: _parse(value or "") for key, value in row.items()} for row in reader]
    return {"record_count": len(rows), "columns": reader.fieldnames, "rows": rows}


def summarize(path: str) -> Dict[str, Any]:
    data = read_csv(path)
    columns, rows = data["columns"], data["rows"]
    types, missing, numeric, categorical = {}, {}, {}, {}
    for column in columns:
        values = [row.get(column) for row in rows]
        missing[column] = sum(value is None for value in values)
        observed = [value for value in values if value is not None]
        is_numeric = bool(observed) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in observed)
        types[column] = "numeric" if is_numeric else "categorical"
        if is_numeric:
            array = np.asarray(observed, dtype=float)
            numeric[column] = {name: float(getattr(np, name)(array)) for name in ("min", "max", "mean", "std")}
        else:
            categorical[column] = {str(key): count for key, count in sorted(Counter(str(value) for value in observed).items())}
    metadata = [column for column in columns if column.endswith("X")]
    outcome_names = [column for column in columns if column in OUTCOMES]
    descriptors = [column for column in columns if column not in metadata and column not in outcome_names]
    roles = {"project_descriptors": descriptors, "metadata_excluded_from_modeling": metadata, "outcomes": outcome_names}
    return {"record_count": data["record_count"], "columns": columns, "inferred_types": types, "missing_values": missing,
            "numeric_statistics": numeric, "categorical_frequencies": categorical, "roles": roles,
            "outcome_directions": {name: OUTCOMES[name] for name in outcome_names},
            "delivered_size": "Kloc+" if "Kloc+" in columns else None,
            "effort": "Effort-" if "Effort-" in columns else None,
            "defects": "Defects-" if "Defects-" in columns else None,
            "sklearn_version": sklearn.__version__}


def format_summary(result: Dict[str, Any]) -> str:
    lines = ["NASA93 historical project summary", f"Record count: {result['record_count']}", "Columns: " + ", ".join(result["columns"]),
             "Inferred data types: " + "; ".join(f"{k}={v}" for k, v in result["inferred_types"].items()),
             "Missing values: " + "; ".join(f"{k}={v}" for k, v in result["missing_values"].items()),
             "Numeric descriptive statistics (min, max, mean, std):"]
    for key, stats in result["numeric_statistics"].items():
        lines.append(f"  {key}: " + ", ".join(f"{name}={value:.6g}" for name, value in stats.items()))
    lines.append("Categorical frequencies:")
    for key, values in result["categorical_frequencies"].items():
        lines.append(f"  {key}: " + ", ".join(f"{name}={count}" for name, count in values.items()))
    lines += ["Roles: " + str(result["roles"]), "Outcome directions: " + str(result["outcome_directions"]),
              "Delivered size: Kloc+; effort: Effort-; defects: Defects-.",
              "X-suffix columns are excluded metadata for future modeling; delivered size is not automatically a predictor.",
              "Parsing: blank and non-finite numeric values become missing; empty CSVs return zero records and empty summaries."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(format_summary(summarize(args.csv_path)), end="")


if __name__ == "__main__":
    main()
