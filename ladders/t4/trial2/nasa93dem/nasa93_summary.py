#!/usr/bin/env python3
"""Independent NASA93 CSV summarizer for experiment Step 1."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import sklearn


MISSING = {"", "?", "na", "n/a", "none", "null"}


def parse_value(value):
    text = value.strip()
    if text.lower() in MISSING:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV input has no header")
        rows = [{key: parse_value(value or "") for key, value in row.items()} for row in reader]
        return reader.fieldnames, rows


def infer_type(values):
    present = [value for value in values if value is not None]
    if not present:
        return "unknown"
    if all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in present):
        return "numeric"
    return "categorical"


def finite_numbers(values):
    return np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)


def summarize(path):
    columns, rows = load_csv(path)
    types = {column: infer_type([row[column] for row in rows]) for column in columns}
    missing = {column: sum(row[column] is None for row in rows) for column in columns}
    numeric = {}
    categorical = {}
    for column in columns:
        values = [row[column] for row in rows]
        if types[column] == "numeric":
            data = finite_numbers(values)
            numeric[column] = {
                "count": int(data.size), "mean": float(np.mean(data)) if data.size else None,
                "std": float(np.std(data, ddof=1)) if data.size > 1 else None,
                "min": float(np.min(data)) if data.size else None,
                "q25": float(np.percentile(data, 25)) if data.size else None,
                "median": float(np.median(data)) if data.size else None,
                "q75": float(np.percentile(data, 75)) if data.size else None,
                "max": float(np.max(data)) if data.size else None,
            }
        else:
            frequencies = {}
            for value in values:
                label = "<MISSING>" if value is None else str(value)
                frequencies[label] = frequencies.get(label, 0) + 1
            categorical[column] = dict(sorted(frequencies.items()))
    excluded_metadata = [column for column in columns if column.endswith("X")]
    outcomes = [column for column in columns if column in {"Kloc+", "Effort-", "Defects-"}]
    descriptors = [column for column in columns if column not in excluded_metadata and column not in outcomes]
    metadata = excluded_metadata
    return {
        "record_count": len(rows), "columns": columns, "inferred_types": types,
        "missing_values": missing, "numeric_statistics": numeric,
        "categorical_frequencies": categorical, "column_roles": {
            "project_descriptors": descriptors, "metadata_excluded_from_future_modeling": metadata,
            "outcomes": outcomes, "outcome_directions": {"Kloc+": "+", "Effort-": "-", "Defects-": "-"},
            "outcome_semantics": {"Kloc+": "delivered size", "Effort-": "effort", "Defects-": "defects"},
        },
        "parsing_assumptions": ["CSV header defines columns.", "Whitespace is stripped.", "Blank, ?, NA, N/A, none, and null are missing.", "Numeric-looking values are parsed as floating-point numbers; other values remain categorical strings.", "Kloc+ is an outcome, not automatically a predictor."],
        "limitations": ["No units are inferred.", "Numeric category codes are not converted into continuous quantities.", "The summary is descriptive and does not establish causal relationships."],
        "tooling": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv_path), indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
