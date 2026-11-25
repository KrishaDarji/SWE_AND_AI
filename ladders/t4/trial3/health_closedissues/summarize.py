"""Summarize model configuration and quality fields in a CSV dataset."""
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version

CONFIGURATION_FIELDS = ["N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth"]
OUTCOME_FIELDS = ["MRE-", "ACC+", "PRED40+"]
NUMERIC_FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators",
                  "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
EXPECTED_FIELDS = OUTCOME_FIELDS + CONFIGURATION_FIELDS


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    missing = {field: 0 for field in fields}
    nonfinite = {field: 0 for field in fields}
    types = {}
    numeric = {}
    categorical = {}
    for field in fields:
        values = [row.get(field, "") for row in rows]
        missing[field] = sum(value is None or value.strip() == "" for value in values)
        if field in NUMERIC_FIELDS:
            numbers = []
            for value in values:
                if value is None or value.strip() == "":
                    continue
                number = finite(value)
                if number is None:
                    nonfinite[field] += 1
                else:
                    numbers.append(number)
            types[field] = "numeric"
            numeric[field] = descriptive(numbers)
        else:
            types[field] = "categorical"
            categorical[field] = dict(sorted(Counter(value for value in values if value.strip()).items()))
    config_keys = [tuple(row.get(field, "") for field in CONFIGURATION_FIELDS) for row in rows]
    config_counts = Counter(config_keys)
    row_keys = [tuple(row.get(field, "") for field in fields) for row in rows]
    row_counts = Counter(row_keys)
    return {
        "record_count": len(rows), "columns": fields, "data_types": types,
        "missing_values": missing, "non_finite_values": nonfinite,
        "numeric_statistics": numeric, "categorical_frequencies": categorical,
        "configuration_fields": CONFIGURATION_FIELDS, "outcome_fields": OUTCOME_FIELDS,
        "optimization_directions": {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"},
        "distinct_configuration_count": len(config_counts),
        "repeated_configurations": {json.dumps(list(key)): count for key, count in sorted(config_counts.items()) if count > 1},
        "duplicate_row_count": sum(count - 1 for count in row_counts.values() if count > 1),
        "sklearn_version": sklearn_version,
        "parsing_assumptions": ["The first CSV row is a header; comma-separated UTF-8 text is expected.",
                                 "Blank cells are missing; numeric tokens are parsed as float.",
                                 "Non-finite or non-numeric numeric cells are excluded from statistics."],
        "limitations": ["No metric formula, unit, valid range, or compute cost is inferred.",
                        "The filename is not interpreted as an issue timeline or closure event."]}


def descriptive(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None,
                "median": None, "q25": None, "q75": None}
    array = np.asarray(values, dtype=float)
    return {"count": int(array.size), "min": float(np.min(array)), "max": float(np.max(array)),
            "mean": float(np.mean(array)), "std": float(np.std(array)),
            "median": float(np.median(array)), "q25": float(np.percentile(array, 25)),
            "q75": float(np.percentile(array, 75))}


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: summarize.py DATASET.csv", file=sys.stderr)
        return 2
    result = summarize(argv[0])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
