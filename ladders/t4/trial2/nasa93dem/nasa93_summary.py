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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy import stats


MISSING = {"", "?", "na", "n/a", "none", "null"}
OUTCOMES = {"Kloc+", "Effort-", "Defects-"}


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


def descriptor_columns(columns):
    return [c for c in columns if not c.endswith("X") and c not in OUTCOMES]


def build_model(predictors):
    prep = ColumnTransformer([("categorical", Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), list(range(len(predictors))))], remainder="drop")
    estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
    return Pipeline([("preprocess", prep), ("estimator", estimator)])


def prepare_learning_data(path):
    columns, rows = load_csv(path)
    predictors = descriptor_columns(columns)
    if "Effort-" not in columns:
        raise ValueError("CSV must contain Effort-")
    X = [{c: ("<MISSING>" if row[c] is None else str(row[c])) for c in predictors} for row in rows]
    y = np.asarray([float(row["Effort-"]) for row in rows], dtype=float)
    configurations = [tuple(item[c] for c in predictors) for item in X]
    group_ids = {configuration: number for number, configuration in enumerate(dict.fromkeys(configurations))}
    groups = np.asarray([group_ids[configuration] for configuration in configurations])
    return columns, predictors, X, y, groups


def evaluate(path):
    columns, predictors, X, y, groups = prepare_learning_data(path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train, test = next(splitter.split(X, y, groups=groups))
    model = build_model(predictors)
    matrix = [[row[c] for c in predictors] for row in X]
    model.fit([matrix[i] for i in train], y[train])
    predictions = model.predict([matrix[i] for i in test])
    baseline = float(np.mean(y[train]))
    return {"columns": columns, "predictors": predictors, "train_count": len(train),
            "test_count": len(test), "model": model, "train_indices": train,
            "test_indices": test, "mae": float(np.mean(np.abs(y[test] - predictions))),
            "baseline_mae": float(np.mean(np.abs(y[test] - baseline))),
            "predictions": predictions, "training_mean": baseline,
            "settings": {"test_size": 0.25, "random_state": 42, "grouping": "identical descriptor configurations",
                         "estimator": "RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)",
                         "missing_values": "explicit <MISSING> categorical token"}}


def fit_final_model(path):
    _, predictors, X, y, _ = prepare_learning_data(path)
    model = build_model(predictors)
    model.fit([[row[c] for c in predictors] for row in X], y)
    return model, predictors


def predict_proposal(model, predictors, proposal):
    row = [("<MISSING>" if proposal.get(c) is None else str(proposal.get(c))) for c in predictors]
    return float(model.predict([row])[0])


def compare_tools(path, practical_threshold=None, tool_a="n", tool_b="h"):
    """Compare observed Effort- for two tool groups using Welch's method."""
    if practical_threshold is not None and practical_threshold < 0:
        raise ValueError("practical_threshold must be nonnegative")
    columns, rows = load_csv(path)
    if "tool" not in columns or "Effort-" not in columns:
        raise ValueError("CSV must contain tool and Effort-")
    groups = {}
    for label in (tool_a, tool_b):
        groups[label] = finite_numbers([r["Effort-"] for r in rows if r["tool"] == label])
        if groups[label].size < 2:
            raise ValueError("each tool group needs at least two finite Effort- outcomes")
    a, b = groups[tool_a], groups[tool_b]
    if np.var(a, ddof=1) == 0 or np.var(b, ddof=1) == 0:
        raise ValueError("Welch comparison requires nonzero variance in both groups")
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    difference = mean_a - mean_b
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se2 = var_a / a.size + var_b / b.size
    df = se2 ** 2 / ((var_a / a.size) ** 2 / (a.size - 1) + (var_b / b.size) ** 2 / (b.size - 1))
    critical = float(stats.t.ppf(0.975, df))
    margin = critical * math.sqrt(se2)
    interval = (float(difference - margin), float(difference + margin))
    p_value = float(2 * stats.t.sf(abs(difference / math.sqrt(se2)), df))
    threshold = float(practical_threshold) if practical_threshold is not None else None
    return {"groups": {tool_a: {"n": int(a.size), "mean": mean_a}, tool_b: {"n": int(b.size), "mean": mean_b}},
            "difference_mean_a_minus_mean_b": float(difference), "confidence_level": 0.95,
            "confidence_interval": interval, "welch_df": float(df), "welch_t": float(difference / math.sqrt(se2)),
            "two_sided_p_value": p_value, "practical_threshold": threshold,
            "point_estimate_exceeds_threshold": None if threshold is None else abs(difference) > threshold,
            "entire_interval_exceeds_threshold": None if threshold is None else (interval[0] > threshold or interval[1] < -threshold),
            "interpretation": "Effort- is observed effort, not verified monetary cost; this observational comparison does not establish a causal process effect."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--compare-tools", action="store_true")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    result = compare_tools(args.csv_path, args.threshold) if args.compare_tools else summarize(args.csv_path)
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
