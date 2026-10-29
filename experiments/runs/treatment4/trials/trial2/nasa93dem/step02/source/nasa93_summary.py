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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv_path), indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
