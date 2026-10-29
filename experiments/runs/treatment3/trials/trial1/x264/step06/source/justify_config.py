#!/usr/bin/env python3
"""Recommend and justify an observed x264 configuration."""
import argparse
import csv
import json
import os
import sys
from statistics import mean

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


TARGET = "Performance-"


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    features = [name for name in rows[0] if name != TARGET]
    if not features or TARGET not in rows[0]:
        raise ValueError("dataset must contain feature columns and Performance-")
    x = [[int(row[name].strip()) for name in features] for row in rows]
    y = [float(row[TARGET].strip()) for row in rows]
    return features, x, y


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=160, random_state=7,
                                         min_samples_leaf=1, n_jobs=1)),
    ])


def recommend(path):
    features, x, y = load_dataset(path)
    if len(x) < 4:
        raise ValueError("at least four observations are required")
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.25, random_state=7)
    model = make_model()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions) if len(test_y) > 1 else float("nan")

    # The recommendation is restricted to configurations actually measured.
    full_model = make_model().fit(x, y)
    scored = [(float(full_model.predict([row])[0]), row, actual)
              for row, actual in zip(x, y)]
    scored.sort(key=lambda item: (item[0], item[2]))
    predicted, config, observed = scored[0]
    config_map = dict(zip(features, config))
    active = [name for name, value in config_map.items() if value]
    lower = [value for value in y if value >= observed]
    percentile = 100.0 * (len(lower) / len(y))
    return {
        "dataset": os.path.abspath(path), "rows": len(x), "features": features,
        "configuration": config_map, "enabled_options": active,
        "observed_performance": observed, "predicted_performance": predicted,
        "dataset_minimum": min(y), "dataset_mean": mean(y),
        "percentile_better_or_equal": percentile, "validation_mae": mae,
        "validation_r2": r2,
    }


def render(result):
    enabled = ", ".join(result["enabled_options"]) or "no optional flags"
    return ("Recommended configuration for the platform administrator\n"
            "======================================================\n"
            f"Dataset: {result['dataset']} ({result['rows']} measured configurations)\n"
            f"Recommendation: use the measured configuration with {enabled}.\n"
            f"Observed Performance-: {result['observed_performance']:.3f}\n"
            f"Model-estimated Performance-: {result['predicted_performance']:.3f}\n"
            f"Dataset minimum: {result['dataset_minimum']:.3f}; dataset mean: {result['dataset_mean']:.3f}\n"
            f"Evidence: this observation is better than or equal to {result['percentile_better_or_equal']:.1f}% of rows; "+
            f"holdout MAE={result['validation_mae']:.3f}, R2={result['validation_r2']:.3f}.\n"
            "Operational note: validate this setting on representative workloads before broad rollout.\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = recommend(args.dataset)
    print(json.dumps(result, indent=2) if args.json else render(result), end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
