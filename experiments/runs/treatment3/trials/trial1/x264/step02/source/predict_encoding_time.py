#!/usr/bin/env python3
"""Train and use a small x264 encoding-time predictor."""
import argparse
import csv
import json
from typing import Dict, Iterable, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

TARGET = "Performance-"

def load_measurements(path: str) -> Tuple[List[str], List[List[float]], List[float]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or TARGET not in rows[0]:
        raise ValueError("dataset is empty or has no Performance- target")
    features = [name for name in rows[0] if name != TARGET]
    matrix = [[float(row[name].strip()) for name in features] for row in rows]
    target = [float(row[TARGET].strip()) for row in rows]
    return features, matrix, target

def fit_predictor(path: str, random_state: int = 17):
    features, matrix, target = load_measurements(path)
    model = RandomForestRegressor(n_estimators=160, random_state=random_state, n_jobs=1)
    model.fit(matrix, target)
    return model, features

def predict_configuration(model, feature_names: Iterable[str], configuration: Dict[str, float]) -> float:
    names = list(feature_names)
    missing = [name for name in names if name not in configuration]
    unknown = [name for name in configuration if name not in names]
    if missing or unknown:
        raise ValueError("configuration mismatch; missing=%s unknown=%s" % (missing, unknown))
    return float(model.predict([[float(configuration[name]) for name in names]])[0])

def evaluate(path: str, random_state: int = 17) -> float:
    _, matrix, target = load_measurements(path)
    x_train, x_test, y_train, y_test = train_test_split(matrix, target, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=160, random_state=random_state, n_jobs=1)
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_test, model.predict(x_test)))

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict x264 encoding time in seconds.")
    parser.add_argument("dataset")
    parser.add_argument("--config", required=True, help="JSON object containing every feature")
    args = parser.parse_args()
    model, features = fit_predictor(args.dataset)
    with open(args.config) as handle:
        config = json.load(handle)
    print("predicted_encoding_time_seconds=%.6f" % predict_configuration(model, features, config))

if __name__ == "__main__":
    main()
