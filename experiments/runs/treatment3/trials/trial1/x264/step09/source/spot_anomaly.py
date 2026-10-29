#!/usr/bin/env python3
"""Detect potentially anomalous x264 benchmark measurements."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset contains no rows")
    target = "Performance-"
    if target not in rows[0]:
        raise ValueError("missing Performance- column")
    features = [name for name in rows[0] if name != target]
    X, y = [], []
    for number, row in enumerate(rows, 2):
        try:
            X.append([float(row[name].strip()) for name in features])
            y.append(float(row[target].strip()))
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid numeric value on CSV line %d" % number) from exc
    return features, X, y


def detect(path, contamination=0.02, random_state=17):
    names, X, y = load_csv(path)
    n = len(y)
    if n < 10:
        raise ValueError("at least 10 rows are required")

    # Configuration-space novelty: catches rows unlike the measured design space.
    forest = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state
    )
    forest.fit(X)
    config_label = forest.predict(X)
    config_score = -forest.score_samples(X)

    # Out-of-fold prediction prevents the model from explaining away its own row.
    folds = min(5, n)
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=1
        ),
    )
    predictions = cross_val_predict(model, X, y, cv=KFold(folds, shuffle=True, random_state=random_state))
    residuals = [abs(actual - predicted) for actual, predicted in zip(y, predictions)]
    center = statistics.median(residuals)
    deviations = [abs(value - center) for value in residuals]
    mad = statistics.median(deviations)
    scale = max(1.4826 * mad, 1e-9)
    residual_score = [(value - center) / scale for value in residuals]
    # A robust residual threshold is deliberately conservative.
    residual_label = [score >= 6.0 for score in residual_score]
    anomaly = [label == -1 or residual_label[i] for i, label in enumerate(config_label)]

    candidates = []
    for i, flagged in enumerate(anomaly):
        if flagged:
            candidates.append({
                "row": i + 2,
                "measurement": y[i],
                "prediction": round(predictions[i], 6),
                "absolute_residual": round(residuals[i], 6),
                "residual_robust_score": round(residual_score[i], 3),
                "configuration_outlier": bool(config_label[i] == -1),
                "possible_measurement_error": bool(residual_label[i]),
                "features": {names[j]: X[i][j] for j in range(len(names))},
            })
    candidates.sort(key=lambda item: (item["possible_measurement_error"], item["absolute_residual"]), reverse=True)
    return {
        "rows": n,
        "features": len(names),
        "flagged_count": len(candidates),
        "flagged_fraction": len(candidates) / n,
        "method": "IsolationForest configuration novelty plus out-of-fold RandomForest residual MAD",
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(detect(args.dataset, args.contamination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
