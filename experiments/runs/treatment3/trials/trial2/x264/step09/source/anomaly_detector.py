#!/usr/bin/env python3
"""Reproducible anomaly screening for the x264 benchmark measurements."""
import argparse
import csv
import json
import math
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

EXPECTED_TARGET = "Performance-"
SEED = 20261029


def median(values):
    values = sorted(values)
    n = len(values)
    return (values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2.0)


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset contains no records")
    fields = list(rows[0])
    if EXPECTED_TARGET not in fields:
        raise ValueError("missing Performance- column")
    feature_names = [name for name in fields if name != EXPECTED_TARGET]
    if not feature_names:
        raise ValueError("dataset has no configuration columns")
    parsed = []
    for number, row in enumerate(rows, 2):
        if set(row) != set(fields) or any(value is None or value == "" for value in row.values()):
            raise ValueError(f"invalid/missing value on CSV line {number}")
        try:
            features = [float(row[name]) for name in feature_names]
            target = float(row[EXPECTED_TARGET])
        except ValueError as exc:
            raise ValueError(f"non-numeric value on CSV line {number}") from exc
        if not all(math.isfinite(value) for value in features + [target]):
            raise ValueError(f"non-finite value on CSV line {number}")
        parsed.append((row, features, target))
    return fields, feature_names, parsed


def detect(path):
    fields, feature_names, records = load_dataset(path)
    x = [record[1] for record in records]
    y = [record[2] for record in records]
    center = median(y)
    deviations = [abs(value - center) for value in y]
    mad = median(deviations)
    robust_scale = 1.4826 * mad
    if robust_scale == 0:
        robust_scale = max(1e-12, (max(y) - min(y)) / 6.0)
    robust_scores = [abs(value - center) / robust_scale for value in y]

    scaled_x = StandardScaler().fit_transform(x)
    detector = IsolationForest(contamination="auto", random_state=SEED, n_estimators=256)
    detector.fit([[*features, target] for features, target in zip(scaled_x, y)])
    isolation_scores = -detector.decision_function([[*features, target] for features, target in zip(scaled_x, y)])
    iso_min, iso_max = min(isolation_scores), max(isolation_scores)
    iso_range = iso_max - iso_min or 1.0
    iso_normalized = [(score - iso_min) / iso_range for score in isolation_scores]
    combined = [0.7 * min(score / 6.0, 1.0) + 0.3 * iso for score, iso in zip(robust_scores, iso_normalized)]
    order = sorted(range(len(records)), key=lambda i: combined[i], reverse=True)
    # 0.45 is a screening threshold: it retains the upper tail candidate even
    # when the dataset has no extreme 3.5-MAD outlier.
    flagged = [i for i in order if robust_scores[i] >= 3.5 or combined[i] >= 0.45]
    top = order[0]
    direction = "lower-than-typical (possible measurement error)" if y[top] < center else "higher-than-typical (possibly legitimate exceptional performance)"
    result = {
        "dataset": str(path), "records": len(records), "features": feature_names,
        "median_performance": center, "mad": mad, "robust_scale": robust_scale,
        "flagged_count": len(flagged), "top_index_zero_based": top,
        "top_record_number": top + 2, "top_performance": y[top],
        "top_robust_score": robust_scores[top], "top_isolation_score": isolation_scores[top],
        "top_combined_score": combined[top], "interpretation": direction,
        "top_configuration": records[top][0],
        "flagged_record_numbers": [i + 2 for i in flagged],
        "assumptions": ["rows are independent benchmark runs", "larger Performance- is better", "configuration columns are numeric switches/parameters"],
        "limitations": ["no repeated-run variance or timestamps are available", "Isolation Forest can flag rare valid configurations", "thresholds are screening heuristics, not proof of error"],
        "confounders": ["machine load, thermal throttling, compiler/runtime variation, and input/cache effects"],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(detect(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
