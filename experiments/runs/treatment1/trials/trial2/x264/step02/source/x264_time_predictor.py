#!/usr/bin/env python3
"""Predict x264 encoding time from historical binary configurations."""
import argparse
import csv
import json
import math
import statistics
import sys


TARGET = "Performance-"


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or TARGET not in rows[0]:
        raise ValueError("dataset must contain rows and a Performance- column")
    features = [name for name in rows[0] if name != TARGET]
    records = []
    for number, row in enumerate(rows, 2):
        try:
            vector = tuple(int(row[name].strip()) for name in features)
            target = float(row[TARGET].strip())
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid row %d: %s" % (number, exc))
        if any(value not in (0, 1) for value in vector):
            raise ValueError("row %d contains a non-binary feature" % number)
        records.append((vector, target))
    return features, records


def predict(features, records, configuration, neighbors=7):
    if set(configuration) != set(features):
        missing = sorted(set(features) - set(configuration))
        extra = sorted(set(configuration) - set(features))
        raise ValueError("configuration keys differ; missing=%s extra=%s" % (missing, extra))
    vector = tuple(int(configuration[name]) for name in features)
    if any(value not in (0, 1) for value in vector):
        raise ValueError("configuration values must be 0 or 1")
    if not records:
        raise ValueError("dataset has no records")
    count = max(1, min(int(neighbors), len(records)))
    ranked = sorted(
        ((sum(a != b for a, b in zip(vector, known)), target) for known, target in records),
        key=lambda item: item[0],
    )[:count]
    weights = [1.0 / (distance + 1.0) for distance, _ in ranked]
    estimate = sum(weight * target for weight, (_, target) in zip(weights, ranked)) / sum(weights)
    values = [target for _, target in ranked]
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "predicted_seconds": estimate,
        "neighbor_stddev_seconds": spread,
        "neighbors_used": count,
        "nearest_distance": ranked[0][0],
        "neighbor_times_seconds": values,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("configuration", help="JSON object mapping each setting to 0 or 1")
    parser.add_argument("-k", type=int, default=7, help="number of historical neighbors")
    args = parser.parse_args(argv)
    features, records = load_dataset(args.dataset)
    try:
        configuration = json.loads(args.configuration)
    except json.JSONDecodeError as exc:
        parser.error("configuration is not valid JSON: %s" % exc)
    result = predict(features, records, configuration, args.k)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
