#!/usr/bin/env python3
"""Predict x264 encoding time from historical configurations."""

import argparse
import csv
import json
import math
from pathlib import Path


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def predict(dataset, configuration, k=5, measure="Performance-"):
    """Return a prediction and diagnostics for *configuration*.

    Distances are normalized per setting.  Numeric settings use absolute
    difference; non-numeric settings use equality.  The prediction is the
    inverse-distance weighted mean of the nearest historical rows.
    """
    if k < 1:
        raise ValueError("k must be positive")
    with Path(dataset).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if measure not in fields:
            raise ValueError("measure column not found: " + measure)
        rows = list(reader)
    if not rows:
        raise ValueError("dataset has no data rows")
    settings = [field for field in fields if field != measure]
    missing = [field for field in settings if field not in configuration]
    if missing:
        raise ValueError("configuration is missing settings: " + ", ".join(missing))

    ranges = {}
    for field in settings:
        values = [_number(row[field]) for row in rows]
        numeric = [value for value in values if value is not None]
        ranges[field] = max(numeric) - min(numeric) if len(numeric) > 1 else 1.0

    neighbors = []
    for row in rows:
        distance = 0.0
        for field in settings:
            target = configuration[field]
            observed = row[field]
            left, right = _number(target), _number(observed)
            if left is not None and right is not None:
                difference = (left - right) / (ranges[field] or 1.0)
            else:
                difference = 0.0 if str(target) == observed else 1.0
            distance += difference * difference
        value = _number(row[measure])
        if value is not None:
            neighbors.append((math.sqrt(distance), value, row))
    neighbors.sort(key=lambda item: item[0])
    selected = neighbors[:min(k, len(neighbors))]
    if not selected:
        raise ValueError("measure column contains no numeric values")
    if selected[0][0] == 0:
        prediction = sum(value for distance, value, _ in selected if distance == 0) / sum(
            distance == 0 for distance, _, _ in selected
        )
    else:
        weights = [1.0 / distance for distance, _, _ in selected]
        prediction = sum(weight * value for weight, (_, value, _) in zip(weights, selected)) / sum(weights)
    values = [value for _, value, _ in selected]
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {"predicted_time": prediction, "neighbor_spread": spread,
            "neighbors_used": len(selected), "settings": settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("configuration", help="JSON object or name=value pairs")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    try:
        configuration = json.loads(args.configuration)
    except json.JSONDecodeError:
        configuration = dict(item.split("=", 1) for item in args.configuration.split(","))
    print(json.dumps(predict(args.dataset, configuration, args.k), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
