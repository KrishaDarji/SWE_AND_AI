#!/usr/bin/env python3
"""Summaries and nearest-neighbor encoding-time predictions for x264 CSV data."""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    settings: tuple
    measures: tuple
    rows: tuple


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_dataset(path):
    """Load a rectangular CSV, classifying binary columns as settings."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("CSV has no data rows")
    columns = tuple(reader.fieldnames)
    numeric = {name: [_number(row[name]) for row in rows] for name in columns}
    settings = tuple(name for name in columns if set(numeric[name]) <= {0.0, 1.0})
    measures = tuple(name for name in columns if name not in settings)
    if not measures:
        raise ValueError("CSV has no non-binary measurement column")
    return Dataset(settings, measures, tuple(rows))


def _stats(values):
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "median": None, "stdev": None}
    return {
        "count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(dataset):
    """Return JSON-serializable dataset, quality, and setting-level summaries."""
    quality = {name: _stats(_number(row[name]) for row in dataset.rows)
               for name in dataset.measures}
    settings = {}
    for setting in dataset.settings:
        by_value = {}
        for value in (0, 1):
            selected = [row for row in dataset.rows if int(float(row[setting])) == value]
            by_value[str(value)] = {
                "runs": len(selected),
                "measures": {name: _stats(_number(row[name]) for row in selected)
                              for name in dataset.measures},
            }
        settings[setting] = by_value
    return {"run_count": len(dataset.rows), "setting_count": len(dataset.settings),
            "settings": list(dataset.settings), "measures": list(dataset.measures),
            "quality": quality, "settings_by_value": settings}


def predict(dataset, configuration, measure=None):
    """Predict a measure using exact-match or minimum-Hamming-distance rows.

    Configuration must provide every binary setting as 0 or 1. Ties are averaged;
    the returned metadata exposes whether the estimate was exact or approximate.
    """
    measure = measure or dataset.measures[0]
    if measure not in dataset.measures:
        raise ValueError("unknown measure: " + measure)
    missing = [name for name in dataset.settings if name not in configuration]
    extra = [name for name in configuration if name not in dataset.settings]
    if missing or extra:
        raise ValueError("configuration must contain exactly the dataset settings")
    try:
        target = {name: int(configuration[name]) for name in dataset.settings}
    except (TypeError, ValueError):
        raise ValueError("setting values must be 0 or 1")
    if any(value not in (0, 1) for value in target.values()):
        raise ValueError("setting values must be 0 or 1")
    candidates = []
    for row in dataset.rows:
        distance = sum(int(float(row[name])) != target[name] for name in dataset.settings)
        value = _number(row[measure])
        if value is not None:
            candidates.append((distance, value))
    best_distance = min(distance for distance, _ in candidates)
    values = [value for distance, value in candidates if distance == best_distance]
    return {"measure": measure, "prediction": statistics.fmean(values),
            "distance": best_distance, "neighbor_count": len(values),
            "exact_match": best_distance == 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize or predict encoder measurements")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--predict-json", metavar="JSON", help="setting values as a JSON object")
    parser.add_argument("--measure", help="measurement column to predict")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.csv_path)
    if args.predict_json is None:
        result = summarize(dataset)
    else:
        result = predict(dataset, json.loads(args.predict_json), args.measure)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
