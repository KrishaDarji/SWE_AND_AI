#!/usr/bin/env python3
"""Reusable standard-library summaries for configuration measurement CSV files."""

import argparse
import csv
import json
import math
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
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
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
    return {
        "run_count": len(dataset.rows),
        "setting_count": len(dataset.settings),
        "settings": list(dataset.settings),
        "measures": list(dataset.measures),
        "quality": quality,
        "settings_by_value": settings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize encoder settings and quality measures")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args(argv)
    result = summarize(load_dataset(args.csv_path))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
