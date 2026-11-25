#!/usr/bin/env python3
"""Summaries, predictions, and statistical certification for x264 measurements."""

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
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def summarize(dataset):
    quality = {name: _stats(_number(row[name]) for row in dataset.rows)
               for name in dataset.measures}
    settings = {}
    for setting in dataset.settings:
        by_value = {}
        for value in (0, 1):
            selected = [row for row in dataset.rows if int(float(row[setting])) == value]
            by_value[str(value)] = {"runs": len(selected),
                "measures": {name: _stats(_number(row[name]) for row in selected)
                              for name in dataset.measures}}
        settings[setting] = by_value
    return {"run_count": len(dataset.rows), "setting_count": len(dataset.settings),
            "settings": list(dataset.settings), "measures": list(dataset.measures),
            "quality": quality, "settings_by_value": settings}


def predict(dataset, configuration, measure=None):
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
        value = _number(row[measure])
        if value is not None:
            distance = sum(int(float(row[name])) != target[name] for name in dataset.settings)
            candidates.append((distance, value))
    best_distance = min(distance for distance, _ in candidates)
    values = [value for distance, value in candidates if distance == best_distance]
    return {"measure": measure, "prediction": statistics.fmean(values),
            "distance": best_distance, "neighbor_count": len(values),
            "exact_match": best_distance == 0}


def certify(dataset, measure=None, alpha=0.05):
    """Test whether each setting's value 1 improves the measure over value 0.

    The p-value uses a Welch statistic with a normal reference distribution.
    It is a conservative, dependency-free screening test, not a replacement
    for a randomized repeated-run analysis.
    """
    measure = measure or dataset.measures[0]
    if measure not in dataset.measures or not 0 < alpha < 1:
        raise ValueError("unknown measure or alpha must be between 0 and 1")
    results = {}
    for setting in dataset.settings:
        groups = [[_number(row[measure]) for row in dataset.rows
                   if int(float(row[setting])) == value] for value in (0, 1)]
        groups = [[x for x in group if x is not None] for group in groups]
        if min(map(len, groups)) < 2:
            results[setting] = {"measure": measure, "mean_0": None, "mean_1": None,
                "improvement": None, "z": None, "p_value": None, "alpha": alpha,
                "credible": False, "counts": [len(groups[0]), len(groups[1])],
                "status": "not_testable"}
            continue
        means = [statistics.fmean(group) for group in groups]
        variances = [statistics.variance(group) for group in groups]
        standard_error = math.sqrt(variances[0] / len(groups[0]) +
                                   variances[1] / len(groups[1]))
        z = (means[1] - means[0]) / standard_error if standard_error else 0.0
        p_value = math.erfc(abs(z) / math.sqrt(2.0)) if standard_error else (0.0 if z else 1.0)
        results[setting] = {"measure": measure, "mean_0": means[0], "mean_1": means[1],
            "improvement": means[1] - means[0], "z": z, "p_value": p_value,
            "alpha": alpha, "credible": means[1] > means[0] and p_value < alpha,
            "counts": [len(groups[0]), len(groups[1])], "status": "tested"}
    credible = [name for name, result in results.items() if result["credible"]]
    return {"measure": measure, "alpha": alpha, "method": "Welch z approximation",
            "settings": results, "credible_improvements": credible,
            "any_credible": bool(credible)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize, predict, or certify encoder measurements")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--predict-json", metavar="JSON")
    parser.add_argument("--measure")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.csv_path)
    if args.certify:
        result = certify(dataset, args.measure, args.alpha)
    elif args.predict_json is not None:
        result = predict(dataset, json.loads(args.predict_json), args.measure)
    else:
        result = summarize(dataset)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
