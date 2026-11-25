#!/usr/bin/env python3
"""Reusable, standard-library-only summaries for experiment CSV datasets."""

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class ColumnSummary:
    name: str
    kind: str
    count: int
    unique: int
    minimum: float = None
    maximum: float = None
    mean: float = None
    median: float = None
    values: Dict[str, int] = None


def _number(value: str):
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None


def summarize_csv(path: Path) -> dict:
    """Return a JSON-serializable summary; this is the reusable API boundary."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        rows = list(reader)
    summaries = []
    for name in reader.fieldnames:
        raw = [(row.get(name) or "").strip() for row in rows]
        numbers = [_number(value) for value in raw]
        numeric = all(value is not None for value in numbers)
        distinct = set(raw)
        binary = numeric and distinct <= {"0", "1"}
        if binary:
            kind = "encoder_setting"
        elif numeric:
            kind = "quality_measure"
        else:
            kind = "categorical"
        summary = ColumnSummary(name=name, kind=kind, count=len(raw),
                                unique=len(distinct))
        if numeric:
            summary.minimum = min(numbers)
            summary.maximum = max(numbers)
            summary.mean = statistics.fmean(numbers)
            summary.median = statistics.median(numbers)
        elif len(distinct) <= 20:
            summary.values = dict(sorted((item, raw.count(item)) for item in distinct))
        summaries.append(summary)
    return {
        "dataset": str(path),
        "rows": len(rows),
        "columns": len(reader.fieldnames),
        "encoder_settings": [asdict(item) for item in summaries if item.kind == "encoder_setting"],
        "quality_measures": [asdict(item) for item in summaries if item.kind == "quality_measure"],
        "other_columns": [asdict(item) for item in summaries if item.kind == "categorical"],
    }


def predict_encoding_time(path: Path, configuration: Dict[str, str]) -> dict:
    """Predict encoding time before running a configuration."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Performance-" not in reader.fieldnames:
            raise ValueError("CSV must contain a Performance- column")
        settings = [name for name in reader.fieldnames if name != "Performance-"]
        rows = list(reader)
    if set(configuration) != set(settings):
        raise ValueError("configuration columns do not match dataset")
    target = {name: str(configuration[name]).strip() for name in settings}
    candidates = []
    for row in rows:
        time = _number(row.get("Performance-", ""))
        if time is None:
            continue
        distance = sum((row.get(name) or "").strip() != target[name] for name in settings)
        candidates.append((distance, time))
    if not candidates:
        raise ValueError("dataset has no numeric Performance- values")
    nearest_distance = min(distance for distance, _ in candidates)
    nearest = [time for distance, time in candidates if distance == nearest_distance]
    if nearest_distance == 0:
        prediction, method = statistics.fmean(nearest), "exact-match"
    else:
        weights = [1.0 / (distance + 1) for distance, _ in candidates]
        prediction = sum(weight * time for weight, (_, time) in zip(weights, candidates)) / sum(weights)
        method = "inverse-hamming-weighted"
    return {"predicted_time": prediction, "nearest_distance": nearest_distance,
            "method": method, "rows_used": len(candidates)}


def suggest_setting_change(path: Path, configuration: Dict[str, str]) -> dict:
    """Suggest the best one-setting change when it is predicted to be faster."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Performance-" not in reader.fieldnames:
            raise ValueError("CSV must contain a Performance- column")
        settings = [name for name in reader.fieldnames if name != "Performance-"]
        rows = list(reader)
    if set(configuration) != set(settings):
        raise ValueError("configuration columns do not match dataset")
    target = {name: str(configuration[name]).strip() for name in settings}
    baseline = predict_encoding_time(path, target)["predicted_time"]
    candidates = []
    for name in settings:
        values = sorted({(row.get(name) or "").strip() for row in rows})
        for value in values:
            if value == target[name]:
                continue
            changed = dict(target)
            changed[name] = value
            prediction = predict_encoding_time(path, changed)["predicted_time"]
            if prediction < baseline:
                candidates.append((prediction, name, value))
    if not candidates:
        return {"configuration": target, "baseline_time": baseline,
                "improvement": 0.0, "suggestion": None}
    prediction, name, value = min(candidates, key=lambda item: (item[0], settings.index(item[1]), item[2]))
    return {"configuration": target, "baseline_time": baseline,
            "improvement": baseline - prediction,
            "suggestion": {"setting": name, "from": target[name],
                            "to": value, "predicted_time": prediction}}


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predict", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--suggest", nargs="*", metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    if args.suggest is not None:
        if any("=" not in item for item in args.suggest):
            parser.error("--suggest values must use NAME=VALUE")
        result = suggest_setting_change(args.csv_path, dict(item.split("=", 1) for item in args.suggest))
    elif args.predict is None:
        result = summarize_csv(args.csv_path)
    else:
        if any("=" not in item for item in args.predict):
            parser.error("--predict values must use NAME=VALUE")
        result = predict_encoding_time(args.csv_path, dict(item.split("=", 1) for item in args.predict))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
