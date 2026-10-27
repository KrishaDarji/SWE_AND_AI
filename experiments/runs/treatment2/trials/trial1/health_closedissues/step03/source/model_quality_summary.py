#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings from a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ColumnSummary:
    name: str
    kind: str
    count: int
    missing: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    values: dict[str, int] | None = None
    mode: str | None = None


@dataclass(frozen=True)
class Prediction:
    """Estimated metrics for a configuration, based on nearby evaluated rows."""
    neighbors: int
    predicted_error: float | None
    predicted_accuracy: float | None
    predicted_metrics: dict[str, float | None]


@dataclass(frozen=True)
class Certification:
    """Evidence about whether one configuration family reliably wins."""
    family_field: str
    family_a: str
    family_b: str
    metrics: dict[str, dict[str, Any]]
    reliable_winner: str | None
    confidence: float
    samples_a: int
    samples_b: int


def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV rows while preserving source field names and values."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def _number(value: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def summarize_column(name: str, values: Iterable[str]) -> ColumnSummary:
    raw = list(values)
    numeric = [_number(value) for value in raw]
    if all(value is not None for value in numeric):
        numbers = [value for value in numeric if value is not None]
        return ColumnSummary(name, "numeric", len(numbers), len(raw) - len(numbers),
                             min(numbers) if numbers else None,
                             max(numbers) if numbers else None,
                             statistics.fmean(numbers) if numbers else None,
                             statistics.median(numbers) if numbers else None)
    present = [value for value in raw if value and value.strip()]
    counts = Counter(present)
    return ColumnSummary(name, "categorical", len(present), len(raw) - len(present),
                         values=dict(sorted(counts.items())),
                         mode=counts.most_common(1)[0][0] if counts else None)


def classify_columns(fields: Sequence[str], metric_suffixes: tuple[str, ...] = ("+", "-")) -> tuple[list[str], list[str]]:
    """Classify quality metrics by suffix; remaining fields are configurations."""
    metrics = [field for field in fields if field.endswith(metric_suffixes)]
    return metrics, [field for field in fields if field not in metrics]


def summarize(path: str | Path) -> dict[str, Any]:
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    columns = [summarize_column(field, (row.get(field, "") for row in rows)) for field in fields]
    return {
        "source": str(path),
        "row_count": len(rows),
        "column_count": len(fields),
        "metric_fields": metric_fields,
        "configuration_fields": configuration_fields,
        "columns": [asdict(column) for column in columns],
    }


def _distance(left: Mapping[str, str], right: Mapping[str, str], fields: Sequence[str]) -> float:
    """Mixed-type distance; numeric differences are normalized by observed range."""
    distance = 0.0
    for field in fields:
        a, b = left.get(field, ""), right.get(field, "")
        na, nb = _number(a), _number(b)
        if na is not None and nb is not None:
            distance += abs(na - nb)
        elif a.strip() != b.strip():
            distance += 1.0
    return distance


def predict(path: str | Path, configuration: Mapping[str, Any], neighbors: int = 5) -> dict[str, Any]:
    """Predict quality metrics without evaluating the requested configuration.

    The estimate is a distance-weighted average of the closest rows in the CSV.
    The CSV's metric columns are treated as historical observations only.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    query = {field: str(configuration.get(field, "")) for field in configuration_fields}
    ranked = sorted(rows, key=lambda row: _distance(row, query, configuration_fields))[:neighbors]
    if not ranked:
        result = Prediction(0, None, None, {field: None for field in metric_fields})
    else:
        weights = [1.0 / (1.0 + _distance(row, query, configuration_fields)) for row in ranked]
        estimates = {}
        for field in metric_fields:
            observed = [_number(row.get(field, "")) for row in ranked]
            usable = [(value, weight) for value, weight in zip(observed, weights) if value is not None]
            estimates[field] = (sum(value * weight for value, weight in usable) / sum(weight for _, weight in usable)
                                if usable else None)
        result = Prediction(len(ranked), estimates.get("MRE-"), estimates.get("ACC+"), estimates)
    return asdict(result) | {"configuration": dict(configuration), "source": str(path)}


def certify(path: str | Path, family_field: str, family_a: Any, family_b: Any,
            metrics: Sequence[str] | None = None, confidence: float = 0.95,
            resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Compare two family values using bootstrap confidence intervals.

    A family is reliable only when every requested metric's interval excludes a
    tie in the same direction.  ``+`` metrics are maximized and ``-`` metrics
    are minimized.  The reported difference is family A minus family B after
    converting all metrics to a higher-is-better scale.
    """
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be between 0 and 1 and resamples must be positive")
    fields, rows = read_rows(path)
    if family_field not in fields:
        raise ValueError(f"unknown family field: {family_field}")
    a, b = str(family_a), str(family_b)
    group_a = [row for row in rows if row.get(family_field, "") == a]
    group_b = [row for row in rows if row.get(family_field, "") == b]
    if not group_a or not group_b:
        raise ValueError("both family values must occur in the dataset")
    metric_fields, _ = classify_columns(fields)
    selected = list(metrics) if metrics is not None else metric_fields
    unknown = [field for field in selected if field not in metric_fields]
    if unknown:
        raise ValueError(f"unknown metric fields: {', '.join(unknown)}")
    rng = random.Random(seed)
    alpha = (1.0 - confidence) / 2.0
    report = {}
    winners = []
    for metric in selected:
        sign = 1.0 if metric.endswith("+") else -1.0
        values_a = [sign * _number(row.get(metric, "")) for row in group_a]
        values_b = [sign * _number(row.get(metric, "")) for row in group_b]
        values_a = [value for value in values_a if value is not None]
        values_b = [value for value in values_b if value is not None]
        if not values_a or not values_b:
            report[metric] = {"difference": None, "interval": [None, None], "winner": None}
            continue
        observed = statistics.fmean(values_a) - statistics.fmean(values_b)
        differences = []
        for _ in range(resamples):
            sample_a = [values_a[rng.randrange(len(values_a))] for _ in values_a]
            sample_b = [values_b[rng.randrange(len(values_b))] for _ in values_b]
            differences.append(statistics.fmean(sample_a) - statistics.fmean(sample_b))
        differences.sort()
        low = differences[min(len(differences) - 1, int(alpha * len(differences)))]
        high = differences[min(len(differences) - 1, int((1 - alpha) * len(differences)))]
        winner = a if low > 0 else b if high < 0 else None
        report[metric] = {"difference": observed, "interval": [low, high], "winner": winner,
                          "direction": "higher_is_better"}
        if winner:
            winners.append(winner)
    reliable = winners[0] if winners and len(winners) == len(selected) and len(set(winners)) == 1 else None
    result = Certification(family_field, a, b, report, reliable, confidence, len(group_a), len(group_b))
    return asdict(result) | {"source": str(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON summary to this path")
    parser.add_argument("--predict", type=json.loads, metavar="JSON", help="Predict metrics for a configuration")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--certify", type=json.loads, metavar="JSON",
                        help="Certify two family values, e.g. '{\"field\":\"criterion\",\"a\":\"poisson\",\"b\":\"squared_error\"}'")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.certify is not None:
        spec = args.certify
        result = certify(args.csv_path, spec["field"], spec["a"], spec["b"],
                         spec.get("metrics"), args.confidence, args.resamples)
    else:
        result = predict(args.csv_path, args.predict, args.neighbors) if args.predict is not None else summarize(args.csv_path)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
