#!/usr/bin/env python3
"""Summarize, predict, certify, evaluate, and blame configuration values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
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
    neighbors: int
    predicted_error: float | None
    predicted_accuracy: float | None
    predicted_metrics: dict[str, float | None]


@dataclass(frozen=True)
class Certification:
    family_field: str
    family_a: str
    family_b: str
    metrics: dict[str, dict[str, Any]]
    reliable_winner: str | None
    confidence: float
    samples_a: int
    samples_b: int


@dataclass(frozen=True)
class BlameResult:
    """Per-hyperparameter evidence for values associated with high error."""
    error_field: str
    high_error_threshold: float
    overall_mean_error: float | None
    parameters: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Justification:
    """Evidence-backed recommendation for a platform configuration."""
    configuration: dict[str, Any]
    validation_results: dict[str, Any]
    rationale: list[str]
    caveats: list[str]


def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        return list(reader.fieldnames), [dict(row) for row in reader]


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
                             min(numbers) if numbers else None, max(numbers) if numbers else None,
                             statistics.fmean(numbers) if numbers else None,
                             statistics.median(numbers) if numbers else None)
    present = [value for value in raw if value and value.strip()]
    counts = Counter(present)
    return ColumnSummary(name, "categorical", len(present), len(raw) - len(present),
                         values=dict(sorted(counts.items())),
                         mode=counts.most_common(1)[0][0] if counts else None)


def classify_columns(fields: Sequence[str], metric_suffixes: tuple[str, ...] = ("+", "-")) -> tuple[list[str], list[str]]:
    metrics = [field for field in fields if field.endswith(metric_suffixes)]
    return metrics, [field for field in fields if field not in metrics]


def summarize(path: str | Path) -> dict[str, Any]:
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    return {"source": str(path), "row_count": len(rows), "column_count": len(fields),
            "metric_fields": metric_fields, "configuration_fields": configuration_fields,
            "columns": [asdict(summarize_column(field, (row.get(field, "") for row in rows))) for field in fields]}


def _distance(left: Mapping[str, str], right: Mapping[str, str], fields: Sequence[str]) -> float:
    distance = 0.0
    for field in fields:
        a, b = left.get(field, ""), right.get(field, "")
        na, nb = _number(a), _number(b)
        distance += abs(na - nb) if na is not None and nb is not None else (0.0 if a.strip() == b.strip() else 1.0)
    return distance


def predict(path: str | Path, configuration: Mapping[str, Any], neighbors: int = 5) -> dict[str, Any]:
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    query = {field: str(configuration.get(field, "")) for field in configuration_fields}
    ranked = sorted(rows, key=lambda row: _distance(row, query, configuration_fields))[:neighbors]
    weights = [1.0 / (1.0 + _distance(row, query, configuration_fields)) for row in ranked]
    estimates = {}
    for field in metric_fields:
        usable = [(_number(row.get(field, "")), weight) for row, weight in zip(ranked, weights)]
        usable = [(value, weight) for value, weight in usable if value is not None]
        estimates[field] = sum(value * weight for value, weight in usable) / sum(weight for _, weight in usable) if usable else None
    result = Prediction(len(ranked), estimates.get("MRE-"), estimates.get("ACC+"), estimates)
    return asdict(result) | {"configuration": dict(configuration), "source": str(path)}


def certify(path: str | Path, family_field: str, family_a: Any, family_b: Any,
            metrics: Sequence[str] | None = None, confidence: float = 0.95,
            resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be between 0 and 1 and resamples must be positive")
    fields, rows = read_rows(path)
    if family_field not in fields:
        raise ValueError(f"unknown family field: {family_field}")
    a, b = str(family_a), str(family_b)
    group_a, group_b = [row for row in rows if row.get(family_field, "") == a], [row for row in rows if row.get(family_field, "") == b]
    if not group_a or not group_b:
        raise ValueError("both family values must occur in the dataset")
    metric_fields, _ = classify_columns(fields)
    selected = list(metrics) if metrics is not None else metric_fields
    if any(metric not in metric_fields for metric in selected):
        raise ValueError("unknown metric field")
    rng, alpha, report, winners = random.Random(seed), (1 - confidence) / 2, {}, []
    for metric in selected:
        sign = 1.0 if metric.endswith("+") else -1.0
        va = [sign * _number(row.get(metric, "")) for row in group_a]
        vb = [sign * _number(row.get(metric, "")) for row in group_b]
        va, vb = [x for x in va if x is not None], [x for x in vb if x is not None]
        if not va or not vb:
            report[metric] = {"difference": None, "interval": [None, None], "winner": None}
            continue
        diffs = [statistics.fmean([va[rng.randrange(len(va))] for _ in va]) - statistics.fmean([vb[rng.randrange(len(vb))] for _ in vb]) for _ in range(resamples)]
        diffs.sort(); low = diffs[min(len(diffs) - 1, int(alpha * len(diffs)))]; high = diffs[min(len(diffs) - 1, int((1 - alpha) * len(diffs)))]
        winner = a if low > 0 else b if high < 0 else None
        report[metric] = {"difference": statistics.fmean(va) - statistics.fmean(vb), "interval": [low, high], "winner": winner, "direction": "higher_is_better"}
        if winner: winners.append(winner)
    result = Certification(family_field, a, b, report, winners[0] if winners and len(winners) == len(selected) and len(set(winners)) == 1 else None, confidence, len(group_a), len(group_b))
    return asdict(result) | {"source": str(path)}


class IncrementalEvaluator:
    """Evaluate complete validation results in arrival order and retain the best result."""

    def __init__(self, metric_fields: Sequence[str]):
        if not metric_fields:
            raise ValueError("at least one metric field is required")
        self.metric_fields = tuple(metric_fields)
        self.results: list[dict[str, Any]] = []

    def submit(self, configuration: Mapping[str, Any], validation_results: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in self.metric_fields if _number(str(validation_results.get(field, ""))) is None]
        if missing:
            raise ValueError(f"missing or invalid validation metrics: {', '.join(missing)}")
        result = {"configuration": dict(configuration), "validation_results": dict(validation_results), "arrival_index": len(self.results)}
        self.results.append(result)
        result["is_best"] = result is self.best_result()
        return dict(result)

    def best_result(self) -> dict[str, Any] | None:
        if not self.results:
            return None
        return min(self.results, key=self._score)

    def _score(self, result: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple((-1 if field.endswith("+") else 1) * float(result["validation_results"][field]) for field in self.metric_fields)

    def report(self) -> dict[str, Any]:
        best = self.best_result()
        return {"evaluated": len(self.results), "best": best, "metric_fields": list(self.metric_fields)}


def evaluate_file(path: str | Path) -> dict[str, Any]:
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    evaluator = IncrementalEvaluator(metric_fields)
    for row in rows:
        evaluator.submit({field: row.get(field, "") for field in configuration_fields}, {field: row.get(field, "") for field in metric_fields})
    return evaluator.report() | {"source": str(path)}


def blame(path: str | Path, error_field: str = "MRE-", quantile: float = 0.75,
          min_samples: int = 1) -> dict[str, Any]:
    """Rank hyperparameter values by mean error and high-error rate.

    The high-error threshold is the requested quantile of observed errors.  This
    is an association analysis, not a causal claim; each parameter is analyzed
    independently and sparse groups are retained only when they meet
    ``min_samples``.
    """
    if not 0 < quantile <= 1 or min_samples < 1:
        raise ValueError("quantile must be in (0, 1] and min_samples must be positive")
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    if error_field not in metric_fields:
        raise ValueError(f"unknown error metric: {error_field}")
    errors = [_number(row.get(error_field, "")) for row in rows]
    usable = [value for value in errors if value is not None]
    if not usable:
        raise ValueError("error metric has no numeric observations")
    ordered = sorted(usable)
    threshold = ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]
    overall = statistics.fmean(usable)
    parameter_reports: dict[str, list[dict[str, Any]]] = {}
    for field in configuration_fields:
        groups: dict[str, list[float]] = {}
        for row, error in zip(rows, errors):
            if error is not None:
                groups.setdefault(row.get(field, ""), []).append(error)
        report = []
        for value, group in groups.items():
            if len(group) < min_samples:
                continue
            report.append({"value": value, "count": len(group),
                           "mean_error": statistics.fmean(group),
                           "high_error_count": sum(error >= threshold for error in group),
                           "high_error_rate": sum(error >= threshold for error in group) / len(group),
                           "mean_error_lift": statistics.fmean(group) - overall})
        parameter_reports[field] = sorted(report, key=lambda item: (-item["mean_error"], item["value"]))
    return asdict(BlameResult(error_field, threshold, overall, parameter_reports)) | {"source": str(path)}


def justify(path: str | Path, neighbors: int = 5, quantile: float = 0.75,
            min_samples: int = 1) -> dict[str, Any]:
    """Recommend the observed best configuration and explain its evidence.

    The recommendation is the best complete validation row under the existing
    multi-metric ordering.  Prediction and blame are supporting evidence; they
    do not turn observational associations into causal claims.
    """
    evaluation = evaluate_file(path)
    best = evaluation["best"]
    if best is None:
        raise ValueError("dataset has no validation rows")
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    configuration = best["configuration"]
    prediction = predict(path, configuration, neighbors)
    blame_report = blame(path, quantile=quantile, min_samples=min_samples)
    rationale = [
        f"Selected the best of {evaluation['evaluated']} observed configurations using metric priorities {metric_fields}.",
        "The selected configuration has validation results " + json.dumps(best["validation_results"], sort_keys=True) + ".",
        f"A {prediction['neighbors']}-neighbor estimate predicts " + json.dumps(prediction['predicted_metrics'], sort_keys=True) + ".",
    ]
    caveats = [
        "This is evidence from the supplied closed-issue dataset, not a causal guarantee.",
        "Blame analysis reports associations for individual hyperparameter values and may be sparse.",
    ]
    return asdict(Justification(configuration, best["validation_results"], rationale, caveats)) | {
        "source": str(path), "evaluation": evaluation, "prediction": prediction,
        "blame": blame_report, "configuration_fields": configuration_fields,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--predict", type=json.loads); parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--certify", type=json.loads); parser.add_argument("--confidence", type=float, default=0.95); parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--evaluate", action="store_true", help="evaluate validation rows as they arrive")
    parser.add_argument("--blame", action="store_true", help="identify hyperparameter values associated with high error")
    parser.add_argument("--error-field", default="MRE-")
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--justify", action="store_true", help="recommend and explain a configuration")
    args = parser.parse_args(argv)
    if args.justify: result = justify(args.csv_path, args.neighbors, args.quantile, args.min_samples)
    elif args.blame: result = blame(args.csv_path, args.error_field, args.quantile, args.min_samples)
    elif args.evaluate: result = evaluate_file(args.csv_path)
    elif args.certify is not None:
        spec = args.certify; result = certify(args.csv_path, spec["field"], spec["a"], spec["b"], spec.get("metrics"), args.confidence, args.resamples)
    elif args.predict is not None: result = predict(args.csv_path, args.predict, args.neighbors)
    else: result = summarize(args.csv_path)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output: args.output.write_text(rendered + "\n", encoding="utf-8")
    else: print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
