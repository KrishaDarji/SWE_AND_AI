#!/usr/bin/env python3
"""Standard-library-only summaries, prediction, and improvement certification."""

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Sequence


class OnlineBenchmarkEvaluator:
    """Evaluate lower-is-better benchmark results as they arrive."""

    def __init__(self, settings=None):
        self.settings = tuple(settings) if settings is not None else None
        self.results = []

    def add_result(self, configuration, performance):
        configuration = {str(key): str(value).strip()
                         for key, value in configuration.items()}
        value = _number(str(performance))
        if value is None:
            raise ValueError("performance must be a finite number")
        if self.settings is None:
            self.settings = tuple(sorted(configuration))
        if set(configuration) != set(self.settings):
            raise ValueError("configuration columns do not match evaluator settings")
        record = {"configuration": configuration, "performance": value,
                  "index": len(self.results) + 1}
        self.results.append(record)
        return self.snapshot()

    def snapshot(self):
        ordered = sorted(self.results, key=lambda item: (item["performance"], item["index"]))
        return {"received": len(self.results),
                "best": ordered[0] if ordered else None,
                "ranking": ordered}


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


def _read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        return reader.fieldnames, list(reader)


def summarize_csv(path: Path) -> dict:
    fieldnames, rows = _read(path)
    summaries = []
    for name in fieldnames:
        raw = [(row.get(name) or "").strip() for row in rows]
        numbers = [_number(value) for value in raw]
        numeric = all(value is not None for value in numbers)
        distinct = set(raw)
        binary = numeric and distinct <= {"0", "1"}
        kind = "encoder_setting" if binary else "quality_measure" if numeric else "categorical"
        summary = ColumnSummary(name=name, kind=kind, count=len(raw), unique=len(distinct))
        if numeric:
            summary.minimum = min(numbers)
            summary.maximum = max(numbers)
            summary.mean = statistics.fmean(numbers)
            summary.median = statistics.median(numbers)
        elif len(distinct) <= 20:
            summary.values = dict(sorted((item, raw.count(item)) for item in distinct))
        summaries.append(summary)
    return {"dataset": str(path), "rows": len(rows), "columns": len(fieldnames),
            "encoder_settings": [asdict(item) for item in summaries if item.kind == "encoder_setting"],
            "quality_measures": [asdict(item) for item in summaries if item.kind == "quality_measure"],
            "other_columns": [asdict(item) for item in summaries if item.kind == "categorical"]}


def predict_encoding_time(path: Path, configuration: Dict[str, str]) -> dict:
    fieldnames, rows = _read(path)
    if "Performance-" not in fieldnames:
        raise ValueError("CSV must contain a Performance- column")
    settings = [name for name in fieldnames if name != "Performance-"]
    if set(configuration) != set(settings):
        raise ValueError("configuration columns do not match dataset")
    target = {name: str(configuration[name]).strip() for name in settings}
    candidates = []
    for row in rows:
        time = _number(row.get("Performance-", ""))
        if time is not None:
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


def certify_improvement(path: Path, baseline: Dict[str, str], candidate: Dict[str, str], alpha=0.05) -> dict:
    """Test whether candidate is faster than baseline on matched contexts.

    For each row pair sharing every setting not changed by the configurations,
    a positive difference means candidate has lower Performance-.  The exact
    one-sided sign-test p-value avoids assumptions about timing distribution.
    """
    fieldnames, rows = _read(path)
    if "Performance-" not in fieldnames or set(baseline) != set(fieldnames) - {"Performance-"} or set(candidate) != set(baseline):
        raise ValueError("baseline and candidate must specify every dataset setting")
    baseline = {key: str(value).strip() for key, value in baseline.items()}
    candidate = {key: str(value).strip() for key, value in candidate.items()}
    changed = [key for key in baseline if baseline[key] != candidate[key]]
    if not changed:
        raise ValueError("baseline and candidate must differ")
    fixed = [key for key in baseline if key not in changed]
    grouped = {}
    for row in rows:
        value = _number(row.get("Performance-", ""))
        if value is None or any((row.get(key) or "").strip() != baseline[key] for key in fixed):
            continue
        key = tuple((row.get(name) or "").strip() for name in fixed)
        grouped.setdefault(key, {})
        if all((row.get(name) or "").strip() == candidate[name] for name in changed):
            grouped[key].setdefault("candidate", []).append(value)
        elif all((row.get(name) or "").strip() == baseline[name] for name in changed):
            grouped[key].setdefault("baseline", []).append(value)
    differences = [baseline_value - candidate_value
                   for item in grouped.values() if "baseline" in item and "candidate" in item
                   for baseline_value, candidate_value in zip(item["baseline"], item["candidate"])]
    nonzero = [difference for difference in differences if difference != 0]
    wins = sum(difference > 0 for difference in nonzero)
    n = len(nonzero)
    p_value = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n) if n else 1.0
    mean = statistics.fmean(differences) if differences else None
    return {"baseline": baseline, "candidate": candidate, "changed_settings": changed,
            "pairs": len(differences), "wins": wins, "ties": len(differences) - n,
            "mean_improvement": mean, "p_value": p_value, "alpha": alpha,
            "credible": bool(differences) and mean > 0 and p_value < alpha,
            "test": "one-sided exact sign test"}


def _configuration(items):
    if any("=" not in item for item in items):
        raise ValueError("configuration values must use NAME=VALUE")
    return dict(item.split("=", 1) for item in items)


def evaluate_stream(records, settings=None):
    """Consume iterable records with configuration and Performance- fields."""
    evaluator = OnlineBenchmarkEvaluator(settings)
    snapshot = None
    for record in records:
        if "configuration" in record:
            configuration = record["configuration"]
        else:
            configuration = {key: value for key, value in record.items()
                            if key != "Performance-"}
        performance = record.get("performance", record.get("Performance-"))
        snapshot = evaluator.add_result(configuration, performance)
    return snapshot or evaluator.snapshot()


def blame_slow_encodes(path: Path, slow_quantile=0.9) -> dict:
    """Rank binary settings associated with the slowest observed encodes.

    A setting's penalty is mean(time | setting=1) - mean(time | setting=0).
    Slow-tail enrichment compares the fraction of its 1-valued rows in the
    slow tail with the overall fraction.  These are associations, not causal
    effects; the ranking exposes both signals for interpretation.
    """
    if not 0 < slow_quantile <= 1:
        raise ValueError("slow_quantile must be in (0, 1]")
    fieldnames, rows = _read(path)
    if "Performance-" not in fieldnames:
        raise ValueError("CSV must contain a Performance- column")
    usable = []
    for row in rows:
        performance = _number(row.get("Performance-", ""))
        if performance is not None:
            usable.append((row, performance))
    if not usable:
        raise ValueError("dataset has no numeric Performance- values")
    count = max(1, math.ceil(len(usable) * (1 - slow_quantile)))
    cutoff = sorted(value for _, value in usable)[-count]
    slow_rows = [(row, value) for row, value in usable if value >= cutoff]
    results = []
    for setting in fieldnames:
        if setting == "Performance-":
            continue
        values = [(row.get(setting) or "").strip() for row, _ in usable]
        if set(values) != {"0", "1"}:
            continue
        groups = {value: [time for (row, time), actual in zip(usable, values)
                          if actual == value] for value in ("0", "1")}
        slow_groups = {value: sum((row.get(setting) or "").strip() == value
                                  for row, _ in slow_rows) for value in ("0", "1")}
        overall_one = len(groups["1"]) / len(usable)
        slow_one = slow_groups["1"] / len(slow_rows)
        mean_zero = statistics.fmean(groups["0"])
        mean_one = statistics.fmean(groups["1"])
        results.append({"setting": setting, "mean_time_0": mean_zero,
                        "mean_time_1": mean_one,
                        "mean_penalty_1_minus_0": mean_one - mean_zero,
                        "overall_fraction_1": overall_one,
                        "slow_fraction_1": slow_one,
                        "slow_tail_enrichment": slow_one - overall_one,
                        "count_0": len(groups["0"]), "count_1": len(groups["1"])})
    results.sort(key=lambda item: (item["slow_tail_enrichment"],
                                   item["mean_penalty_1_minus_0"]), reverse=True)
    return {"dataset": str(path), "rows_used": len(usable),
            "slow_quantile": slow_quantile, "slow_rows": len(slow_rows),
            "slow_cutoff": cutoff, "settings": results}


def justify_recommendation(path: Path, configuration=None) -> dict:
    """Build an administrator-facing justification for a lower-time configuration.

    With no configuration, the fastest observed configuration is recommended.
    The recommendation is evidence-based but observational: it reports the
    measured row, comparison to the dataset mean, and important limitations.
    """
    fieldnames, rows = _read(path)
    if "Performance-" not in fieldnames:
        raise ValueError("CSV must contain a Performance- column")
    settings = [name for name in fieldnames if name != "Performance-"]
    observations = []
    for index, row in enumerate(rows, 1):
        performance = _number(row.get("Performance-", ""))
        if performance is not None:
            observations.append((performance, index, {name: (row.get(name) or "").strip()
                                                       for name in settings}))
    if not observations:
        raise ValueError("dataset has no numeric Performance- values")
    mean_time = statistics.fmean(item[0] for item in observations)
    if configuration is None:
        performance, row_number, selected = min(observations, key=lambda item: (item[0], item[1]))
        selection = "fastest observed configuration"
    else:
        if set(configuration) != set(settings):
            raise ValueError("configuration columns do not match dataset")
        selected = {name: str(configuration[name]).strip() for name in settings}
        matches = [(time, index) for time, index, row in observations if row == selected]
        if not matches:
            prediction = predict_encoding_time(path, selected)
            performance, row_number = prediction["predicted_time"], None
            selection = "requested configuration with predicted performance"
        else:
            performance, row_number = min(matches)
            selection = "requested configuration with observed performance"
    improvement = mean_time - performance
    percent = (improvement / mean_time * 100.0) if mean_time else 0.0
    rationale = ("Recommend this configuration because it has the lowest observed encoding time "
                 "in the supplied benchmark data. It is estimated to be "
                 f"{improvement:.3f} time units ({percent:.2f}%) faster than the dataset mean. "
                 "Validate on representative production clips before rollout.")
    return {"dataset": str(path), "recommendation": selected,
            "selection": selection, "observed_row": row_number,
            "performance": performance, "dataset_mean": mean_time,
            "improvement_vs_mean": improvement,
            "improvement_percent_vs_mean": percent,
            "rationale": rationale,
            "limitations": ["Benchmark associations are not causal.",
                            "The dataset may not represent production content.",
                            "A single fastest row may need repeat measurements."],
            "evidence_rows": len(observations)}


def select_best_configuration(path: Path, quality_column: str,
                              minimum_quality=None, maximum_quality=None) -> dict:
    """Select the fastest observed configuration satisfying a quality bound.

    Quality bounds are inclusive.  Encoding performance remains the objective
    and is minimized, while the selected row supplies the configuration.
    """
    if minimum_quality is None and maximum_quality is None:
        raise ValueError("at least one quality bound is required")
    if (minimum_quality is not None and maximum_quality is not None
            and minimum_quality > maximum_quality):
        raise ValueError("minimum quality cannot exceed maximum quality")
    fieldnames, rows = _read(path)
    if quality_column not in fieldnames:
        raise ValueError("quality column is not present in dataset")
    if "Performance-" not in fieldnames:
        raise ValueError("CSV must contain a Performance- column")
    settings = [name for name in fieldnames
                if name not in (quality_column, "Performance-")]
    eligible = []
    for index, row in enumerate(rows, 1):
        quality = _number(row.get(quality_column, ""))
        performance = _number(row.get("Performance-", ""))
        if quality is None or performance is None:
            continue
        if minimum_quality is not None and quality < minimum_quality:
            continue
        if maximum_quality is not None and quality > maximum_quality:
            continue
        configuration = {name: (row.get(name) or "").strip() for name in settings}
        eligible.append((performance, index, quality, configuration))
    if not eligible:
        raise ValueError("no observed configuration satisfies the quality constraint")
    performance, row_number, quality, configuration = min(
        eligible, key=lambda item: (item[0], item[1]))
    return {"dataset": str(path), "quality_column": quality_column,
            "minimum_quality": minimum_quality, "maximum_quality": maximum_quality,
            "recommendation": configuration, "quality": quality,
            "performance": performance, "observed_row": row_number,
            "eligible_rows": len(eligible), "selection": "fastest eligible observed configuration"}


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predict", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--baseline", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--candidate", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--stream", action="store_true",
                        help="read one JSON benchmark result per stdin line")
    parser.add_argument("--blame", action="store_true",
                        help="rank settings associated with slow encodes")
    parser.add_argument("--slow-quantile", type=float, default=0.9)
    parser.add_argument("--justify", nargs="*", metavar="NAME=VALUE",
                        help="justify a requested or fastest observed configuration")
    parser.add_argument("--quality-column", help="numeric column constrained during selection")
    parser.add_argument("--minimum-quality", type=float)
    parser.add_argument("--maximum-quality", type=float)
    args = parser.parse_args(argv)
    if args.stream:
        records = (json.loads(line) for line in __import__("sys").stdin if line.strip())
        result = evaluate_stream(records)
    elif args.quality_column is not None:
        result = select_best_configuration(args.csv_path, args.quality_column,
                                           args.minimum_quality, args.maximum_quality)
    elif args.justify is not None:
        result = justify_recommendation(args.csv_path,
                                        _configuration(args.justify) if args.justify else None)
    elif args.blame:
        result = blame_slow_encodes(args.csv_path, args.slow_quantile)
    elif args.baseline is not None or args.candidate is not None:
        if args.baseline is None or args.candidate is None:
            parser.error("--baseline and --candidate are both required")
        result = certify_improvement(args.csv_path, _configuration(args.baseline), _configuration(args.candidate), args.alpha)
    elif args.predict is not None:
        result = predict_encoding_time(args.csv_path, _configuration(args.predict))
    else:
        result = summarize_csv(args.csv_path)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
