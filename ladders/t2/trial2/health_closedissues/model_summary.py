"""Summarize, predict, and certify model-configuration families."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summarize_numeric(values):
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "minimum": ordered[0],
            "maximum": ordered[-1]}


def summarize_rows(rows, quality_columns=None, configuration_columns=None):
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    columns = list(rows[0])
    quality_columns = list(quality_columns or columns[:3])
    configuration_columns = list(configuration_columns or columns[3:])
    missing = [c for c in quality_columns + configuration_columns if c not in columns]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(missing))
    quality = {}
    for column in quality_columns:
        usable = [_number(row[column]) for row in rows]
        usable = [v for v in usable if v is not None]
        quality[column] = _summarize_numeric(usable) if usable else {"count": 0}
    configuration = {}
    for column in configuration_columns:
        values = [row[column] for row in rows]
        numeric = [_number(v) for v in values]
        if all(v is not None for v in numeric):
            configuration[column] = {"kind": "numeric", **_summarize_numeric(numeric)}
        else:
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            configuration[column] = {"kind": "categorical", "count": len(values),
                                     "unique": len(counts), "values": dict(sorted(counts.items()))}
    return {"row_count": len(rows), "quality_metrics": quality,
            "configuration": configuration}


def summarize_csv(path, quality_columns=None, configuration_columns=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        return summarize_rows(reader, quality_columns, configuration_columns)


def predict_quality(rows, target_configuration, quality_columns=("MRE-", "ACC+"),
                    configuration_columns=None, neighbors=25):
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    configuration_columns = list(configuration_columns or
                                 [c for c in rows[0] if c not in quality_columns])
    missing = [c for c in list(quality_columns) + configuration_columns if c not in rows[0]]
    missing += [c for c in configuration_columns if c not in target_configuration]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(dict.fromkeys(missing)))
    numeric_ranges = {}
    for column in configuration_columns:
        values = [_number(row[column]) for row in rows]
        if all(value is not None for value in values):
            span = max(values) - min(values)
            numeric_ranges[column] = (min(values), span or 1.0)
    scored = []
    for row in rows:
        distance = 0.0
        for column in configuration_columns:
            if column in numeric_ranges:
                low, span = numeric_ranges[column]
                distance += ((_number(row[column]) - _number(target_configuration[column])) / span) ** 2
            elif row[column] != str(target_configuration[column]):
                distance += 1.0
        quality = [_number(row[column]) for column in quality_columns]
        if all(value is not None for value in quality):
            scored.append((math.sqrt(distance), quality))
    if not scored:
        raise ValueError("No rows have usable quality metrics")
    scored.sort(key=lambda item: item[0])
    chosen = scored[:max(1, min(neighbors, len(scored)))]
    weights = [1.0 / (distance + 1e-9) for distance, _ in chosen]
    prediction = [sum(weight * quality[index] for weight, (_, quality) in zip(weights, chosen)) /
                  sum(weights) for index in range(len(quality_columns))]
    return {"predictions": dict(zip(quality_columns, prediction)),
            "neighbors_used": len(chosen), "nearest_distance": chosen[0][0],
            "baseline": {column: statistics.fmean(item[1][index] for item in scored)
                         for index, column in enumerate(quality_columns)}}


def certify_families(rows, family_column, family_a, family_b, metric="ACC+",
                     higher_is_better=True, effect_threshold=0.0, resamples=2000,
                     seed=0, confidence=0.95):
    """Test whether family A reliably beats B using a bootstrap CI for A-B.

    A family is a categorical value in ``family_column``. Reliability means the
    confidence interval excludes zero in the desired direction and the observed
    effect reaches ``effect_threshold``. The returned verdict is conservative.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    if family_column not in rows[0] or metric not in rows[0]:
        raise ValueError("Unknown family or metric column")
    a = [_number(row[metric]) for row in rows if str(row[family_column]) == str(family_a)]
    b = [_number(row[metric]) for row in rows if str(row[family_column]) == str(family_b)]
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if not a or not b:
        raise ValueError("Both families need at least one usable metric value")
    observed = statistics.fmean(a) - statistics.fmean(b)
    rng = random.Random(seed)
    differences = []
    for _ in range(max(1, resamples)):
        differences.append(statistics.fmean(rng.choice(a) for _ in a) -
                           statistics.fmean(rng.choice(b) for _ in b))
    differences.sort()
    alpha = (1.0 - confidence) / 2.0
    low = differences[min(len(differences) - 1, int(alpha * len(differences)))]
    high = differences[min(len(differences) - 1, int((1.0 - alpha) * len(differences)))]
    signed_effect = observed if higher_is_better else -observed
    signed_low = low if higher_is_better else -high
    reliable = signed_low > effect_threshold
    return {"family_column": family_column, "family_a": str(family_a), "family_b": str(family_b),
            "metric": metric, "sample_sizes": {"a": len(a), "b": len(b)},
            "mean_a": statistics.fmean(a), "mean_b": statistics.fmean(b),
            "observed_effect_a_minus_b": observed, "confidence_interval": [low, high],
            "confidence": confidence, "effect_threshold": effect_threshold,
            "higher_is_better": higher_is_better, "reliably_better": reliable,
            "verdict": "A reliably better than B" if reliable else "No reliable advantage for A"}


def blame_hyperparameters(rows, error_column="MRE-", configuration_columns=None,
                          high_error_quantile=0.75):
    """Describe which configuration values are associated with high error.

    High error is defined as error at or above the requested empirical
    quantile. Numeric parameters use high-vs-low error means and a Pearson
    correlation with the high-error indicator. Categorical parameters use the
    high-error rate for each value and the spread between those rates.
    Results are ranked by the absolute strength of the reported association.
    This is descriptive evidence, not a causal model.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    if not 0.0 <= high_error_quantile <= 1.0:
        raise ValueError("high_error_quantile must be between 0 and 1")
    columns = list(rows[0])
    if error_column not in columns:
        raise ValueError("Unknown error column: " + error_column)
    configuration_columns = list(configuration_columns or
                                 [c for c in columns if c != error_column])
    missing = [c for c in configuration_columns if c not in columns]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(missing))
    usable = [(row, _number(row[error_column])) for row in rows]
    usable = [(row, error) for row, error in usable if error is not None]
    if not usable:
        raise ValueError("No rows have usable error values")
    errors = sorted(error for _, error in usable)
    threshold = errors[int((len(errors) - 1) * high_error_quantile)]
    high = [error >= threshold for _, error in usable]
    high_count = sum(high)
    low_count = len(high) - high_count
    results = []
    for column in configuration_columns:
        numeric = [_number(row[column]) for row, _ in usable]
        if all(value is not None for value in numeric):
            high_values = [value for value, flag in zip(numeric, high) if flag]
            low_values = [value for value, flag in zip(numeric, high) if not flag]
            mean_high = statistics.fmean(high_values)
            mean_low = statistics.fmean(low_values) if low_values else mean_high
            mean_difference = mean_high - mean_low
            p_high = high_count / len(high)
            # Pearson correlation with the binary high-error indicator.
            mean_value = statistics.fmean(numeric)
            mean_flag = p_high
            covariance = sum((value - mean_value) * (flag - mean_flag)
                             for value, flag in zip(numeric, high))
            denom_value = math.sqrt(sum((value - mean_value) ** 2 for value in numeric))
            denom_flag = math.sqrt(sum((flag - mean_flag) ** 2 for flag in high))
            correlation = covariance / (denom_value * denom_flag) if denom_value and denom_flag else 0.0
            results.append({"hyperparameter": column, "kind": "numeric",
                            "association": mean_difference, "mean_high_error": mean_high,
                            "mean_low_error": mean_low, "pearson_high_error": correlation,
                            "strength": abs(correlation)})
        else:
            groups = {}
            for (row, _), flag in zip(usable, high):
                value = str(row[column])
                item = groups.setdefault(value, {"count": 0, "high_error_count": 0})
                item["count"] += 1
                item["high_error_count"] += int(flag)
            rates = {value: item["high_error_count"] / item["count"]
                     for value, item in groups.items()}
            ordered = sorted(rates.items(), key=lambda item: (-item[1], item[0]))
            spread = ordered[0][1] - ordered[-1][1]
            results.append({"hyperparameter": column, "kind": "categorical",
                            "association": spread, "high_error_rate_by_value": rates,
                            "strength": spread, "highest_rate_value": ordered[0][0]})
    results.sort(key=lambda item: (-item["strength"], item["hyperparameter"]))
    return {"error_column": error_column, "high_error_quantile": high_error_quantile,
            "high_error_threshold": threshold, "row_count": len(usable),
            "high_error_count": high_count, "associations": results}


def justify_configuration(rows, target_configuration, quality_columns=("MRE-", "ACC+"),
                          configuration_columns=None, error_column="MRE-",
                          accuracy_column="ACC+", neighbors=25, margin=0.0):
    """Create an evidence-based recommendation for a candidate configuration.

    The recommendation compares nearest-neighbor predictions with dataset-wide
    baselines and states the decision rule explicitly.  It is decision support,
    not a replacement for prospective validation.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    quality_columns = list(quality_columns)
    configuration_columns = list(configuration_columns or
                                 [c for c in rows[0] if c not in quality_columns])
    prediction = predict_quality(rows, target_configuration, quality_columns,
                                 configuration_columns, neighbors)
    if error_column not in prediction["predictions"] or accuracy_column not in prediction["predictions"]:
        raise ValueError("quality_columns must include error and accuracy metrics")
    error_delta = prediction["baseline"][error_column] - prediction["predictions"][error_column]
    accuracy_delta = prediction["predictions"][accuracy_column] - prediction["baseline"][accuracy_column]
    recommended = error_delta >= margin and accuracy_delta >= margin
    reasons = []
    reasons.append("predicted error is below the dataset baseline" if error_delta >= margin
                   else "predicted error does not improve on the dataset baseline")
    reasons.append("predicted accuracy is at or above the dataset baseline" if accuracy_delta >= margin
                   else "predicted accuracy is below the dataset baseline")
    return {"recommendation": "recommend" if recommended else "do_not_recommend",
            "configuration": dict(target_configuration), "predictions": prediction["predictions"],
            "baseline": prediction["baseline"],
            "deltas": {"error_reduction": error_delta, "accuracy_gain": accuracy_delta},
            "neighbors_used": prediction["neighbors_used"],
            "nearest_distance": prediction["nearest_distance"], "margin": margin,
            "reasons": reasons,
            "caveat": "Validate prospectively before production deployment."}


class SequentialEvaluator:
    """Evaluate configurations incrementally as validation results arrive.

    Results are kept in arrival order; the best result is selected using the
    requested metric direction, with earlier arrivals breaking ties.
    """

    def __init__(self, metric="ACC+", higher_is_better=True):
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.results = []
        self._best = None

    def add_result(self, configuration, validation_result):
        if not isinstance(configuration, dict) or not configuration:
            raise ValueError("configuration must be a non-empty dictionary")
        value = _number(validation_result if not isinstance(validation_result, dict)
                        else validation_result.get(self.metric))
        if value is None:
            raise ValueError("validation result must contain a finite metric value")
        record = {"configuration": dict(configuration), self.metric: value,
                  "index": len(self.results)}
        self.results.append(record)
        if (self._best is None or
                (value > self._best[self.metric] if self.higher_is_better
                 else value < self._best[self.metric])):
            self._best = record
        return self.snapshot()

    def snapshot(self):
        return {"metric": self.metric, "higher_is_better": self.higher_is_better,
                "evaluated": len(self.results),
                "best": None if self._best is None else dict(self._best)}

    def summary(self):
        return {**self.snapshot(), "results": [dict(result) for result in self.results]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--predict", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--certify", nargs=2, metavar=("FAMILY_A", "FAMILY_B"))
    parser.add_argument("--family-column", default="criterion")
    parser.add_argument("--metric", default="ACC+")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--sequential", action="store_true",
                        help="evaluate JSON lines: {configuration, validation_result}")
    parser.add_argument("--blame", action="store_true",
                        help="rank hyperparameters associated with high error")
    parser.add_argument("--error", default="MRE-", dest="error_column")
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--justify", action="store_true",
                        help="justify a candidate configuration against the baseline")
    parser.add_argument("--target", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--quality", nargs="+")
    parser.add_argument("--configuration", nargs="+")
    args = parser.parse_args(argv)
    if args.sequential:
        evaluator = SequentialEvaluator(args.metric, not args.lower_is_better)
        for line in args.csv_path.open(encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                evaluator.add_result(item["configuration"], item["validation_result"])
        print(json.dumps(evaluator.summary(), indent=2, sort_keys=True))
        return
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.justify:
        target = dict(item.split("=", 1) for item in (args.target or []))
        result = justify_configuration(rows, target, args.quality or ["MRE-", "ACC+"],
                                       args.configuration, args.error_column, args.metric,
                                       margin=args.margin)
    elif args.blame:
        result = blame_hyperparameters(rows, args.error_column, args.configuration,
                                       args.high_quantile)
    elif args.certify:
        result = certify_families(rows, args.family_column, *args.certify, args.metric,
                                  not args.lower_is_better)
    elif args.predict is not None:
        target = dict(item.split("=", 1) for item in args.predict)
        result = predict_quality(rows, target, args.quality or ["MRE-", "ACC+"], args.configuration)
    else:
        result = summarize_rows(rows, args.quality, args.configuration)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
