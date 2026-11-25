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


def blame(dataset, measure=None, slow_fraction=0.10):
    """Rank settings associated with the slowest measurements.

    The slow group is the highest ``slow_fraction`` of observed values.  For
    each binary setting, ``slow_rate_difference`` compares the fraction of
    slow rows when the setting is 1 with that fraction when it is 0.  This is
    an association screen, not a causal claim; settings can be confounded by
    the experimental design.
    """
    measure = measure or dataset.measures[0]
    if measure not in dataset.measures or not 0 < slow_fraction <= 1:
        raise ValueError("unknown measure or slow_fraction must be between 0 and 1")
    usable = [(row, _number(row[measure])) for row in dataset.rows]
    usable = [(row, value) for row, value in usable if value is not None]
    if not usable:
        raise ValueError("measure has no numeric observations")
    slow_count = max(1, math.ceil(len(usable) * slow_fraction))
    cutoff = sorted(value for _, value in usable)[-slow_count]
    slow_rows = {id(row) for row, value in usable if value >= cutoff}
    results = []
    for setting in dataset.settings:
        groups = {}
        for value in (0, 1):
            selected = [(row, result) for row, result in usable
                        if int(float(row[setting])) == value]
            slow_rate = (sum(id(row) in slow_rows for row, _ in selected) /
                         len(selected)) if selected else None
            values = [result for _, result in selected]
            groups[str(value)] = {"runs": len(selected), "slow_runs":
                sum(id(row) in slow_rows for row, _ in selected),
                "slow_rate": slow_rate,
                "mean": statistics.fmean(values) if values else None}
        rate_0, rate_1 = groups["0"]["slow_rate"], groups["1"]["slow_rate"]
        mean_0, mean_1 = groups["0"]["mean"], groups["1"]["mean"]
        results.append({"setting": setting, "by_value": groups,
                        "slow_rate_difference": (rate_1 - rate_0
                                                  if rate_0 is not None and rate_1 is not None
                                                  else None),
                        "mean_difference": (mean_1 - mean_0
                                             if mean_0 is not None and mean_1 is not None
                                             else None)})
    results.sort(key=lambda item: (item["slow_rate_difference"] is None,
                                   -(item["slow_rate_difference"] or 0),
                                   -(item["mean_difference"] or 0), item["setting"]))
    return {"measure": measure, "slow_fraction": slow_fraction,
            "slow_count": slow_count, "cutoff": cutoff,
            "method": "slow-tail association", "settings": results}


class OnlineEvaluator:
    """Evaluate completed benchmark results incrementally, one at a time.

    A result is retained as a running count and mean for its binary setting
    configuration.  The current best configuration is the highest observed
    mean (appropriate for the Performance- measure), with deterministic
    lexicographic tie-breaking.
    """

    def __init__(self, settings, measure="Performance-"):
        self.settings = tuple(settings)
        if not self.settings:
            raise ValueError("at least one setting is required")
        self.measure = measure
        self._groups = {}
        self._results_seen = 0

    def _configuration_key(self, configuration):
        missing = [name for name in self.settings if name not in configuration]
        extra = [name for name in configuration if name not in self.settings]
        if missing or extra:
            raise ValueError("configuration must contain exactly the evaluator settings")
        try:
            values = tuple(int(configuration[name]) for name in self.settings)
        except (TypeError, ValueError):
            raise ValueError("setting values must be 0 or 1")
        if any(value not in (0, 1) for value in values):
            raise ValueError("setting values must be 0 or 1")
        return values

    def add_result(self, configuration, value):
        """Add one completed benchmark result and return the current state."""
        key = self._configuration_key(configuration)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("benchmark result must be numeric")
        if not math.isfinite(value):
            raise ValueError("benchmark result must be finite")
        count, mean = self._groups.get(key, (0, 0.0))
        count += 1
        mean += (value - mean) / count
        self._groups[key] = (count, mean)
        self._results_seen += 1
        return self.snapshot()

    def best(self):
        if not self._groups:
            return None
        key, (count, mean) = min(
            self._groups.items(), key=lambda item: (-item[1][1], item[0]))
        return {"configuration": dict(zip(self.settings, key)),
                "mean": mean, "count": count, "measure": self.measure}

    def snapshot(self):
        groups = {}
        for key in sorted(self._groups):
            count, mean = self._groups[key]
            groups["|".join(map(str, key))] = {"configuration": dict(zip(self.settings, key)),
                                                "count": count, "mean": mean}
        return {"measure": self.measure, "results_seen": self._results_seen,
                "configurations_seen": len(self._groups), "groups": groups,
                "best": self.best()}


def evaluate_rows(dataset):
    """Feed dataset rows to OnlineEvaluator in file order."""
    evaluator = OnlineEvaluator(dataset.settings, dataset.measures[0])
    for row in dataset.rows:
        configuration = {name: row[name] for name in dataset.settings}
        evaluator.add_result(configuration, row[dataset.measures[0]])
    return evaluator.snapshot()


def justify(dataset, measure=None, alpha=0.05, slow_fraction=0.10):
    """Build a concise, evidence-backed recommendation for an administrator.

    The recommendation is the highest observed configuration in the dataset;
    supporting certification and slow-tail evidence are included without
    presenting observational associations as causal proof.
    """
    measure = measure or dataset.measures[0]
    if measure not in dataset.measures:
        raise ValueError("unknown measure: " + measure)
    online = evaluate_rows(dataset)
    best = online["best"]
    certification = certify(dataset, measure, alpha)
    tail = blame(dataset, measure, slow_fraction)
    evidence = [
        "It is the highest observed mean in the supplied benchmark data.",
        "The result is based on {} observed configuration(s) and {} completed run(s).".format(
            online["configurations_seen"], online["results_seen"]),
    ]
    if certification["credible_improvements"]:
        evidence.append("Credible positive setting effects under the screening test: {}.".format(
            ", ".join(certification["credible_improvements"])))
    else:
        evidence.append("No setting passed the configured statistical screening threshold.")
    concerns = [
        "Validate with representative production clips and a staged rollout.",
        "The dataset is observational; certification and slow-tail results do not establish causality.",
    ]
    return {
        "measure": measure,
        "recommendation": best,
        "decision": "recommend",
        "evidence": evidence,
        "certification": certification,
        "slow_tail": tail,
        "limitations": concerns,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize, predict, or certify encoder measurements")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--predict-json", metavar="JSON")
    parser.add_argument("--measure")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--online", action="store_true",
                        help="evaluate rows incrementally and report current best")
    parser.add_argument("--blame", action="store_true",
                        help="rank settings associated with the slowest rows")
    parser.add_argument("--slow-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--justify", action="store_true",
                        help="justify the best observed configuration for an administrator")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.csv_path)
    if args.justify:
        result = justify(dataset, args.measure, args.alpha, args.slow_fraction)
    elif args.blame:
        result = blame(dataset, args.measure, args.slow_fraction)
    elif args.certify:
        result = certify(dataset, args.measure, args.alpha)
    elif args.predict_json is not None:
        result = predict(dataset, json.loads(args.predict_json), args.measure)
    elif args.online:
        result = evaluate_rows(dataset)
    else:
        result = summarize(dataset)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
