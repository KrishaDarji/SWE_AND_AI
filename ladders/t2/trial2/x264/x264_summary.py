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


def choose(dataset, minimum_quality, quality_measure=None, objective_measure=None):
    """Select the best observed configuration satisfying a quality threshold.

    Quality is constrained as ``quality >= minimum_quality``.  The objective
    defaults to the quality measure and is maximized, matching the
    Performance- convention used by the earlier online evaluator.  Ties are
    deterministic: lower setting tuples win.
    """
    quality_measure = quality_measure or dataset.measures[0]
    objective_measure = objective_measure or quality_measure
    for measure in (quality_measure, objective_measure):
        if measure not in dataset.measures:
            raise ValueError("unknown measure: " + measure)
    try:
        minimum_quality = float(minimum_quality)
    except (TypeError, ValueError):
        raise ValueError("minimum_quality must be numeric")
    if not math.isfinite(minimum_quality):
        raise ValueError("minimum_quality must be finite")
    candidates = []
    for row in dataset.rows:
        quality = _number(row[quality_measure])
        objective = _number(row[objective_measure])
        if quality is None or objective is None or quality < minimum_quality:
            continue
        configuration = {name: int(float(row[name])) for name in dataset.settings}
        key = tuple(configuration[name] for name in dataset.settings)
        candidates.append((objective, key, configuration, quality))
    if not candidates:
        return {"decision": "no_feasible_configuration",
                "quality_measure": quality_measure,
                "objective_measure": objective_measure,
                "minimum_quality": minimum_quality,
                "feasible_count": 0, "recommendation": None}
    objective, key, configuration, quality = min(
        candidates, key=lambda item: (-item[0], item[1]))
    return {"decision": "recommend", "quality_measure": quality_measure,
            "objective_measure": objective_measure,
            "minimum_quality": minimum_quality,
            "feasible_count": len(candidates),
            "recommendation": {"configuration": configuration,
                                "quality": quality, "objective": objective,
                                "measure": objective_measure}}


def improve(dataset, configuration, measure=None):
    """Recommend the smallest (one-setting) change that improves a slow setup.

    The baseline and each candidate are scored with the existing nearest-neighbor
    predictor.  Ties are deterministic by setting order and resulting values.
    No recommendation is made unless a one-bit flip has a strictly higher
    predicted measure than the supplied configuration.
    """
    measure = measure or dataset.measures[0]
    baseline = predict(dataset, configuration, measure)
    candidates = []
    target = dict(configuration)
    for setting in dataset.settings:
        neighbor = dict(target)
        neighbor[setting] = 1 - int(neighbor[setting])
        prediction = predict(dataset, neighbor, measure)
        candidates.append({"setting": setting, "from": int(target[setting]),
                           "to": int(neighbor[setting]),
                           "configuration": neighbor,
                           "prediction": prediction["prediction"],
                           "improvement": prediction["prediction"] - baseline["prediction"],
                           "distance": prediction["distance"],
                           "neighbor_count": prediction["neighbor_count"]})
    candidates.sort(key=lambda item: (-item["prediction"], item["setting"]))
    recommendation = next((item for item in candidates
                           if item["improvement"] > 0), None)
    return {"measure": measure, "decision": "recommend" if recommendation else "no_improvement",
            "baseline": {"configuration": dict(target),
                         "prediction": baseline["prediction"],
                         "distance": baseline["distance"],
                         "neighbor_count": baseline["neighbor_count"]},
            "recommendation": recommendation, "candidate_count": len(candidates),
            "candidates": candidates,
            "limitations": ["Predictions use observed nearest neighbors; validate the change on representative clips."]}


def detect_anomalies(dataset, measure=None, threshold=3.5):
    """Flag unusually large benchmark values using a robust MAD score.

    Repeated rows with the same binary configuration are compared within that
    configuration.  If every configuration is unique, the comparison falls
    back to all usable observations.  A zero MAD uses a small exact-deviation
    rule, and results are sorted by descending robust score then row number.
    This is a screening signal, not proof of a measurement error.
    """
    measure = measure or dataset.measures[0]
    if measure not in dataset.measures:
        raise ValueError("unknown measure: " + measure)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("threshold must be numeric")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    usable = [(index, row, _number(row[measure]))
              for index, row in enumerate(dataset.rows, 1)]
    usable = [(index, row, value) for index, row, value in usable if value is not None]
    if not usable:
        raise ValueError("measure has no numeric observations")
    grouped = {}
    for index, row, value in usable:
        key = tuple(int(float(row[name])) for name in dataset.settings)
        grouped.setdefault(key, []).append((index, row, value))
    scopes = list(grouped.values()) if any(len(group) > 1 for group in grouped.values()) else [usable]
    flagged = []
    for scope in scopes:
        values = [value for _, _, value in scope]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations)
        scale = 1.4826 * mad
        for index, row, value in scope:
            score = abs(value - median) / scale if scale else (float("inf") if value != median else 0.0)
            if score >= threshold:
                flagged.append({"row_number": index,
                                "configuration": {name: int(float(row[name])) for name in dataset.settings},
                                "value": value, "median": median, "mad": mad,
                                "robust_z": score,
                                "reason": "measurement differs materially from its comparison median"})
    flagged.sort(key=lambda item: (-item["robust_z"], item["row_number"]))
    return {"measure": measure, "threshold": threshold, "method": "median/MAD robust screen",
            "observations": len(usable), "comparison_groups": len(scopes),
            "anomalies": flagged, "anomaly_count": len(flagged),
            "has_anomaly": bool(flagged),
            "limitations": ["An anomaly is a review signal; it does not establish a bad benchmark run."]}


def compare_search(dataset, budget, measure=None, batch_size=8):
    """Compare staged racing with a deterministic sequential baseline.

    Each strategy evaluates rows from the supplied finite candidate stream at most
    once.  Both consume exactly ``budget`` rows and maximize the same measure.
    Racing evaluates equal-sized batches, retains the better half, and fills the
    next batch from the retained candidates; the baseline evaluates the first
    budget rows.  This is a reproducible comparison, not a claim of causality.
    """
    measure = measure or dataset.measures[0]
    if measure not in dataset.measures:
        raise ValueError("unknown measure: " + measure)
    if not isinstance(budget, int) or budget <= 0 or budget > len(dataset.rows):
        raise ValueError("budget must be a positive integer no greater than row count")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    def record(indices):
        observations = [(index, _number(dataset.rows[index][measure])) for index in indices]
        observations = [(index, value) for index, value in observations if value is not None]
        if not observations:
            raise ValueError("measure has no numeric observations")
        index, value = max(observations, key=lambda item: (item[1], -item[0]))
        row = dataset.rows[index]
        return {"configuration": {name: int(float(row[name])) for name in dataset.settings},
                "objective": value, "measure": measure, "evaluations": len(indices)}

    baseline_indices = list(range(budget))
    # A race is staged over the same candidate order.  At each stage, all
    # survivors receive the next available observation; this preserves a fixed
    # budget while making elimination explicit and deterministic.
    remaining = list(range(len(dataset.rows)))
    survivors = remaining[:min(batch_size, len(remaining))]
    evaluated = []
    while survivors and len(evaluated) < budget:
        take = min(len(survivors), budget - len(evaluated))
        current = survivors[:take]
        evaluated.extend(current)
        if len(evaluated) == budget:
            break
        scores = [(index, _number(dataset.rows[index][measure])) for index in current]
        scores = [(index, value) for index, value in scores if value is not None]
        keep = max(1, (len(scores) + 1) // 2)
        survivors = [index for index, _ in sorted(scores, key=lambda item: (-item[1], item[0]))[:keep]]
        used = set(evaluated) | set(survivors)
        survivors.extend(index for index in remaining if index not in used)
        survivors = survivors[:batch_size]
    racing = record(evaluated)
    baseline = record(baseline_indices)
    return {"measure": measure, "budget": budget, "batch_size": batch_size,
            "system": {"name": "staged_race", **racing},
            "baseline": {"name": "sequential_prefix", **baseline},
            "same_budget": racing["evaluations"] == baseline["evaluations"] == budget,
            "system_gain": racing["objective"] - baseline["objective"],
            "limitations": ["Rows are pre-recorded candidates; results measure search selection on this dataset, not live encoding cost."]}


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
    parser.add_argument("--choose", action="store_true",
                        help="choose the best observed configuration under a quality constraint")
    parser.add_argument("--min-quality", type=float,
                        help="minimum permitted quality measure value")
    parser.add_argument("--improve-json", metavar="JSON",
                        help="slow configuration to improve with one setting flip")
    parser.add_argument("--quality-measure")
    parser.add_argument("--objective-measure")
    parser.add_argument("--anomalies", action="store_true",
                        help="screen for anomalous benchmark measurements")
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--compare-search", action="store_true")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.csv_path)
    if args.compare_search:
        result = compare_search(dataset, args.budget, args.measure, args.batch_size)
    elif args.anomalies:
        result = detect_anomalies(dataset, args.measure, args.threshold)
    elif args.choose:
        if args.min_quality is None:
            parser.error("--choose requires --min-quality")
        result = choose(dataset, args.min_quality, args.quality_measure,
                        args.objective_measure)
    elif args.improve_json is not None:
        result = improve(dataset, json.loads(args.improve_json), args.measure)
    elif args.justify:
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
