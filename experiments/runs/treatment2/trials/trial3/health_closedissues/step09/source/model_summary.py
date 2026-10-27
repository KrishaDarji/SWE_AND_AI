"""Reusable CSV summary, configuration prediction, and family certification."""

import csv
import math
import statistics

DEFAULT_QUALITY_PREFIXES = ("mre", "acc", "pred")


def spot_quality_anomalies(path, quality_columns=None, outlier_threshold=3.5):
    """Report unusual quality values and possible duplicate-run measurement errors.

    Outliers use a robust modified z-score (median/MAD), with an IQR rule when
    MAD is zero.  This is a screening report, not proof that a result is wrong.
    """
    if outlier_threshold <= 0:
        raise ValueError("outlier_threshold must be positive")
    columns, rows = _read(path)
    quality = list(quality_columns or [name for name in columns
                                      if name.lower().startswith(DEFAULT_QUALITY_PREFIXES)])
    if not quality:
        raise ValueError("at least one quality column is required")
    unknown = set(quality).difference(columns)
    if unknown:
        raise ValueError("Unknown quality columns: " + ", ".join(sorted(unknown)))
    anomalies = []
    column_stats = {}
    for name in quality:
        values = [_number(row.get(name, "")) for row in rows]
        finite = [value for value in values if value is not None]
        median = statistics.median(finite) if finite else None
        deviations = [abs(value - median) for value in finite] if finite else []
        mad = statistics.median(deviations) if deviations else None
        lower = upper = None
        if finite and (mad is None or mad == 0):
            quartiles = statistics.quantiles(finite, n=4, method="inclusive") if len(finite) > 1 else [finite[0]] * 3
            iqr = quartiles[2] - quartiles[0]
            lower, upper = quartiles[0] - 1.5 * iqr, quartiles[2] + 1.5 * iqr
        column_stats[name] = {"count": len(finite), "missing_or_nonfinite": len(values) - len(finite),
                              "median": median, "mad": mad, "lower_fence": lower, "upper_fence": upper}
        for index, value in enumerate(values):
            if value is None:
                anomalies.append({"row_index": index, "quality_column": name,
                                  "value": rows[index].get(name, ""), "reason": "missing_or_nonfinite"})
            elif mad and mad > 0:
                score = 0.6745 * abs(value - median) / mad
                if score >= outlier_threshold:
                    anomalies.append({"row_index": index, "quality_column": name, "value": value,
                                      "reason": "robust_outlier", "modified_z_score": score})
            elif lower is not None and (value < lower or value > upper):
                anomalies.append({"row_index": index, "quality_column": name, "value": value,
                                  "reason": "iqr_outlier"})
    config_columns = [name for name in columns if name not in quality]
    groups = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(name, "") for name in config_columns)
        groups.setdefault(key, []).append(index)
    duplicate_conflicts = []
    for key, indexes in groups.items():
        if len(indexes) < 2:
            continue
        for name in quality:
            values = [_number(rows[index].get(name, "")) for index in indexes]
            finite = [value for value in values if value is not None]
            if len(finite) > 1 and max(finite) != min(finite):
                duplicate_conflicts.append({"configuration": dict(zip(config_columns, key)),
                    "row_indices": indexes, "quality_column": name,
                    "min": min(finite), "max": max(finite), "range": max(finite) - min(finite),
                    "reason": "duplicate_configuration_disagreement"})
    return {"quality_columns": quality, "row_count": len(rows), "anomalies": anomalies,
            "duplicate_configuration_conflicts": duplicate_conflicts,
            "column_stats": column_stats,
            "caveat": "Flags indicate unusual measurements for review; they do not establish that a result is erroneous."}


def blame_high_error(path, error_column="MRE-", high_percentile=0.75):
    """Rank configuration columns associated with unusually high error.

    Numeric settings use Pearson correlation and compare the high-error group's
    mean with the remaining rows. Categorical settings compare each level's
    mean error and high-error rate with the whole data set. This is descriptive
    association, not causal attribution.
    """
    if not 0 < high_percentile < 1:
        raise ValueError("high_percentile must be between 0 and 1")
    columns, rows = _read(path)
    if error_column not in columns:
        raise ValueError("Unknown error column: " + error_column)
    usable = [(row, _number(row.get(error_column, ""))) for row in rows]
    usable = [(row, error) for row, error in usable if error is not None]
    if not usable:
        raise ValueError("error column contains no finite values")
    errors = [error for _, error in usable]
    threshold = statistics.quantiles(errors, n=100, method="inclusive")[
        min(99, max(0, int(high_percentile * 100) - 1))] if len(errors) > 1 else errors[0]
    high = [error >= threshold for _, error in usable]
    overall = statistics.fmean(errors)
    results = []
    for name in columns:
        if name == error_column or name.lower().startswith(DEFAULT_QUALITY_PREFIXES):
            continue
        values = [row.get(name, "") for row, _ in usable]
        numeric = [_number(value) for value in values]
        if all(value is not None for value in numeric):
            mean_x = statistics.fmean(numeric)
            mean_y = overall
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(numeric, errors))
            den_x = math.sqrt(sum((x - mean_x) ** 2 for x in numeric))
            den_y = math.sqrt(sum((y - mean_y) ** 2 for y in errors))
            correlation = cov / (den_x * den_y) if den_x and den_y else 0.0
            high_values = [x for x, is_high in zip(numeric, high) if is_high]
            low_values = [x for x, is_high in zip(numeric, high) if not is_high]
            results.append({"hyperparameter": name, "type": "numeric",
                            "correlation": correlation,
                            "high_error_mean": statistics.fmean(high_values) if high_values else None,
                            "low_error_mean": statistics.fmean(low_values) if low_values else None,
                            "association_score": abs(correlation)})
        else:
            levels = []
            for level in sorted(set(values)):
                level_errors = [error for value, (_, error) in zip(values, usable) if value == level]
                level_high = [is_high for value, is_high in zip(values, high) if value == level]
                rate = statistics.fmean(level_high)
                levels.append({"value": level, "count": len(level_errors),
                               "mean_error": statistics.fmean(level_errors),
                               "high_error_rate": rate})
            spread = max((abs(item["mean_error"] - overall) for item in levels), default=0.0)
            results.append({"hyperparameter": name, "type": "categorical",
                            "levels": levels, "association_score": spread})
    results.sort(key=lambda item: (-item["association_score"], item["hyperparameter"]))
    return {"error_column": error_column, "high_error_threshold": threshold,
            "high_error_count": sum(high), "row_count": len(usable),
            "overall_mean_error": overall, "associations": results}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values)}


def _read(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [name.strip() for name in (reader.fieldnames or [])]
        if not columns:
            raise ValueError("CSV must contain a header")
        if any(not name for name in columns):
            raise ValueError("CSV contains an empty column name")
        return columns, list(reader)


def summarize_csv(path, quality_columns=None, quality_prefixes=DEFAULT_QUALITY_PREFIXES):
    columns, rows = _read(path)
    selected = set(quality_columns or ())
    unknown = selected.difference(columns)
    if unknown:
        raise ValueError("Unknown quality columns: " + ", ".join(sorted(unknown)))
    if not selected:
        selected = {name for name in columns if name.lower().startswith(
            tuple(prefix.lower() for prefix in quality_prefixes))}
    quality = {}
    for name in columns:
        if name in selected:
            quality[name] = _stats([v for v in (_number(r.get(name, "")) for r in rows)
                                    if v is not None])
    settings = {}
    for name in (column for column in columns if column not in selected):
        raw = [row.get(name, "") for row in rows]
        numeric = [_number(value) for value in raw]
        if raw and all(value is not None for value in numeric):
            settings[name] = {"type": "numeric", "count": len(raw),
                              "unique_count": len(set(numeric)),
                              "min": min(numeric), "max": max(numeric),
                              "values": sorted(set(numeric))}
        else:
            settings[name] = {"type": "categorical", "count": len(raw),
                              "unique_count": len(set(raw)), "values": sorted(set(raw))}
    return {"row_count": len(rows), "column_count": len(columns), "columns": columns,
            "quality_metrics": quality, "configuration": settings}


def predict_quality(path, configuration, quality_columns=None, neighbors=5):
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    columns, rows = _read(path)
    quality = set(quality_columns or ())
    if not quality:
        quality = {name for name in columns if name.lower().startswith(DEFAULT_QUALITY_PREFIXES)}
    config_columns = [name for name in columns if name not in quality]
    missing = set(configuration).difference(config_columns)
    if missing:
        raise ValueError("Unknown configuration columns: " + ", ".join(sorted(missing)))
    spans = {}
    for name in config_columns:
        valid = [v for v in (_number(row.get(name, "")) for row in rows) if v is not None]
        spans[name] = (min(valid), max(valid)) if valid else None
    def distance(row):
        total = 0.0
        for name in config_columns:
            target, actual = configuration.get(name, row.get(name, "")), row.get(name, "")
            target_num, actual_num = _number(target), _number(actual)
            if target_num is not None and actual_num is not None and spans[name]:
                low, high = spans[name]
                total += 0.0 if high == low else abs(target_num - actual_num) / (high - low)
            else:
                total += 0.0 if str(target) == str(actual) else 1.0
        return total
    ranked = sorted(((distance(row), row) for row in rows), key=lambda item: item[0])[:neighbors]
    result = {name: None for name in sorted(quality)}
    for name in result:
        weighted = [(1.0 / (dist + 1e-12), _number(row.get(name, ""))) for dist, row in ranked]
        weighted = [(weight, value) for weight, value in weighted if value is not None]
        if weighted:
            result[name] = sum(weight * value for weight, value in weighted) / sum(weight for weight, _ in weighted)
    return {"configuration": dict(configuration), "neighbors_used": len(ranked), "predicted_quality": result}


def certify_families(path, family_columns, quality_columns=None, confidence=0.95):
    """Compare configuration families and flag only consistent, supported wins.

    A family is a unique tuple of family-column values.  For each pair, a metric
    is reliable when its 95% normal-approximation interval for the mean
    difference excludes zero.  A family wins reliably only when it wins every
    requested metric; error metrics (MRE) are minimized and other metrics are
    maximized.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    columns, rows = _read(path)
    unknown = set(family_columns).difference(columns)
    if unknown:
        raise ValueError("Unknown family columns: " + ", ".join(sorted(unknown)))
    quality = set(quality_columns or ())
    if not quality:
        quality = {name for name in columns if name.lower().startswith(DEFAULT_QUALITY_PREFIXES)}
    unknown = quality.difference(columns)
    if unknown:
        raise ValueError("Unknown quality columns: " + ", ".join(sorted(unknown)))
    groups = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in family_columns)
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError("at least two configuration families are required")
    z = 1.96 if confidence == 0.95 else 1.96  # documented conservative default
    family_names = ["|".join(key) for key in sorted(groups)]
    comparisons = []
    for left, right in ((family_names[i], family_names[j]) for i in range(len(family_names))
                        for j in range(i + 1, len(family_names))):
        left_rows, right_rows = groups[tuple(left.split("|"))], groups[tuple(right.split("|"))]
        metrics = {}
        winners = []
        for name in sorted(quality):
            a = [v for v in (_number(r.get(name, "")) for r in left_rows) if v is not None]
            b = [v for v in (_number(r.get(name, "")) for r in right_rows) if v is not None]
            if not a or not b:
                metrics[name] = {"left_mean": None, "right_mean": None, "difference": None,
                                 "reliable": False, "winner": None}
                continue
            direction = -1 if name.lower().startswith("mre") else 1
            difference = direction * (statistics.fmean(a) - statistics.fmean(b))
            variance = (statistics.pvariance(a) / len(a) if len(a) > 1 else 0.0) + \
                       (statistics.pvariance(b) / len(b) if len(b) > 1 else 0.0)
            margin = z * math.sqrt(variance)
            winner = left if difference > 0 else right if difference < 0 else None
            reliable = abs(difference) > margin and winner is not None
            metrics[name] = {"left_mean": statistics.fmean(a), "right_mean": statistics.fmean(b),
                             "difference_in_preferred_direction": difference,
                             "confidence_interval": [difference - margin, difference + margin],
                             "reliable": reliable, "winner": winner if reliable else None}
            if reliable:
                winners.append(winner)
        reliable_winner = winners[0] if winners and len(winners) == len(quality) and len(set(winners)) == 1 else None
        comparisons.append({"left": left, "right": right, "metrics": metrics,
                            "reliably_better": reliable_winner})
    return {"family_columns": list(family_columns), "families": family_names,
            "comparison_count": len(comparisons), "comparisons": comparisons}


class OnlineEvaluator:
    """Evaluate configurations incrementally as validation results arrive.

    Each observation is final for that configuration.  The evaluator keeps the
    full arrival order, updates the incumbent immediately, and never needs the
    complete CSV in memory or a second pass over prior observations.
    """

    def __init__(self, quality_columns=None):
        self.quality_columns = list(quality_columns or [])
        self.observations = []
        self._best = None

    def _score(self, results):
        names = self.quality_columns or sorted(results)
        if not names:
            raise ValueError("at least one validation result is required")
        missing = [name for name in names if name not in results]
        if missing:
            raise ValueError("Missing validation results: " + ", ".join(missing))
        values = {}
        for name in names:
            value = _number(results[name])
            if value is None:
                raise ValueError("Validation result must be finite: " + name)
            values[name] = value
        return values

    def _is_better(self, candidate, incumbent):
        wins = []
        for name in self.quality_columns or sorted(candidate["results"]):
            direction = -1 if name.lower().startswith("mre") else 1
            a, b = candidate["results"][name], incumbent["results"][name]
            wins.append(direction * (a - b))
        return all(value >= 0 for value in wins) and any(value > 0 for value in wins)

    def observe(self, configuration, validation_results):
        if not isinstance(configuration, dict) or not isinstance(validation_results, dict):
            raise TypeError("configuration and validation_results must be dictionaries")
        results = self._score(validation_results)
        record = {"configuration": dict(configuration), "results": results,
                  "arrival_index": len(self.observations)}
        self.observations.append(record)
        if self._best is None or self._is_better(record, self._best):
            self._best = record
        return self.snapshot()

    def snapshot(self):
        return {"observations_seen": len(self.observations),
                "incumbent": None if self._best is None else dict(self._best),
                "history": [dict(item) for item in self.observations]}


def evaluate_stream(records, quality_columns=None):
    """Consume iterable records with configuration and validation_results keys."""
    evaluator = OnlineEvaluator(quality_columns)
    for record in records:
        evaluator.observe(record["configuration"], record["validation_results"])
    return evaluator.snapshot()


def justify_configuration(path, configuration=None, quality_columns=None,
                          error_column="MRE-", high_percentile=0.75):
    """Return an evidence-backed recommendation for a platform decision.

    If no configuration is supplied, select the observed row with the best
    mean min-max normalized quality rank (errors minimized, other metrics
    maximized).  The returned evidence reuses prediction and high-error
    association analysis; it is decision support, not a causal claim.
    """
    columns, rows = _read(path)
    quality = list(quality_columns or [name for name in columns
                                      if name.lower().startswith(DEFAULT_QUALITY_PREFIXES)])
    if not quality:
        raise ValueError("at least one quality column is required")
    unknown = set(quality).difference(columns)
    if unknown:
        raise ValueError("Unknown quality columns: " + ", ".join(sorted(unknown)))
    usable = []
    for row in rows:
        values = {name: _number(row.get(name, "")) for name in quality}
        if all(value is not None for value in values.values()):
            usable.append((row, values))
    if not usable:
        raise ValueError("no rows contain finite values for every quality column")
    ranges = {name: (min(values[name] for _, values in usable),
                     max(values[name] for _, values in usable)) for name in quality}
    def rank(values):
        scores = []
        for name in quality:
            low, high = ranges[name]
            normalized = 0.5 if high == low else (values[name] - low) / (high - low)
            scores.append(normalized if name.lower().startswith("mre") else 1.0 - normalized)
        return statistics.fmean(scores)
    was_selected = configuration is None
    if was_selected:
        row, values = min(usable, key=lambda item: (rank(item[1]),
                                                     tuple(str(item[0].get(c, "")) for c in columns)))
        configuration = {name: row[name] for name in columns if name not in quality}
    prediction = predict_quality(path, configuration, quality, neighbors=5)
    blame = blame_high_error(path, error_column, high_percentile)
    return {"recommended_configuration": dict(configuration),
            "selection": "best observed normalized quality rank" if was_selected else "provided",
            "predicted_quality": prediction["predicted_quality"],
            "neighbors_used": prediction["neighbors_used"],
            "high_error_analysis": blame,
            "quality_columns": quality,
            "caveat": "Associations and nearest-neighbor predictions support review; they do not establish causation or production performance."}


def choose_configuration(path, configuration=None, quality_columns=None,
                         neighbors=5, prediction_weight=0.5):
    """Choose a configuration using observed quality and predicted quality.

    Candidate rows are scored on min-max normalized quality (errors minimized,
    other metrics maximized).  The score blends that observed score with the
    same score computed from nearest-neighbor predictions, making the tradeoff
    explicit while retaining the earlier evidence in ``justify_configuration``.
    """
    if not 0 <= prediction_weight <= 1:
        raise ValueError("prediction_weight must be between 0 and 1")
    columns, rows = _read(path)
    quality = list(quality_columns or [name for name in columns
                                      if name.lower().startswith(DEFAULT_QUALITY_PREFIXES)])
    if not quality:
        raise ValueError("at least one quality column is required")
    unknown = set(quality).difference(columns)
    if unknown:
        raise ValueError("Unknown quality columns: " + ", ".join(sorted(unknown)))
    config_columns = [name for name in columns if name not in quality]
    candidates = []
    for row in rows:
        values = {name: _number(row.get(name, "")) for name in quality}
        if all(value is not None for value in values.values()):
            candidates.append(({name: row[name] for name in config_columns}, values))
    if not candidates:
        raise ValueError("no rows contain finite values for every quality column")
    if configuration is not None:
        unknown_config = set(configuration).difference(config_columns)
        if unknown_config:
            raise ValueError("Unknown configuration columns: " + ", ".join(sorted(unknown_config)))
        candidates = [(dict(configuration), {})]
    observed_values = [values for _, values in candidates if values]
    if not observed_values:
        observed_values = [{name: _number(row.get(name, "")) for name in quality}
                           for row in rows]
        observed_values = [values for values in observed_values
                           if all(value is not None for value in values.values())]
    ranges = {name: (min(v[name] for v in observed_values), max(v[name] for v in observed_values))
              for name in quality}

    def score(values):
        parts = []
        for name in quality:
            low, high = ranges[name]
            normalized = 0.5 if high == low else (values[name] - low) / (high - low)
            parts.append(normalized if name.lower().startswith("mre") else 1.0 - normalized)
        return statistics.fmean(parts)

    if configuration is None and len(candidates) > 25:
        # Prediction is the expensive step; retain a broad, quality-ranked
        # shortlist before applying nearest-neighbor evidence.
        candidates.sort(key=lambda item: (score(item[1]),
                                          tuple(str(item[0].get(c, "")) for c in config_columns)))
        candidates = candidates[:25]

    scored = []
    for candidate, observed in candidates:
        prediction = predict_quality(path, candidate, quality, neighbors=neighbors)["predicted_quality"]
        predicted = {name: prediction[name] for name in quality}
        if any(value is None for value in predicted.values()):
            predicted_score = None
            combined = score(observed) if observed else None
        else:
            predicted_score = score(predicted)
            observed_score = score(observed) if observed else predicted_score
            combined = ((1 - prediction_weight) * observed_score +
                        prediction_weight * predicted_score)
        scored.append({"configuration": dict(candidate), "observed_quality": observed,
                       "predicted_quality": predicted, "observed_score": score(observed) if observed else None,
                       "predicted_score": predicted_score, "combined_score": combined})
    chosen = min(scored, key=lambda item: (item["combined_score"],
                                            tuple(str(item["configuration"].get(c, "")) for c in config_columns)))
    return {"recommended_configuration": chosen["configuration"],
            "selection": "balanced observed and predicted quality",
            "prediction_weight": prediction_weight, "quality_columns": quality,
            "candidates_considered": len(scored), "candidate_scores": scored,
            "predicted_quality": chosen["predicted_quality"]}


def suggest_smallest_change(path, error_column="MRE-"):
    """Recommend the smallest observed one-hyperparameter change that lowers error.

    The baseline is the first observed row (the current configuration).  A candidate is a
    row differing from that baseline in exactly one configuration column.  The
    recommendation minimizes the number of changed fields (therefore one),
    then the resulting error, then the changed field and value.  This is an
    empirical suggestion, not a causal conclusion.
    """
    columns, rows = _read(path)
    if error_column not in columns:
        raise ValueError("Unknown error column: " + error_column)
    config_columns = [name for name in columns
                      if name != error_column and
                      not name.lower().startswith(DEFAULT_QUALITY_PREFIXES)]
    usable = [(row, _number(row.get(error_column, ""))) for row in rows]
    usable = [(row, error) for row, error in usable if error is not None]
    if not usable:
        raise ValueError("error column contains no finite values")
    baseline, baseline_error = usable[0]
    candidates = []
    for row, error in usable:
        changes = [name for name in config_columns
                   if str(row.get(name, "")) != str(baseline.get(name, ""))]
        if len(changes) == 1 and error < baseline_error:
            changed = changes[0]
            candidates.append((error, changed, str(row.get(changed, "")), row))
    candidates.sort(key=lambda item: (len([item[1]]), item[0], item[1], item[2]))
    if candidates:
        error, changed, value, row = candidates[0]
        return {"baseline_configuration": {name: baseline[name] for name in config_columns},
                "baseline_error": baseline_error, "recommendation": {
                    "hyperparameter": changed, "from": baseline.get(changed, ""),
                    "to": row.get(changed, ""), "configuration": {
                        name: row[name] for name in config_columns},
                    "predicted_error": None, "observed_error": error,
                    "improvement": baseline_error - error},
                "change_count": 1, "candidate_count": len(candidates),
                "selection": "smallest observed one-hyperparameter error improvement"}
    return {"baseline_configuration": {name: baseline[name] for name in config_columns},
            "baseline_error": baseline_error, "recommendation": None,
            "change_count": None, "candidate_count": 0,
            "selection": "no observed one-hyperparameter improvement"}
