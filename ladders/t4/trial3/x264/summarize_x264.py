#!/usr/bin/env python3
"""Summarize an X264 measurements CSV without assuming domain-specific units."""
import argparse
import csv
import json
import math
import copy
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array
from scipy import stats


OUTCOME = "Performance-"
MISSING_SETTING = "__MISSING_SETTING__"


class IncrementalBenchmark:
    """One-pass lower-is-better benchmark accumulator with immutable snapshots."""
    def __init__(self, configuration_fields=None):
        self.configuration_fields = list(configuration_fields or [])
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.best = None

    def update(self, record, record_id=None):
        """Consume one mapping; invalid or non-finite outcomes are ignored."""
        outcome = _number(record.get(OUTCOME) if hasattr(record, "get") else None)
        if outcome is None:
            return self.snapshot()
        if not self.configuration_fields:
            self.configuration_fields = [k for k in record if k != OUTCOME]
        identifier = self.count if record_id is None else record_id
        configuration = {k: copy.deepcopy(record.get(k)) for k in self.configuration_fields}
        self.count += 1
        delta = outcome - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (outcome - self.mean)
        if self.best is None or outcome < self.best["outcome"]:
            self.best = {"record_id": copy.deepcopy(identifier), "outcome": outcome,
                         "configuration": configuration}
        return self.snapshot()

    def snapshot(self):
        variance = self._m2 / (self.count - 1) if self.count > 1 else None
        return copy.deepcopy({"count": self.count, "mean": self.mean if self.count else None,
            "sample_variance": variance, "best": self.best,
            "outcome": OUTCOME, "direction": "lower is better",
            "invalid_outcomes_ignored": True})


def benchmark_flow(records, configuration_fields=None):
    """Yield a snapshot after each input record, without buffering the iterator.

    CSV order is a demonstration sequence, not a verified timeline. Empty input
    yields no snapshots; ties retain the first best record. Missing/non-finite
    outcomes leave the accumulator unchanged but still produce a snapshot.
    """
    accumulator = IncrementalBenchmark(configuration_fields)
    for record_id, record in enumerate(records):
        yield accumulator.update(record, record_id)


def benchmark_csv(path):
    with Path(path).open(newline="") as handle:
        yield from benchmark_flow(csv.DictReader(handle))


def certify(path, threshold=None):
    """Certify fixed A/B settings with a two-sided Welch test and interval."""
    threshold_source = "demonstration_default" if threshold is None else "supplied"
    threshold = 5.0 if threshold is None else threshold
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    settings = ["no_8x8dct"]
    groups = {0: [], 1: []}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if any(field not in fields for field in settings + [OUTCOME]):
            raise ValueError("missing required comparison columns")
        for row in reader:
            setting = _number(row.get("no_8x8dct", ""))
            outcome = _number(row.get(OUTCOME, ""))
            if setting in groups and outcome is not None:
                groups[int(setting)].append(outcome)
    a, b = groups[0], groups[1]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each comparison group needs at least two finite outcomes")
    if np.var(a, ddof=1) == 0 or np.var(b, ddof=1) == 0:
        raise ValueError("zero sample variance is not supported")
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    difference = mean_a - mean_b
    variance_a, variance_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    standard_error = math.sqrt(variance_a / len(a) + variance_b / len(b))
    df = (variance_a / len(a) + variance_b / len(b)) ** 2 / ((variance_a / len(a)) ** 2 / (len(a) - 1) + (variance_b / len(b)) ** 2 / (len(b) - 1))
    t_stat = difference / standard_error
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    critical = float(stats.t.ppf(0.975, df))
    interval = [difference - critical * standard_error, difference + critical * standard_error]
    positive_supported = interval[0] > threshold
    negative_supported = interval[1] < -threshold
    if positive_supported:
        favored = "B"
        interpretation = "interval supports practical importance favoring B"
    elif negative_supported:
        favored = "A"
        interpretation = "interval supports practical importance favoring A"
    else:
        favored = "uncertain"
        interpretation = "practical importance is uncertain because the interval crosses zero or the practical threshold"
    return {"comparison": {"A": {"no_8x8dct": 0, "n": len(a), "mean": mean_a}, "B": {"no_8x8dct": 1, "n": len(b), "mean": mean_b}, "outcome": OUTCOME, "lower_is_better": True}, "mean_difference_A_minus_B": difference, "welch": {"t": t_stat, "df": df, "p_value_two_sided": p_value, "confidence_level": 0.95, "interval_A_minus_B": interval}, "practical_effect": {"threshold": float(threshold), "threshold_source": threshold_source, "point_estimate_magnitude_exceeds_threshold": abs(difference) > threshold, "interval_entirely_above_positive_threshold": positive_supported, "interval_entirely_below_negative_threshold": negative_supported, "favored_group": favored, "interpretation": interpretation}, "limitations": ["Rows are different configurations, not repeated-run measurements; configuration variation cannot establish measurement noise or causal effects.", "Failure to detect a difference does not establish equivalence."]}


def _number(value):
    if value is None:
        return None
    try:
        number = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(path):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    settings = [field for field in fields if field != OUTCOME]
    result = {
        "record_count": len(rows), "encoder_settings": settings,
        "outcome": {"name": OUTCOME, "direction": "lower is better"},
        "setting_frequencies": {}, "numeric_statistics": {},
        "missing_values": {}, "parsing_assumptions": [
            "The first CSV row is the header; comma separates fields.",
            "Blank or non-finite numeric cells are missing; other non-numeric cells are missing for numeric summaries.",
            "All non-outcome columns are reported as encoder settings; no units or formulas are inferred.",
        ],
        "limitations": [
            "Only the available Performance- outcome is summarized.",
            "Frequency counts are exact trimmed cell values and do not imply semantic categories.",
        ],
    }
    for field in fields:
        values = [row.get(field, "") for row in rows]
        result["missing_values"][field] = sum(_number(v) is None for v in values)
        if field != OUTCOME:
            counts = {}
            for value in values:
                key = value.strip()
                counts[key] = counts.get(key, 0) + 1
            result["setting_frequencies"][field] = dict(sorted(counts.items()))
    outcome_values = [_number(row.get(OUTCOME, "")) for row in rows]
    present = [v for v in outcome_values if v is not None]
    if present:
        # check_array supplies scikit-learn's finite numeric validation.
        checked = check_array(present, ensure_2d=False, dtype=float)
        mean = sum(checked) / len(checked)
        variance = sum((v - mean) ** 2 for v in checked) / len(checked)
        result["numeric_statistics"][OUTCOME] = {
            "count": len(checked), "min": min(checked), "max": max(checked),
            "mean": mean, "std_population": math.sqrt(variance),
            "median": sorted(checked)[len(checked) // 2] if len(checked) % 2 else
                (sorted(checked)[len(checked)//2-1] + sorted(checked)[len(checked)//2]) / 2,
        }
    else:
        result["numeric_statistics"][OUTCOME] = {"count": 0}
    return result


def _read_valid_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if OUTCOME not in fields:
            raise ValueError(f"missing target column {OUTCOME!r}")
        settings = [f for f in fields if f != OUTCOME]
        valid = []
        invalid_target = 0
        for row in reader:
            target = _number(row.get(OUTCOME, ""))
            if target is None:
                invalid_target += 1
                continue
            valid.append((row, target))
    if not valid:
        raise ValueError("no rows with finite Performance- targets")
    return settings, valid, invalid_target


def _frame(rows, settings):
    return [[_setting(row.get(field, "")) for field in settings] for row in rows]


def _setting(value):
    """Return one canonical numeric setting value, or None for missing input."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return str(number)


def _pipeline(settings, seed=42):
    prep = ColumnTransformer([
        ("settings", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=None, keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), list(range(len(settings))))
    ])
    return Pipeline([("preprocess", prep), ("model", RandomForestRegressor(
        n_estimators=200, random_state=seed, n_jobs=1))])


def evaluate(path, random_state=42):
    settings, valid, invalid_target = _read_valid_rows(path)
    rows = [x[0] for x in valid]
    y = np.asarray([x[1] for x in valid], dtype=float)
    groups = np.asarray(["\x1f".join((_setting(r.get(f, "")) or MISSING_SETTING) for f in settings) for r in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train, test = next(splitter.split(rows, y, groups))
    model = _pipeline(settings, random_state)
    train_frame = _frame([rows[i] for i in train], settings)
    for column in range(len(settings)):
        if all(row[column] is None for row in train_frame):
            train_frame = [row[:column] + [MISSING_SETTING] + row[column + 1:] for row in train_frame]
    model.fit(train_frame, y[train])
    predictions = model.predict(_frame([rows[i] for i in test], settings))
    baseline = float(np.mean(y[train]))
    return {
        "predictor_names": settings, "train_count": int(len(train)), "test_count": int(len(test)),
        "test_groups": groups[test].tolist(),
        "group_count": int(len(set(groups.tolist()))), "held_out_mae": float(mean_absolute_error(y[test], predictions)),
        "training_mean_baseline_mae": float(mean_absolute_error(y[test], np.full(len(test), baseline))),
        "training_mean": baseline, "invalid_target_rows": invalid_target,
        "model_settings": {"preprocessing": "most_frequent imputation + one-hot encoding fitted on training rows", "estimator": "RandomForestRegressor", "n_estimators": 200, "random_state": random_state, "group_split": "GroupShuffleSplit(test_size=0.2)"},
        "model": model, "settings": settings, "training_frame": train_frame,
    }


def predict(path, configurations, random_state=42):
    result = evaluate(path, random_state)
    model = result["model"]
    missing = [f for f in result["settings"] if any(f not in c for c in configurations)]
    if missing:
        raise ValueError(f"missing settings in configuration: {missing}")
    prediction_frame = _frame(configurations, result["settings"])
    for column in range(len(result["settings"])):
        if all(row[column] is None for row in result["training_frame"]):
            prediction_frame = [row[:column] + [MISSING_SETTING] + row[column + 1:] for row in prediction_frame]
    values = model.predict(prediction_frame)
    return [float(v) for v in values]


def _blame_value(value):
    """Canonical explanatory value; invalid setting cells become one level."""
    normalized = _setting(value)
    return MISSING_SETTING if normalized is None else normalized


def blame(path):
    """Rank settings by marginal eta-squared association with Performance-.

    Each setting is treated as a categorical explanatory variable, so the same
    primary score is used for binary, multi-value, and missing-value settings.
    Scores are descriptive associations, not causal effects.
    """
    settings, valid, invalid_target = _read_valid_rows(path)
    y = np.asarray([target for _, target in valid], dtype=float)
    overall = float(np.mean(y))
    total = float(np.sum((y - overall) ** 2))
    ranking = []
    for order, field in enumerate(settings):
        groups = {}
        for (row, target) in valid:
            groups.setdefault(_blame_value(row.get(field, "")), []).append(target)
        summaries = []
        between = 0.0
        for value, outcomes in sorted(groups.items(), key=lambda item: item[0]):
            mean = float(np.mean(outcomes))
            between += len(outcomes) * (mean - overall) ** 2
            summaries.append({"value": value, "count": len(outcomes),
                              "mean": mean, "min": float(np.min(outcomes)),
                              "max": float(np.max(outcomes))})
        eligible = len(summaries) >= 2 and all(item["count"] >= 2 for item in summaries)
        if not eligible or total == 0.0:
            score = 0.0
            score_status = "undefined; reported as 0.0"
        else:
            score = between / total
            score_status = "defined"
        fastest = min(summaries, key=lambda item: (item["mean"], item["value"])) if summaries else None
        slowest = max(summaries, key=lambda item: (item["mean"], item["value"])) if summaries else None
        ranking.append({"setting": field, "primary_score": float(score),
                        "score_name": "eta_squared_between_group",
                        "score_status": score_status, "group_count": len(summaries),
                        "groups": summaries, "fastest_group": fastest,
                        "slowest_group": slowest, "input_order": order})
    ranking.sort(key=lambda item: (-item["primary_score"], item["setting"]))
    return {"outcome": OUTCOME, "direction": "lower is better", "record_count": len(valid),
            "invalid_target_rows": invalid_target, "overall_mean": overall,
            "ranking": ranking, "method": "marginal eta-squared: between-group sum of squares / total sum of squares; missing/non-finite settings are one level; groups need at least two observations; ties sort by setting name",
            "interpretation": "These are observational marginal associations. Correlated or mutually exclusive settings can share explanatory information; a low score does not prove irrelevance, and changing a setting is not guaranteed to change the outcome."}


def justify(path):
    """Return a retrospective recommendation and auditable first-record comparison."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if OUTCOME not in fields:
            raise ValueError(f"missing target column {OUTCOME!r}")
        settings = [field for field in fields if field != OUTCOME]
        rows = list(reader)
    valid = [(i, row, _number(row.get(OUTCOME, ""))) for i, row in enumerate(rows)]
    valid = [(i, row, value) for i, row, value in valid if value is not None]
    if not valid:
        raise ValueError("no usable records with finite Performance- values")
    best_id, best_row, best_value = min(valid, key=lambda item: (item[2], item[0]))
    first_value = _number(rows[0].get(OUTCOME, "")) if rows else None
    comparison = None if first_value is None else {
        "historical_reference": "first CSV record",
        "reference_record_id": 0,
        "reference_performance": first_value,
        "recommended_minus_reference": best_value - first_value,
        "reference_minus_recommended": first_value - best_value,
        "lower_is_better": True,
    }
    return {
        "recommendation": {
            "record_id": best_id, "configuration": {k: best_row.get(k, "") for k in settings},
            "performance": best_value, "evidence_label": "measured observed dataset record",
        },
        "historical_comparison": comparison,
        "usable_record_count": len(valid), "invalid_or_nonfinite_outcome_count": len(rows) - len(valid),
        "outcome": {"name": OUTCOME, "direction": "lower is better"},
        "interpretation": "Retrospective dataset evidence only; it does not guarantee future performance or establish statistical superiority. Associations do not establish the effect of changing a setting, and correlated settings can share explanatory information.",
    }


def _configuration_key(configuration):
    try:
        return json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError("configurations must be JSON-serializable mappings") from exc


def choose_configuration(candidates, evaluate, budget=20, random_state=42):
    """Select a low measured Performance- using only a counted callback.

    Candidate settings are visible, but outcomes enter this routine only through
    ``evaluate(configuration)``. Unique candidates are shuffled reproducibly;
    measured ties retain the earlier evaluation order.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")
    unique = []
    seen = set()
    for configuration in candidates:
        if not hasattr(configuration, "keys"):
            raise ValueError("each candidate must be a mapping")
        key = _configuration_key(configuration)
        if key not in seen:
            seen.add(key)
            unique.append((key, copy.deepcopy(dict(configuration))))
    if not unique:
        raise ValueError("candidates must contain a unique configuration")
    order = list(range(len(unique)))
    rng = np.random.RandomState(random_state)
    rng.shuffle(order)
    selected = order[:min(budget, len(order))]
    observations = []
    for evaluation_index, candidate_index in enumerate(selected):
        key, configuration = unique[candidate_index]
        outcome = _number(evaluate(copy.deepcopy(configuration)))
        if outcome is None:
            raise ValueError("evaluation callback must return a finite numeric outcome")
        observations.append({"evaluation_index": evaluation_index, "candidate_key": key,
                             "configuration": configuration, "outcome": outcome})
    best = min(observations, key=lambda item: (item["outcome"], item["evaluation_index"]))
    return {"budget": budget, "budget_used": len(observations), "candidate_count": len(candidates),
            "unique_candidate_count": len(unique), "evaluation_order": observations,
            "recommended_identifier": best["candidate_key"],
            "recommendation": {"configuration": copy.deepcopy(best["configuration"]),
                                "performance": best["outcome"]},
            "strategy": "reproducibly shuffled unique candidates, measured through callback; lower is better",
            "limitations": ["Random search can miss the global optimum.",
                            "The recommendation is supported only by the measured evaluations."]}


def audit_observed_minimum(records):
    """Retrospectively report the full-data minimum; never used by selection."""
    usable = [(i, row, _number(row.get(OUTCOME, ""))) for i, row in enumerate(records)]
    usable = [(i, row, value) for i, row, value in usable if value is not None]
    if not usable:
        raise ValueError("no usable audit records")
    minimum = min(usable, key=lambda item: (item[2], item[0]))
    return {"observed_minimum": {"record_id": minimum[0], "performance": minimum[2]},
            "audit_only": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--blame", action="store_true", help="rank settings by marginal association")
    parser.add_argument("--justify", action="store_true", help="justify the best observed configuration")
    args = parser.parse_args()
    summary = justify(args.dataset) if args.justify else (blame(args.dataset) if args.blame else summarize(args.dataset))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
