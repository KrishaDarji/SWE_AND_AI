#!/usr/bin/env python3
"""Summarize an X264 measurements CSV without assuming domain-specific units."""
import argparse
import csv
import json
import math
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
    if value is None or not value.strip():
        return None
    try:
        number = float(value.strip())
    except ValueError:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary = summarize(args.dataset)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
