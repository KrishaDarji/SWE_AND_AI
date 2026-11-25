"""Summarize historical encoder runs using only Python's standard library."""
import argparse
import csv
import json
import math
import statistics
import sys
import random

MISSING = {"", "?", "na", "n/a", "null", "none", "missing"}

def is_missing(value):
    return value.strip().lower() in MISSING

def read_runs(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return [], [], ["input has no header"]
        names = [name.strip() if name else "" for name in reader.fieldnames]
        if any(not name for name in names):
            raise ValueError("header contains an empty column name")
        rows = []
        for number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError("row %d has more fields than the header" % number)
            rows.append({name: (raw.get(name) or "").strip() for name in names})
        return names, rows, []

def numeric_stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "median": None, "stdev": None}
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}

def summarize(path):
    names, rows, warnings = read_runs(path)
    outcome = next((n for n in names if n.rstrip().lower() == "performance-"), None)
    if outcome is None and names:
        raise ValueError("required measured outcome Performance- is absent")
    settings = [n for n in names if n != outcome]
    frequencies = {n: {} for n in settings}
    invalid = []
    outcome_values = []
    missing_outcomes = 0
    for index, row in enumerate(rows, 2):
        for name in settings:
            value = row[name]
            key = "<missing>" if is_missing(value) else value
            frequencies[name][key] = frequencies[name].get(key, 0) + 1
        value = row.get(outcome, "") if outcome else ""
        if is_missing(value):
            missing_outcomes += 1
            continue
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            outcome_values.append(number)
        except ValueError:
            invalid.append({"row": index, "value": value})
    return {"record_count": len(rows), "columns": names, "settings": settings,
            "outcome": {"name": outcome, "direction": "lower is better",
                        "statistics": numeric_stats(outcome_values),
                        "missing_count": missing_outcomes, "invalid": invalid},
            "setting_frequencies": frequencies, "warnings": warnings,
            "assumptions": ["First row is the CSV header.",
                            "Performance- is the sole measured outcome.",
                            "Blank, ?, NA, and common null words are missing."]}

def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def _groups(rows, settings):
    groups = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(name, "") for name in settings)
        groups.setdefault(key, []).append(index)
    return groups

def split_groups(rows, settings, seed=42, test_fraction=0.2):
    """Return disjoint row indexes, assigning identical settings together."""
    groups = _groups(rows, settings)
    keys = sorted(groups, key=lambda key: tuple(str(v) for v in key))
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("at least two configuration groups are required")
    target = max(1, int(round(len(keys) * test_fraction)))
    target = min(target, len(keys) - 1)
    test_keys = set(keys[:target])
    test = sorted(i for key in test_keys for i in groups[key])
    train = sorted(i for key in keys if key not in test_keys for i in groups[key])
    return train, test

class NearestMeanModel:
    """k-nearest mean predictor with train-only numeric ranges and categories."""
    def __init__(self, settings, k=5):
        self.settings, self.k = list(settings), k
        self.numeric, self.categories, self.ranges = {}, {}, {}

    def fit(self, rows, targets):
        if not rows or len(rows) != len(targets):
            raise ValueError("nonempty training data with matching targets is required")
        self.rows, self.targets = list(rows), list(targets)
        for name in self.settings:
            nums = [_finite(row.get(name, "")) for row in self.rows]
            if all(value is not None for value in nums):
                lo, hi = min(nums), max(nums)
                self.numeric[name], self.ranges[name] = True, (lo, hi)
            else:
                self.numeric[name] = False
                self.categories[name] = set(row.get(name, "") for row in self.rows)
        self.mean = statistics.fmean(self.targets)
        return self

    def _distance(self, left, right):
        gaps = []
        for name in self.settings:
            if self.numeric[name]:
                a, b = _finite(left.get(name, "")), _finite(right.get(name, ""))
                lo, hi = self.ranges[name]
                gaps.append(0.5 if a is None or b is None else (0.0 if hi == lo else abs(a-b)/(hi-lo)))
            else:
                gaps.append(0.0 if left.get(name, "") == right.get(name, "") else 1.0)
        return math.sqrt(statistics.fmean([gap * gap for gap in gaps])) if gaps else 0.0

    def predict_one(self, row):
        if set(row) != set(self.settings):
            missing = sorted(set(self.settings) - set(row)); extra = sorted(set(row) - set(self.settings))
            raise ValueError("configuration fields must match; missing=%s extra=%s" % (missing, extra))
        if any(is_missing(str(value)) for value in row.values()):
            raise ValueError("missing or invalid setting value")
        for name in self.settings:
            if self.numeric[name] and _finite(row[name]) is None:
                raise ValueError("invalid numeric setting value for %s" % name)
        nearest = sorted((self._distance(row, old), i) for i, old in enumerate(self.rows))[:max(1, min(self.k, len(self.rows)))]
        return statistics.fmean(self.targets[i] for _, i in nearest)

def evaluate(path, seed=42):
    names, rows, _ = read_runs(path)
    outcome = next((n for n in names if n.rstrip().lower() == "performance-"), None)
    if outcome is None: raise ValueError("required measured outcome Performance- is absent")
    settings = [n for n in names if n != outcome]
    valid = [(i, _finite(row.get(outcome, ""))) for i, row in enumerate(rows)]
    valid = [(i, y) for i, y in valid if y is not None]
    train_i, test_i = split_groups(rows, settings, seed)
    train = [(i, y) for i, y in valid if i in set(train_i)]
    test = [(i, y) for i, y in valid if i in set(test_i)]
    if not train or not test: raise ValueError("finite targets are required in both train and test")
    model = NearestMeanModel(settings).fit([rows[i] for i, _ in train], [y for _, y in train])
    predictions = [model.predict_one({name: rows[i][name] for name in settings}) for i, _ in test]
    mae = statistics.fmean(abs(p-y) for p, (_, y) in zip(predictions, test))
    baseline = statistics.fmean(y for _, y in train)
    base_mae = statistics.fmean(abs(baseline-y) for _, y in test)
    return {"model": model, "settings": settings, "train_rows": len(train), "test_rows": len(test),
            "train_groups": len(set(tuple(rows[i][n] for n in settings) for i, _ in train)),
            "test_groups": len(set(tuple(rows[i][n] for n in settings) for i, _ in test)),
            "held_out_mae": mae, "baseline_mae": base_mae, "training_mean": baseline,
            "excluded_invalid_targets": len(rows)-len(valid), "train_indexes": train_i, "test_indexes": test_i}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize X264 historical runs")
    parser.add_argument("dataset", help="CSV dataset path")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize(args.dataset), indent=2, sort_keys=True))
    except (OSError, ValueError, csv.Error) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
