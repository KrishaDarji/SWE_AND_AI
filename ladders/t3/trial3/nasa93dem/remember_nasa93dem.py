#!/usr/bin/env python3
"""Independent descriptive and relationship analysis for nasa93dem.csv."""
import csv
import math
import os
import sys
from collections import Counter
from statistics import mean, median, stdev

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

OUTCOMES = ("Kloc+", "Effort-", "Defects-", "MonthsX")
RATING_ORDER = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no data rows")
    headers = list(rows[0])
    if any(value is None for value in headers):
        raise ValueError("CSV has an unnamed column")
    missing = [name for name in OUTCOMES if name not in headers]
    if missing:
        raise ValueError("Missing outcome columns: " + ", ".join(missing))
    return headers, rows


def numeric(rows, column):
    values = []
    bad = 0
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            bad += 1
    return values, bad


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denominator = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denominator if denominator else None


def summarize_numeric(rows, column):
    values, bad = numeric(rows, column)
    if not values:
        return "no numeric values"
    spread = stdev(values) if len(values) > 1 else 0.0
    return (f"n={len(values)}, missing_or_invalid={bad}, mean={mean(values):.2f}, "
            f"median={median(values):.2f}, sd={spread:.2f}, min={min(values):.2f}, max={max(values):.2f}")


def main(dataset_path):
    headers, rows = load_csv(dataset_path)
    features = [h for h in headers if h not in OUTCOMES]
    errors = []
    for row_number, row in enumerate(rows, 2):
        if any(row.get(h, "") == "" for h in headers):
            errors.append(f"row {row_number}: blank field")
    print("NASA93DEM historical project summary")
    print(f"dataset={os.path.abspath(dataset_path)}")
    print(f"rows={len(rows)} columns={len(headers)}")
    print(f"project_features ({len(features)}): {', '.join(features)}")
    print(f"outcomes ({len(OUTCOMES)}): {', '.join(OUTCOMES)}")
    print(f"validation_errors={len(errors)}")
    if errors:
        print("validation_detail=" + " | ".join(errors[:5]))

    print("\nOutcome distributions")
    for outcome in OUTCOMES:
        print(f"{outcome}: {summarize_numeric(rows, outcome)}")
    print("\nCategorical feature levels")
    for feature in features:
        values = [row[feature].strip().lower() for row in rows]
        print(f"{feature}: levels={len(set(values))}, top={Counter(values).most_common(4)}")

    print("\nOutcome relationships")
    outcome_values = {o: numeric(rows, o)[0] for o in OUTCOMES}
    for feature in features:
        encoded = []
        label_values = []
        for row in rows:
            raw = row[feature].strip().lower()
            encoded.append(RATING_ORDER.get(raw, float("nan")))
        for outcome in OUTCOMES:
            pairs = [(x, y) for x, y in zip(encoded, outcome_values[outcome]) if not math.isnan(x)]
            if len(pairs) >= 3:
                r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
                if r is not None:
                    label_values.append(f"{outcome}:r={r:+.2f}")
        if label_values:
            print(f"{feature}: " + ", ".join(label_values))

    print("\nSimple predictive check")
    categorical = [[row[f].strip().lower() for f in features] for row in rows]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # compatibility with older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    design = encoder.fit_transform(categorical)
    for outcome in OUTCOMES:
        target = outcome_values[outcome]
        model = LinearRegression().fit(design, target)
        print(f"{outcome}: in_sample_R2={model.score(design, target):.3f}, predictors={design.shape[1]}")

    print("\nAssumptions and limitations")
    print("- Columns ending in + or - and MonthsX are treated as delivery/effort outcomes; all others are project features.")
    print("- Rating codes vl/l/n/h/vh/xh are mapped to increasing ordinal values for correlations; this is an approximation.")
    print("- Correlations are unadjusted associations, not causal effects; the in-sample regression check is descriptive, not validation.")
    print("- The file has no explicit delivery-date or on-time-success field, so MonthsX is used as duration and Effort-/Defects- as outcomes.")
    print("- Potential quality issues include compact categorical codes, historical measurement conventions, and possible skew/outliers in numeric outcomes.")
    return 0 if not errors else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: remember_nasa93dem.py DATASET", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
