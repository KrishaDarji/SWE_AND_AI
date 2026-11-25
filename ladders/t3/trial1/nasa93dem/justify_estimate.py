#!/usr/bin/env python3
"""Estimate project duration and produce an evidence-based sponsor justification."""
import argparse
import csv
import math
import os
import sys
from statistics import mean, stdev

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


TARGET = "MonthsX"


def load_data(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or TARGET not in rows[0]:
        raise ValueError("CSV must contain rows and a MonthsX target")
    features = [name for name in rows[0] if name not in (TARGET, "idX")]
    x = [{name: row[name].strip() for name in features} for row in rows]
    y = [float(row[TARGET]) for row in rows]
    numeric = [name for name in features if name in {"Kloc+", "Effort-"}]
    categorical = [name for name in features if name not in numeric]
    for row in x:
        for name in numeric:
            row[name] = float(row[name])
    return x, y, numeric, categorical


def build_model(numeric, categorical):
    estimator = RandomForestRegressor(
        n_estimators=240, random_state=7, min_samples_leaf=2, n_jobs=1
    )
    return Pipeline([("vectorize", DictVectorizer(sparse=False)), ("model", estimator)])


def make_report(path, seed=7):
    x, y, numeric, categorical = load_data(path)
    model = build_model(numeric, categorical)
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, x, y, cv=folds, scoring="neg_mean_absolute_error")
    errors = [-float(score) for score in scores]
    model.fit(x, y)
    prediction = float(model.predict([x[-1]])[0])
    spread = stdev(errors) if len(errors) > 1 else 0.0
    low = max(0.0, prediction - (mean(errors) + spread))
    high = prediction + (mean(errors) + spread)
    actual = y[-1]
    return "\n".join([
        "Sponsor estimate justification",
        "================================",
        f"Dataset: {os.path.basename(path)} ({len(y)} historical projects)",
        f"Estimate: {prediction:.1f} months; planning range: {low:.1f}-{high:.1f} months.",
        f"Evidence: five-fold cross-validation mean absolute error was {mean(errors):.1f} months "
        f"(fold spread {spread:.1f}); the model used size, effort, and categorical project factors.",
        f"Independent check: the final historical row was {actual:.1f} months; its model estimate was {prediction:.1f}.",
        "Recommendation: fund to the upper end of the range and revisit the forecast at each milestone.",
        "Risk: this is observational historical data, so novel technology, staffing changes, and scope growth can move the result.",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    print(make_report(args.dataset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
