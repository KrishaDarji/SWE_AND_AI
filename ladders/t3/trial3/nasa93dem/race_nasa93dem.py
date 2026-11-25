#!/usr/bin/env python3
"""Treatment 3 race: effort estimation versus a historical median baseline."""
import csv
import math
import os
import sys
import time
from statistics import median

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET = "moot/optimize/process/nasa93dem.csv"
TARGET = "Effort-"
SIZE = "Kloc+"
REPEATS = 3
FOLDS = 5
SEED = 3103


def load_dataset(path=DATASET):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    fields = list(rows[0])
    required = {TARGET, SIZE}
    if not required.issubset(fields):
        raise ValueError("missing required columns: %s" % sorted(required - set(fields)))
    for number, row in enumerate(rows, 2):
        if set(row) != set(fields):
            raise ValueError("inconsistent columns at CSV row %d" % number)
        try:
            float(row[TARGET]); float(row[SIZE])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid numeric value at CSV row %d" % number) from exc
    return fields, rows


def _metric(actual, predicted):
    errors = [a - p for a, p in zip(actual, predicted)]
    ape = [abs(e) / a for a, e in zip(actual, errors) if a != 0]
    return {"mae": mean_absolute_error(actual, predicted),
            "rmse": math.sqrt(mean_squared_error(actual, predicted)),
            "mdape": median(ape) * 100.0}


def evaluate(path=DATASET):
    fields, rows = load_dataset(path)
    features = [f for f in fields if f not in {TARGET, "idX"}]
    numeric = [f for f in features if f in {SIZE, "YearX", "MonthsX"}]
    categorical = [f for f in features if f not in numeric]
    numeric_indices = [features.index(f) for f in numeric]
    categorical_indices = [features.index(f) for f in categorical]
    transformer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_indices),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_indices),
    ])
    model = Pipeline([("features", transformer), ("forest", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1))])
    x = [[float(row[name]) if name in numeric else row[name].strip()
          for name in features] for row in rows]
    y = [float(row[TARGET]) for row in rows]
    scores = {"treatment3": [], "historical_baseline": []}
    splitter = RepeatedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    started = time.perf_counter()
    for train, test in splitter.split(x):
        train_x = [x[i] for i in train]
        test_x = [x[i] for i in test]
        train_y = [y[i] for i in train]
        test_y = [y[i] for i in test]
        model.fit(train_x, train_y)
        scores["treatment3"].append(_metric(test_y, model.predict(test_x)))
        baseline = [median(train_y)] * len(test_y)
        scores["historical_baseline"].append(_metric(test_y, baseline))
    elapsed = time.perf_counter() - started
    summary = {}
    for name, folds in scores.items():
        summary[name] = {metric: sum(item[metric] for item in folds) / len(folds)
                         for metric in ("mae", "rmse", "mdape")}
    return {"rows": len(rows), "features": len(features), "folds": FOLDS * REPEATS,
            "runtime_seconds": elapsed, "scores": summary}


def format_report(result):
    t = result["scores"]["treatment3"]
    b = result["scores"]["historical_baseline"]
    better = "Treatment 3" if t["mae"] < b["mae"] else "historical baseline"
    lines = ["NASA93DEM Treatment 3 race", "rows=%d features=%d evaluation_folds=%d" %
             (result["rows"], result["features"], result["folds"]),
             "protocol=3 repeats of shuffled 5-fold CV; seed=%d; all preprocessing fit on training folds" % SEED,
             "method,MAE,RMSE,median_APE_percent",
             "Treatment 3,%.4f,%.4f,%.4f" % (t["mae"], t["rmse"], t["mdape"]),
             "historical baseline,%.4f,%.4f,%.4f" % (b["mae"], b["rmse"], b["mdape"]),
             "winner_by_mean_MAE=%s" % better,
             "baseline=median effort of the training fold; Treatment 3=one-hot categorical/numeric features plus random forest",
             "conditions=results apply to random train/test splits resembling this small historical sample; baseline can be preferable when data are sparse or feature relationships do not generalize",
             "uncertainty=fold averages vary across splits; no confidence intervals are claimed, and repeated CV folds are not independent observations",
             "limitations=93 projects, observational data, possible time/process and selection effects, no external validation, and effort is right-skewed",
             "bias_sources=project-era changes, measurement conventions, omitted staffing/process variables, and tuning choices can favor either method",
             "runtime_seconds=%.6f" % result["runtime_seconds"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else DATASET
    print(format_report(evaluate(dataset)), end="")
