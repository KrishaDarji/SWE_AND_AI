"""NASA93 effort estimator and historical-baseline comparison."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Effort-"
DROP_COLUMNS = {"idX", TARGET, "Defects-", "MonthsX"}


def read_dataset(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or TARGET not in rows[0]:
        raise ValueError("dataset must contain rows and an Effort- column")
    fields = [field for field in rows[0] if field not in DROP_COLUMNS]
    x = [[row[field].strip() for field in fields] for row in rows]
    y = np.asarray([float(row[TARGET].strip()) for row in rows], dtype=float)
    numeric = []
    categorical = []
    for field in fields:
        try:
            [float(row[fields.index(field)]) for row in x]
            numeric.append(field)
        except ValueError:
            categorical.append(field)
    for row in x:
        for field in numeric:
            index = fields.index(field)
            row[index] = float(row[index])
    return x, y, [fields.index(field) for field in numeric], [fields.index(field) for field in categorical]


def build_model(numeric, categorical):
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    forest = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1
    )
    return Pipeline([("preprocess", prep), ("regressor", forest)])


def run(path, test_size=0.25, random_state=42):
    x, y, numeric, categorical = read_dataset(path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    model = build_model(numeric, categorical)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    baseline_prediction = np.full(len(y_test), np.median(y_train))
    return {
        "rows": len(y),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "model": {"mae": mean_absolute_error(y_test, prediction),
                  "rmse": mean_squared_error(y_test, prediction) ** 0.5},
        "historical_baseline": {
            "mae": mean_absolute_error(y_test, baseline_prediction),
            "rmse": mean_squared_error(y_test, baseline_prediction) ** 0.5,
            "prediction_rule": "median training-set Effort-",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
