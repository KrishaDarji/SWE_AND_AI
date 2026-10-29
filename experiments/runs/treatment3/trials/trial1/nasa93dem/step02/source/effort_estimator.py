"""Estimate software effort from the NASA93 dataset using scikit-learn."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


TARGET = "Effort-"
DROP_COLUMNS = {"idX", TARGET, "Defects-", "MonthsX"}


def read_dataset(path: str | Path) -> Tuple[List[Dict[str, Any]], List[float]]:
    rows: List[Dict[str, Any]] = []
    targets: List[float] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            targets.append(float(raw[TARGET]))
            row: Dict[str, Any] = {}
            for key, value in raw.items():
                if key in DROP_COLUMNS or value is None:
                    continue
                value = value.strip()
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    if not rows:
        raise ValueError("dataset contains no rows")
    return rows, targets


def build_model(rows: List[Dict[str, Any]]) -> Pipeline:
    return Pipeline([
        ("features", DictVectorizer(sparse=False)),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=17, min_samples_leaf=2, n_jobs=1
        )),
    ])


def fit_and_score(dataset_path: str | Path) -> Tuple[Pipeline, Dict[str, float]]:
    rows, targets = read_dataset(dataset_path)
    model = build_model(rows)
    train_x, test_x, train_y, test_y = train_test_split(
        rows, targets, test_size=0.2, random_state=17
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    return model, {
        "mae": float(mean_absolute_error(test_y, predicted)),
        "rmse": float(mean_squared_error(test_y, predicted) ** 0.5),
        "test_rows": float(len(test_y)),
    }


def estimate(dataset_path: str | Path, proposal: Mapping[str, Any]) -> float:
    rows, targets = read_dataset(dataset_path)
    model = build_model(rows)
    model.fit(rows, targets)
    return float(model.predict([dict(proposal)])[0])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Estimate project effort in person-months.")
    parser.add_argument("dataset")
    parser.add_argument("proposal_json", help="JSON object with project feature names")
    args = parser.parse_args()
    proposal = json.loads(args.proposal_json)
    value = estimate(args.dataset, proposal)
    print(json.dumps({"estimated_effort_person_months": round(value, 3)}))


if __name__ == "__main__":
    main()
