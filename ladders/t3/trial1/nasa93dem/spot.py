"""Spot an unusually inconsistent NASA software project record."""
import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = {"idX", "centerX", "Kloc+", "Effort-", "Defects-", "MonthsX"}


def read_records(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no records")
    return reader.fieldnames, rows


def build_matrix(fields: List[str], rows: List[Dict[str, str]]):
    features = [f for f in fields if f not in TARGET_COLUMNS]
    numeric = [f for f in features if f in {"YearX"}]
    numeric += [f for f in features if f not in numeric and _is_numeric(rows, f)]
    categorical = [f for f in features if f not in numeric]
    transformers = []
    numeric_indices = [features.index(f) for f in numeric]
    categorical_indices = [features.index(f) for f in categorical]
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_indices))
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_indices))
    prep = ColumnTransformer(transformers=transformers)
    matrix = [[row.get(field, "").strip() for field in features] for row in rows]
    return features, prep.fit_transform(matrix)


def _is_numeric(rows, field):
    values = [r.get(field, "").strip() for r in rows if r.get(field, "").strip()]
    if not values:
        return False
    try:
        [float(v) for v in values]
        return True
    except ValueError:
        return False


def spot(path: str, contamination: float = 0.1, random_state: int = 19):
    fields, rows = read_records(path)
    _, matrix = build_matrix(fields, rows)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)
    model.fit(matrix)
    scores = model.decision_function(matrix)
    labels = model.predict(matrix)
    ranked = sorted(range(len(rows)), key=lambda i: scores[i])
    index = ranked[0]
    row = rows[index]
    return {"record": row, "row_number": index + 2, "anomaly_score": float(scores[index]), "isolation_label": int(labels[index]), "ranked_scores": [{"row_number": i + 2, "id": rows[i].get("idX"), "score": float(scores[i])} for i in ranked]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--contamination", type=float, default=0.1)
    args = parser.parse_args()
    started = time.perf_counter()
    result = spot(args.csv_path, args.contamination)
    result["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
