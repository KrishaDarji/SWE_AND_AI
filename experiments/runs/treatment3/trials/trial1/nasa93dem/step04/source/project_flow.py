"""Assess software project proposals as they arrive.

The model is deliberately small and self-contained: scikit-learn performs
encoding, imputation, and random-forest regression; the standard library
handles CSV/JSON and the arrival-time flow state.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGETS = ("Effort-", "Defects-", "MonthsX")
ID_COLUMNS = {"idX"}


def _clean(value):
    if value is None or str(value).strip() == "":
        return np.nan
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text.lower()


class ProposalFlow:
    """Fit on NASA history and score proposals in arrival order."""

    def __init__(self, data_path, random_state=17):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.model = None
        self.feature_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.baseline = {}

    def fit(self):
        with self.data_path.open(newline="") as handle:
            rows = [{k: _clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        self.feature_columns = [k for k in rows[0] if k not in TARGETS and k not in ID_COLUMNS]
        self.numeric_columns = [k for k in self.feature_columns if any(isinstance(r[k], (int, float)) and not isinstance(r[k], bool) for r in rows)]
        self.categorical_columns = [k for k in self.feature_columns if k not in self.numeric_columns]
        x = [[r.get(k, np.nan) for k in self.feature_columns] for r in rows]
        y = np.array([[float(r[t]) for t in TARGETS] for r in rows])
        numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
        categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
        prep = ColumnTransformer([("num", numeric, list(range(len(self.numeric_columns)))), ("cat", categorical, list(range(len(self.numeric_columns), len(self.feature_columns))))])
        self.model = Pipeline([("prepare", prep), ("regress", RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=self.random_state, n_jobs=1))])
        # Feature order is numeric first, categorical second for the transformer.
        reordered = [[r.get(k, np.nan) for k in self.numeric_columns + self.categorical_columns] for r in rows]
        self.model.fit(reordered, y)
        self.baseline = {t: float(np.median(y[:, i])) for i, t in enumerate(TARGETS)}
        return self

    def assess(self, proposal, available_effort=None):
        if self.model is None:
            raise RuntimeError("fit() must be called before assess()")
        ordered = [[_clean(proposal.get(k)) for k in self.numeric_columns + self.categorical_columns]]
        prediction = np.maximum(self.model.predict(ordered)[0], 0.0)
        effort, defects, months = map(float, prediction)
        risk = min(1.0, 0.55 * defects / max(self.baseline["Defects-"], 1.0) + 0.45 * months / max(self.baseline["MonthsX"], 1.0))
        decision = "review"
        if available_effort is not None and effort <= float(available_effort) and risk < 0.9:
            decision = "accept"
        elif available_effort is not None and effort > float(available_effort):
            decision = "defer"
        return {"name": proposal.get("name", "unnamed"), "predicted_effort": round(effort, 2), "predicted_defects": round(defects, 2), "predicted_months": round(months, 2), "risk_score": round(risk, 3), "decision": decision}


def main():
    parser = argparse.ArgumentParser(description="Assess proposal JSON objects from stdin.")
    parser.add_argument("data", help="path to nasa93dem.csv")
    parser.add_argument("--effort-budget", type=float)
    args = parser.parse_args()
    flow = ProposalFlow(args.data).fit()
    for line in __import__("sys").stdin:
        if line.strip():
            print(json.dumps(flow.assess(json.loads(line), args.effort_budget), sort_keys=True))


if __name__ == "__main__":
    main()
