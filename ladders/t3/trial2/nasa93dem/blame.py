#!/usr/bin/env python3
"""Independent association analysis for NASA93DEM project effort."""
import argparse
import csv
import math
import os
import re
import statistics
import sys
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_validate(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError("dataset is empty or has no header")
    headers = list(rows[0])
    target = next((h for h in headers if h.lower().startswith("effort")), None)
    if target is None:
        raise ValueError("could not identify effort target")
    if len(rows) < 10:
        raise ValueError("at least 10 rows are required")
    numeric = {}
    for h in headers:
        vals = []
        for row in rows:
            if row[h].strip() == "":
                vals.append(None)
            else:
                try:
                    vals.append(float(row[h]))
                except ValueError:
                    vals.append(None)
        numeric[h] = vals
    efforts = numeric[target]
    if any(v is None or not math.isfinite(v) for v in efforts):
        raise ValueError("effort target contains missing or non-finite values")
    factors = [h for h in headers if h != target and h not in {"idX"}]
    if not factors:
        raise ValueError("no project factors remain after excluding identifiers")
    return rows, headers, target, numeric, factors


def build_model(rows, factors, numeric):
    X = []
    for row in rows:
        item = []
        for h in factors:
            item.append(numeric[h][len(X)] if numeric[h][len(X)] is not None else row[h].strip())
        X.append(item)
    numeric_factors = [i for i, h in enumerate(factors) if all(isinstance(x[i], float) for x in X)]
    categorical_factors = [i for i, h in enumerate(factors) if i not in numeric_factors]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformer = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_factors),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", encoder)]), categorical_factors),
    ], remainder="drop")
    model = Pipeline([("features", transformer), ("classifier", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=0))])
    return model, X, numeric_factors, categorical_factors


def analyze(path):
    rows, headers, target, numeric, factors = load_validate(path)
    effort = numeric[target]
    cutoff = statistics.median(effort)
    y = [int(v >= cutoff) for v in effort]
    model, X, nums, cats = build_model(rows, factors, numeric)
    folds = min(5, min(y.count(0), y.count(1)))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    predictions = [int(p >= 0.5) for p in probabilities]
    model.fit(X, y)
    names = list(model.named_steps["features"].get_feature_names_out())
    coefficients = model.named_steps["classifier"].coef_[0]
    grouped = defaultdict(list)
    for name, coefficient in zip(names, coefficients):
        clean = name.split("__", 1)[-1]
        match = re.match(r"x(\d+)", clean)
        factor = factors[int(match.group(1))] if match else next((f for f in factors if clean.startswith(f)), clean)
        grouped[factor].append(abs(float(coefficient)))
    ranking = sorted(((factor, max(values), len(values)) for factor, values in grouped.items()), key=lambda x: (-x[1], x[0]))
    report = []
    report.append("NASA93DEM blame skill: effort association analysis")
    report.append(f"Rows: {len(rows)}; effort target: {target}; excessive threshold (median): {cutoff:.3f}")
    report.append(f"Excessive projects: {sum(y)}/{len(y)}; model: balanced logistic regression with one-hot categorical factors")
    report.append(f"{folds}-fold CV ROC-AUC: {roc_auc_score(y, probabilities):.3f}")
    report.append(f"{folds}-fold CV accuracy: {accuracy_score(y, predictions):.3f}; balanced accuracy: {balanced_accuracy_score(y, predictions):.3f}")
    report.append("Ranked factors (maximum absolute standardized/encoded coefficient):")
    for i, (factor, score, levels) in enumerate(ranking, 1):
        report.append(f"  {i}. {factor}: {score:.3f} ({levels} encoded level(s))")
    report.append("Interpretation: larger coefficients indicate stronger adjusted association with the high-effort label in this sample; signs and levels are available in the fitted encoding, but rankings use magnitude.")
    report.append("Caution: association is not causation. The median cutoff is sample-relative; n=93 limits precision. Results may be confounded by project size (Kloc+), year, organization/site, process maturity, and correlated ratings. Cross-validation is exploratory and does not establish a causal effect.")
    return "\n".join(report) + "\n", {"rows": len(rows), "target": target, "cutoff": cutoff, "auc": roc_auc_score(y, probabilities), "accuracy": accuracy_score(y, predictions), "balanced_accuracy": balanced_accuracy_score(y, predictions), "ranking": ranking}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    report, _ = analyze(args.dataset)
    print(report, end="")


if __name__ == "__main__":
    main()
