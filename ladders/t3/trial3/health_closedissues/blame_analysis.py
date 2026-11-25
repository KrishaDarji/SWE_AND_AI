#!/usr/bin/env python3
"""Association analysis for high model error in Health-ClosedIssues."""
import csv
import math
import os
import sys
from collections import defaultdict
from statistics import mean, median

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
NUMERIC = ["ACC+", "PRED40+", "N_estimators", "Min_sample_leaves",
           "Min_impurity_decrease", "Max_depth"]
CATEGORICAL = ["criterion"]


def load(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != REQUIRED:
        raise ValueError("unexpected or missing columns")
    if any(any(v == "" for v in row.values()) for row in rows):
        raise ValueError("missing value")
    for row in rows:
        for name in REQUIRED:
            if name != "criterion":
                row[name] = float(row[name])
    return rows


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def analyze(path, seed=17):
    rows = load(path)
    errors = [r["MRE-"] for r in rows]
    cutoff = sorted(errors)[math.ceil(.75 * len(errors)) - 1]
    y = [int(v >= cutoff) for v in errors]
    x = [[r[c] for c in NUMERIC + CATEGORICAL] for r in rows]
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(range(len(NUMERIC)))),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [len(NUMERIC)]),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed)
    model.fit(x_train, y_train)
    predicted = model.predict_proba(x_test)[:, 1]
    perm = permutation_importance(model, x_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed)
    names = NUMERIC + CATEGORICAL
    ranked = sorted(zip(names, perm.importances_mean, perm.importances_std), key=lambda z: z[1], reverse=True)
    associations = []
    for name in NUMERIC:
        values = [r[name] for r in rows]
        associations.append((name, pearson(values, y), mean(v for v, hi in zip(values, y) if hi), mean(v for v, hi in zip(values, y) if not hi)))
    associations.sort(key=lambda z: abs(z[1]), reverse=True)
    return rows, cutoff, sum(y), model, ranked, associations, roc_auc_score(y_test, predicted), accuracy_score(y_test, [int(p >= .5) for p in predicted])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "moot/optimize/hpo/Health-ClosedIssues0000.csv"
    rows, cutoff, high, model, ranked, associations, auc, accuracy = analyze(path)
    print("dataset_rows=%d" % len(rows))
    print("target=MRE-; high_error_definition=MRE- >= 75th_percentile")
    print("high_error_cutoff=%.12f; high_error_count=%d; prevalence=%.6f" % (cutoff, high, high / len(rows)))
    print("holdout_auc=%.6f; holdout_accuracy=%.6f" % (auc, accuracy))
    print("permutation_importance_rank (mean_auc_decrease, std):")
    for name, value, std in ranked:
        print("%s,%.8f,%.8f" % (name, value, std))
    print("numeric_univariate_association (pearson_with_high_error, high_mean, nonhigh_mean):")
    for name, corr, high_mean, low_mean in associations:
        print("%s,%.8f,%.8f,%.8f" % (name, corr, high_mean, low_mean))


if __name__ == "__main__":
    main()
