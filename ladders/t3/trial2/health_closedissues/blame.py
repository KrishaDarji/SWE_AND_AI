#!/usr/bin/env python3
"""Independent blame analysis for Health-ClosedIssues hyperparameters."""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "MRE-"
EXPECTED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def quantile(values, p):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def ranks(values):
    pairs = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1
        for _, idx in pairs[i:j + 1]:
            out[idx] = rank
        i = j + 1
    return out


def correlation(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def load_dataset(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    if set(rows[0]) != set(EXPECTED):
        raise ValueError("unexpected columns: " + repr(list(rows[0])))
    data = []
    for line, row in enumerate(rows, 2):
        try:
            item = {key: row[key] for key in EXPECTED}
            for key in ("MRE-", "ACC+", "PRED40+", "N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"):
                item[key] = float(item[key])
            item["criterion"] = str(item["criterion"])
            if not (0 <= item["MRE-"] <= 1):
                raise ValueError("MRE- outside [0, 1]")
            data.append(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid row %d: %s" % (line, exc)) from exc
    return data


def analyze(path, seed=20261029):
    rows = load_dataset(path)
    threshold = quantile([r[TARGET] for r in rows], 0.75)
    y = [int(r[TARGET] >= threshold) for r in rows]
    features = [k for k in EXPECTED if k != TARGET]
    numeric = [k for k in features if k != "criterion"]
    X = [[r[k] for k in features] for r in rows]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [3])], remainder="passthrough")
    model = Pipeline([("preprocess", prep), ("model", ExtraTreesClassifier(n_estimators=250, random_state=seed, n_jobs=-1, class_weight="balanced"))])
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    perm = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=seed, scoring="roc_auc", n_jobs=-1)
    # Repeated raw-column permutation is unavailable through the transformed matrix;
    # this reports one importance per original feature by permuting test dictionaries.
    baseline = model.score(x_test, y_test)
    importances = {}
    for key in features:
        drops = []
        for rep in range(10):
            shuffled = list(x_test)
            column = features.index(key)
            values = [row[column] for row in shuffled]
            shift = (rep * 37 + 1) % len(values)
            values = values[shift:] + values[:shift]
            altered = [row[:column] + [value] + row[column + 1:] for row, value in zip(shuffled, values)]
            drops.append(baseline - model.score(altered, y_test))
        importances[key] = {"mean_accuracy_drop": sum(drops) / len(drops), "std_accuracy_drop": (sum((v - sum(drops) / len(drops)) ** 2 for v in drops) / len(drops)) ** 0.5}
    associations = {}
    for key in numeric:
        associations[key] = {"spearman_with_MRE": correlation([r[key] for r in rows], [r[TARGET] for r in rows]), "high_error_rate_by_quartile": []}
        values = [r[key] for r in rows]
        cuts = [quantile(values, q) for q in (0.25, 0.5, 0.75)]
        for low, high in zip([-math.inf] + cuts, cuts + [math.inf]):
            group = [i for i, value in enumerate(values) if low < value <= high]
            associations[key]["high_error_rate_by_quartile"].append(sum(y[i] for i in group) / len(group) if group else None)
    for key in ["criterion"]:
        groups = defaultdict(list)
        for row, flag in zip(rows, y):
            groups[row[key]].append(flag)
        associations[key] = {"high_error_rate_by_value": {v: sum(flags) / len(flags) for v, flags in sorted(groups.items())}}
    ranked = sorted(importances, key=lambda k: importances[k]["mean_accuracy_drop"], reverse=True)
    return {"dataset": os.path.basename(path), "rows": len(rows), "target": TARGET, "high_error_definition": "MRE- >= empirical 75th percentile", "threshold": threshold, "high_error_rate": sum(y) / len(y), "heldout_accuracy": accuracy, "model": "ExtraTreesClassifier (250 trees; one-hot criterion)", "permutation_importance": importances, "ranked_hyperparameters": ranked, "univariate_associations": associations, "assumptions_limitations": ["Rows are treated as independent observations.", "The 75th percentile is a relative, analysis-defined threshold, not a domain failure limit.", "Permutation importance measures predictive association on this split; it is not causal attribution.", "Correlated or discretely sampled hyperparameters can share or mask importance; seed and split add uncertainty.", "Observed associations may reflect experimental design, interactions, or unmeasured confounding."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.dataset)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"rows": result["rows"], "threshold": result["threshold"], "high_error_rate": result["high_error_rate"], "heldout_accuracy": result["heldout_accuracy"], "ranked_hyperparameters": result["ranked_hyperparameters"]}, indent=2))


if __name__ == "__main__":
    main()
