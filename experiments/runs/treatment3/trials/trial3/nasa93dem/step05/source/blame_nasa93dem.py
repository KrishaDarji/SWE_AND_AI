#!/usr/bin/env python3
"""Association analysis for excessive effort in NASA93DEM."""
import csv
import math
import os
import random
import sys
from collections import Counter, defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/process/nasa93dem.csv"))


def load_validate(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError("dataset is empty or has no header")
    fields = list(rows[0])
    target = next((f for f in fields if f.lower().replace("-", "") == "effort"), None)
    if target is None:
        raise ValueError("could not identify effort target")
    if len(rows) != 93:
        raise ValueError(f"expected 93 rows, found {len(rows)}")
    missing = [f for r in rows for f in fields if not r[f].strip()]
    if missing:
        raise ValueError(f"missing cells: {len(missing)}")
    efforts = []
    for r in rows:
        try:
            efforts.append(float(r[target]))
        except ValueError as exc:
            raise ValueError("effort target is not numeric") from exc
    if any(x <= 0 for x in efforts):
        raise ValueError("effort must be positive")
    return rows, fields, target, efforts


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def analyze(path=DATASET, seed=17):
    rows, fields, target, efforts = load_validate(path)
    threshold = percentile(efforts, .75)
    y = [int(v > threshold) for v in efforts]
    features = [f for f in fields if f not in (target, "idX")]
    X = [[r[f].strip() for f in features] for r in rows]
    encoder = OneHotEncoder(handle_unknown="ignore")
    model = Pipeline([("encode", ColumnTransformer([("categorical", encoder, list(range(len(features))))])),
                      ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(model, X, y, cv=cv, scoring=("roc_auc", "accuracy"), n_jobs=1)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.30, stratify=y, random_state=seed)
    model.fit(train_x, train_y)
    perm = permutation_importance(model, test_x, test_y, scoring="roc_auc", n_repeats=40, random_state=seed)
    # Permutation importance is measured per raw project factor because each is a feature group.
    importances = dict(zip(features, perm.importances_mean))
    stds = dict(zip(features, perm.importances_std))
    ranked = sorted(features, key=lambda f: (importances[f], f), reverse=True)
    # Bootstrap the fitted-model ranking signal on held-out predictions for a compact uncertainty interval.
    rng = random.Random(seed)
    boot = defaultdict(list)
    for _ in range(120):
        idx = [rng.randrange(len(test_y)) for _ in test_y]
        bx, by = [test_x[i] for i in idx], [test_y[i] for i in idx]
        if len(set(by)) < 2:
            continue
        p = permutation_importance(model, bx, by, scoring="roc_auc", n_repeats=3, random_state=rng.randrange(1_000_000))
        for f, value in zip(features, p.importances_mean):
            boot[f].append(value)
    return {"rows": len(rows), "features": features, "target": target, "threshold": threshold,
            "high_count": sum(y), "cv_auc": scores["test_roc_auc"].mean(),
            "cv_auc_sd": scores["test_roc_auc"].std(), "cv_accuracy": scores["test_accuracy"].mean(),
            "ranked": [(f, importances[f], stds[f], percentile(boot[f], .025) if boot.get(f) else float("nan"),
                        percentile(boot[f], .975) if boot.get(f) else float("nan")) for f in ranked]}


def main():
    result = analyze(sys.argv[1] if len(sys.argv) > 1 else DATASET)
    print(f"validated_rows={result['rows']} target={result['target']} features={len(result['features'])}")
    print(f"excessive_effort=Effort- > {result['threshold']:.3f} (75th percentile); high_count={result['high_count']}")
    print(f"cv_roc_auc={result['cv_auc']:.3f} +/- {result['cv_auc_sd']:.3f}; cv_accuracy={result['cv_accuracy']:.3f}")
    print("rank,factor,permutation_auc_delta,repeat_sd,bootstrap_95pct_low,bootstrap_95pct_high")
    for i, (f, imp, sd, low, high) in enumerate(result["ranked"], 1):
        print(f"{i},{f},{imp:.5f},{sd:.5f},{low:.5f},{high:.5f}")
    print("interpretation=These are predictive associations with high effort, not causal effects; correlated planning factors and historical/project selection confounding remain possible.")


if __name__ == "__main__":
    main()
