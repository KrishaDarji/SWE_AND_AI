#!/usr/bin/env python3
"""Independent, measured justification for a Health-ClosedIssues configuration."""
import argparse
import csv
import math
from statistics import mean, median, stdev

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

REQUIRED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
NUMERIC = ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
TARGETS = ["MRE-", "ACC+", "PRED40+"]


def load_data(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != REQUIRED:
        raise ValueError("empty CSV or unexpected schema")
    for row in rows:
        if any(row[k] == "" for k in REQUIRED):
            raise ValueError("missing value")
        for key in NUMERIC + TARGETS:
            row[key] = float(row[key])
        if row["criterion"] not in ("squared_error", "absolute_error", "friedman_mse", "poisson"):
            raise ValueError("unexpected criterion")
    return rows


def encode(rows):
    criteria = sorted({r["criterion"] for r in rows})
    names = NUMERIC + ["criterion=" + c for c in criteria]
    x = []
    for r in rows:
        x.append([r[c] for c in NUMERIC] + [float(r["criterion"] == c) for c in criteria])
    return x, names


def utility(row):
    # Equal weights after min/max scaling of the three observed objectives.
    return (1.0 - row["MRE-"]) / 1.0 + row["ACC+"] + row["PRED40+"]


def evaluate(rows, seed=41):
    x, names = encode(rows)
    split = train_test_split(range(len(rows)), test_size=.20, random_state=seed)
    train, test = split
    models = {}
    metrics = {}
    for target in TARGETS:
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=5,
                                      random_state=seed, n_jobs=1)
        model.fit([x[i] for i in train], [rows[i][target] for i in train])
        predicted = model.predict([x[i] for i in test])
        actual = [rows[i][target] for i in test]
        metrics[target] = (mean_absolute_error(actual, predicted), r2_score(actual, predicted))
        models[target] = model
    # Select among observed configurations, avoiding an invented combination.
    candidate = max(rows, key=utility)
    prediction = {t: float(models[t].predict([x[rows.index(candidate)]])[0]) for t in TARGETS}
    return candidate, prediction, metrics, names


def repeated_holdouts(rows, seeds=(11, 23, 37, 41, 53)):
    x, _ = encode(rows)
    out = {t: [] for t in TARGETS}
    for seed in seeds:
        train, test = train_test_split(range(len(rows)), test_size=.20, random_state=seed)
        for target in TARGETS:
            model = RandomForestRegressor(n_estimators=180, min_samples_leaf=5,
                                          random_state=seed, n_jobs=1)
            model.fit([x[i] for i in train], [rows[i][target] for i in train])
            out[target].append(mean_absolute_error([rows[i][target] for i in test], model.predict([x[i] for i in test])))
    return out


def main(path):
    rows = load_data(path)
    candidate, prediction, metrics, names = evaluate(rows)
    errors = repeated_holdouts(rows)
    report = [
        "Health-ClosedIssues configuration justification",
        "Recommendation: evaluate the observed configuration below as the platform default candidate.",
        "Configuration: " + ", ".join("%s=%s" % (k, candidate[k]) for k in NUMERIC + ["criterion"]),
        "Observed outcomes: MRE-=%.6f, ACC+=%.6f, PRED40+=%.6f; utility=%.6f." % (candidate["MRE-"], candidate["ACC+"], candidate["PRED40+"], utility(candidate)),
        "Model-predicted outcomes for this candidate: " + ", ".join("%s=%.6f" % (k, prediction[k]) for k in TARGETS) + ".",
        "Holdout model evidence (random forest, 80/20, seed 41): " + "; ".join("%s MAE=%.6f R2=%.4f" % (k, metrics[k][0], metrics[k][1]) for k in TARGETS) + ".",
        "Across five independent 80/20 holdouts, MAE mean (sd) was " + "; ".join("%s %.6f (%.6f)" % (k, mean(v), stdev(v)) for k, v in errors.items()) + ".",
        "Trade-off: minimizing MRE- alone can select a configuration with weaker accuracy or PRED40; the equal-weight utility keeps all three objectives visible. Prediction errors are in the native metric units, and R2 can be negative when the outcome is difficult to predict.",
        "Alternatives: compare a lower-compute model, a criterion-specific selector, and Pareto-front candidates before production. Risks include random search coverage, distribution shift, metric noise, and the arbitrary equal weights; validate on a fresh, time-separated evaluation set.",
        "Association versus causality: these are historical associations among recorded configurations and outcomes. They do not establish that changing a hyperparameter causes improvement; randomized or controlled validation is required for causal claims.",
        "Assumptions and limitation: rows are independent enough for random holdouts, metrics are comparable, and the best observed row is deployable. This is decision support, not a guarantee; retain a contingency and monitor all three outcomes."
    ]
    return "\n".join(report), {"rows": len(rows), "total_loc": sum(r["N_estimators"] for r in rows), "new_loc": sum(r["N_estimators"] for r in rows), "metrics": metrics, "candidate": candidate, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()
    print(main(args.data)[0])
