#!/usr/bin/env python3
"""Standalone NASA93DEM effort justification."""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline

TARGET = "Effort-"
REQUIRED = {"Kloc+", TARGET}
ID_COLUMNS = {"idX"}


def _number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_validate(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    columns = set(rows[0])
    missing = REQUIRED - columns
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if any(set(row) != columns for row in rows):
        raise ValueError("rows do not share the header schema")
    for row_number, row in enumerate(rows, 2):
        for column in ("Kloc+", TARGET):
            if _number(row[column]) is None:
                raise ValueError(f"non-numeric {column} at row {row_number}")
    return rows


def feature_rows(rows):
    features = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in ID_COLUMNS or key == TARGET:
                continue
            number = _number(value)
            item[key] = number if number is not None else value.strip()
        features.append(item)
    return features


def representative_proposal(rows):
    proposal = {}
    for key in rows[0]:
        if key in ID_COLUMNS or key == TARGET:
            continue
        numbers = [_number(row[key]) for row in rows]
        if all(value is not None for value in numbers):
            proposal[key] = statistics.median(numbers)
        else:
            values = [row[key].strip() for row in rows]
            proposal[key] = statistics.multimode(values)[0]
    return proposal


def model_and_metrics(rows, seed=42):
    X = feature_rows(rows)
    y = [_number(row[TARGET]) for row in rows]
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(matrix, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2, n_jobs=1)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    pipeline = make_pipeline(DictVectorizer(sparse=True), RandomForestRegressor(
        n_estimators=200, random_state=seed, min_samples_leaf=2, n_jobs=1))
    oof = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=1)
    residuals = [abs(actual - predicted) for actual, predicted in zip(y, oof)]
    return model, vectorizer, {
        "mae": mean_absolute_error(test_y, prediction),
        "rmse": math.sqrt(mean_squared_error(test_y, prediction)),
        "r2": r2_score(test_y, prediction),
        "oof_mae": mean_absolute_error(y, oof),
        "oof_q90_abs_error": sorted(residuals)[math.ceil(0.90 * len(residuals)) - 1],
        "n": len(rows), "test_n": len(test_y),
    }


def render_report(rows, proposal, model, vectorizer, metrics):
    encoded = vectorizer.transform([proposal])
    estimate = float(model.predict(encoded)[0])
    interval = metrics["oof_q90_abs_error"]
    values = [_number(row[TARGET]) for row in rows]
    median = statistics.median(values)
    lower = max(0.0, estimate - interval)
    upper = estimate + interval
    lines = [
        "NASA93DEM sponsor-facing effort justification",
        "",
        f"Recommendation: plan for {estimate:.1f} person-months of effort.",
        f"Indicative 90% prediction band: {lower:.1f} to {upper:.1f} person-months.",
        "This band reflects historical out-of-fold absolute errors; it is not a guarantee.",
        "",
        "Historical evidence:",
        f"- {len(rows)} NASA93DEM projects; observed effort median {median:.1f}, range {min(values):.1f}–{max(values):.1f} person-months.",
        f"- Representative proposal uses numeric medians and categorical modes from the dataset (KLOC median {proposal['Kloc+']:.1f}).",
        "",
        "Model evidence:",
        f"- Random forest holdout: MAE {metrics['mae']:.1f}, RMSE {metrics['rmse']:.1f}, R² {metrics['r2']:.2f} (n={metrics['test_n']}).",
        f"- Five-fold out-of-fold MAE: {metrics['oof_mae']:.1f}; 90th-percentile absolute error: {interval:.1f}.",
        "",
        "Risks and assumptions:",
        "- The proposal is synthetic and representative, not a fully reviewed project plan.",
        "- Estimates are sensitive to size, capability, complexity, reliability, reuse, tools, and schedule attributes; missing or optimistic inputs can bias the result.",
        "- NASA93DEM projects may not match the sponsor's team, technology, constraints, or current prices; contingency and staged re-estimation are prudent.",
        "",
        "Alternatives and limitations:",
        "- A median baseline is transparent but ignores project-specific attributes; the model adds pattern detection but is less explainable.",
        "- This is observational historical association, not causal evidence: the data cannot establish that changing one attribute will cause effort to change.",
        "- Validate with an independently elicited size/work-breakdown estimate and update after requirements and architecture milestones.",
    ]
    return "\n".join(lines) + "\n", estimate, (lower, upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--proposal", help="JSON object overriding representative proposal fields")
    args = parser.parse_args()
    started = time.perf_counter()
    rows = load_validate(args.dataset)
    proposal = representative_proposal(rows)
    if args.proposal:
        proposal.update(json.loads(args.proposal))
    model, vectorizer, metrics = model_and_metrics(rows)
    report, estimate, interval = render_report(rows, proposal, model, vectorizer, metrics)
    print(report, end="")
    print("Runtime summary:")
    print(json.dumps({"rows": len(rows), "estimate_person_months": round(estimate, 3),
                      "prediction_interval": [round(x, 3) for x in interval],
                      "runtime_seconds": round(time.perf_counter() - started, 4),
                      "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
