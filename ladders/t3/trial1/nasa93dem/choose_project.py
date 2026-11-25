#!/usr/bin/env python3
"""Choose a high-value NASA93 project configuration under a data-derived budget."""
import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


TARGETS = ("Kloc+", "Effort-", "Defects-", "MonthsX")
FEATURES = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
            "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
            "apex", "plex", "ltex", "tool", "site", "sced")


def load_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    for row in rows:
        for key in TARGETS:
            row[key] = float(row[key])
        for key in FEATURES:
            row[key] = row[key].strip()
    return rows


def make_model(seed=7):
    return Pipeline([("prep", DictVectorizer(sparse=False)), ("model", RandomForestRegressor(
        n_estimators=180, min_samples_leaf=2, random_state=seed, n_jobs=1))])


def choose(path, seed=7):
    rows = load_rows(path)
    x = [{k: row[k] for k in FEATURES} for row in rows]
    budget = sorted(row["Effort-"] for row in rows)[len(rows) // 2]
    models = {}
    for target in TARGETS:
        model = make_model(seed)
        model.fit(x, [row[target] for row in rows])
        models[target] = model

    predictions = []
    for index, row in enumerate(rows):
        setting = x[index]
        pred = {target: float(models[target].predict([setting])[0]) for target in TARGETS}
        # Value rewards delivered size and subtracts predicted quality/time risk.
        value = pred["Kloc+"] / (1.0 + pred["Defects-"] / 1000.0)
        value /= (1.0 + max(pred["MonthsX"], 0.0) / 24.0)
        if pred["Effort-"] <= budget:
            predictions.append((value, setting, pred, row["idX"]))
    if not predictions:
        raise ValueError("no feasible settings under the budget")
    value, setting, pred, source_id = max(predictions, key=lambda item: item[0])
    return {"budget_effort": budget, "source_project": source_id,
            "settings": setting, "predicted": pred, "delivered_value": value,
            "feasible_count": len(predictions), "project_count": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    result = choose(args.dataset, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
