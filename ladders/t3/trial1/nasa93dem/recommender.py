#!/usr/bin/env python3
"""Recommend the cheapest one-step process change that lowers estimated risk."""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Defects-"
ID_COLUMNS = {"idX", "centerX", "YearX"}
OUTCOMES = {"Kloc+", "Effort-", "Defects-", "MonthsX"}
PROCESS_COLUMNS = [
    "prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse",
    "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex",
    "ltex", "tool", "site", "sced",
]

# COCOMO-style ratings are ordinal, but their direction differs by attribute.
# Each tuple lists values from least to most favorable for reducing defects.
RATING_ORDER = {
    c: ["vl", "l", "n", "h", "vh", "eh"] for c in PROCESS_COLUMNS
}
RATING_ORDER.update({"ruse": ["n", "l", "h", "vh", "eh"], "docu": ["n", "l", "h", "vh", "eh"],
                     "tool": ["n", "l", "h", "vh"], "site": ["n", "l", "h", "vh"],
                     "sced": ["vl", "l", "n", "h", "vh"]})


def read_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{k: (v or "").strip().lower() for k, v in row.items()} for row in reader]
        return reader.fieldnames or [], rows


def build_model(rows: List[Dict[str, str]], features: List[str]) -> Pipeline:
    x = [[row[f] for f in features] for row in rows]
    y = np.asarray([float(row[TARGET]) for row in rows])
    prep = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(len(features))))])
    model = Pipeline([("prep", prep), ("regressor", RandomForestRegressor(
        n_estimators=300, random_state=17, min_samples_leaf=2))])
    model.fit(x, y)
    return model


def recommend(path: str) -> Dict[str, object]:
    fields, rows = read_rows(path)
    features = [c for c in PROCESS_COLUMNS if c in fields]
    if not rows or TARGET not in fields or not features:
        raise ValueError("dataset must contain rows, Defects-, and process drivers")
    model = build_model(rows, features)
    baseline = float(np.mean(model.predict([[row[f] for f in features] for row in rows])))
    candidates = []
    for feature in features:
        values = sorted({row[feature] for row in rows}, key=lambda v: RATING_ORDER[feature].index(v) if v in RATING_ORDER[feature] else 999)
        for old in values:
            order = RATING_ORDER[feature]
            if old not in order:
                continue
            pos = order.index(old)
            if pos == len(order) - 1:
                continue
            new = order[pos + 1]
            changed = [[row[f] if f != feature else new for f in features] for row in rows]
            estimate = float(np.mean(model.predict(changed)))
            candidates.append({"feature": feature, "from": old, "to": new,
                               "cost": 1, "estimated_risk": estimate,
                               "risk_reduction": baseline - estimate})
    improving = [c for c in candidates if c["risk_reduction"] > 0]
    if improving:
        best = min(improving, key=lambda c: (c["cost"], -c["risk_reduction"], c["feature"], c["from"]))
        status = "change_recommended"
    else:
        best = {"feature": None, "from": None, "to": None, "cost": 0,
                "estimated_risk": baseline, "risk_reduction": 0.0}
        status = "no_improving_change_found"
    return {"status": status, "rows": len(rows), "baseline_estimated_risk": baseline,
            "recommendation": best, "candidate_count": len(candidates)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    try:
        print(json.dumps(recommend(args.dataset), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
