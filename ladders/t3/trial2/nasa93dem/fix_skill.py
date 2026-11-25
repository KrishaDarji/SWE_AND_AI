#!/usr/bin/env python3
"""Treatment 3 / Trial 2 / Step 08: cheapest one-step process change."""
import argparse, csv, json, math, time
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

TARGET = "Defects-"
OUTCOMES = {"Kloc+", "Effort-", "Defects-", "MonthsX"}
ID_COLS = {"idX"}
LEVEL_ORDER = ["vl", "l", "n", "h", "vh", "xh"]

def load_data(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows: raise ValueError("dataset is empty")
    required = {"idX", TARGET, "team", "pmat"}
    missing = required - set(rows[0])
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    for row in rows:
        for key, value in row.items():
            if not value or not value.strip(): raise ValueError(f"blank value in {key}")
        for key in row:
            if key in ID_COLS or key in OUTCOMES: float(row[key])
    return rows

def mode(values):
    return Counter(values).most_common(1)[0][0]

def build_model(rows):
    features = [k for k in rows[0] if k not in ID_COLS | OUTCOMES]
    categorical = [k for k in features if not _is_number(rows[0][k])]
    numeric = [k for k in features if k not in categorical]
    prep = DictVectorizer(sparse=True)
    model = Pipeline([("prep", prep), ("rf", RandomForestRegressor(n_estimators=300, random_state=3, min_samples_leaf=2, n_jobs=1))])
    model.fit([{k: r[k].strip() if k in categorical else float(r[k]) for k in features} for r in rows], [float(r[TARGET]) for r in rows])
    return model, features, categorical, numeric

def _is_number(value):
    try: float(value); return True
    except ValueError: return False

def run(path):
    rows = load_data(path)
    model, features, categorical, numeric = build_model(rows)
    baseline = {k: (mode([r[k].strip() for r in rows]) if k in categorical else sum(float(r[k]) for r in rows) / len(rows)) for k in features}
    baseline_risk = float(model.predict([baseline])[0])
    candidates = []
    for col in categorical:
        current = baseline[col]
        if current not in LEVEL_ORDER: continue
        for level in LEVEL_ORDER:
            if abs(LEVEL_ORDER.index(level) - LEVEL_ORDER.index(current)) != 1: continue
            candidate = dict(baseline); candidate[col] = level
            risk = float(model.predict([candidate])[0])
            candidates.append({"attribute": col, "from": current, "to": level, "cost": 1, "risk": risk, "reduction": baseline_risk - risk})
    beneficial = [c for c in candidates if c["reduction"] > 1e-9]
    chosen = min(beneficial, key=lambda c: (c["cost"], -c["reduction"], c["attribute"])) if beneficial else None
    kf = KFold(n_splits=5, shuffle=True, random_state=3)
    X = [{k: r[k].strip() if k in categorical else float(r[k]) for k in features} for r in rows]
    y = [float(r[TARGET]) for r in rows]
    cv_mae = float((-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")).mean())
    return {"rows": len(rows), "baseline": baseline, "baseline_risk": baseline_risk, "candidate_count": len(candidates), "chosen": chosen, "cv_mae": cv_mae}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); args = ap.parse_args()
    started = time.perf_counter(); result = run(args.dataset); elapsed = time.perf_counter() - started
    print(json.dumps({"dataset": args.dataset, "rows": result["rows"], "baseline_estimated_defects": round(result["baseline_risk"], 3), "candidates_evaluated": result["candidate_count"], "recommended_change": result["chosen"], "cross_validated_mae": round(result["cv_mae"], 3), "runtime_seconds": round(elapsed, 4)}, sort_keys=True))

if __name__ == "__main__": main()
