#!/usr/bin/env python3
"""Independent NASA93DEM effort-estimate justification."""
import argparse, csv, json, math, os, sys
from statistics import median, quantiles
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Effort-"
REQUIRED = {"Kloc+", TARGET}

def load_data(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError("missing required columns or empty CSV")
    numeric = {"Kloc+", "YearX", "centerX"}
    for row in rows:
        for col in numeric | {TARGET}:
            try: row[col] = float(row[col])
            except (TypeError, ValueError): raise ValueError("non-numeric value in %s" % col)
    return rows

def mode(rows, col):
    vals = [r[col].strip() for r in rows if r.get(col, "").strip()]
    return max(set(vals), key=vals.count)

def representative(rows, columns):
    proposal = {}
    for col in columns:
        vals = [r[col] for r in rows]
        if col == "Kloc+": proposal[col] = median(vals)
        elif isinstance(vals[0], float): proposal[col] = median(vals)
        else: proposal[col] = mode(rows, col)
    return proposal

def main(data_path, proposal_path=None):
    rows = load_data(data_path)
    feature_cols = [c for c in rows[0] if c != TARGET and c != "idX"]
    cat = [c for c in feature_cols if not isinstance(rows[0][c], float)]
    num = [c for c in feature_cols if c not in cat]
    X = [[r[c] for c in feature_cols] for r in rows]
    y = [r[TARGET] for r in rows]
    num_idx = [feature_cols.index(c) for c in num]; cat_idx = [feature_cols.index(c) for c in cat]
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_idx),
                              ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_idx)])
    model = Pipeline([("prep", prep), ("reg", RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2))])
    tr, te = train_test_split(range(len(y)), test_size=.2, random_state=42)
    model.fit([X[i] for i in tr], [y[i] for i in tr])
    pred = model.predict([X[i] for i in te])
    mae, rmse = mean_absolute_error([y[i] for i in te], pred), math.sqrt(mean_squared_error([y[i] for i in te], pred))
    r2 = r2_score([y[i] for i in te], pred)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv)
    abs_err = sorted(abs(a-b) for a,b in zip(y, oof))
    q = abs_err[min(len(abs_err)-1, math.ceil(.9*len(abs_err))-1)]
    proposal = json.load(open(proposal_path)) if proposal_path else representative(rows, feature_cols)
    proposal_vector = [proposal[c] for c in feature_cols] if isinstance(proposal, dict) else proposal
    estimate = float(model.predict([proposal_vector])[0])
    lo, hi = max(0.0, estimate-q), estimate+q
    return ("NASA93DEM effort justification\n"
      "Sponsor recommendation: plan for %.1f person-months; a data-based 80%% range is %.1f–%.1f.\n"
      "Proposal: %s\n\nHistorical evidence: %d completed projects; effort median %.1f, range %.1f–%.1f person-months. "
      "The proposal uses median numeric inputs and modal categorical inputs unless supplied.\n"
      "Model evidence: random forest on a fixed 80/20 holdout (seed 42) had MAE %.1f, RMSE %.1f, R² %.2f. "
      "The interval is an empirical 90th-percentile out-of-fold absolute-error band, not a guarantee.\n"
      "Risks/assumptions: the estimate assumes the proposal is comparable to this historical sample, inputs are accurate, and scope and productivity conditions remain stable. "
      "Small sample size, project heterogeneity, categorical encoding, and possible outliers can widen error. Validate scope, staffing, dependencies, and contingency with the team.\n"
      "Causality and alternatives: associations in this observational dataset do not show that any factor causes effort. Consider an interpretable linear/regularized baseline, expert three-point estimate, and scenario analysis; update with actuals."
      % (estimate, lo, hi, proposal, len(rows), median(y), min(y), max(y), mae, rmse, r2)), {
        "rows": len(rows), "total_loc": sum(r["Kloc+"] for r in rows), "new_loc": sum(r["Kloc+"] for r in rows), "mae": mae, "rmse": rmse, "r2": r2, "estimate": estimate
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--proposal")
    args = ap.parse_args(); report, _ = main(args.data, args.proposal); print(report)
