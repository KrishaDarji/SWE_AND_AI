"""Predict MRE- and ACC+ from model configuration settings."""
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIGURATION_FIELDS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
TARGETS = ["MRE-", "ACC+"]
NUMERIC_FIELDS = ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
MISSING = "__MISSING__"

def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan

def _criterion(value):
    value = "" if value is None else str(value).strip()
    return value if value else MISSING

def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _frame(rows):
    return np.array([[*[_num(r.get(f)) for f in NUMERIC_FIELDS], _criterion(r.get("criterion"))] for r in rows], dtype=object)

def _pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=MISSING)),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([("preprocess", ColumnTransformer([("numeric", numeric, [0, 1, 2, 3]),
                                                        ("criterion", categorical, [4])])),
                     ("estimator", Ridge(alpha=1.0))])

def _target(rows, name):
    return np.array([_num(r.get(name)) for r in rows], dtype=float)

def _groups(X):
    return np.array(["|".join(MISSING if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) for row in X])

def evaluate(path, test_size=0.2, random_state=42):
    rows = _rows(path)
    if len(rows) < 2:
        raise ValueError("insufficient data: at least two rows are required")
    X = _frame(rows); groups = _groups(X)
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(split.split(X, groups=groups))
    result = {"predictor_names": CONFIGURATION_FIELDS, "train_count": int(len(train)), "test_count": int(len(test)),
              "estimator_settings": {"model": "Ridge", "alpha": 1.0, "split": "GroupShuffleSplit", "test_size": test_size, "random_state": random_state}, "targets": {}}
    for name in TARGETS:
        y = _target(rows, name)
        valid = np.isfinite(y[train])
        if not valid.any():
            raise ValueError("insufficient valid training targets for " + name)
        model = _pipeline().fit(X[train][valid], y[train][valid])
        test_valid = np.isfinite(y[test])
        predictions = model.predict(X[test][test_valid]) if test_valid.any() else np.array([])
        baseline = float(np.mean(y[train][valid]))
        result["targets"][name] = {"held_out_mae": float(mean_absolute_error(y[test][test_valid], predictions)) if test_valid.any() else None,
                                    "training_mean_baseline_mae": float(mean_absolute_error(y[test][test_valid], np.full(sum(test_valid), baseline))) if test_valid.any() else None,
                                    "training_mean": baseline, "valid_test_count": int(sum(test_valid))}
    return result

def fit_and_predict(path, configuration):
    rows = _rows(path); X = _frame(rows); proposal = _frame([configuration])
    output = {}
    for name in TARGETS:
        y = _target(rows, name); valid = np.isfinite(y)
        if not valid.any(): raise ValueError("insufficient valid targets for " + name)
        output[name] = float(_pipeline().fit(X[valid], y[valid]).predict(proposal)[0])
    return output
