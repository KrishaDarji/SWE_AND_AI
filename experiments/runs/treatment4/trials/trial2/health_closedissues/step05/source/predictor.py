"""Predict MRE- and ACC+ from configuration settings with grouped evaluation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from summarizer import CONFIGURATION_FIELDS

TARGETS = ["MRE-", "ACC+"]
NUMERIC_CONFIG = ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
CATEGORICAL_CONFIG = ["criterion"]


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float:
    return float(value) if value != "" else np.nan


def _features(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [[_number(row.get(f, "")) for f in NUMERIC_CONFIG] + [row.get("criterion", "")] for row in rows]


def _groups(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
    return ["|".join(row.get(f, "") for f in CONFIGURATION_FIELDS) for row in rows]


def _pipeline(seed: int) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), [0, 1, 2, 3]),
        ("criterion", Pipeline([("missing", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]), [4]),
    ])
    return Pipeline([("preprocess", prep), ("estimator", RandomForestRegressor(
        n_estimators=100, random_state=seed, n_jobs=1))])


def evaluate(path: str | Path, random_state: int = 42) -> dict[str, Any]:
    """Return grouped held-out metrics, baselines, and fitted split models."""
    rows = load_rows(path)
    usable = [r for r in rows if all(r.get(t, "") != "" for t in TARGETS)]
    if not usable:
        raise ValueError("no rows have both target outcomes")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_i, test_i = next(splitter.split(np.arange(len(usable)), groups=_groups(usable)))
    train, test = [usable[i] for i in train_i], [usable[i] for i in test_i]
    train_groups, test_groups = set(_groups(train)), set(_groups(test))
    if train_groups & test_groups:
        raise AssertionError("configuration group leakage")
    models, metrics, baselines, test_predictions = {}, {}, {}, {}
    for target in TARGETS:
        model = _pipeline(random_state)
        y_train = np.array([float(r[target]) for r in train])
        y_test = np.array([float(r[target]) for r in test])
        model.fit(_features(train), y_train)
        predictions = model.predict(_features(test))
        test_predictions[target] = predictions.tolist()
        metrics[target] = {"mae": float(mean_absolute_error(y_test, predictions)),
                           "baseline_mae": float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))}
        baselines[target] = float(y_train.mean())
        models[target] = model
    return {"predictor_names": CONFIGURATION_FIELDS, "train_count": len(train), "test_count": len(test),
            "train_groups": len(train_groups), "test_groups": len(test_groups), "metrics": metrics,
            "training_mean_baselines": baselines, "test_predictions": test_predictions, "estimator_settings": {"type": "RandomForestRegressor", "n_estimators": 100, "random_state": random_state, "n_jobs": 1},
            "models": models, "train_rows": train, "test_rows": test}


def fit_final(path: str | Path, random_state: int = 42) -> dict[str, Pipeline]:
    """Fit separate final models on all rows having each required target."""
    rows = load_rows(path)
    result = {}
    for target in TARGETS:
        usable = [r for r in rows if r.get(target, "") != ""]
        model = _pipeline(random_state)
        model.fit(_features(usable), [float(r[target]) for r in usable])
        result[target] = model
    return result


def predict_configuration(models: dict[str, Pipeline], configuration: dict[str, Any]) -> dict[str, float]:
    """Predict both targets for one supplied configuration, including missing settings."""
    row = {f: configuration.get(f, "") for f in CONFIGURATION_FIELDS}
    return {target: float(model.predict(_features([row]))[0]) for target, model in models.items()}
