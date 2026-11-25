"""Treatment 3 race skill for Health-ClosedIssues0000.csv.

The CSV is treated as an ordered historical HPO log.  A selector learns the
three quality metrics from configuration values, then chooses the candidate
with the highest predicted composite utility.  The baseline chooses the
highest-utility configuration observed before each evaluation window.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

METRICS = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


def load_dataset(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    expected = set(METRICS + CONFIG)
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"expected columns {sorted(expected)}")
    out = []
    for i, row in enumerate(rows):
        try:
            item = {k: row[k] for k in CONFIG}
            item.update({k: float(row[k]) for k in METRICS})
            item["_index"] = i
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid row {i}: {exc}") from exc
        if not all(math.isfinite(item[k]) for k in METRICS):
            raise ValueError(f"non-finite metric in row {i}")
        out.append(item)
    return out


def utility(rows: list[dict] | np.ndarray) -> np.ndarray:
    """Equal-weight utility: normalized ACC and PRED, minus normalized MRE."""
    if isinstance(rows, list):
        a = np.array([[r[k] for k in METRICS] for r in rows], dtype=float)
    else:
        a = np.asarray(rows, dtype=float)
    lo, hi = a.min(axis=0), a.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    z = (a - lo) / span
    return (-z[:, 0] + z[:, 1] + z[:, 2]) / 3.0


def _pipeline() -> object:
    # One-hot encoding is explicit so categorical criterion is not ordered.
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [1])], remainder="passthrough")
    return make_pipeline(prep, ExtraTreesRegressor(n_estimators=160, random_state=3, min_samples_leaf=3, n_jobs=1))


@dataclass
class Result:
    method: str
    n: int
    mean_utility: float
    mean_mre: float
    mean_acc: float
    mean_pred40: float
    top_half_rate: float


def _features(rows: list[dict]) -> list[list[object]]:
    return [[r["N_estimators"], r["criterion"], r["Min_sample_leaves"], r["Min_impurity_decrease"], r["Max_depth"]] for r in rows]


def evaluate(rows: list[dict], train_end: int = 6000, eval_end: int = 10000) -> tuple[list[Result], float]:
    if not (0 < train_end < eval_end <= len(rows)):
        raise ValueError("need 0 < train_end < eval_end <= row count")
    train, test = rows[:train_end], rows[train_end:eval_end]
    # Utility scale is fixed from the historical training period (no test leakage).
    train_metrics = np.array([[r[k] for k in METRICS] for r in train])
    lo, hi = train_metrics.min(axis=0), train_metrics.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    def score(a):
        z = (np.asarray(a) - lo) / span
        return (-z[:, 0] + z[:, 1] + z[:, 2]) / 3
    y = train_metrics
    model = _pipeline()
    model.fit(_features(train), y)
    pred = model.predict(_features(test))
    # Each observed row is a candidate; model ranks each candidate independently.
    pred_score = score(pred)
    # Since every row is evaluated once, ranking is measured as top-k selection
    # quality over prefixes: compare the selector's top half against baseline's.
    actual = np.array([[r[k] for k in METRICS] for r in test])
    selected_idx = int(np.argmax(pred_score))
    hist_best = max(train, key=lambda r: float(score([[r[k] for k in METRICS]])[0]))
    picked = [test[selected_idx]]
    baseline = [hist_best]
    def result(name, rs):
        vals = np.array([[r[k] for k in METRICS] for r in rs])
        return Result(name, len(rs), float(np.mean(score(vals))), *map(float, vals.mean(axis=0)), float(np.mean(score(vals) >= np.median(score(actual)))))
    return [result("treatment3_selector", picked), result("historical_baseline", baseline)], time.time()


def run(path: str | Path) -> tuple[list[Result], float]:
    start = time.perf_counter()
    results, _ = evaluate(load_dataset(path))
    return results, time.perf_counter() - start


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    args = p.parse_args()
    for r in run(args.dataset)[0]:
        print(f"{r.method}: n={r.n} utility={r.mean_utility:.6f} MRE={r.mean_mre:.6f} ACC={r.mean_acc:.6f} PRED40={r.mean_pred40:.6f} top_half={r.top_half_rate:.3f}")
