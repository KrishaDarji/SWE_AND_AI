"""Rank configuration settings by marginal association with lower MRE-.

The primary score is a proportion of finite outcome variation explained: r^2
for numeric settings and eta^2 for criterion.  These are comparable descriptive
association measures, not causal effects or predictive guarantees.
"""
import csv
import math
from pathlib import Path

import numpy as np

SETTINGS = ["N_estimators", "criterion", "Min_sample_leaves",
            "Min_impurity_decrease", "Max_depth"]
NUMERIC_SETTINGS = set(SETTINGS) - {"criterion"}
EXCLUDED = {"MRE-", "ACC+", "PRED40+"}


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric(rows, setting):
    pairs = [(_number(row.get(setting)), _number(row.get("MRE-"))) for row in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return _base(setting, "numeric", len(pairs), None, None, None)
    x, y = np.asarray(pairs, dtype=float).T
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        score = 0.0
        slope = None
    else:
        r = float(np.corrcoef(x, y)[0, 1])
        score, slope = r * r, float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    return _base(setting, "numeric", len(pairs), score, slope, {"min": float(x.min()), "max": float(x.max())})


def _base(setting, kind, count, score, direction, observed):
    return {"setting": setting, "type": kind, "valid_count": count,
            "score": score, "direction": direction, "observed": observed}


def _categorical(rows, setting):
    groups = {}
    for row in rows:
        y = _number(row.get("MRE-"))
        if y is not None:
            key = str(row.get(setting) or "__MISSING__").strip() or "__MISSING__"
            groups.setdefault(key, []).append(y)
    values = [v for group in groups.values() for v in group]
    if len(values) < 2 or len(groups) < 2:
        result = _base(setting, "categorical", len(values), 0.0 if values else None, None, None)
    else:
        overall = float(np.mean(values))
        between = sum(len(v) * (float(np.mean(v)) - overall) ** 2 for v in groups.values())
        total = sum((v - overall) ** 2 for v in values)
        score = float(between / total) if total else 0.0
        means = {k: float(np.mean(v)) for k, v in sorted(groups.items())}
        result = _base(setting, "categorical", len(values), score, None, means)
    result["groups"] = {k: {"count": len(v), "mean_mre": float(np.mean(v))}
                         for k, v in sorted(groups.items())}
    return result


def rank(path):
    """Return deterministic descending association ranking and outcome summary."""
    rows = _rows(path)
    outcome = [_number(row.get("MRE-")) for row in rows]
    finite = [v for v in outcome if v is not None]
    results = [_categorical(rows, s) if s == "criterion" else _numeric(rows, s) for s in SETTINGS]
    results.sort(key=lambda r: (-(r["score"] if r["score"] is not None else -1), r["setting"]))
    for i, item in enumerate(results, 1):
        item["rank"] = i
    return {"outcome": {"name": "MRE-", "direction": "minimize", "valid_count": len(finite),
                         "mean": float(np.mean(finite)) if finite else None,
                         "min": float(np.min(finite)) if finite else None,
                         "max": float(np.max(finite)) if finite else None},
            "method": "marginal variance explained; r^2 for numeric and eta^2 for categorical",
            "excluded_outcomes": sorted(EXCLUDED), "settings": results}


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    print(json.dumps(rank(parser.parse_args().dataset), indent=2, sort_keys=True))
