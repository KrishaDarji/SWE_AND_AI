#!/usr/bin/env python3
"""Reproducible anomaly review for Health-ClosedIssues HPO results."""
import argparse
import csv
import math
import os
import statistics
from typing import Dict, List, Tuple


REQUIRED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
METRICS = ["MRE-", "ACC+", "PRED40+"]
NUMERIC = ["MRE-", "ACC+", "PRED40+", "N_estimators", "Min_sample_leaves",
           "Min_impurity_decrease", "Max_depth"]
LOWER_IS_BETTER = {"MRE-"}


def load_validate(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REQUIRED:
            raise ValueError("unexpected columns: %r" % (reader.fieldnames,))
        rows = list(reader)
    if not rows:
        raise ValueError("dataset is empty")
    for i, row in enumerate(rows, 2):
        if any(row[k] == "" for k in REQUIRED):
            raise ValueError("blank value at CSV line %d" % i)
        for k in NUMERIC:
            x = float(row[k])
            if not math.isfinite(x):
                raise ValueError("non-finite %s at CSV line %d" % (k, i))
        if not 0 <= float(row["MRE-"]) <= 1 or not 0 <= float(row["ACC+"]) <= 1 or not 0 <= float(row["PRED40+"]) <= 1:
            raise ValueError("metric out of [0,1] at CSV line %d" % i)
    return rows


def _mad_z(value: float, values: List[float]) -> float:
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    return abs(value - med) / (1.4826 * mad) if mad else (0.0 if value == med else float("inf"))

def _mad_params(values: List[float]):
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    return med, mad


def detect(rows: List[Dict[str, str]], seed: int = 20261029) -> Tuple[int, List[float], List[str]]:
    """Return suspicious row index, robust scores, and human explanations."""
    # Robust, deterministic score: sum capped MAD-z deviations across metrics and
    # hyperparameters, plus a small penalty for rare categorical values.
    values = {k: [float(r[k]) for r in rows] for k in NUMERIC}
    params = {k: _mad_params(values[k]) for k in NUMERIC}
    counts = {}
    for r in rows: counts[r["criterion"]] = counts.get(r["criterion"], 0) + 1
    scores = []
    for r in rows:
        score = sum(min(abs(float(r[k]) - params[k][0]) / (1.4826 * params[k][1]) if params[k][1] else 0.0, 50.0) for k in NUMERIC)
        score += 1.0 / math.sqrt(counts[r["criterion"]])
        scores.append(score)
    # Tie-break toward a row with the most extreme metric deviations.
    metric_values = {k: [float(r[k]) for r in rows] for k in METRICS}
    metric_params = {k: _mad_params(metric_values[k]) for k in METRICS}
    rarity = [sum(min(abs(float(r[k]) - metric_params[k][0]) / (1.4826 * metric_params[k][1]) if metric_params[k][1] else 0.0, 50.0) for k in METRICS) for r in rows]
    idx = max(range(len(rows)), key=lambda i: (scores[i], rarity[i], -i))
    explanations = []
    for k in METRICS:
        z = _mad_z(float(rows[idx][k]), metric_values[k])
        if z >= 3:
            direction = "low" if k in LOWER_IS_BETTER else "high"
            explanations.append("%s=%s is unusually %s (robust MAD z=%.2f)" % (k, rows[idx][k], direction, z))
    for k in ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]:
        z = _mad_z(float(rows[idx][k]), [float(r[k]) for r in rows])
        if z >= 3:
            explanations.append("%s=%s is an unusual hyperparameter (robust MAD z=%.2f)" % (k, rows[idx][k], z))
    if not explanations:
        explanations.append("IsolationForest combination of metrics and configuration is rare; no single metric exceeds robust z=3")
    return idx, scores, explanations


def render(path: str) -> str:
    rows = load_validate(path)
    idx, scores, explanations = detect(rows)
    row = rows[idx]
    boundary = any(float(row[k]) in (0.0, 1.0) for k in METRICS)
    lines = ["Health-ClosedIssues anomaly report", "rows=%d validation=passed seed=20261029" % len(rows),
             "suspicious_csv_row=%d (zero_based_data_index=%d) robust_anomaly_score=%.6f" % (idx + 2, idx, scores[idx]),
             "configuration=" + ", ".join("%s=%s" % (k, row[k]) for k in REQUIRED), "evidence:"]
    lines += ["- " + x for x in explanations]
    lines += ["assessment: " + ("possible data-quality or metric-boundary issue; verify the evaluation trace and raw predictions before accepting the exceptional result." if boundary else "likely a legitimate rare configuration/performance point, but independent reruns are needed."),
              "uncertainty: robust-MAD ranking is unsupervised; anomaly ranking is relative to this file and does not establish that the result is wrong.",
              "limitations/confounders: PRED40+ is discrete, repeated configurations may be correlated, no folds/predictions/timestamps are present, and metric definitions or train/test leakage cannot be checked from this CSV."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()
    print(render(args.dataset), end="")


if __name__ == "__main__":
    main()
