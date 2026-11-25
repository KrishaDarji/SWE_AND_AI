#!/usr/bin/env python3
"""Reproducible anomaly screening for the x264 benchmark measurements."""
import csv
import json
import math
import os
import sys
from pathlib import Path

from sklearn.ensemble import IsolationForest

DATASET = Path(__file__).resolve().parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
SEED = 303


def load_dataset(path=DATASET):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError("dataset must contain a header and at least one record")
    header = rows[0]
    if not header or header[-1] != "Performance-":
        raise ValueError("expected Performance- as the final column")
    if len(set(header)) != len(header):
        raise ValueError("column names must be unique")
    records = []
    for line_no, row in enumerate(rows[1:], 2):
        if len(row) != len(header):
            raise ValueError(f"line {line_no}: expected {len(header)} fields, got {len(row)}")
        try:
            features = [int(value.strip()) for value in row[:-1]]
            performance = float(row[-1].strip())
        except ValueError as exc:
            raise ValueError(f"line {line_no}: non-numeric value") from exc
        if any(value not in (0, 1) for value in features):
            raise ValueError(f"line {line_no}: feature is not binary")
        if not math.isfinite(performance):
            raise ValueError(f"line {line_no}: performance is not finite")
        records.append({"line": line_no, "features": features, "performance": performance})
    return header, records


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def analyze(path=DATASET):
    header, records = load_dataset(path)
    x = [record["features"] + [record["performance"]] for record in records]
    y = [record["performance"] for record in records]
    center = median(y)
    mad = median([abs(value - center) for value in y])
    robust_scale = max(1.4826 * mad, 1e-9)
    for record in records:
        record["robust_z"] = abs(record["performance"] - center) / robust_scale
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED)
    model.fit(x)
    scores = -model.decision_function(x)
    for record, score, prediction in zip(records, scores, model.predict(x)):
        record["isolation_score"] = float(score)
        record["isolation_flag"] = int(prediction == -1)
    flagged = sorted(records, key=lambda r: (r["robust_z"] + r["isolation_score"], r["line"]), reverse=True)
    most = flagged[0]
    # A possible measurement error is an extreme outcome that is also isolated;
    # a legitimate exceptional result is isolated mainly because of its settings.
    possible_error = most["robust_z"] >= 3.5 and most["isolation_flag"] == 1
    result = {
        "dataset": str(path), "rows": len(records), "features": len(header) - 1,
        "seed": SEED, "median_performance": center, "mad": mad,
        "most_anomalous": {k: most[k] for k in ("line", "performance", "robust_z", "isolation_score", "isolation_flag")},
        "configuration": dict(zip(header[:-1], most["features"])),
        "suspicious_records": [{"line": r["line"], "performance": r["performance"], "robust_z": r["robust_z"], "isolation_score": r["isolation_score"]} for r in flagged[:10]],
        "classification": "possible_measurement_error" if possible_error else "legitimate_exception_or_needs_review",
        "explanation": "The record is unusual in outcome magnitude and joint configuration/outcome space; rerun it to distinguish a measurement error from a real setting effect." if possible_error else "The record is unusual but does not meet both conservative error indicators; treat it as an exceptional configuration pending replication.",
    }
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET
    print(json.dumps(analyze(path), indent=2, sort_keys=True))
