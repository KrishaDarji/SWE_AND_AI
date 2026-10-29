#!/usr/bin/env python3
"""Independent, reproducible spotter for NASA93DEM project records."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


REQUIRED = ["idX", "Kloc+", "Effort-", "Defects-", "MonthsX"]
TARGETS = ["Kloc+", "Effort-", "Defects-", "MonthsX"]
DROP = {"idX"}


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no data rows")
    missing = [name for name in REQUIRED if name not in rows[0]]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))
    for row_number, row in enumerate(rows, 2):
        for name in TARGETS:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                raise ValueError("non-numeric %s at CSV row %d" % (name, row_number))
            if not math.isfinite(value):
                raise ValueError("non-finite %s at CSV row %d" % (name, row_number))
    return rows


def build_matrix(rows):
    columns = [key for key in rows[0] if key not in DROP]
    numeric = set(TARGETS) | {"centerX", "YearX"}
    matrix = []
    encodings = {}
    for column in columns:
        if column not in numeric:
            values = sorted({row[column].strip() for row in rows})
            encodings[column] = {value: index for index, value in enumerate(values)}
    for row in rows:
        vector = []
        for column in columns:
            value = row[column].strip()
            vector.append(float(value) if column in numeric else float(encodings[column][value]))
        matrix.append(vector)
    return columns, matrix


def robust_reasons(rows, row_index, columns):
    reasons = []
    for column in columns:
        values = [float(row[column]) for row in rows if column in TARGETS or column in {"centerX", "YearX"}]
        if column not in TARGETS and column not in {"centerX", "YearX"}:
            counts = {}
            for row in rows:
                key = row[column].strip()
                counts[key] = counts.get(key, 0) + 1
            key = rows[row_index][column].strip()
            if counts[key] <= 2:
                reasons.append({"feature": column, "value": key, "kind": "rare_category", "score": 1.0 / counts[key]})
            continue
        current = float(rows[row_index][column])
        ordered = sorted(values)
        median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2
        deviations = sorted(abs(value - median) for value in values)
        mad = deviations[len(deviations)//2] or 1e-9
        score = abs(current - median) / (1.4826 * mad)
        reasons.append({"feature": column, "value": current, "kind": "robust_numeric_deviation", "score": round(score, 3), "median": round(median, 3)})
    return sorted(reasons, key=lambda item: item["score"], reverse=True)


def detect(path, contamination=0.1):
    rows = load_dataset(path)
    columns, matrix = build_matrix(rows)
    scaled = StandardScaler().fit_transform(matrix)
    model = IsolationForest(n_estimators=400, contamination=contamination, random_state=20261029, n_jobs=1)
    model.fit(scaled)
    scores = -model.score_samples(scaled)
    order = sorted(range(len(rows)), key=lambda index: scores[index], reverse=True)
    records = []
    for index in order[:max(1, math.ceil(len(rows) * contamination))]:
        records.append({"row": index + 2, "id": rows[index]["idX"].strip(), "anomaly_score": round(float(scores[index]), 6), "reasons": robust_reasons(rows, index, columns)[:6]})
    return {"rows": len(rows), "columns": len(columns) + 1, "contamination": contamination, "method": "IsolationForest(n_estimators=400, random_state=20261029) on StandardScaler ordinal mixed-type matrix", "suspicious": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = detect(args.dataset)
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
