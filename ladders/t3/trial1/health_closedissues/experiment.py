"""Independent spot check for anomalous model-quality records."""
import argparse
import csv
import math
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

METRICS = ["MRE-", "ACC+", "PRED40+"]
NUMERIC = METRICS + ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
CATEGORICAL = ["criterion"]
REQUIRED = NUMERIC + CATEGORICAL


def load_and_validate(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    missing = set(REQUIRED) - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    bad = []
    for i, row in enumerate(rows):
        for col in NUMERIC:
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                bad.append((i, col, "not numeric")); continue
            if not math.isfinite(value):
                bad.append((i, col, "non-finite"))
        for col in METRICS:
            try: value = float(row[col])
            except (TypeError, ValueError): continue
            if not 0 <= value <= 1: bad.append((i, col, "outside [0,1]"))
    return rows, bad


def analyze(path, contamination=0.01, random_state=42):
    rows, data_errors = load_and_validate(path)
    numeric = [[float(row[col]) for col in NUMERIC] for row in rows]
    categories = [[row[col] for col in CATEGORICAL] for row in rows]
    transformer = Pipeline([("scale", StandardScaler()),])
    numeric_scaled = transformer.fit_transform(numeric)
    category_encoded = OneHotEncoder(handle_unknown="ignore").fit_transform(categories).toarray()
    features = __import__("numpy").hstack((numeric_scaled, category_encoded))
    detector = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=1)
    detector.fit(features)
    scores = -detector.score_samples(features)
    index = max(range(len(rows)), key=lambda i: scores[i])
    row = rows[index]
    metric_reason = (float(row["MRE-"]) >= 0.95 or float(row["ACC+"]) <= 0.05 or
                     float(row["PRED40+"]) <= 0.05)
    reasons = ["isolated joint metric/configuration profile"]
    if metric_reason: reasons.append("extreme model-quality metric")
    status = "possible data-quality issue" if index in {e[0] for e in data_errors} else "possible legitimate exceptional performance"
    return {"n_rows": len(rows), "data_errors": data_errors, "index": index,
            "row": row, "anomaly_score": float(scores[index]), "status": status,
            "reasons": reasons, "flagged_count": int((scores >= sorted(scores)[int((1-contamination)*len(scores))-1]).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = analyze(args.dataset)
    print(f"rows={result['n_rows']}")
    print(f"validation_errors={len(result['data_errors'])}")
    print(f"flagged_records={result['flagged_count']}")
    print(f"most_anomalous_row={result['index'] + 2}")
    print(f"anomaly_score={result['anomaly_score']:.6f}")
    print(f"status={result['status']}")
    print("record=" + repr(result["row"]))
    print("why=" + "; ".join(result["reasons"]))
    print("uncertainty=IsolationForest is unsupervised; contamination=0.01 and random_state=42 are assumptions.")
    print("limitations=Anomaly is not proof of error; no ground-truth outcomes or repeated trials are available.")


if __name__ == "__main__":
    main()
