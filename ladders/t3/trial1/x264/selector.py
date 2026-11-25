"""Select the fastest x264 configuration that satisfies a quality constraint."""
import argparse
import csv
import json
import time
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUALITY_SENSITIVE = ("no_8x8dct", "no_cabac", "no_deblock", "no_fast_pskip", "no_mbtree", "no_mixed_refs", "no_weightb")

def read_dataset(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    target = "Performance-"
    features = [k for k in rows[0] if k != target]
    for row in rows:
        for key in features:
            row[key] = int(row[key].strip())
        row[target] = float(row[target].strip())
    return rows, features, target

def choose(path):
    rows, features, target = read_dataset(path)
    feasible = [r for r in rows if all(r[k] == 0 for k in QUALITY_SENSITIVE)]
    if not feasible:
        raise ValueError("quality constraint leaves no feasible configuration")
    x = [[r[k] for k in features] for r in rows]
    y = [r[target] for r in rows]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    predictions = model.predict([[r[k] for k in features] for r in feasible])
    predicted_best = feasible[min(range(len(feasible)), key=lambda i: predictions[i])]
    measured_best = min(feasible, key=lambda r: r[target])
    return {"selected_configuration": {k: measured_best[k] for k in features}, "measured_performance": measured_best[target], "feasible_count": len(feasible), "total_count": len(rows), "quality_constraint": "all quality-sensitive no_* switches equal 0", "model_check": {"test_r2": float(model.score(x_test, y_test)), "predicted_best_measured_performance": predicted_best[target]}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    started = time.perf_counter()
    result = choose(args.dataset)
    result["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
