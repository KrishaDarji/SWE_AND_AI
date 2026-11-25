"""Compare the inherited NASA93 estimator with a training-mean baseline."""
import argparse
import csv
import math
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

try:
    from .nasa93_effort import TARGET, _frame, _usable, build_pipeline, predictor_names
    from .nasa93_summary import read_csv
except ImportError:
    from nasa93_effort import TARGET, _frame, _usable, build_pipeline, predictor_names
    from nasa93_summary import read_csv

TIE_TOLERANCE = 1e-12

def _group(row: Dict[str, Any], names: List[str]) -> str:
    return "\x1f".join("__MISSING__" if row.get(name) is None else str(row.get(name)) for name in names)

def paired_difference(model_mae: float, baseline_mae: float) -> float:
    return float(model_mae - baseline_mae)

def classify_difference(difference: float, tolerance: float = TIE_TOLERANCE) -> str:
    if difference < -tolerance:
        return "win"
    if difference > tolerance:
        return "loss"
    return "tie"

def compare(path: str, seeds=range(42, 52), test_size: float = 0.25) -> Dict[str, Any]:
    data = read_csv(path)
    rows = _usable(data["rows"])
    names = predictor_names(data["columns"])
    if len(rows) < 2 or not names:
        raise ValueError("insufficient usable rows or descriptors")
    groups = [_group(row, names) for row in rows]
    results = []
    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(rows, groups=groups))
        train = [rows[i] for i in train_idx]
        test = [rows[i] for i in test_idx]
        model = build_pipeline(names)
        model.fit(_frame(train, names), [float(row[TARGET]) for row in train])
        predictions = [float(value) for value in model.predict(_frame(test, names))]
        observed = [float(row[TARGET]) for row in test]
        training_mean = float(np.mean([float(row[TARGET]) for row in train]))
        model_mae = float(mean_absolute_error(observed, predictions))
        baseline_predictions = [training_mean] * len(test)
        baseline_mae = float(mean_absolute_error(observed, baseline_predictions))
        results.append({"seed": seed, "train_count": len(train), "test_count": len(test),
                        "train_groups": sorted({groups[i] for i in train_idx}),
                        "test_groups": sorted({groups[i] for i in test_idx}),
                        "group_separation": not ({groups[i] for i in train_idx} & {groups[i] for i in test_idx}),
                        "train_record_identifiers": [row.get("idX") for row in train],
                        "test_record_identifiers": [row.get("idX") for row in test],
                        "predictions": predictions, "baseline_predictions": baseline_predictions,
                        "observed_efforts": observed, "model_mae": model_mae,
                        "baseline_mae": baseline_mae, "paired_difference": paired_difference(model_mae, baseline_mae),
                        "classification": classify_difference(paired_difference(model_mae, baseline_mae)),
                        "negative_model_prediction_count": sum(value < 0 for value in predictions),
                        "training_mean": training_mean})
    counts = {label: sum(item["classification"] == label for item in results) for label in ("win", "tie", "loss")}
    return {"predictor_names": names, "tie_tolerance": TIE_TOLERANCE, "results": results, "counts": counts,
            "paired_differences": [item["paired_difference"] for item in results],
            "disclosure": "Repeated grouped splits overlap and are not independent datasets or independent LLM trials.",
            "statistical_superiority": "not claimed; the project's final same procedure is pending if unavailable"}

def write_metrics(result: Dict[str, Any], path: str) -> None:
    fields = ["step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc", "test_pass", "runtime_seconds", "error_count", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result["metrics"])

def format_report(result: Dict[str, Any]) -> str:
    lines = ["NASA93 estimator race against training-mean baseline", "Predictors: " + ", ".join(result["predictor_names"]),
             f"Tie tolerance fixed before results: {result['tie_tolerance']}", ""]
    for item in result["results"]:
        lines.append(f"seed={item['seed']} train/test={item['train_count']}/{item['test_count']} groups_separated={item['group_separation']} test_ids={item['test_record_identifiers']} predictions={item['predictions']} observed={item['observed_efforts']} model_mae={item['model_mae']:.6g} baseline_mae={item['baseline_mae']:.6g} paired_difference={item['paired_difference']:.6g} classification={item['classification']} negative_predictions={item['negative_model_prediction_count']}")
    lines += [f"Descriptive wins/ties/losses: {result['counts']}", result["disclosure"], result["statistical_superiority"], "Negative predictions are reported, not clipped; effort units and deployment reliability are not inferred."]
    return "\n".join(lines) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(format_report(compare(args.csv_path)), end="")

if __name__ == "__main__":
    main()
