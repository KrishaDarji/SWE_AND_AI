import csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from flow import evaluate_csv

evaluator = evaluate_csv(sys.argv[1])
print("snapshot_count", len(evaluator.snapshots))
print(json.dumps(evaluator.final(), sort_keys=True))
with open(sys.argv[2], "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["step", "record_id", "outcome", "count", "mean", "sample_variance", "best", "best_record_id", "best_configuration"])
    for snap in evaluator.snapshots:
        for outcome, data in snap["outcomes"].items():
            writer.writerow([4, snap["record_id"], outcome, data["count"], data["mean"], data["sample_variance"], data["best"], data["best_record_id"], json.dumps(data["best_configuration"], sort_keys=True)])
