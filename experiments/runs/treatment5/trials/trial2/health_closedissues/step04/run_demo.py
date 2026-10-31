import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from flow import evaluate_csv, write_metrics

root = os.path.dirname(__file__)
dataset = "/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv"
evaluator = evaluate_csv(dataset)
write_metrics(os.path.join(root, "metrics_detail.csv"), evaluator)
print("snapshot_count", len(evaluator.snapshots))
for name, state in evaluator.summary().items():
    print(name, state)
