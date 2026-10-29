import argparse
import json
import time
from flow_evaluator import evaluate_csv

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
started = time.perf_counter()
state = evaluate_csv(args.dataset)
elapsed = time.perf_counter() - started
best = None if state.best is None else {k: v for k, v in state.best.items() if k != "_key"}
print(json.dumps({"observations": state.observations, "valid": state.valid,
                  "invalid": state.invalid, "best": best,
                  "running_mean": state.running,
                  "runtime_seconds": elapsed}, sort_keys=True))
