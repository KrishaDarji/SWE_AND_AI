#!/usr/bin/env python3
"""Incrementally evaluate model-configuration results from a CSV stream."""
import argparse, csv, json
from pathlib import Path
TARGETS = ("MRE-", "ACC+", "PRED40+")
CONFIGURATION = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
class ConfigurationFlow:
    def __init__(self):
        self.records_seen = 0; self._sums = {n: 0.0 for n in TARGETS}; self._best = {n: None for n in TARGETS}
    def add(self, result):
        missing = [n for n in CONFIGURATION + TARGETS if n not in result]
        if missing: raise ValueError("missing fields: " + ", ".join(missing))
        values = {n: float(result[n]) for n in TARGETS}; config = {n: result[n] for n in CONFIGURATION}; self.records_seen += 1
        for name, value in values.items():
            self._sums[name] += value; best = self._best[name]
            if best is None or (value < best[name] if name == "MRE-" else value > best[name]): self._best[name] = dict(config, **{name: value})
    def snapshot(self):
        averages = {n: self._sums[n] / self.records_seen for n in TARGETS} if self.records_seen else {}
        return {"records_seen": self.records_seen, "average_metrics": averages, "best_configurations": self._best.copy()}
def read_csv(path):
    flow = ConfigurationFlow()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle): flow.add(row)
    if not flow.records_seen: raise ValueError("dataset contains no evaluation results")
    return flow.snapshot()
def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("dataset", type=Path)
    print(json.dumps(read_csv(parser.parse_args().dataset), sort_keys=True, indent=2))
if __name__ == "__main__": main()
