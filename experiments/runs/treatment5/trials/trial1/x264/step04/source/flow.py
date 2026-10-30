"""Streaming benchmark statistics for X264 configurations."""
import csv
import math
import sys
from copy import deepcopy


class Flow:
    """Update-only summary; lower Performance- is the champion."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.best_value = None
        self.best_record_id = None
        self.best_configuration = None

    def update(self, record_id, configuration, outcome):
        try:
            value = float(outcome)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        if self.best_value is None or value < self.best_value:
            self.best_value = value
            self.best_record_id = record_id
            self.best_configuration = deepcopy(configuration)
        return self.snapshot()

    def snapshot(self):
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return deepcopy({"count": self.count, "mean": self.mean,
                         "sample_variance": variance,
                         "best_performance": self.best_value,
                         "best_record_id": self.best_record_id,
                         "best_configuration": self.best_configuration})


def evaluate(records, outcome_key="Performance-"):
    flow = Flow()
    snapshots = []
    for record_id, row in enumerate(records, 1):
        configuration = {k: deepcopy(v) for k, v in row.items() if k != outcome_key}
        snapshot = flow.update(record_id, configuration, row.get(outcome_key))
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def csv_records(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: python flow.py DATASET.csv")
    snapshots = evaluate(csv_records(argv[1]))
    if snapshots:
        print(snapshots[-1])
    else:
        print({"count": 0, "mean": 0.0, "sample_variance": 0.0,
               "best_performance": None, "best_record_id": None,
               "best_configuration": None})


if __name__ == "__main__":
    main(sys.argv)
