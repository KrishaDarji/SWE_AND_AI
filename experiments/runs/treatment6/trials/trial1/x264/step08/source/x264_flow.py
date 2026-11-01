"""Incremental flow evaluation for X264 benchmark results."""
import csv
import json
import math
import sys
from copy import deepcopy

TARGET = "Performance-"

def _number(value):
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None

class Flow:
    """One-pass Welford statistics and a defensive best-record snapshot."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.best = None

    def update(self, record_id, configuration, outcome):
        value = _number(outcome)
        if value is None:
            return self.snapshot()
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        if self.best is None or value < self.best["outcome"]:
            self.best = {"record_id": record_id, "outcome": value,
                         "configuration": deepcopy(configuration)}
        return self.snapshot()

    def snapshot(self):
        return {"count": self.count,
                "mean": self.mean if self.count else None,
                "sample_variance": (self.m2 / (self.count - 1)
                                    if self.count > 1 else 0.0),
                "best": deepcopy(self.best),
                "assumptions": ["lower Performance- is better",
                                "CSV order is a demonstration sequence, not a verified timeline"]}

def records(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if TARGET not in fields:
            raise ValueError("dataset must contain Performance-")
        settings = [name for name in fields if name != TARGET]
        for index, row in enumerate(reader):
            yield index, {name: row.get(name, "") for name in settings}, row.get(TARGET, "")

def flow(path):
    state = Flow()
    snapshots = []
    for record_id, configuration, outcome in records(path):
        snapshots.append(state.update(record_id, configuration, outcome))
    return snapshots

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("usage: x264_flow.py DATASET", file=sys.stderr)
        return 2
    print(json.dumps(flow(argv[0]), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
