#!/usr/bin/env python3
"""Streaming validation summaries for the Health closed-issues sequence."""
import csv, json, math, sys

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
MINIMIZE = {"MRE-"}

def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

class Running:
    def __init__(self, name):
        self.name, self.n, self.mean, self.m2, self.best = name, 0, 0.0, 0.0, None
        self.best_config, self.best_record = None, None
    def add(self, value, config, record_id):
        value = _number(value)
        if value is None:
            return False
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        better = self.best is None or (value < self.best if self.name in MINIMIZE else value > self.best)
        if better:  # ties retain the earliest record in CSV order
            self.best, self.best_config, self.best_record = value, tuple(config), record_id
        return True
    def snapshot(self):
        return {"valid_count": self.n, "mean": self.mean if self.n else None,
                "sample_variance": self.m2 / (self.n - 1) if self.n > 1 else None,
                "best_value": self.best, "best_config": list(self.best_config) if self.best_config else None,
                "best_record_id": self.best_record}

def _copy_state(stats):
    return {name: dict(value) for name, value in stats.items()}

def process_rows(rows):
    """Consume an iterator of dictionaries and yield independent snapshots."""
    stats = {name: Running(name) for name in OUTCOMES}
    for record_id, row in enumerate(rows, 1):
        config = tuple(row.get(field, "") for field in CONFIG)
        for name in OUTCOMES:
            stats[name].add(row.get(name), config, record_id)
        yield record_id, _copy_state({name: stats[name].snapshot() for name in OUTCOMES})

def csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)

def run(path, snapshot_path=None):
    snapshots = []
    for record_id, snapshot in process_rows(csv_rows(path)):
        snapshots.append({"record_id": record_id, "outcomes": snapshot})
    if snapshot_path:
        with open(snapshot_path, "w", encoding="utf-8") as out:
            for item in snapshots: out.write(json.dumps(item, sort_keys=True) + "\n")
    final = snapshots[-1]["outcomes"] if snapshots else {n: Running(n).snapshot() for n in OUTCOMES}
    return snapshots, final

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) not in (1, 2):
        raise SystemExit("usage: flow.py INPUT.csv [SNAPSHOTS.jsonl]")
    snapshots, final = run(argv[0], argv[1] if len(argv) == 2 else None)
    print(json.dumps({"snapshot_count": len(snapshots), "final": final}, sort_keys=True))

if __name__ == "__main__": main()
