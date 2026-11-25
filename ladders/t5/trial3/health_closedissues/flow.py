"""Streaming validation summaries for the Health closed-issues configuration stream."""
import csv
import math

CONFIG_FIELDS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _copy_config(row):
    return {field: row.get(field) for field in CONFIG_FIELDS}


class RunningStat:
    """One-pass count/mean/sample-variance accumulator (Welford update)."""
    def __init__(self, direction):
        self.direction = direction
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.best = None
        self.best_record_id = None
        self.best_configuration = None

    def update(self, value, record_id, configuration):
        value = _finite(value)
        if value is None:
            return False
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        better = self.best is None or (value < self.best if self.direction == "min" else value > self.best)
        if better:  # ties retain the earliest record deterministically
            self.best = value
            self.best_record_id = record_id
            self.best_configuration = dict(configuration)
        return True

    def snapshot(self):
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return {"count": self.count, "mean": self.mean, "sample_variance": variance,
                "best": self.best, "best_record_id": self.best_record_id,
                "best_configuration": None if self.best_configuration is None else dict(self.best_configuration)}


class StreamEvaluator:
    def __init__(self):
        self.stats = {name: RunningStat("min" if name == "MRE-" else "max") for name in OUTCOMES}
        self.snapshots = []
        self.received = 0

    def receive(self, row, record_id=None):
        if record_id is None:
            record_id = self.received + 1
        configuration = _copy_config(row)
        self.received += 1
        for outcome in OUTCOMES:
            self.stats[outcome].update(row.get(outcome), record_id, configuration)
        self.snapshots.append({"record_id": record_id, "received": self.received,
                               "outcomes": {key: stat.snapshot() for key, stat in self.stats.items()}})
        return self.snapshots[-1]

    def final(self):
        return {key: stat.snapshot() for key, stat in self.stats.items()}


def rows_from_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield row


def evaluate_csv(path):
    evaluator = StreamEvaluator()
    for record_id, row in enumerate(rows_from_csv(path), 1):
        evaluator.receive(row, record_id)
    return evaluator
