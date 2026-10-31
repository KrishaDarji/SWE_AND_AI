"""Streaming benchmark-result evaluator for lower-is-better Performance-."""
import csv
import math
from copy import deepcopy


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class RunningStats:
    """One-pass count, mean, and sample variance (Welford accumulation)."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, value):
        number = _finite(value)
        if number is None:
            return False
        self.count += 1
        delta = number - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (number - self.mean)
        return True

    @property
    def sample_variance(self):
        return self._m2 / (self.count - 1) if self.count > 1 else None


class FlowEvaluator:
    """Consumes result records in arrival order and emits immutable snapshots."""
    def __init__(self, outcome_key="Performance-"):
        self.outcome_key = outcome_key
        self.stats = RunningStats()
        self.best_value = None
        self.best_record_id = None
        self.best_configuration = None
        self._next_id = 1
        self.snapshots = []

    def update(self, record):
        current_id = self._next_id
        self._next_id += 1
        copied = deepcopy(dict(record))
        outcome = _finite(copied.get(self.outcome_key))
        accepted = self.stats.add(outcome)
        if accepted and (self.best_value is None or outcome < self.best_value):
            self.best_value = outcome
            self.best_record_id = current_id
            self.best_configuration = deepcopy(copied)
        snapshot = {
            "record_id": current_id,
            "observations_received": current_id,
            "valid_observations": self.stats.count,
            "mean": self.stats.mean if self.stats.count else None,
            "sample_variance": self.stats.sample_variance,
            "best_value": self.best_value,
            "best_record_id": self.best_record_id,
            "best_configuration": deepcopy(self.best_configuration),
            "accepted": accepted,
        }
        self.snapshots.append(deepcopy(snapshot))
        return deepcopy(snapshot)

    def process(self, records):
        for record in records:
            yield self.update(record)


def csv_records(path):
    """Yield dictionaries one row at a time; the CSV order is preserved."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield row


def evaluate_csv(path):
    evaluator = FlowEvaluator()
    for snapshot in evaluator.process(csv_records(path)):
        pass
    return evaluator


def main(path):
    evaluator = evaluate_csv(path)
    print("observations_received={}".format(len(evaluator.snapshots)))
    print("valid_observations={}".format(evaluator.stats.count))
    print("mean={}".format(evaluator.stats.mean if evaluator.stats.count else None))
    print("sample_variance={}".format(evaluator.stats.sample_variance))
    print("best_record_id={}".format(evaluator.best_record_id))
    print("best_value={}".format(evaluator.best_value))
    print("best_configuration={}".format(evaluator.best_configuration))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
