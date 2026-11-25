"""Streaming benchmark statistics for ordered encoder observations."""
import csv
import math
from copy import deepcopy


class Flow:
    """Incremental count/mean/sample variance and minimum Performance-."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.best = None

    def update(self, record, record_id):
        row = deepcopy(dict(record))
        raw = row.get("Performance-")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self.snapshot()
        if not math.isfinite(value):
            return self.snapshot()
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)
        if self.best is None or value < self.best["value"]:
            self.best = {"record_id": record_id, "value": value,
                         "configuration": deepcopy(row)}
        return self.snapshot()

    def snapshot(self):
        return {"count": self.count, "mean": self.mean,
                "sample_variance": self._m2 / (self.count - 1)
                if self.count > 1 else None,
                "best": deepcopy(self.best)}


def process(records):
    flow = Flow()
    for record_id, record in enumerate(records, 1):
        yield record_id, flow.update(record, record_id)


def csv_records(path):
    with open(path, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def main(path):
    for record_id, snapshot in process(csv_records(path)):
        print("record_id={}, count={}, mean={:.6f}".format(
            record_id, snapshot["count"], snapshot["mean"]))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: stream_stats.py DATASET.csv")
    main(sys.argv[1])
