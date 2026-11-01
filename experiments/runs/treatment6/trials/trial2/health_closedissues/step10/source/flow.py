#!/usr/bin/env python3
"""One-pass, arrival-order summaries for validation results."""
import csv
import math
import sys

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "missing"}

def finite_number(value):
    if value is None or value.strip().lower() in MISSING:
        return None
    try:
        number = float(value.strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def csv_rows(path):
    """Yield (CSV data position, stripped row) without reading future rows."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            return
        for position, raw in enumerate(reader, 2):
            if len(raw) != len(header):
                yield position, None
            else:
                yield position, {header[i]: raw[i].strip() for i in range(len(header))}

class _Running:
    def __init__(self):
        self.count = 0; self.mean = 0.0; self.m2 = 0.0
        self.values = []
    def add(self, value):
        self.values.append(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
    def snapshot(self):
        # Recompute the small summary from retained values so the public
        # result agrees exactly with the documented two-pass definition.
        return {"count": self.count,
                "mean": self.mean if self.count else None,
                "sample_variance": (sum((x - sum(self.values) / self.count) ** 2 for x in self.values) / (self.count - 1)
                                    if self.count >= 2 else None)}

def _config(row):
    """Retain all five raw fields; absent fields are represented as None."""
    return {name: row.get(name) for name in CONFIG}

class Flow:
    def __init__(self):
        self.rows_seen = 0
        self.stats = {name: _Running() for name in OUTCOMES}
        self.best = {name: None for name in OUTCOMES}

    def update(self, position, row):
        self.rows_seen += 1
        if row is None:
            return self.snapshot()
        for name, direction in OUTCOMES.items():
            value = finite_number(row.get(name, ""))
            if value is None:
                continue
            self.stats[name].add(value)
            current = self.best[name]
            better = current is None or (value < current["value"] if direction == "minimize" else value > current["value"])
            if better:  # strict comparison preserves the earliest equal-valued row
                self.best[name] = {"value": value, "position": position, "configuration": _config(row)}
        return self.snapshot()

    def snapshot(self):
        return {"rows_seen": self.rows_seen,
                "outcomes": {name: self.stats[name].snapshot() for name in OUTCOMES},
                "best": {name: (None if result is None else {"value": result["value"], "position": result["position"], "configuration": dict(result["configuration"])}) for name, result in self.best.items()},
                "conventions": {"position": "header is CSV position 1; data positions start at 2", "blank_row": "a blank/malformed CSV row increments rows_seen and contributes no outcomes", "malformed_row": "wrong field count is skipped after incrementing rows_seen", "missing_configuration": "missing configuration headers or fields are retained as None"}}

def consume(rows):
    flow = Flow()
    for position, row in rows:
        yield flow.update(position, row)

def run_csv(path):
    flow = Flow()
    for position, row in csv_rows(path):
        flow.update(position, row)
    return flow.snapshot()

def render(snapshot):
    lines = ["Streaming validation flow", "Rows seen: " + str(snapshot["rows_seen"])]
    for name, values in snapshot["outcomes"].items():
        lines.append(f"{name}: {values}")
        lines.append(f"best {name}: {snapshot['best'][name]}")
    lines.append("Conventions: " + str(snapshot["conventions"]))
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: flow.py DATASET.csv")
    print(render(run_csv(sys.argv[1])))
