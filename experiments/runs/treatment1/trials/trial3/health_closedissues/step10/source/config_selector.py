#!/usr/bin/env python3
"""Select a model configuration and compare it with an ACC-only baseline."""
import argparse
import csv
import math
import os
import time

METRICS = ("MRE-", "ACC+", "PRED40+")


def read_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(METRICS)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("dataset is missing required outcome columns")
        rows = []
        for index, row in enumerate(reader):
            try:
                values = {name: float(row[name]) for name in METRICS}
            except (TypeError, ValueError) as exc:
                raise ValueError("non-numeric metric at row %d" % (index + 2)) from exc
            row["_index"] = index
            row["_values"] = values
            rows.append(row)
    if len(rows) < 4:
        raise ValueError("at least four rows are required")
    return rows


def utility(row, bounds=None):
    values = row["_values"]
    if bounds is None:
        return (values["ACC+"] + values["PRED40+"] + (1.0 - values["MRE-"])) / 3.0
    total = 0.0
    for name in METRICS:
        low, high = bounds[name]
        span = high - low
        scaled = 0.5 if span == 0 else (values[name] - low) / span
        total += (1.0 - scaled) if name == "MRE-" else scaled
    return total / 3.0


def bounds(rows):
    return {name: (min(r["_values"][name] for r in rows), max(r["_values"][name] for r in rows)) for name in METRICS}


def select(rows):
    train_size = int(len(rows) * 0.7)
    train, evaluation = rows[:train_size], rows[train_size:]
    scale = bounds(train)
    selected = max(train, key=lambda r: (utility(r, scale), -r["_index"]))
    baseline = max(train, key=lambda r: (r["_values"]["ACC+"], -r["_index"]))
    # The protocol applies the same evaluation utility to both selected rows.
    return train, evaluation, selected, baseline, scale


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    rows = read_rows(args.dataset)
    train, evaluation, selected, baseline, scale = select(rows)
    selector_score = utility(selected, scale)
    baseline_score = utility(baseline, scale)
    print("rows=%d train=%d holdout=%d" % (len(rows), len(train), len(evaluation)))
    print("selector_index=%d selector_utility=%.9f" % (selected["_index"] + 1, selector_score))
    print("baseline_index=%d baseline_utility=%.9f" % (baseline["_index"] + 1, baseline_score))
    print("selector_minus_baseline=%.9f" % (selector_score - baseline_score))
    print("runtime_seconds=%.6f" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
