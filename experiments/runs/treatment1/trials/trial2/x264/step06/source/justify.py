#!/usr/bin/env python3
"""Recommend an observed x264 configuration and justify it from measurements."""

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Summary:
    values: tuple

    @property
    def count(self):
        return len(self.values)

    @property
    def mean(self):
        return statistics.fmean(self.values)

    @property
    def minimum(self):
        return min(self.values)

    @property
    def maximum(self):
        return max(self.values)

    @property
    def stdev(self):
        return statistics.stdev(self.values) if self.count > 1 else 0.0


def load_measurements(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError("CSV must contain a header and at least one data row")
    header = [cell.strip() for cell in rows[0]]
    if "Performance-" not in header:
        raise ValueError("CSV is missing the Performance- column")
    score_index = header.index("Performance-")
    config_columns = header[:score_index]
    if not config_columns:
        raise ValueError("CSV has no configuration columns")
    grouped = {}
    for line_number, row in enumerate(rows[1:], 2):
        if len(row) != len(header):
            raise ValueError("row {} has {} fields; expected {}".format(line_number, len(row), len(header)))
        try:
            key = tuple(int(row[i].strip()) for i in range(score_index))
            score = float(row[score_index].strip())
        except ValueError as exc:
            raise ValueError("invalid numeric value on row {}".format(line_number)) from exc
        if any(bit not in (0, 1) for bit in key):
            raise ValueError("configuration values must be 0 or 1 on row {}".format(line_number))
        if not math.isfinite(score):
            raise ValueError("performance must be finite on row {}".format(line_number))
        grouped.setdefault(key, []).append(score)
    return config_columns, {key: Summary(tuple(values)) for key, values in grouped.items()}


def recommend(path):
    columns, summaries = load_measurements(path)
    # The dataset labels the objective Performance; higher is therefore preferred.
    key, summary = max(summaries.items(), key=lambda item: (item[1].mean, item[1].minimum, item[0]))
    ordered = sorted(summaries.values(), key=lambda item: item.mean, reverse=True)
    rank = 1 + sum(item.mean > summary.mean for item in ordered)
    return columns, key, summary, len(ordered), rank


def render(path):
    columns, key, summary, observed_count, rank = recommend(path)
    enabled = [name for name, bit in zip(columns, key) if bit]
    disabled = [name for name, bit in zip(columns, key) if not bit]
    lines = [
        "x264 configuration recommendation",
        "Dataset: {}".format(path),
        "Recommendation: {}".format("all defaults (no flags enabled)" if not enabled else ", ".join(enabled)),
        "Configuration bits: {}".format(" ".join(str(bit) for bit in key)),
        "Evidence: mean Performance- {:.3f}; range {:.3f}-{:.3f}; standard deviation {:.3f}; n={}".format(
            summary.mean, summary.minimum, summary.maximum, summary.stdev, summary.count),
        "Decision: rank {}/{} among {} observed configurations by mean Performance- (higher is better).".format(
            rank, observed_count, observed_count),
        "Administrator justification: deploy this configuration as the measured performance leader, subject to a representative validation run and a rollback threshold.",
        "Disabled options: {}".format(", ".join(disabled)),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        sys.stdout.write(render(args.csv_path))
    except (OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
