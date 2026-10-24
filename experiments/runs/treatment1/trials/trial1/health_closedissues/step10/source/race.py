#!/usr/bin/env python3
"""Compare an adaptive historical median with an expanding-history baseline."""
import argparse
import csv
import math
import statistics


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values):
    return statistics.median(values)


def evaluate(path, window=20, split=0.5, columns=None):
    """Evaluate sequential forecasts without using future observations.

    The system uses the most recent ``window`` observations. The baseline uses
    every observation seen so far. Rows with invalid target values are skipped.
    """
    if window < 1 or not 0 < split < 1:
        raise ValueError("window must be positive and split must be between 0 and 1")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError("CSV has no data rows or header")
    names = list(rows[0])
    columns = columns or [name for name in names if name.endswith(("-", "+"))]
    missing = [name for name in columns if name not in names]
    if missing:
        raise ValueError("unknown column(s): " + ", ".join(missing))
    cut = max(1, min(len(rows) - 1, int(len(rows) * split)))
    result = {"rows": len(rows), "split_row": cut + 2, "columns": {}}
    for column in columns:
        history = []
        adaptive_errors = []
        baseline_errors = []
        adaptive_wins = baseline_wins = ties = invalid = 0
        for index, row in enumerate(rows):
            actual = _number(row.get(column))
            if index >= cut:
                if actual is None or not history:
                    invalid += 1
                    continue
                adaptive = _median(history[-window:])
                baseline = _median(history)
                ae, be = abs(actual - adaptive), abs(actual - baseline)
                adaptive_errors.append(ae)
                baseline_errors.append(be)
                if ae < be:
                    adaptive_wins += 1
                elif be < ae:
                    baseline_wins += 1
                else:
                    ties += 1
            if actual is not None:
                history.append(actual)
        if not adaptive_errors:
            raise ValueError("no valid evaluation rows for " + column)
        result["columns"][column] = {
            "evaluated": len(adaptive_errors),
            "invalid": invalid,
            "adaptive_mae": statistics.fmean(adaptive_errors),
            "baseline_mae": statistics.fmean(baseline_errors),
            "adaptive_wins": adaptive_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    report = evaluate(args.csv_path, args.window)
    print("rows={} split_row={}".format(report["rows"], report["split_row"]))
    for name, item in report["columns"].items():
        print("{} evaluated={} adaptive_mae={:.6g} baseline_mae={:.6g} wins={}/{} ties={}".format(
            name, item["evaluated"], item["adaptive_mae"], item["baseline_mae"],
            item["adaptive_wins"], item["baseline_wins"], item["ties"]))


if __name__ == "__main__":
    main()
