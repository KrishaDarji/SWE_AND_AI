#!/usr/bin/env python3
"""Compare a driver-aware effort estimator with a historical baseline."""
import argparse
import csv
import math
import statistics

DRIVERS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
           "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
           "apex", "plex", "ltex", "tool", "site", "sced")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("dataset is empty")
    required = set(DRIVERS) | {"Effort-"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    return rows


def fit(rows):
    """Fit smoothed categorical effects in log-effort space."""
    logs = [math.log(float(row["Effort-"])) for row in rows]
    overall = statistics.fmean(logs)
    effects = {}
    for driver in DRIVERS:
        groups = {}
        for row in rows:
            groups.setdefault(row[driver].strip(), []).append(math.log(float(row["Effort-"])))
        effects[driver] = {
            level: (len(values) * statistics.fmean(values) + overall) / (len(values) + 1) - overall
            for level, values in groups.items()
        }
    return overall, effects


def estimate(row, model):
    overall, effects = model
    return math.exp(overall + sum(effects[d].get(row[d].strip(), 0.0) for d in DRIVERS))


def evaluate(rows):
    """Return leave-one-out absolute errors for model and median baseline."""
    model_errors, baseline_errors = [], []
    for index, row in enumerate(rows):
        training = rows[:index] + rows[index + 1:]
        actual = float(row["Effort-"])
        model_errors.append(abs(estimate(row, fit(training)) - actual))
        baseline_errors.append(abs(statistics.median(float(r["Effort-"]) for r in training) - actual))
    return {"projects": len(rows), "model_mae": statistics.fmean(model_errors),
            "baseline_mae": statistics.fmean(baseline_errors),
            "model_wins": sum(a < b for a, b in zip(model_errors, baseline_errors))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    result = evaluate(load_rows(args.dataset))
    print("projects={projects}".format(**result))
    print("model=driver_smoothed_log_effects mae={:.6f}".format(result["model_mae"]))
    print("historical_baseline=leave_one_out_median mae={:.6f}".format(result["baseline_mae"]))
    print("model_wins={}".format(result["model_wins"]))
    print("model_better={}".format(result["model_mae"] < result["baseline_mae"]))


if __name__ == "__main__":
    main()
