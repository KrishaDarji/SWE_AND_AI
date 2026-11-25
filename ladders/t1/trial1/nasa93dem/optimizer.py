#!/usr/bin/env python3
"""Suggest the smallest one-driver change that lowers estimated effort."""
import argparse
import csv
import math
import statistics
from pathlib import Path

DRIVERS = ["prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
           "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
           "apex", "plex", "ltex", "tool", "site", "sced"]
LEVEL_ORDER = ("vl", "l", "n", "h", "vh", "xh")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    for row in rows:
        for key in DRIVERS + ["Effort-", "Kloc+"]:
            if key not in row:
                raise ValueError("missing column: " + key)
    return rows


def _log_mean(values):
    return statistics.fmean(math.log(float(value)) for value in values)


def fit_model(rows):
    """Fit a smoothed additive model in log-effort space.

    Each level effect is its log-effort mean relative to the global mean,
    shrunk toward zero according to the amount of supporting data.
    """
    global_mean = _log_mean([row["Effort-"] for row in rows])
    effects = {}
    for driver in DRIVERS:
        groups = {}
        for row in rows:
            groups.setdefault(row[driver].strip(), []).append(row["Effort-"])
        effects[driver] = {}
        for level, values in groups.items():
            # One pseudo-observation at the global mean prevents tiny groups
            # from dominating the recommendation.
            effects[driver][level] = (len(values) * _log_mean(values) + global_mean) / (len(values) + 1) - global_mean
    return global_mean, effects


def predict(row, model):
    global_mean, effects = model
    value = global_mean + sum(effects[d].get(row[d].strip(), 0.0) for d in DRIVERS)
    return math.exp(value)


def suggest(row, model):
    """Return the best strictly improving one-level change, or None."""
    baseline = predict(row, model)
    candidates = []
    _, effects = model
    for driver in DRIVERS:
        current = row[driver].strip()
        for level in effects[driver]:
            if level == current:
                continue
            candidate = dict(row)
            candidate[driver] = level
            estimate = predict(candidate, model)
            if estimate < baseline:
                candidates.append((estimate, driver, level, current))
    if not candidates:
        return {"change": None, "baseline": baseline, "estimated_effort": baseline}
    estimate, driver, level, current = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {"change": (driver, current, level), "baseline": baseline, "estimated_effort": estimate}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--row", type=int, default=None, help="1-based project id; default is first row")
    args = parser.parse_args(argv)
    rows = load_rows(args.dataset)
    model = fit_model(rows)
    row = rows[0] if args.row is None else next(r for r in rows if int(r["idX"]) == args.row)
    result = suggest(row, model)
    print("project_id=" + row["idX"])
    print("baseline_estimated_effort={:.6f}".format(result["baseline"]))
    if result["change"] is None:
        print("recommendation=none")
    else:
        driver, old, new = result["change"]
        print("recommendation={} {} -> {}".format(driver, old, new))
        print("new_estimated_effort={:.6f}".format(result["estimated_effort"]))


if __name__ == "__main__":
    main()
