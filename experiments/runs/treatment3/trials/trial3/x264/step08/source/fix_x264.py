#!/usr/bin/env python3
"""Independently find the smallest measured x264 fix for a slow configuration."""
import csv
import os
import sys


DATASET = "moot/optimize/config/X264_AllMeasurements.csv"
METRIC = "Performance-"


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    fields = list(rows[0])
    if METRIC not in fields:
        raise ValueError("missing Performance- column")
    settings = [name for name in fields if name != METRIC]
    if not settings:
        raise ValueError("no setting columns")
    for number, row in enumerate(rows, 2):
        if set(row) != set(fields):
            raise ValueError("inconsistent columns at CSV row %d" % number)
        for name in settings:
            if row[name] not in {"0", "1"}:
                raise ValueError("non-binary setting %s at CSV row %d" % (name, number))
        try:
            row[METRIC] = float(row[METRIC].strip())
        except ValueError as exc:
            raise ValueError("invalid metric at CSV row %d" % number) from exc
    if len({tuple(row[name] for name in settings) for row in rows}) != len(rows):
        raise ValueError("duplicate configurations")
    return fields, settings, rows


def hamming_distance(left, right, settings):
    return sum(left[name] != right[name] for name in settings)


def find_fix(rows, settings):
    baseline = max(rows, key=lambda row: row[METRIC])
    neighbors = [
        row for row in rows
        if hamming_distance(row, baseline, settings) == 1
        and row[METRIC] < baseline[METRIC]
    ]
    neighbors.sort(key=lambda row: (row[METRIC], tuple(row[name] for name in settings)))
    if not neighbors:
        return baseline, None
    revised = neighbors[0]
    changed = next(name for name in settings if baseline[name] != revised[name])
    return baseline, (changed, revised)


def format_result(dataset, fields, settings, rows):
    baseline, fix = find_fix(rows, settings)
    out = []
    out.append("x264 Step 08 minimal measured fix")
    out.append("dataset=%s" % dataset)
    out.append("rows=%d settings=%d" % (len(rows), len(settings)))
    out.append("baseline_selection=slowest observed configuration (higher Performance- is slower)")
    out.append("baseline_performance=%.3f" % baseline[METRIC])
    out.append("baseline_configuration=" + ",".join("%s=%s" % (n, baseline[n]) for n in settings))
    if fix is None:
        out.append("recommendation=none; no measured one-setting improvement exists")
        out.append("revised_performance=not available")
        out.append("gain=not available")
    else:
        changed, revised = fix
        gain = baseline[METRIC] - revised[METRIC]
        out.append("changed_setting=%s: %s -> %s" % (changed, baseline[changed], revised[changed]))
        out.append("revised_performance=%.3f" % revised[METRIC])
        out.append("gain=%.3f (%.2f%% lower)" % (gain, 100.0 * gain / baseline[METRIC]))
        out.append("revised_configuration=" + ",".join("%s=%s" % (n, revised[n]) for n in settings))
    out.append("assumptions=Performance- is a runtime-like cost to minimize; one-bit neighbors are valid when observed in the dataset")
    out.append("uncertainty=single measurements have no replication/error bars; ties are resolved deterministically")
    out.append("limitations=search is limited to observed one-setting neighbors and does not test interactions or unseen configurations")
    out.append("causation=the comparison is an observed association, not proof that changing the setting alone causes the gain")
    return "\n".join(out) + "\n"


def main(argv=None):
    path = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else DATASET
    fields, settings, rows = load_dataset(path)
    sys.stdout.write(format_result(path, fields, settings, rows))


if __name__ == "__main__":
    main()
