#!/usr/bin/env python3
"""Find unusual quality results in a hyperparameter-results CSV."""
import argparse
import csv
import math
import statistics
from collections import defaultdict


QUALITY = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(QUALITY + CONFIG) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        rows = []
        for number, raw in enumerate(reader, 2):
            try:
                values = {name: float(raw[name]) for name in QUALITY}
                values.update({name: raw[name] for name in CONFIG})
                values["_line"] = number
                rows.append(values)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid row at CSV line %d: %s" % (number, exc))
    if not rows:
        raise ValueError("no data rows")
    return rows


def median_abs_deviation(values):
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def inspect(rows):
    findings = []
    for metric in QUALITY:
        values = [row[metric] for row in rows]
        center = statistics.median(values)
        mad = median_abs_deviation(values)
        scale = 1.4826 * mad
        if scale == 0:
            scale = statistics.pstdev(values) or 1.0
        for row in rows:
            score = abs(row[metric] - center) / scale
            if score >= 6:
                findings.append({"line": row["_line"], "kind": "robust_outlier",
                                 "metric": metric, "score": score,
                                 "detail": "value=%.9g median=%.9g" % (row[metric], center)})

    for row in rows:
        if not all(0 <= row[m] <= 1 for m in QUALITY):
            findings.append({"line": row["_line"], "kind": "range_error",
                             "metric": "quality", "score": 0,
                             "detail": "one or more quality values outside [0,1]"})
        if row["ACC+"] > 0.95 and row["PRED40+"] == 0:
            findings.append({"line": row["_line"], "kind": "discordant_quality",
                             "metric": "ACC+/PRED40+", "score": row["ACC+"],
                             "detail": "high ACC+ with zero PRED40+"})

    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in CONFIG)].append(row)
    for key, group in groups.items():
        if len(group) > 1:
            for metric in QUALITY:
                spread = max(r[metric] for r in group) - min(r[metric] for r in group)
                if spread >= 0.25:
                    findings.append({"line": min(r["_line"] for r in group),
                                     "kind": "duplicate_disagreement", "metric": metric,
                                     "score": spread,
                                     "detail": "config repeated %d times; spread=%.6g" % (len(group), spread)})
    return sorted(findings, key=lambda item: (-item["score"], item["line"], item["kind"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    rows = read_rows(args.csv_path)
    findings = inspect(rows)
    print("rows=%d findings=%d" % (len(rows), len(findings)))
    for item in findings[:max(0, args.limit)]:
        print("line=%d kind=%s metric=%s score=%.6g %s" %
              (item["line"], item["kind"], item["metric"], item["score"], item["detail"]))
    if not findings:
        print("No unusual result or possible measurement error detected by these rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
