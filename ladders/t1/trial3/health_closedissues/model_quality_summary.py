#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings in a CSV dataset."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter


QUALITY_COLUMNS = ("MRE-", "ACC+", "PRED40+")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_summary(values):
    result = {"count": len(values)}
    if not values:
        return result
    result.update(
        minimum=min(values),
        maximum=max(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
    )
    if len(values) > 1:
        result["stdev"] = statistics.stdev(values)
    return result


def summarize(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fields = list(reader.fieldnames)
        missing = [name for name in QUALITY_COLUMNS if name not in fields]
        if missing:
            raise ValueError("missing quality columns: " + ", ".join(missing))
        rows = list(reader)

    quality = {}
    for name in QUALITY_COLUMNS:
        values = [_number(row.get(name)) for row in rows]
        usable = [value for value in values if value is not None]
        quality[name] = _numeric_summary(usable)

    configurations = {}
    for name in fields:
        if name in QUALITY_COLUMNS:
            continue
        counts = Counter((row.get(name) or "").strip() for row in rows)
        configurations[name] = {
            "unique": len(counts),
            "values": dict(sorted(counts.items())),
        }

    return {
        "dataset": path,
        "rows": len(rows),
        "quality_metrics": quality,
        "configuration_settings": configurations,
    }


def render_text(summary):
    lines = ["Model quality metrics", "======================"]
    for name, stats in summary["quality_metrics"].items():
        formatted = ", ".join(
            key + "=" + (str(value) if key == "count" else f"{value:.6g}")
            for key, value in stats.items()
        )
        lines.append(f"{name}: {formatted}")
    lines += ["", "Configuration settings", "======================="]
    for name, details in summary["configuration_settings"].items():
        values = ", ".join(f"{key}={value}" for key, value in details["values"].items())
        lines.append(f"{name} (unique={details['unique']}): {values}")
    lines.append(f"Rows: {summary['rows']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.csv_path)
    except (OSError, ValueError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
