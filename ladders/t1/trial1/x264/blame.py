#!/usr/bin/env python3
"""Identify x264 settings associated with slower encodes."""

import argparse
import csv
import statistics
import sys


def read_measurements(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no data rows")
    if not rows[0] or "Performance-" not in rows[0]:
        raise ValueError("dataset must contain a Performance- column")
    return rows


def setting_effect(rows, setting):
    """Return mean-on, mean-off, slowdown, and sample counts for a setting."""
    on = [float(row["Performance-"]) for row in rows if row[setting].strip() == "1"]
    off = [float(row["Performance-"]) for row in rows if row[setting].strip() == "0"]
    if not on or not off:
        return None
    return {
        "setting": setting,
        "mean_on": statistics.fmean(on),
        "mean_off": statistics.fmean(off),
        "slowdown": statistics.fmean(on) - statistics.fmean(off),
        "on_count": len(on),
        "off_count": len(off),
    }


def analyze(rows):
    settings = [name for name in rows[0] if name != "Performance-"]
    results = [setting_effect(rows, name) for name in settings]
    return sorted((item for item in results if item is not None),
                  key=lambda item: item["slowdown"], reverse=True)


def format_report(rows, limit=None):
    results = analyze(rows)
    if limit is not None:
        results = results[:limit]
    lines = [
        "x264 slow-encode blame report",
        "Higher Performance- values mean slower encodes.",
        "Setting             mean-on  mean-off  slowdown  samples(on/off)",
    ]
    for item in results:
        lines.append(
            f"{item['setting']:<18} {item['mean_on']:8.3f} {item['mean_off']:9.3f} "
            f"{item['slowdown']:8.3f}  {item['on_count']}/{item['off_count']}"
        )
    if results:
        worst = results[0]
        lines.append("")
        lines.append(
            f"Most associated with slow encodes: {worst['setting']} "
            f"({worst['slowdown']:.3f} slower on average when enabled)."
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        print(format_report(read_measurements(args.csv_path), args.limit))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
