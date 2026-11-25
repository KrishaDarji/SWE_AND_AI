#!/usr/bin/env python3
"""Choose the fastest observed X264 configuration meeting a quality limit."""

import argparse
import csv
import sys


def choose_configuration(path, max_performance=300.0):
    """Return (configuration, performance) for the best eligible CSV row."""
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "Performance-" not in reader.fieldnames:
            raise ValueError("CSV must contain a Performance- column")
        feature_names = [name for name in reader.fieldnames if name != "Performance-"]
        best = None
        for line_number, row in enumerate(reader, start=2):
            try:
                performance = float(row["Performance-"].strip())
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("invalid Performance- at CSV line %d" % line_number) from exc
            if performance <= max_performance and (best is None or performance < best[1]):
                best = ({name: row[name] for name in feature_names}, performance)
    if best is None:
        raise ValueError("no configuration satisfies Performance- <= %.6g" % max_performance)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Choose the lowest-Performance- X264 configuration under a limit."
    )
    parser.add_argument("csv_path")
    parser.add_argument(
        "--max-performance", type=float, default=300.0,
        help="maximum allowed Performance- (default: 300.0)",
    )
    args = parser.parse_args(argv)
    if args.max_performance < 0:
        parser.error("--max-performance must be non-negative")
    try:
        configuration, performance = choose_configuration(args.csv_path, args.max_performance)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("best_performance=%.6f" % performance)
    print("max_performance=%.6f" % args.max_performance)
    print("configuration=" + ",".join("%s=%s" % item for item in configuration.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
