#!/usr/bin/env python3
"""Choose the fastest feasible x264 configuration from a CSV measurement file."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def choose_configuration(csv_path, max_performance):
    """Return the row with minimum Performance- not exceeding the constraint."""
    if not math.isfinite(max_performance):
        raise ValueError("max_performance must be finite")
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Performance-" not in reader.fieldnames:
            raise ValueError("CSV must contain a Performance- column")
        best = None
        for line_number, row in enumerate(reader, start=2):
            try:
                value = float(row["Performance-"].strip())
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("invalid Performance- at line %d" % line_number) from exc
            if not math.isfinite(value):
                raise ValueError("non-finite Performance- at line %d" % line_number)
            if value <= max_performance and (best is None or value < best["performance"]):
                best = {"performance": value, "configuration": {
                    key: value for key, value in row.items() if key != "Performance-"
                }}
    if best is None:
        raise ValueError("no configuration satisfies the quality constraint")
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--max-performance", type=float, required=True,
                        help="maximum allowed measured Performance-")
    args = parser.parse_args(argv)
    try:
        result = choose_configuration(args.csv_path, args.max_performance)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
